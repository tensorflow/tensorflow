/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/stream_executor/cuda/cuda_command_buffer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/command_buffer_kernels.h"
#include "xla/stream_executor/cuda/cuda_context.h"
#include "xla/stream_executor/cuda/cuda_kernel.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/scoped_update_mode.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"  // IWYU pragma: keep
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {
absl::StatusOr<CUgraph> CreateGraph() {
  CUgraph graph = nullptr;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuGraphCreate(&graph, /*flags=*/0),
                                    "Failed to create CUDA graph"));
  VLOG(2) << "Create CUDA graph " << graph;
  return graph;
}

CUdeviceptr AsDevicePtr(const DeviceMemoryBase& mem) {
  return absl::bit_cast<CUdeviceptr>(mem.opaque());
}

using NodeHandle = CommandBuffer::NodeHandle;
using ConditionalHandle = CommandBuffer::ConditionalHandle;

// Converts a platform independent NodeHandle into a CUDA specific
// CUgraphNode.
CUgraphNode ToCudaGraphHandle(NodeHandle handle) {
  return absl::bit_cast<CUgraphNode>(handle);
}

// Converts a platform independent ConditionalHandle into a CUDA
// specific CUgraphConditionalHandle.
CUgraphConditionalHandle ToCudaGraphHandle(ConditionalHandle handle) {
  return absl::bit_cast<CUgraphConditionalHandle>(handle);
}

// Converts a list of platform independent std::vector<NodeHandle>
// into a list of CUDA specific CUgraphNode.
std::vector<CUgraphNode> ToCudaGraphHandles(
    absl::Span<const NodeHandle> opaque_handles) {
  std::vector<CUgraphNode> handles;
  for (const NodeHandle opaque_handle : opaque_handles) {
    handles.push_back(ToCudaGraphHandle(opaque_handle));
  }
  return handles;
}

// Converts a CUDA specific CUgraphNode into a platform independent
// NodeHandle.
NodeHandle FromCudaGraphHandle(CUgraphNode handle) {
  return absl::bit_cast<NodeHandle>(handle);
}

// Converts a CUDA specific CUgraphConditionalHandle into a platform
// independent ConditionalHandle.
ConditionalHandle FromCudaGraphHandle(CUgraphConditionalHandle handle) {
  return absl::bit_cast<ConditionalHandle>(handle);
}

std::string ConditionalTypeToString(CommandBuffer::ConditionType type) {
  switch (type) {
    case CommandBuffer::ConditionType::kIf:
      return "IF";
    case CommandBuffer::ConditionType::kWhile:
      return "WHILE";
  }
}

absl::Status GraphInstantiate(CUgraphExec* exec, CUgraph graph) {
  VLOG(2) << "Instantiate CUDA executable graph from graph " << graph;

#if CUDA_VERSION >= 12000
  uint64_t cu_flags = 0;
  return cuda::ToStatus(cuGraphInstantiate(exec, graph, cu_flags),
                        "Failed to instantiate CUDA graph");
#else
  return cuda::ToStatus(cuGraphInstantiate(exec, graph, nullptr, nullptr, 0),
                        "Failed to instantiate CUDA graph");
#endif  // CUDA_VERSION >= 12000
}

}  // namespace

absl::StatusOr<std::unique_ptr<CudaCommandBuffer>> CudaCommandBuffer::Create(
    Mode mode, StreamExecutor* stream_executor, CudaContext* cuda_context) {
  TF_ASSIGN_OR_RETURN(CUgraph graph, CreateGraph());
  return std::unique_ptr<CudaCommandBuffer>(
      new CudaCommandBuffer(mode, stream_executor, cuda_context, graph,
                            /*is_owned_graph=*/true));
}

absl::StatusOr<NodeHandle> CudaCommandBuffer::CreateLaunchNode(
    Dependencies dependencies, const ThreadDim& threads, const BlockDim& blocks,
    const Kernel& kernel, const KernelArgs& args) {
  if (finalized_) {
    return absl::InternalError(
        "Trying to create launch node in a finalized command buffer");
  }

  // If arguments are already packed we can just launch the kernel.
  if (auto* packed = DynCast<KernelArgsPackedArrayBase>(&args)) {
    return CreateKernelNode(dependencies, threads, blocks, kernel, *packed);
  }

  // For device memory array we rely on a custom kernel arguments packing.
  if (auto* device_mem = DynCast<KernelArgsDeviceMemoryArray>(&args)) {
    auto& pack = kernel.args_packing();
    if (!pack) {
      return absl::InternalError(
          "Kernel is missing a custom arguments packing function for device "
          "memory arguments array");
    }

    TF_ASSIGN_OR_RETURN(auto packed, pack(kernel, *device_mem));
    return CreateKernelNode(dependencies, threads, blocks, kernel, *packed);
  }

  return absl::InternalError("Unsupported kernel arguments type");
}

absl::Status CudaCommandBuffer::UpdateLaunchNode(NodeHandle node,
                                                 const ThreadDim& threads,
                                                 const BlockDim& blocks,
                                                 const Kernel& kernel,
                                                 const KernelArgs& args) {
  CHECK(finalized_)
      << "Trying to update launch node when command buffer is not finalized";

  // If arguments are already packed we can just launch the kernel.
  if (auto* packed = DynCast<KernelArgsPackedArrayBase>(&args)) {
    return UpdateKernelNode(node, threads, blocks, kernel, *packed);
  }

  // For device memory array we rely on a custom kernel arguments packing.
  if (auto* device_mem = DynCast<KernelArgsDeviceMemoryArray>(&args)) {
    auto& pack = kernel.args_packing();
    if (!pack) {
      return absl::InternalError(
          "Kernel is missing a custom arguments packing function for device "
          "memory arguments array");
    }

    TF_ASSIGN_OR_RETURN(auto packed, pack(kernel, *device_mem));
    return UpdateKernelNode(node, threads, blocks, kernel, *packed);
  }
  return absl::InternalError("Unsupported kernel arguments type");
}

absl::StatusOr<NodeHandle> CudaCommandBuffer::CreateIfElseConditionNode(
    Dependencies dependencies, ConditionalHandle then_conditional,
    ConditionalHandle else_conditional, DeviceMemory<bool> predicate) {
  if (!set_if_else_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec,
                        cuda::GetSetIfElseConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_if_else_condition_kernel_,
        SetIfElseConditionKernel::FactoryType::Create(stream_executor_, spec));
  }
  return CreateTypedLaunchNode(dependencies, ThreadDim(), BlockDim(),
                               set_if_else_condition_kernel_,
                               ToCudaGraphHandle(then_conditional),
                               ToCudaGraphHandle(else_conditional), predicate);
}

absl::Status CudaCommandBuffer::UpdateIfElseConditionNode(
    NodeHandle node, ConditionalHandle then_conditional,
    ConditionalHandle else_conditional, DeviceMemory<bool> predicate) {
  CHECK(set_if_else_condition_kernel_)
      << "SetIfElseConditionKernel is not initialized";
  return UpdateTypedLaunchNode(node, ThreadDim(), BlockDim(),
                               set_if_else_condition_kernel_,
                               ToCudaGraphHandle(then_conditional),
                               ToCudaGraphHandle(else_conditional), predicate);
}

absl::StatusOr<NodeHandle> CudaCommandBuffer::CreateIfConditionNode(
    Dependencies dependencies, ConditionalHandle then_conditional,
    DeviceMemory<bool> predicate) {
  if (!set_if_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec, cuda::GetSetIfConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_if_condition_kernel_,
        SetIfConditionKernel::FactoryType::Create(stream_executor_, spec));
  }
  return CreateTypedLaunchNode(dependencies, ThreadDim(), BlockDim(),
                               set_if_condition_kernel_,
                               ToCudaGraphHandle(then_conditional), predicate);
}

absl::Status CudaCommandBuffer::UpdateIfConditionNode(
    NodeHandle node, ConditionalHandle then_conditional,
    DeviceMemory<bool> predicate) {
  CHECK(set_if_condition_kernel_) << "SetIfConditionKernel is not initialized";
  return UpdateTypedLaunchNode(node, ThreadDim(), BlockDim(),
                               set_if_condition_kernel_,
                               ToCudaGraphHandle(then_conditional), predicate);
}

absl::StatusOr<NodeHandle> CudaCommandBuffer::CreateForConditionNode(
    Dependencies dependencies, ConditionalHandle condition,
    DeviceMemory<int32_t> loop_counter, int32_t iterations) {
  if (!set_for_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec, cuda::GetSetForConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_for_condition_kernel_,
        SetForConditionKernel::FactoryType::Create(stream_executor_, spec));
  }
  return CreateTypedLaunchNode(
      dependencies, ThreadDim(), BlockDim(), set_for_condition_kernel_,
      ToCudaGraphHandle(condition), loop_counter, iterations);
}

absl::Status CudaCommandBuffer::UpdateForConditionNode(
    NodeHandle node, ConditionalHandle condition,
    DeviceMemory<int32_t> loop_counter, int32_t iterations) {
  CHECK(set_for_condition_kernel_)
      << "SetForConditionKernel is not initialized";
  return UpdateTypedLaunchNode(
      node, ThreadDim(), BlockDim(), set_for_condition_kernel_,
      ToCudaGraphHandle(condition), loop_counter, iterations);
}

absl::StatusOr<NodeHandle> CudaCommandBuffer::CreateWhileConditionNode(
    Dependencies dependencies, ConditionalHandle condition,
    DeviceMemory<bool> predicate) {
  if (!set_while_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec,
                        cuda::GetSetWhileConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_while_condition_kernel_,
        SetWhileConditionKernel::FactoryType::Create(stream_executor_, spec));
  }
  return CreateTypedLaunchNode(dependencies, ThreadDim(), BlockDim(),
                               set_while_condition_kernel_,
                               ToCudaGraphHandle(condition), predicate);
}

absl::Status CudaCommandBuffer::UpdateWhileConditionNode(
    NodeHandle node, ConditionalHandle condition,
    DeviceMemory<bool> predicate) {
  CHECK(set_while_condition_kernel_)
      << "SetWhileConditionKernel is not initialized";
  return UpdateTypedLaunchNode(node, ThreadDim(), BlockDim(),
                               set_while_condition_kernel_,
                               ToCudaGraphHandle(condition), predicate);
}

absl::StatusOr<NodeHandle> CudaCommandBuffer::CreateCaseConditionNode(
    Dependencies dependencies, std::array<ConditionalHandle, 8> conditions,
    DeviceMemory<uint8_t> index, bool index_is_bool, int32_t batch_offset,
    int32_t num_branches, bool enable_conditional_default) {
  if (!set_case_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec, cuda::GetSetCaseConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_case_condition_kernel_,
        SetCaseConditionKernel::FactoryType::Create(stream_executor_, spec));
  }

  return CreateTypedLaunchNode(
      dependencies, ThreadDim(), BlockDim(), set_case_condition_kernel_,
      ToCudaGraphHandle(conditions[0]), ToCudaGraphHandle(conditions[1]),
      ToCudaGraphHandle(conditions[2]), ToCudaGraphHandle(conditions[3]),
      ToCudaGraphHandle(conditions[4]), ToCudaGraphHandle(conditions[5]),
      ToCudaGraphHandle(conditions[6]), ToCudaGraphHandle(conditions[7]), index,
      index_is_bool, batch_offset, num_branches, enable_conditional_default);
}

absl::Status CudaCommandBuffer::UpdateCaseConditionNode(
    NodeHandle node, std::array<ConditionalHandle, 8> conditions,
    DeviceMemory<uint8_t> index, bool index_is_bool, int32_t batch_offset,
    int32_t num_branches, bool enable_conditional_default) {
  CHECK(set_case_condition_kernel_)
      << "SetCaseConditionKernel is not initialized";
  return UpdateTypedLaunchNode(
      node, ThreadDim(), BlockDim(), set_case_condition_kernel_,
      ToCudaGraphHandle(conditions[0]), ToCudaGraphHandle(conditions[1]),
      ToCudaGraphHandle(conditions[2]), ToCudaGraphHandle(conditions[3]),
      ToCudaGraphHandle(conditions[4]), ToCudaGraphHandle(conditions[5]),
      ToCudaGraphHandle(conditions[6]), ToCudaGraphHandle(conditions[7]), index,
      index_is_bool, batch_offset, num_branches, enable_conditional_default);
}

absl::StatusOr<CudaCommandBuffer::NoOpKernel*>
CudaCommandBuffer::GetNoOpKernel() {
  if (!noop_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec, cuda::GetNoOpKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        noop_kernel_, NoOpKernel::FactoryType::Create(stream_executor_, spec));
  }
  return &noop_kernel_;
}

absl::StatusOr<CommandBuffer::ConditionalNodeResult>
CudaCommandBuffer::CreateConditionalNode(Dependencies dependencies,
                                         ConditionalHandle conditional,
                                         CommandBuffer::ConditionType type) {
#if CUDA_VERSION >= 12030
  // Add conditional node to a graph.
  VLOG(2) << "Add conditional node to a graph " << graph_
          << "; type: " << ConditionalTypeToString(type);

  CUgraphNodeParams cu_params;
  std::memset(&cu_params, 0, sizeof(cu_params));

  cu_params.type = CU_GRAPH_NODE_TYPE_CONDITIONAL;
  cu_params.conditional.handle = ToCudaGraphHandle(conditional);
  cu_params.conditional.ctx = cuda_context_->context();
  cu_params.conditional.size = 1;

  switch (type) {
    case CommandBuffer::ConditionType::kIf:
      cu_params.conditional.type = CU_GRAPH_COND_TYPE_IF;
      break;
    case CommandBuffer::ConditionType::kWhile:
      cu_params.conditional.type = CU_GRAPH_COND_TYPE_WHILE;
      break;
  }

  std::vector<CUgraphNode> deps = ToCudaGraphHandles(dependencies);
  CUgraphNode node_handle = nullptr;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuGraphAddNode(&node_handle, graph_, deps.data(),
                                    deps.size(), &cu_params),
                     "Failed to add conditional node to a CUDA graph"));

  VLOG(2) << "Created conditional CUDA graph "
          << cu_params.conditional.phGraph_out[0];

  return ConditionalNodeResult{
      FromCudaGraphHandle(node_handle),
      std::unique_ptr<CudaCommandBuffer>(
          new CudaCommandBuffer(Mode::kNested, stream_executor_, cuda_context_,
                                cu_params.conditional.phGraph_out[0],
                                /*is_owned_graph=*/false))};
#else
  return absl::UnimplementedError("unsupported node type");
#endif  // CUDA_VERSION >= 12030
}

absl::StatusOr<NodeHandle> CudaCommandBuffer::CreateMemsetNode(
    Dependencies dependencies, DeviceMemoryBase destination,
    BitPattern bit_pattern, size_t num_elements) {
  VLOG(2) << "Add memset node to a graph " << graph_
          << "; dst: " << destination.opaque()
          << "; bit_pattern: " << bit_pattern.ToString()
          << "; num_elements: " << num_elements
          << "; dependencies: " << GraphNodeHandlesToString(dependencies)
          << "; context: " << cuda_context_->context();

  CUDA_MEMSET_NODE_PARAMS params{};
  params.dst = AsDevicePtr(destination);
  params.elementSize = bit_pattern.GetElementSize();
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = bit_pattern.GetPatternBroadcastedToUint32();
  params.width = num_elements;

  std::vector<CUgraphNode> deps = ToCudaGraphHandles(dependencies);

  CUgraphNode node_handle = nullptr;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuGraphAddMemsetNode(&node_handle, graph_, deps.data(), deps.size(),
                           &params, cuda_context_->context()),
      "Failed to add memset node to a CUDA graph"));

  return FromCudaGraphHandle(node_handle);
}

absl::Status CudaCommandBuffer::UpdateMemsetNode(NodeHandle node_handle,
                                                 DeviceMemoryBase destination,
                                                 BitPattern bit_pattern,
                                                 size_t num_elements) {
  VLOG(2) << "Set memset node params " << node_handle << " in graph executable "
          << exec_ << "; dst: " << destination.opaque()
          << "; bit_pattern: " << bit_pattern.ToString()
          << "; num_elements: " << num_elements
          << "; context: " << cuda_context_->context();

  CUDA_MEMSET_NODE_PARAMS params{};
  params.dst = AsDevicePtr(destination);
  params.elementSize = bit_pattern.GetElementSize();
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = bit_pattern.GetPatternBroadcastedToUint32();
  params.width = num_elements;

  return cuda::ToStatus(
      cuGraphMemsetNodeSetParams(ToCudaGraphHandle(node_handle), &params),
      "Failed to set memset node params");
}

absl::StatusOr<NodeHandle> CudaCommandBuffer::CreateMemcpyD2DNode(
    Dependencies dependencies, DeviceMemoryBase destination,
    DeviceMemoryBase source, uint64_t size) {
  VLOG(2) << "Add memcpy d2d node to a graph " << graph_
          << "; dependencies: " << GraphNodeHandlesToString(dependencies)
          << "; dst: " << destination.opaque() << "; src: " << source.opaque()
          << "; size: " << size << "; context: " << cuda_context_->context();

  CUDA_MEMCPY3D params{};
  params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  params.srcDevice = AsDevicePtr(source);
  params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  params.dstDevice = AsDevicePtr(destination);
  params.WidthInBytes = size;
  params.Height = 1;
  params.Depth = 1;

  std::vector<CUgraphNode> deps = ToCudaGraphHandles(dependencies);

  CUgraphNode node_handle = nullptr;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuGraphAddMemcpyNode(&node_handle, graph_, deps.data(), deps.size(),
                           &params, cuda_context_->context()),
      "Failed to add memcpy d2d node to a CUDA graph"));
  return FromCudaGraphHandle(node_handle);
}

absl::Status CudaCommandBuffer::UpdateMemcpyD2DNode(
    NodeHandle node_handle, DeviceMemoryBase destination,
    DeviceMemoryBase source, uint64_t size) {
  VLOG(2) << "Set memcpy d2d node params " << node_handle
          << " in graph executable " << exec_
          << "; dst: " << destination.opaque() << "; src: " << source.opaque()
          << "; size: " << size << "; context: " << cuda_context_->context();

  CUDA_MEMCPY3D params{};
  params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  params.srcDevice = AsDevicePtr(source);
  params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  params.dstDevice = AsDevicePtr(destination);
  params.WidthInBytes = size;
  params.Height = 1;
  params.Depth = 1;

  return cuda::ToStatus(
      cuGraphMemcpyNodeSetParams(ToCudaGraphHandle(node_handle), &params),
      "Failed to set memcpy d2d node params");
}

absl::StatusOr<NodeHandle> CudaCommandBuffer::CreateChildNode(
    Dependencies dependencies, const CommandBuffer& nested) {
  const CUgraph& child_graph =
      tensorflow::down_cast<const CudaCommandBuffer&>(nested).graph_;
  VLOG(2) << "Create a new node by cloning the child graph " << child_graph
          << " and add it to " << graph_
          << "; dependencies: " << GraphNodeHandlesToString(dependencies);

  std::vector<CUgraphNode> deps = ToCudaGraphHandles(dependencies);

  CUgraphNode node_handle;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuGraphAddChildGraphNode(&node_handle, graph_, deps.data(), deps.size(),
                               child_graph),
      "Failed to create a child graph node and add it to a CUDA graph"));

  return FromCudaGraphHandle(node_handle);
}

absl::Status CudaCommandBuffer::UpdateChildNode(NodeHandle node_handle,
                                                const CommandBuffer& nested) {
  CUgraph child_graph =
      tensorflow::down_cast<const CudaCommandBuffer&>(nested).graph_;
  VLOG(2) << "Set child node params " << node_handle << " in graph executable "
          << exec_ << " to params contained in " << child_graph;

  CUgraphNodeParams cu_params;
  std::memset(&cu_params, 0, sizeof(cu_params));

  cu_params.type = CU_GRAPH_NODE_TYPE_GRAPH;
  cu_params.graph.graph = child_graph;

  return cuda::ToStatus(
      cuGraphNodeSetParams(ToCudaGraphHandle(node_handle), &cu_params),
      "Failed to set CUDA graph child node params");
}

absl::StatusOr<NodeHandle> CudaCommandBuffer::CreateKernelNode(
    Dependencies dependencies, const ThreadDim& threads, const BlockDim& blocks,
    const Kernel& kernel, const KernelArgsPackedArrayBase& args) {
  const uint64_t shared_mem_bytes = args.number_of_shared_bytes();
  VLOG(2) << "Add kernel node to a graph " << graph_
          << "; dependencies: " << GraphNodeHandlesToString(dependencies)
          << "; kernel: " << kernel.name() << "; gdx: " << blocks.x
          << " gdy: " << blocks.y << " gdz: " << blocks.z
          << " bdx: " << threads.x << " bdy: " << threads.y
          << " bdz: " << threads.z << "; shmem: " << shared_mem_bytes;

  CUDA_KERNEL_NODE_PARAMS params{};

  CUfunction function = static_cast<const CudaKernel&>(kernel).gpu_function();
  params.func = function;
  params.gridDimX = blocks.x;
  params.gridDimY = blocks.y;
  params.gridDimZ = blocks.z;
  params.blockDimX = threads.x;
  params.blockDimY = threads.y;
  params.blockDimZ = threads.z;
  params.sharedMemBytes = shared_mem_bytes;
  params.kernelParams = const_cast<void**>(args.argument_addresses().data());
  params.extra = nullptr;

  // TODO(ezhulenev): Why do we do it on every call to launch kernel? This
  // should be moved one level up to se::Kernel level, and done just once (or
  // updated once we get a new larger shared memory request).
  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(cuda::ToStatus(
        cuFuncSetAttribute(function,
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_mem_bytes),
        "Failed to set shared memory size"));
  }

  std::vector<CUgraphNode> deps = ToCudaGraphHandles(dependencies);

  CUgraphNode node_handle = nullptr;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuGraphAddKernelNode(&node_handle, graph_, deps.data(),
                                          deps.size(), &params),
                     "Failed to add kernel node to a CUDA graph"));
  return FromCudaGraphHandle(node_handle);
}

absl::Status CudaCommandBuffer::UpdateKernelNode(
    NodeHandle node_handle, const ThreadDim& threads, const BlockDim& blocks,
    const Kernel& kernel, const KernelArgsPackedArrayBase& args) {
  const uint64_t shared_mem_bytes = args.number_of_shared_bytes();

  VLOG(2) << "Set kernel node params " << node_handle << " in graph executable "
          << exec_ << "; kernel: " << kernel.name() << "; gdx: " << blocks.x
          << " gdy: " << blocks.y << " gdz: " << blocks.z
          << " bdx: " << threads.x << " bdy: " << threads.y
          << " bdz: " << threads.z << "; shmem: " << shared_mem_bytes;

  CUDA_KERNEL_NODE_PARAMS params{};
  CUfunction function = static_cast<const CudaKernel&>(kernel).gpu_function();
  params.func = function;
  params.gridDimX = blocks.x;
  params.gridDimY = blocks.y;
  params.gridDimZ = blocks.z;
  params.blockDimX = threads.x;
  params.blockDimY = threads.y;
  params.blockDimZ = threads.z;
  params.sharedMemBytes = shared_mem_bytes;
  params.kernelParams = const_cast<void**>(args.argument_addresses().data());
  params.extra = nullptr;

  // TODO(ezhulenev): Why do we do it on every call to launch kernel? This
  // should be moved one level up to se::Kernel level, and done just once (or
  // updated once we get a new larger shared memory request).
  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(cuda::ToStatus(
        cuFuncSetAttribute(function,
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_mem_bytes),
        "Failed to set shared memory size"));
  }

  return cuda::ToStatus(
      cuGraphKernelNodeSetParams(ToCudaGraphHandle(node_handle), &params),
      "Failed to set CUDA graph kernel node params");
}

absl::StatusOr<NodeHandle> CudaCommandBuffer::CreateEmptyNode(
    Dependencies dependencies) {
  if (stream_executor_->GetDeviceDescription().driver_version() <
      SemanticVersion(12, 4, 0)) {
    // Instead of empty nodes we create no-op kernel nodes as barriers because
    // CUDA 12.3 does not support empty nodes inside conditional command
    // buffers.
    TF_ASSIGN_OR_RETURN(NoOpKernel * noop, GetNoOpKernel());
    return CreateKernelNode(dependencies, ThreadDim{1, 1, 1}, BlockDim{1, 1, 1},
                            **noop, KernelArgsPackedArray<0>());
  }

  VLOG(2) << "Add empty node to a graph " << graph_
          << " with dependencies: " << GraphNodeHandlesToString(dependencies);
  CUgraphNode barrier_handle = nullptr;
  std::vector<CUgraphNode> deps = ToCudaGraphHandles(dependencies);
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuGraphAddEmptyNode(&barrier_handle, graph_, deps.data(), deps.size()),
      "Failed to add empty node to a CUDA graph"));

  return FromCudaGraphHandle(barrier_handle);
}

absl::Status CudaCommandBuffer::Trace(
    Stream* stream, absl::AnyInvocable<absl::Status()> function) {
#if CUDA_VERSION < 12030
  return absl::UnimplementedError(
      "StreamBeginCaptureToGraph is not implemented for CUDA below version "
      "12.3. Therefore tracing is not supported.");
#else
  if (stream_executor_->GetDeviceDescription().driver_version() <
      SemanticVersion{12, 3, 0}) {
    return absl::UnimplementedError(
        "StreamBeginCaptureToGraph is not implemented for CUDA below version "
        "12.3. Therefore tracing is not supported.");
  }

  CHECK(!finalized_) << "Trying to trace a finalized command buffer";

  VLOG(5) << "Trace into GPU command buffer graph " << graph_
          << " on a stream: " << stream;

  CUstream stream_handle = AsGpuStreamValue(stream);

  // Switch stream into the capture mode.
  uint64_t start_nanos = tsl::Env::Default()->NowNanos();

  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuStreamBeginCaptureToGraph(stream_handle, graph_,
                                  /*dependencies=*/nullptr,
                                  /*dependencyData=*/nullptr,
                                  /*numDependencies=*/0,
                                  CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
      "Failed to begin stream capture to graph"));
  auto traced = function();

  // Always stop capturing the stream before checking `traced` result.
  VLOG(5) << "End stream " << stream << " capture";
  CUgraph captured_graph;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuStreamEndCapture(stream_handle, &captured_graph),
                     "Failed to end stream capture"));
  DCHECK(captured_graph == graph_) << "Stream capture should update graph_";
  uint64_t end_nanos = tsl::Env::Default()->NowNanos();

  if (!traced.ok())
    return absl::InternalError(
        absl::StrCat("Failed to capture gpu graph: ", traced.message()));

  VLOG(5) << "Traced into the GPU command buffer graph " << graph_ << " (took "
          << (end_nanos - start_nanos) / 1000 << " μs)";

  return absl::OkStatus();
#endif
}

/*static*/ int64_t CudaCommandBuffer::NotifyExecCreated() {
  alive_execs.fetch_add(1, std::memory_order_relaxed);
  return allocated_execs.fetch_add(1, std::memory_order_relaxed);
}

/*static*/ int64_t CudaCommandBuffer::NotifyExecDestroyed() {
  DCHECK_GE(alive_execs.load(std::memory_order_relaxed), 1);
  return alive_execs.fetch_sub(1, std::memory_order_relaxed) - 1;
}

/*static*/ int64_t CudaCommandBuffer::AliveExecs() {
  return alive_execs.load(std::memory_order_relaxed);
}

absl::Status CudaCommandBuffer::Finalize() {
  TF_RETURN_IF_ERROR(PrepareFinalization());

  // Maybe dump created CUDA graph to a dot file for debugging.
  if (VLOG_IS_ON(10) && mode() == Mode::kPrimary) {
    std::string path = tsl::io::GetTempFilename(/*extension=*/"dot");
    TF_RETURN_IF_ERROR(WriteGraphToDotFile(path));
    if (VLOG_IS_ON(100)) {
      std::string dot_file_contents;
      TF_RETURN_IF_ERROR(
          tsl::ReadFileToString(tsl::Env::Default(), path, &dot_file_contents));
      VLOG(100) << "Contents of " << path << " is:\n" << dot_file_contents;
    }
  }

  // Collect number of nodes and conditionals for logging below.
  size_t num_nodes = 0, num_cond_cmd_buffers = 0;

  if (mode() == Mode::kPrimary) {
    uint64_t start_nanos = tsl::Env::Default()->NowNanos();

    if (exec_ == nullptr) {
      // If this is the first time we finalize command buffer after
      // construction, we need to instantiate it to an executable graph.
      auto instantiated = InstantiateGraph();

      if (instantiated.code() == absl::StatusCode::kResourceExhausted) {
        return absl::ResourceExhaustedError(absl::StrFormat(
            "Underlying backend ran out of memory trying to instantiate "
            "graph "
            "with %d nodes and %d conditionals (total of %d alive graphs "
            "in the process). You can try to (a) Give more memory to the "
            "driver by reducing XLA_CLIENT_MEM_FRACTION (b) Disable "
            "command buffers with "
            "'XLA_FLAGS=--xla_gpu_enable_command_buffer=' "
            "(empty set). Original error: %s",
            num_nodes, num_cond_cmd_buffers, AliveExecs(),
            instantiated.message()));
      }
      TF_RETURN_IF_ERROR(instantiated);

      uint64_t end_nanos = tsl::Env::Default()->NowNanos();
      TF_ASSIGN_OR_RETURN(auto node_count, GetNodeCount());
      auto exec_num = NotifyExecCreated();
      VLOG(5) << "Instantiated executable graph #" << exec_num << " in "
              << (end_nanos - start_nanos) / 1000 << " μs"
              << "; nodes: " << node_count
              << "; conditionals: " << num_cond_cmd_buffers
              << "; alive executable graphs: " << AliveExecs();
    } else {
      VLOG(5) << "Update existing executor " << exec_
              << " by updating with graph " << graph_;
      CUgraphExecUpdateResultInfo resultInfo;
      TF_RETURN_IF_ERROR(
          cuda::ToStatus(cuGraphExecUpdate(exec_, graph_, &resultInfo),
                         "Failed to update CUDA graph executable"));
      VLOG(5) << "Updated CUDA graph executable " << exec_;
    }
  } else {
    // Nested command buffers do not have executable graphs.
    VLOG(5) << "Finalize nested command buffer without instantiating "
               "executable graph";
  }

  finalized_ = true;
  return absl::OkStatus();
}

absl::Status CudaCommandBuffer::Submit(Stream* stream) {
  if (mode() != Mode::kPrimary) {
    return absl::InvalidArgumentError(
        "Can't submit non-primary command buffer for execution");
  }

  if (!finalized_) {
    return absl::InvalidArgumentError(
        "Can't submit command buffer that is not finalized");
  }

  return LaunchGraph(stream);
}

absl::Status CudaCommandBuffer::LaunchGraph(Stream* stream) {
  VLOG(3) << "Launch command buffer executable graph " << exec_
          << " on a stream: " << stream;
  return cuda::ToStatus(cuGraphLaunch(exec_, AsGpuStreamValue(stream)),
                        "Failed to launch CUDA graph");
}

absl::StatusOr<size_t> CudaCommandBuffer::GetNodeCount() const {
  size_t num_nodes;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuGraphGetNodes(graph_, /*nodes=*/nullptr, &num_nodes)));
  return num_nodes;
}

absl::Status CudaCommandBuffer::PrepareFinalization() {
  // TODO(b/362769658): Remove this workaround when cuda supports conditionals
  // with empty graphs.
  TF_ASSIGN_OR_RETURN(auto node_count, GetNodeCount());
  if (node_count > 0) {
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(NoOpKernel * noop, GetNoOpKernel());
  TF_ASSIGN_OR_RETURN(
      auto node,
      CreateLaunchNode(absl::InlinedVector<NodeHandle, 1>{}, ThreadDim(),
                       BlockDim(), **noop, KernelArgsPackedArray<0>()));
  (void)node;
  return absl::OkStatus();
}

absl::StatusOr<ConditionalHandle> CudaCommandBuffer::CreateConditionalHandle() {
  constexpr int kDefaultLaunchValue = 0;
  constexpr int kNoFlags = 0;
  VLOG(2) << "Create conditional handle for a graph " << graph_
          << "; context: " << cuda_context_
          << "; default_launch_value: " << kDefaultLaunchValue
          << "; flags: " << kNoFlags;

#if CUDA_VERSION >= 12030
  CUgraphConditionalHandle handle;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuGraphConditionalHandleCreate(&handle, graph_, cuda_context_->context(),
                                     kDefaultLaunchValue, kNoFlags),
      "Failed to create conditional handle for a CUDA graph"));
  return FromCudaGraphHandle(handle);
#else
  return absl::UnimplementedError(
      "CUDA graph conditional nodes are not implemented");
#endif  // CUDA_VERSION >= 12030
}

absl::Status CudaCommandBuffer::WriteGraphToDotFile(absl::string_view path) {
#if CUDA_VERSION >= 12000
  VLOG(2) << "Print CUDA graph " << graph_ << " debug dot file to " << path;

  int flags = CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE;
  return cuda::ToStatus(
      cuGraphDebugDotPrint(graph_, std::string{path}.c_str(), flags),
      "Failed to print gpu graph debug file");
#endif  // CUDA_VERSION >= 12000

  return absl::UnimplementedError(
      "CUDA graph debug dot print is not supported.");
}

absl::Status CudaCommandBuffer::InstantiateGraph() {
  // If we get a "resource exhausted error" we retry instantiating Gpu graph
  // one more time after releasing unused device memory allocated for graphs.
  auto instantiated = GraphInstantiate(&exec_, graph_);
  if (instantiated.code() == absl::StatusCode::kResourceExhausted) {
    LOG(WARNING) << "Retry CUDA graph instantiation after OOM error";
    CUdevice device;
    TF_RETURN_IF_ERROR(
        cuda::ToStatus(cuDeviceGet(&device, stream_executor_->device_ordinal()),
                       "Failed call to cuDeviceGet"));
    TF_RETURN_IF_ERROR(cuda::ToStatus(cuDeviceGraphMemTrim(device),
                                      "Failed to trim device graph memory"));
    TF_RETURN_IF_ERROR(GraphInstantiate(&exec_, graph_));
  } else {
    TF_RETURN_IF_ERROR(instantiated);
  }

  VLOG(1) << "Instantiate graph " << std::hex << exec_;

  return absl::OkStatus();
}

CudaCommandBuffer::~CudaCommandBuffer() {
  if (exec_ != nullptr && mode() == Mode::kPrimary) {
    auto exec_num = NotifyExecDestroyed();
    VLOG(5) << "Destroy GPU command buffer executable graph " << exec_ << " "
            << "(remaining alive executable graphs: " << exec_num << ")";
    if (auto status = cuda::ToStatus(cuGraphExecDestroy(exec_),
                                     "Failed to destroy CUDA executable graph");
        !status.ok()) {
      LOG(ERROR) << status.message() << " executable " << exec_ << " graph "
                 << graph_;
    }
  }
  if (graph_ != nullptr && is_owned_graph_) {
    if (auto status = cuda::ToStatus(cuGraphDestroy(graph_),
                                     "Failed to destroy CUDA graph");
        !status.ok()) {
      LOG(ERROR) << status.message();
    }
  }
}

absl::Status CudaCommandBuffer::CheckCanBeUpdated() {
  if (exec_ == nullptr) {
    return absl::InternalError(
        "Command buffer has to have a graph executable to be updated.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<NodeHandle>> CudaCommandBuffer::GetNodeDependencies(
    NodeHandle node) {
  VLOG(2) << "Get CUDA graph node " << node << " dependencies";

  std::vector<CUgraphNode> dependencies;

  size_t num_dependencies = 0;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuGraphNodeGetDependencies(ToCudaGraphHandle(node),
                                                nullptr, &num_dependencies),
                     "Failed to get CUDA graph node depedencies size"));

  dependencies.resize(num_dependencies, nullptr);
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuGraphNodeGetDependencies(ToCudaGraphHandle(node), dependencies.data(),
                                 &num_dependencies),
      "Failed to get CUDA graph node depedencies"));

  std::vector<NodeHandle> result;
  result.reserve(dependencies.size());
  absl::c_transform(
      dependencies, std::back_inserter(result),
      static_cast<NodeHandle (*)(CUgraphNode)>(&FromCudaGraphHandle));

  return result;
}

}  // namespace stream_executor::gpu
