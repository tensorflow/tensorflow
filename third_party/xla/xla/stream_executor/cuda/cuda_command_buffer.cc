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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/command_buffer_kernels.h"
#include "xla/stream_executor/cuda/cuda_context.h"
#include "xla/stream_executor/cuda/cuda_kernel.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/typed_kernel_factory.h"  // IWYU pragma: keep
#include "tsl/platform/casts.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {
absl::StatusOr<CUgraph> CreateGraph() {
  VLOG(2) << "Create new CUDA graph";
  CUgraph graph = nullptr;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuGraphCreate(&graph, /*flags=*/0),
                                    "Failed to create CUDA graph"));
  VLOG(2) << "Created CUDA graph " << graph;
  return graph;
}

CUdeviceptr AsDevicePtr(const DeviceMemoryBase& mem) {
  return absl::bit_cast<CUdeviceptr>(mem.opaque());
}

struct BitPatternToString {
  std::string operator()(uint8_t pattern) {
    return absl::StrCat("u8:", pattern);
  }
  std::string operator()(uint16_t pattern) {
    return absl::StrCat("u16:", pattern);
  }
  std::string operator()(uint32_t pattern) {
    return absl::StrCat("u32:", pattern);
  }
};

// Broadcasts a pattern value of 1/2/4 bytes to a 4 byte value.
struct BitPatternToValue {
  std::pair<unsigned, unsigned> operator()(uint8_t pattern) {
    unsigned value = pattern;
    return {(value << 24) | (value << 16) | (value << 8) | value,
            /*element_size=*/1};
  }
  std::pair<unsigned, unsigned> operator()(uint16_t pattern) {
    unsigned value = pattern;
    return {(value << 16) | value, /*element_size=*/2};
  }
  std::pair<unsigned, unsigned> operator()(uint32_t pattern) {
    return {pattern, /*element_size=*/4};
  }
};

// Takes a list of GpuGraphNodeInfo instances and converts them to a list of
// CUgraphNode handles.
std::vector<CUgraphNode> AsNodeHandles(
    absl::Span<const GpuCommandBuffer::GpuGraphNodeInfo* const> nodes) {
  std::vector<CUgraphNode> handles;
  handles.reserve(nodes.size());
  for (const GpuCommandBuffer::GpuGraphNodeInfo* node : nodes) {
    handles.push_back(node->handle);
  }
  return handles;
}
}  // namespace

absl::StatusOr<std::unique_ptr<CudaCommandBuffer>> CudaCommandBuffer::Create(
    Mode mode, GpuExecutor* parent) {
  TF_ASSIGN_OR_RETURN(CUgraph graph, CreateGraph());
  return std::unique_ptr<CudaCommandBuffer>(
      new CudaCommandBuffer(mode, parent, graph,
                            /*is_owned_graph=*/true));
}

absl::StatusOr<CudaCommandBuffer::SetIfConditionKernel*>
CudaCommandBuffer::GetSetIfConditionKernel() {
  if (!set_if_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec, cuda::GetSetIfConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_if_condition_kernel_,
        SetIfConditionKernel::FactoryType::Create(parent_, spec));
  }
  return &set_if_condition_kernel_;
}

absl::StatusOr<CudaCommandBuffer::SetIfElseConditionKernel*>
CudaCommandBuffer::GetSetIfElseConditionKernel() {
  if (!set_if_else_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec,
                        cuda::GetSetIfElseConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_if_else_condition_kernel_,
        SetIfElseConditionKernel::FactoryType::Create(parent_, spec));
  }
  return &set_if_else_condition_kernel_;
}

absl::StatusOr<CudaCommandBuffer::SetCaseConditionKernel*>
CudaCommandBuffer::GetSetCaseConditionKernel() {
  if (!set_case_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec, cuda::GetSetCaseConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_case_condition_kernel_,
        SetCaseConditionKernel::FactoryType::Create(parent_, spec));
  }
  return &set_case_condition_kernel_;
}

absl::StatusOr<CudaCommandBuffer::SetForConditionKernel*>
CudaCommandBuffer::GetSetForConditionKernel() {
  if (!set_for_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec, cuda::GetSetForConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_for_condition_kernel_,
        SetForConditionKernel::FactoryType::Create(parent_, spec));
  }
  return &set_for_condition_kernel_;
}

absl::StatusOr<CudaCommandBuffer::SetWhileConditionKernel*>
CudaCommandBuffer::GetSetWhileConditionKernel() {
  if (!set_while_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec,
                        cuda::GetSetWhileConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_while_condition_kernel_,
        SetWhileConditionKernel::FactoryType::Create(parent_, spec));
  }
  return &set_while_condition_kernel_;
}

absl::StatusOr<CudaCommandBuffer::NoOpKernel*>
CudaCommandBuffer::GetNoOpKernel() {
  if (!noop_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec, cuda::GetNoOpKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(noop_kernel_,
                        NoOpKernel::FactoryType::Create(parent_, spec));
  }
  return &noop_kernel_;
}

std::unique_ptr<GpuCommandBuffer> CudaCommandBuffer::CreateNestedCommandBuffer(
    CUgraph graph) {
  return std::unique_ptr<CudaCommandBuffer>(
      new CudaCommandBuffer(Mode::kNested, parent_, graph,
                            /*is_owned_graph=*/false));
}

absl::StatusOr<CudaCommandBuffer::GpuGraphNodeInfo*>
CudaCommandBuffer::CreateMemsetNode(const Dependencies& dependencies,
                                    DeviceMemoryBase destination,
                                    BitPattern bit_pattern,
                                    size_t num_elements) {
  CudaContext* cuda_context =
      tensorflow::down_cast<CudaContext*>(parent_->gpu_context());
  VLOG(2) << "Add memset node to a graph " << graph_
          << "; dst: " << destination.opaque()
          << "; bit_pattern: " << std::visit(BitPatternToString(), bit_pattern)
          << "; num_elements: " << num_elements
          << "; context: " << cuda_context->context()
          << "; deps: " << dependencies.size();

  CUDA_MEMSET_NODE_PARAMS params;
  std::memset(&params, 0, sizeof(params));

  auto [value, element_size] = std::visit(BitPatternToValue(), bit_pattern);

  params.dst = AsDevicePtr(destination);
  params.elementSize = element_size;
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = value;
  params.width = num_elements;

  std::vector<CUgraphNode> deps = AsNodeHandles(dependencies);

  CUgraphNode node_handle = nullptr;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuGraphAddMemsetNode(&node_handle, graph_, deps.data(), deps.size(),
                           &params, cuda_context->context()),
      "Failed to add memset node to a CUDA graph"));

  node_storage_.push_back(std::make_unique<CudaGraphNode>(node_handle, this));
  return node_storage_.back().get();
}

absl::Status CudaCommandBuffer::CudaGraphNode::UpdateMemsetNode(
    DeviceMemoryBase destination, BitPattern bit_pattern, size_t num_elements) {
  CudaContext* cuda_context = tensorflow::down_cast<CudaContext*>(
      command_buffer_->parent_->gpu_context());
  VLOG(2) << "Set memset node params " << handle << " in graph executable "
          << command_buffer_->exec_ << "; dst: " << destination.opaque()
          << "; bit_pattern: " << std::visit(BitPatternToString(), bit_pattern)
          << "; num_elements: " << num_elements
          << "; context: " << cuda_context->context();

  CUDA_MEMSET_NODE_PARAMS params;
  std::memset(&params, 0, sizeof(params));

  auto [value, element_size] = std::visit(BitPatternToValue(), bit_pattern);

  params.dst = AsDevicePtr(destination);
  params.elementSize = element_size;
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = value;
  params.width = num_elements;

  return cuda::ToStatus(
      cuGraphExecMemsetNodeSetParams(command_buffer_->exec_, handle, &params,
                                     cuda_context->context()),
      "Failed to set memset node params");
}

absl::StatusOr<CudaCommandBuffer::GpuGraphNodeInfo*>
CudaCommandBuffer::CreateMemcpyD2DNode(const Dependencies& dependencies,
                                       DeviceMemoryBase destination,
                                       DeviceMemoryBase source, uint64_t size) {
  CudaContext* cuda_context =
      tensorflow::down_cast<CudaContext*>(parent_->gpu_context());
  VLOG(2) << "Add memcpy d2d node to a graph " << graph_
          << "; dst: " << destination.opaque() << "; src: " << source.opaque()
          << "; size: " << size << "; context: " << cuda_context->context()
          << "; deps: " << dependencies.size();

  CUDA_MEMCPY3D params;
  std::memset(&params, 0, sizeof(params));

  params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  params.srcDevice = AsDevicePtr(source);
  params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  params.dstDevice = AsDevicePtr(destination);
  params.WidthInBytes = size;
  params.Height = 1;
  params.Depth = 1;

  std::vector<CUgraphNode> deps = AsNodeHandles(dependencies);

  CUgraphNode node_handle = nullptr;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuGraphAddMemcpyNode(&node_handle, graph_, deps.data(), deps.size(),
                           &params, cuda_context->context()),
      "Failed to add memcpy d2d node to a CUDA graph"));
  node_storage_.push_back(std::make_unique<CudaGraphNode>(node_handle, this));
  return node_storage_.back().get();
}

absl::Status CudaCommandBuffer::CudaGraphNode::UpdateMemcpyD2DNode(
    DeviceMemoryBase destination, DeviceMemoryBase source, uint64_t size) {
  CudaContext* cuda_context = tensorflow::down_cast<CudaContext*>(
      command_buffer_->parent_->gpu_context());
  VLOG(2) << "Set memcpy d2d node params " << handle << " in graph executable "
          << command_buffer_->exec_ << "; dst: " << destination.opaque()
          << "; src: " << source.opaque() << "; size: " << size
          << "; context: " << cuda_context->context();

  CUDA_MEMCPY3D params;
  std::memset(&params, 0, sizeof(params));

  params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  params.srcDevice = AsDevicePtr(source);
  params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  params.dstDevice = AsDevicePtr(destination);
  params.WidthInBytes = size;
  params.Height = 1;
  params.Depth = 1;

  return cuda::ToStatus(
      cuGraphExecMemcpyNodeSetParams(command_buffer_->exec_, handle, &params,
                                     cuda_context->context()),
      "Failed to set memcpy d2d node params");
}

absl::StatusOr<CudaCommandBuffer::GpuGraphNodeInfo*>
CudaCommandBuffer::CreateChildNode(const Dependencies& dependencies,
                                   const CommandBuffer& nested) {
  CUgraph child_graph =
      tensorflow::down_cast<const CudaCommandBuffer&>(nested).graph_;
  VLOG(2) << "Create a new node by cloning the child graph " << child_graph
          << " and add it to " << graph_ << "; deps: " << dependencies.size();

  std::vector<CUgraphNode> deps = AsNodeHandles(dependencies);

  GpuGraphNodeInfo node_info;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuGraphAddChildGraphNode(&node_info.handle, graph_, deps.data(),
                               deps.size(), child_graph),
      "Failed to create a child graph node and add it to a CUDA graph"));

  node_storage_.push_back(
      std::make_unique<CudaGraphNode>(node_info.handle, this));
  return node_storage_.back().get();
}

absl::Status CudaCommandBuffer::CudaGraphNode::UpdateChildNode(
    const CommandBuffer& nested) {
  CUgraph child_graph =
      tensorflow::down_cast<const CudaCommandBuffer&>(nested).graph_;
  VLOG(2) << "Set child node params " << handle << " in graph executable "
          << command_buffer_->exec_ << "to params contained in " << child_graph;

  return cuda::ToStatus(cuGraphExecChildGraphNodeSetParams(
                            command_buffer_->exec_, handle, child_graph),
                        "Failed to set CUDA graph child node params");
}

absl::StatusOr<CudaCommandBuffer::GpuGraphNodeInfo*>
CudaCommandBuffer::CreateKernelNode(const Dependencies& dependencies,
                                    const ThreadDim& threads,
                                    const BlockDim& blocks,
                                    const Kernel& kernel,
                                    const KernelArgsPackedArrayBase& args) {
  const uint64_t shared_mem_bytes = args.number_of_shared_bytes();

  VLOG(2) << "Add kernel node to a graph " << graph_
          << "; kernel: " << kernel.name() << "; gdx: " << blocks.x
          << " gdy: " << blocks.y << " gdz: " << blocks.z
          << " bdx: " << threads.x << " bdy: " << threads.y
          << " bdz: " << threads.z << "; shmem: " << shared_mem_bytes
          << "; deps: " << dependencies.size();

  CUDA_KERNEL_NODE_PARAMS params;
  std::memset(&params, 0, sizeof(params));

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

  std::vector<CUgraphNode> deps = AsNodeHandles(dependencies);

  CUgraphNode node_handle = nullptr;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuGraphAddKernelNode(&node_handle, graph_, deps.data(),
                                          deps.size(), &params),
                     "Failed to add kernel node to a CUDA graph"));
  node_storage_.push_back(std::make_unique<CudaGraphNode>(node_handle, this));
  return node_storage_.back().get();
}

absl::Status CudaCommandBuffer::CudaGraphNode::UpdateKernelNode(
    const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
    const KernelArgsPackedArrayBase& args) {
  const uint64_t shared_mem_bytes = args.number_of_shared_bytes();

  VLOG(2) << "Set kernel node params " << handle << " in graph executable "
          << command_buffer_->exec_ << "; kernel: " << kernel.name()
          << "; gdx: " << blocks.x << " gdy: " << blocks.y
          << " gdz: " << blocks.z << " bdx: " << threads.x
          << " bdy: " << threads.y << " bdz: " << threads.z
          << "; shmem: " << shared_mem_bytes;

  CUDA_KERNEL_NODE_PARAMS params;
  std::memset(&params, 0, sizeof(params));

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
      cuGraphExecKernelNodeSetParams(command_buffer_->exec_, handle, &params),
      "Failed to set CUDA graph kernel node params");
}

absl::StatusOr<GpuCommandBuffer::GpuGraphNodeInfo*>
CudaCommandBuffer::CreateBarrierNode(const Dependencies& dependencies) {
  if (parent_->GetDeviceDescription().driver_version() <
      SemanticVersion(12, 4, 0)) {
    // Instead of empty nodes we create no-op kernel nodes as barriers because
    // CUDA 12.3 does not support empty nodes inside conditional command
    // buffers.
    TF_ASSIGN_OR_RETURN(NoOpKernel * noop, GetNoOpKernel());
    return CreateKernelNode(dependencies, ThreadDim{1, 1, 1}, BlockDim{1, 1, 1},
                            **noop, KernelArgsPackedArray<0>());
  }

  VLOG(2) << "Add empty node to a graph " << graph_
          << "; deps: " << dependencies.size();

  CUgraphNode barrier_handle = nullptr;
  std::vector<CUgraphNode> deps = AsNodeHandles(dependencies);
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuGraphAddEmptyNode(&barrier_handle, graph_, deps.data(), deps.size()),
      "Failed to add empty node to a CUDA graph"));

  node_storage_.push_back(
      std::make_unique<CudaGraphNode>(barrier_handle, this));
  return node_storage_.back().get();
}

absl::Status CudaCommandBuffer::Trace(
    Stream* stream, absl::AnyInvocable<absl::Status()> function) {
  if (parent_->GetDeviceDescription().driver_version() <
      SemanticVersion{12, 4, 0}) {
    return absl::UnimplementedError(
        "StreamBeginCaptureToGraph is not implemented for CUDA 12.3 and "
        "below. Therefore tracing is not supported.");
  }

  TF_RETURN_IF_ERROR(CheckNotFinalized());

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
}

}  // namespace stream_executor::gpu
