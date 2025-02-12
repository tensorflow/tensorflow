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
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/scoped_update_mode.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
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

using GraphNodeHandle = GpuCommandBuffer::GraphNodeHandle;
using GraphConditionalHandle = GpuCommandBuffer::GraphConditionalHandle;

// Converts a platform independent GraphNodeHandle into a CUDA specific
// CUgraphNode.
CUgraphNode ToCudaGraphHandle(GraphNodeHandle handle) {
  return absl::bit_cast<CUgraphNode>(handle);
}

// Converts a platform independent GraphConditionalHandle into a CUDA specific
// CUgraphConditionalHandle.
CUgraphConditionalHandle ToCudaGraphHandle(GraphConditionalHandle handle) {
  return absl::bit_cast<CUgraphConditionalHandle>(handle);
}

// Converts a list of platform independent GraphNodeHandles into a list of
// CUDA specific CUgraphNode.
std::vector<CUgraphNode> ToCudaGraphHandles(
    absl::Span<const GraphNodeHandle> opaque_handles) {
  std::vector<CUgraphNode> handles;
  handles.reserve(opaque_handles.size());
  for (const GraphNodeHandle opaque_handle : opaque_handles) {
    handles.push_back(ToCudaGraphHandle(opaque_handle));
  }
  return handles;
}

// Converts a CUDA specific CUgraphNode into a platform independent
// GraphNodeHandle.
GraphNodeHandle FromCudaGraphHandle(CUgraphNode handle) {
  return absl::bit_cast<GraphNodeHandle>(handle);
}

// Converts a CUDA specific CUgraphConditionalHandle into a platform
// independent GraphConditionalHandle.
GraphConditionalHandle FromCudaGraphHandle(CUgraphConditionalHandle handle) {
  return absl::bit_cast<GraphConditionalHandle>(handle);
}

std::string ConditionalTypeToString(GpuCommandBuffer::ConditionType type) {
  switch (type) {
    case GpuCommandBuffer::ConditionType::kIf:
      return "IF";
    case GpuCommandBuffer::ConditionType::kWhile:
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
    Mode mode, StreamExecutor* parent, CudaContext* cuda_context) {
  TF_ASSIGN_OR_RETURN(CUgraph graph, CreateGraph());
  return std::unique_ptr<CudaCommandBuffer>(
      new CudaCommandBuffer(mode, parent, cuda_context, graph,
                            /*is_owned_graph=*/true));
}

absl::Status CudaCommandBuffer::LaunchSetIfElseConditionKernel(
    ExecutionScopeId execution_scope_id, GraphConditionalHandle if_conditional,
    GraphConditionalHandle else_conditional, DeviceMemory<bool> predicate) {
  if (!set_if_else_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec,
                        cuda::GetSetIfElseConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_if_else_condition_kernel_,
        SetIfElseConditionKernel::FactoryType::Create(parent_, spec));
  }
  return Launch(set_if_else_condition_kernel_, execution_scope_id, ThreadDim(),
                BlockDim(), ToCudaGraphHandle(if_conditional),
                ToCudaGraphHandle(else_conditional), predicate);
}
absl::Status CudaCommandBuffer::LaunchSetIfConditionKernel(
    ExecutionScopeId execution_scope_id, GraphConditionalHandle if_conditional,
    DeviceMemory<bool> predicate) {
  if (!set_if_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec, cuda::GetSetIfConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_if_condition_kernel_,
        SetIfConditionKernel::FactoryType::Create(parent_, spec));
  }
  return Launch(set_if_condition_kernel_, execution_scope_id, ThreadDim(),
                BlockDim(), ToCudaGraphHandle(if_conditional), predicate);
}
absl::Status CudaCommandBuffer::LaunchSetForConditionKernel(
    ExecutionScopeId execution_scope_id, GraphConditionalHandle conditional,
    DeviceMemory<int32_t> loop_counter, int32_t iterations) {
  if (!set_for_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec, cuda::GetSetForConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_for_condition_kernel_,
        SetForConditionKernel::FactoryType::Create(parent_, spec));
  }

  return CommandBuffer::Launch(
      set_for_condition_kernel_, execution_scope_id, ThreadDim(), BlockDim(),
      ToCudaGraphHandle(conditional), loop_counter, iterations);
}
absl::Status CudaCommandBuffer::LaunchSetWhileConditionKernel(
    ExecutionScopeId execution_scope_id, GraphConditionalHandle conditional,
    DeviceMemory<bool> predicate) {
  if (!set_while_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec,
                        cuda::GetSetWhileConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_while_condition_kernel_,
        SetWhileConditionKernel::FactoryType::Create(parent_, spec));
  }

  return Launch(set_while_condition_kernel_, execution_scope_id, ThreadDim(),
                BlockDim(), ToCudaGraphHandle(conditional), predicate);
}

absl::Status CudaCommandBuffer::LaunchSetCaseConditionKernel(
    ExecutionScopeId execution_scope_id, GraphConditionalHandles conditionals,
    DeviceMemory<int32_t> index, int32_t batch_offset,
    bool enable_conditional_default) {
  constexpr int kCaseBranchBatchSize = 8;
  CHECK(conditionals.size() <= kCaseBranchBatchSize);

  if (!set_case_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec, cuda::GetSetCaseConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_case_condition_kernel_,
        SetCaseConditionKernel::FactoryType::Create(parent_, spec));
  }

  // Pad handles up to size 8 with a default initialized handle.
  std::vector<CUgraphConditionalHandle> padded_handles{};
  padded_handles.resize(kCaseBranchBatchSize);
  std::transform(conditionals.begin(), conditionals.end(),
                 padded_handles.begin(),
                 [](GraphConditionalHandle conditional) {
                   return ToCudaGraphHandle(conditional);
                 });

  return Launch(set_case_condition_kernel_, execution_scope_id, ThreadDim(),
                BlockDim(), padded_handles[0], padded_handles[1],
                padded_handles[2], padded_handles[3], padded_handles[4],
                padded_handles[5], padded_handles[6], padded_handles[7], index,
                batch_offset, static_cast<int32_t>(conditionals.size()),
                enable_conditional_default);
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

absl::StatusOr<GpuCommandBuffer::ConditionalNodeResult>
CudaCommandBuffer::CreateConditionalNode(const Dependencies& dependencies,
                                         GraphConditionalHandle conditional,
                                         ConditionType type) {
#if CUDA_VERSION >= 12030
  // Add conditional node to a graph.
  VLOG(2) << "Add conditional node to a graph " << graph_
          << "; type: " << ConditionalTypeToString(type)
          << "; deps: " << dependencies.size();

  CUgraphNodeParams cu_params;
  std::memset(&cu_params, 0, sizeof(cu_params));

  cu_params.type = CU_GRAPH_NODE_TYPE_CONDITIONAL;
  cu_params.conditional.handle = ToCudaGraphHandle(conditional);
  cu_params.conditional.ctx = cuda_context_->context();
  cu_params.conditional.size = 1;

  switch (type) {
    case GpuCommandBuffer::ConditionType::kIf:
      cu_params.conditional.type = CU_GRAPH_COND_TYPE_IF;
      break;
    case GpuCommandBuffer::ConditionType::kWhile:
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
          new CudaCommandBuffer(Mode::kNested, parent_, cuda_context_,
                                cu_params.conditional.phGraph_out[0],
                                /*is_owned_graph=*/false))};
#else
  return absl::UnimplementedError("unsupported node type");
#endif  // CUDA_VERSION >= 12030
}

absl::StatusOr<GraphNodeHandle> CudaCommandBuffer::CreateMemsetNode(
    const Dependencies& dependencies, DeviceMemoryBase destination,
    BitPattern bit_pattern, size_t num_elements) {
  VLOG(2) << "Add memset node to a graph " << graph_
          << "; dst: " << destination.opaque()
          << "; bit_pattern: " << bit_pattern.ToString()
          << "; num_elements: " << num_elements
          << "; context: " << cuda_context_->context()
          << "; deps: " << dependencies.size();

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

absl::Status CudaCommandBuffer::UpdateMemsetNode(GraphNodeHandle node_handle,
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
      cuGraphExecMemsetNodeSetParams(exec_, ToCudaGraphHandle(node_handle),
                                     &params, cuda_context_->context()),
      "Failed to set memset node params");
}

absl::StatusOr<GraphNodeHandle> CudaCommandBuffer::CreateMemcpyD2DNode(
    const Dependencies& dependencies, DeviceMemoryBase destination,
    DeviceMemoryBase source, uint64_t size) {
  VLOG(2) << "Add memcpy d2d node to a graph " << graph_
          << "; dst: " << destination.opaque() << "; src: " << source.opaque()
          << "; size: " << size << "; context: " << cuda_context_->context()
          << "; deps: " << dependencies.size();

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
    GraphNodeHandle node_handle, DeviceMemoryBase destination,
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
      cuGraphExecMemcpyNodeSetParams(exec_, ToCudaGraphHandle(node_handle),
                                     &params, cuda_context_->context()),
      "Failed to set memcpy d2d node params");
}

absl::StatusOr<GraphNodeHandle> CudaCommandBuffer::CreateChildNode(
    const Dependencies& dependencies, const CommandBuffer& nested) {
  CUgraph child_graph =
      tensorflow::down_cast<const CudaCommandBuffer&>(nested).graph_;
  VLOG(2) << "Create a new node by cloning the child graph " << child_graph
          << " and add it to " << graph_ << "; deps: " << dependencies.size();

  std::vector<CUgraphNode> deps = ToCudaGraphHandles(dependencies);

  CUgraphNode node_handle;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuGraphAddChildGraphNode(&node_handle, graph_, deps.data(), deps.size(),
                               child_graph),
      "Failed to create a child graph node and add it to a CUDA graph"));

  return FromCudaGraphHandle(node_handle);
}

absl::Status CudaCommandBuffer::UpdateChildNode(GraphNodeHandle node_handle,
                                                const CommandBuffer& nested) {
  CUgraph child_graph =
      tensorflow::down_cast<const CudaCommandBuffer&>(nested).graph_;
  VLOG(2) << "Set child node params " << node_handle << " in graph executable "
          << exec_ << "to params contained in " << child_graph;

  return cuda::ToStatus(cuGraphExecChildGraphNodeSetParams(
                            exec_, ToCudaGraphHandle(node_handle), child_graph),
                        "Failed to set CUDA graph child node params");
}

absl::StatusOr<GraphNodeHandle> CudaCommandBuffer::CreateKernelNode(
    const Dependencies& dependencies, const ThreadDim& threads,
    const BlockDim& blocks, const Kernel& kernel,
    const KernelArgsPackedArrayBase& args) {
  const uint64_t shared_mem_bytes = args.number_of_shared_bytes();

  VLOG(2) << "Add kernel node to a graph " << graph_
          << "; kernel: " << kernel.name() << "; gdx: " << blocks.x
          << " gdy: " << blocks.y << " gdz: " << blocks.z
          << " bdx: " << threads.x << " bdy: " << threads.y
          << " bdz: " << threads.z << "; shmem: " << shared_mem_bytes
          << "; deps: " << dependencies.size();

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
    GraphNodeHandle node_handle, const ThreadDim& threads,
    const BlockDim& blocks, const Kernel& kernel,
    const KernelArgsPackedArrayBase& args) {
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

  return cuda::ToStatus(cuGraphExecKernelNodeSetParams(
                            exec_, ToCudaGraphHandle(node_handle), &params),
                        "Failed to set CUDA graph kernel node params");
}

absl::StatusOr<GraphNodeHandle> CudaCommandBuffer::CreateBarrierNode(
    const Dependencies& dependencies) {
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
  if (parent_->GetDeviceDescription().driver_version() <
      SemanticVersion{12, 3, 0}) {
    return absl::UnimplementedError(
        "StreamBeginCaptureToGraph is not implemented for CUDA below version "
        "12.3. Therefore tracing is not supported.");
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
#endif
}

absl::Status CudaCommandBuffer::SetNodeExecutionEnabled(
    GraphNodeHandle node_handle, bool enabled) {
  // Node is enabled if value != 0, otherwise the node is disabled.
  unsigned value = enabled ? 1 : 0;
  VLOG(2) << "Set CUDA executable graph " << exec_ << " node " << node_handle
          << " enabled flag to " << value;
  return cuda::ToStatus(
      cuGraphNodeSetEnabled(exec_, ToCudaGraphHandle(node_handle), value),
      "Failed to set CUDA graph node enabled flag");
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
  TF_RETURN_IF_ERROR(CommandBuffer::Launch(*noop, kDefaultExecutionScope,
                                           ThreadDim(), BlockDim()));

  return absl::OkStatus();
}

absl::StatusOr<GraphConditionalHandle>
CudaCommandBuffer::CreateConditionalHandle() {
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
        cuda::ToStatus(cuDeviceGet(&device, parent_->device_ordinal()),
                       "Failed call to cuDeviceGet"));
    TF_RETURN_IF_ERROR(cuda::ToStatus(cuDeviceGraphMemTrim(device),
                                      "Failed to trim device graph memory"));
    TF_RETURN_IF_ERROR(GraphInstantiate(&exec_, graph_));
  } else {
    TF_RETURN_IF_ERROR(instantiated);
  }

  return absl::OkStatus();
}

std::unique_ptr<ScopedUpdateMode> CudaCommandBuffer::ActivateUpdateMode(
    GpuCommandBuffer* nested_cmd_buffer) {
  auto nested_cuda_cmd_buffer =
      static_cast<CudaCommandBuffer*>(nested_cmd_buffer);

  auto scoped_graph_exec = std::make_unique<ScopedCudaGraphExec>(
      &nested_cuda_cmd_buffer->exec_,
      &nested_cuda_cmd_buffer->is_owned_graph_exec_);

  // We need to store the graph exec handle in the nested command buffer.
  // The scoped_graph_exec will restore the old state once we are done.
  nested_cuda_cmd_buffer->exec_ = exec_;
  nested_cuda_cmd_buffer->is_owned_graph_exec_ = false;

  return std::move(scoped_graph_exec);
}

CudaCommandBuffer::~CudaCommandBuffer() {
  if (exec_ != nullptr && is_owned_graph_exec_) {
    VLOG(5) << "Destroy GPU command buffer executable graph " << exec_ << " "
            << "(remaining alive executable graphs: " << NotifyExecDestroyed()
            << ")";
    if (auto status = cuda::ToStatus(cuGraphExecDestroy(exec_),
                                     "Failed to destroy CUDA executable graph");
        !status.ok()) {
      LOG(ERROR) << status.message();
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

absl::StatusOr<std::vector<GraphNodeHandle>>
CudaCommandBuffer::GetNodeDependencies(GraphNodeHandle node) {
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

  std::vector<GraphNodeHandle> result;
  result.reserve(dependencies.size());
  absl::c_transform(
      dependencies, std::back_inserter(result),
      static_cast<GraphNodeHandle (*)(CUgraphNode)>(&FromCudaGraphHandle));

  return result;
}

}  // namespace stream_executor::gpu
