/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/runtime2/graph.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "third_party/iree/runtime/src/iree/vm/api.h"  // IWYU pragma: keep
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/runtime2/hal.h"
#include "tensorflow/compiler/xla/service/gpu/runtime2/kernel.h"
#include "tensorflow/compiler/xla/service/gpu/runtime2/vm.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_graph.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_types.h"
#include "tensorflow/compiler/xla/stream_executor/kernel.h"
#include "tensorflow/compiler/xla/stream_executor/launch_dim.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"

namespace xla::gpu {

//===-----------------------------------------------------------------------===/
// XLA:GPU graph API
//===-----------------------------------------------------------------------===/

StatusOr<se::gpu::OwnedGpuGraph> CreateGraph() {
  return se::gpu::CreateGpuGraph();
}

StatusOr<se::gpu::GpuGraphNodeHandle> CreateKernelNode(
    const vm::ExecutionContext& ctx, vm::Graph& graph,
    absl::Span<vm::GraphNode*> dependencies, vm::Kernel& kernel,
    iree_hal_allocator_t* device_allocator,
    absl::Span<iree_hal_buffer_view_t*> args, const LaunchDimensions& dims) {
  se::Stream* stream = ctx.run_options->stream();
  se::StreamExecutor* executor = stream->parent();

  absl::MutexLock lock(&kernel.mutex);
  se::KernelBase* loaded_kernel = nullptr;

  if (auto it = kernel.loaded.find(executor); it != kernel.loaded.end()) {
    loaded_kernel = it->second.get();
  } else {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<se::KernelBase> kernel_base,
        CreateKernel(kernel.kernel_name, args.size(), ctx.executable_source.ptx,
                     ctx.executable_source.cubin, executor,
                     kernel.shared_memory_bytes));
    loaded_kernel = (kernel.loaded[executor] = std::move(kernel_base)).get();
  }

  absl::InlinedVector<se::DeviceMemoryBase, 8> device_args;
  for (iree_hal_buffer_view_t* arg : args) {
    TF_ASSIGN_OR_RETURN(device_args.emplace_back(),
                        GetDeviceMemory(device_allocator, arg));
  }

  absl::InlinedVector<se::gpu::GpuGraphNodeHandle, 4> deps;
  for (auto* node : dependencies) deps.push_back(node->handle);

  LaunchDimensions::Dim3D thread_counts = dims.thread_counts_per_block();
  LaunchDimensions::Dim3D block_counts = dims.block_counts();

  static constexpr int kKernelArgsLimit = 1024;
  std::unique_ptr<se::KernelArgsArrayBase> kernel_args;

  // The KernelArgsArray structure requires at a minimum 48 * args.size()
  // bytes. It can be expensive to allocate, say, 48KiB, so we add
  // specializations for smaller sizes. 64 arguments are likely to fit in a
  // 4KiB page.
  if (args.size() <= 64) {
    kernel_args = se::MakeKernelArgs<64>(device_args, dims.SharedMemBytes());
  } else if (args.size() <= 256) {
    kernel_args = se::MakeKernelArgs<256>(device_args, dims.SharedMemBytes());
  } else {
    kernel_args = se::MakeKernelArgs<kKernelArgsLimit>(device_args,
                                                       dims.SharedMemBytes());
  }

  return se::gpu::AddKernelNode(
      &*graph.graph, absl::MakeSpan(deps),
      se::ThreadDim(thread_counts.x, thread_counts.y, thread_counts.z),
      se::BlockDim(block_counts.x, block_counts.y, block_counts.z),
      *loaded_kernel, *kernel_args);
}

StatusOr<se::gpu::GpuGraphNodeHandle> CreateMemcpyD2DNode(
    const vm::ExecutionContext& ctx, vm::Graph& graph,
    absl::Span<vm::GraphNode*> dependencies,
    iree_hal_allocator_t* device_allocator, iree_hal_buffer_view_t* dst,
    iree_hal_buffer_view_t* src) {
  se::Stream* stream = ctx.run_options->stream();
  se::StreamExecutor* executor = stream->parent();

  se::gpu::GpuExecutor* gpu_executor = se::gpu::ExtractGpuExecutor(executor);

  absl::InlinedVector<se::gpu::GpuGraphNodeHandle, 4> deps;
  for (auto* node : dependencies) deps.push_back(node->handle);

  TF_ASSIGN_OR_RETURN(auto dst_mem, GetDeviceMemory(device_allocator, dst));
  TF_ASSIGN_OR_RETURN(auto src_mem, GetDeviceMemory(device_allocator, src));

  return se::gpu::AddMemcpyD2DNode(gpu_executor->gpu_context(), &*graph.graph,
                                   absl::MakeSpan(deps), dst_mem, src_mem);
}

Status ExecuteGraph(const vm::ExecutionContext& ctx, vm::Graph& graph) {
  TF_ASSIGN_OR_RETURN(auto exec,
                      se::gpu::InstantiateGpuGraph(std::move(graph.graph)));
  return exec.Launch(ctx.run_options->stream());
}

//===-----------------------------------------------------------------------===/
// XLA:GPU gemm custom module API
//===-----------------------------------------------------------------------===/

namespace vm {

GraphAPI::GraphAPI(iree_hal_allocator_t* device_allocator)
    : device_allocator_(device_allocator) {}

iree::StatusOr<iree::vm::ref<vm::Graph>> GraphAPI::GraphCreate(
    iree::vm::ref<ExecutionContext> ctx) {
  auto graph = CreateGraph();
  if (!graph.ok()) return FromStatus(graph.status());

  auto ref = iree::vm::make_ref<vm::Graph>();
  ref->graph = std::move(*graph);
  return ref;
}

iree::StatusOr<iree::vm::ref<vm::GraphNode>> GraphAPI::GraphKernelNodeCreate(
    iree::vm::ref<ExecutionContext> ctx, iree::vm::ref<Graph> graph,
    iree::vm::ref<iree_vm_list_t> dependencies, iree::vm::ref<Kernel> kernel,
    iree::vm::ref<iree_vm_list_t> args,
    // Workgroup size (block size)
    int32_t workgroup_size_x, int32_t workgroup_size_y,
    int32_t workgroup_size_z,
    // Workload size (grid size)
    int32_t workload_size_x, int32_t workload_size_y, int32_t workload_size_z) {
  // Kernel launch dimensions + shared memory requirement.
  LaunchDimensions launch_dimensions(
      {workload_size_x, workload_size_y, workload_size_z},
      {workgroup_size_x, workgroup_size_y, workgroup_size_z});
  launch_dimensions.SetSharedMemBytes(kernel->shared_memory_bytes);

  IREE_ASSIGN_OR_RETURN(auto deps, GetGraphNodeVector(dependencies.get()));
  IREE_ASSIGN_OR_RETURN(auto buffer_views, GetBufferViewVector(args.get()));

  auto node = CreateKernelNode(*ctx, *graph, absl::MakeSpan(deps), *kernel,
                               device_allocator_, absl::MakeSpan(buffer_views),
                               launch_dimensions);
  if (!node.ok()) return FromStatus(node.status());

  auto ref = iree::vm::make_ref<vm::GraphNode>();
  ref->handle = std::move(*node);
  return ref;
}

iree::StatusOr<iree::vm::ref<vm::GraphNode>> GraphAPI::GraphMemcpyD2DNodeCreate(
    iree::vm::ref<ExecutionContext> ctx, iree::vm::ref<Graph> graph,
    iree::vm::ref<iree_vm_list_t> dependencies,
    iree::vm::ref<iree_hal_buffer_view_t> dst,
    iree::vm::ref<iree_hal_buffer_view_t> src) {
  IREE_ASSIGN_OR_RETURN(auto deps, GetGraphNodeVector(dependencies.get()));
  auto node = CreateMemcpyD2DNode(*ctx, *graph, absl::MakeSpan(deps),
                                  device_allocator_, dst.get(), src.get());
  if (!node.ok()) return FromStatus(node.status());

  auto ref = iree::vm::make_ref<vm::GraphNode>();
  ref->handle = std::move(*node);
  return ref;
}

iree::Status GraphAPI::GraphExecute(iree::vm::ref<ExecutionContext> ctx,
                                    iree::vm::ref<Graph> graph) {
  return FromStatus(ExecuteGraph(*ctx, *graph));
}

iree::StatusOr<absl::InlinedVector<GraphNode*, 4>> GraphAPI::GetGraphNodeVector(
    iree_vm_list_t* list) {
  iree_host_size_t size = iree_vm_list_size(list);
  absl::InlinedVector<GraphNode*, 4> vector(size);

  for (iree_host_size_t i = 0; i < size; ++i) {
    iree_vm_ref_t ref{nullptr};
    IREE_RETURN_IF_ERROR(iree_vm_list_get_ref_assign(list, i, &ref));
    IREE_RETURN_IF_ERROR(graph_node_check_deref(ref, &vector[i]));
  }
  return vector;

  return vector;
}

}  // namespace vm
}  // namespace xla::gpu

//===----------------------------------------------------------------------===//
// Register types with IREE VM
//===----------------------------------------------------------------------===//

IREE_VM_DEFINE_TYPE_ADAPTERS(graph, xla::gpu::vm::Graph);
IREE_VM_DEFINE_TYPE_ADAPTERS(graph_node, xla::gpu::vm::GraphNode);
