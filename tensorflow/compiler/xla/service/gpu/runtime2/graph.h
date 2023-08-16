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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME2_GRAPH_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME2_GRAPH_H_

#include <cstdint>

#include "absl/types/span.h"
#include "third_party/iree/runtime/src/iree/hal/api.h"  // IWYU pragma: keep
#include "third_party/iree/runtime/src/iree/vm/api.h"   // IWYU pragma: keep
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/runtime2/kernel.h"
#include "tensorflow/compiler/xla/service/gpu/runtime2/vm.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_graph.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_types.h"

namespace xla::gpu {

//===-----------------------------------------------------------------------===/
// XLA:GPU graph API custom types
//===-----------------------------------------------------------------------===/

namespace vm {

struct Graph : public iree::vm::RefObject<Graph> {
  se::gpu::OwnedGpuGraph graph;
};

struct GraphNode : public iree::vm::RefObject<GraphNode> {
  se::gpu::GpuGraphNodeHandle handle;
};

}  // namespace vm

//===-----------------------------------------------------------------------===/
// XLA:GPU graph API
//===-----------------------------------------------------------------------===/

StatusOr<se::gpu::OwnedGpuGraph> CreateGraph();

StatusOr<se::gpu::GpuGraphNodeHandle> CreateKernelNode(
    const vm::ExecutionContext& ctx, vm::Graph& graph,
    absl::Span<vm::GraphNode*> dependencies, vm::Kernel& kernel,
    iree_hal_allocator_t* device_allocator,
    absl::Span<iree_hal_buffer_view_t*> args, const LaunchDimensions& dims);

StatusOr<se::gpu::GpuGraphNodeHandle> CreateMemcpyD2DNode(
    const vm::ExecutionContext& ctx, vm::Graph& graph,
    absl::Span<vm::GraphNode*> dependencies,
    iree_hal_allocator_t* device_allocator, iree_hal_buffer_view_t* dst,
    iree_hal_buffer_view_t* src);

Status ExecuteGraph(const vm::ExecutionContext& ctx, vm::Graph& graph);

//===-----------------------------------------------------------------------===/
// XLA:GPU gemm custom module API
//===-----------------------------------------------------------------------===/

namespace vm {

class GraphAPI {
 public:
  explicit GraphAPI(iree_hal_allocator_t* device_allocator);

  iree::StatusOr<iree::vm::ref<vm::Graph>> GraphCreate(
      iree::vm::ref<ExecutionContext> ctx);

  iree::StatusOr<iree::vm::ref<vm::GraphNode>> GraphKernelNodeCreate(
      iree::vm::ref<ExecutionContext> ctx, iree::vm::ref<Graph> graph,
      iree::vm::ref<iree_vm_list_t> dependencies, iree::vm::ref<Kernel> kernel,
      iree::vm::ref<iree_vm_list_t> args,
      // Workgroup size (block size)
      int32_t workgroup_size_x, int32_t workgroup_size_y,
      int32_t workgroup_size_z,
      // Workload size (grid size)
      int32_t workload_size_x, int32_t workload_size_y,
      int32_t workload_size_z);

  iree::StatusOr<iree::vm::ref<vm::GraphNode>> GraphMemcpyD2DNodeCreate(
      iree::vm::ref<ExecutionContext> ctx, iree::vm::ref<Graph> graph,
      iree::vm::ref<iree_vm_list_t> dependencies,
      iree::vm::ref<iree_hal_buffer_view_t> dst,
      iree::vm::ref<iree_hal_buffer_view_t> src);

  iree::Status GraphExecute(iree::vm::ref<ExecutionContext> ctx,
                            iree::vm::ref<Graph> graph);

 private:
  iree::StatusOr<absl::InlinedVector<GraphNode*, 4>> GetGraphNodeVector(
      iree_vm_list_t* list);

  iree_hal_allocator_t* device_allocator_;
};

}  // namespace vm
}  // namespace xla::gpu

//===----------------------------------------------------------------------===//
// Register types with IREE VM
//===----------------------------------------------------------------------===//

IREE_VM_DECLARE_TYPE_ADAPTERS(graph, xla::gpu::vm::Graph);
IREE_VM_DECLARE_TYPE_ADAPTERS(graph_node, xla::gpu::vm::GraphNode);

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME2_GRAPH_H_
