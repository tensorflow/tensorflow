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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/graph_conditional.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"

namespace stream_executor::gpu {

// This class implements GpuCommandBuffer for Nvidia GPUs.
class CudaCommandBuffer final : public GpuCommandBuffer {
 public:
  // Creates a new CUDA command buffer and the underlying CUDA graph.
  static absl::StatusOr<std::unique_ptr<CudaCommandBuffer>> Create(
      Mode mode, GpuExecutor* parent);

 private:
  friend class CudaGraphNode;

  CudaCommandBuffer(Mode mode, GpuExecutor* parent, CUgraph graph,
                    bool is_owned_graph)
      : GpuCommandBuffer(mode, parent, graph, is_owned_graph),
        parent_(parent) {}

  absl::Status LaunchSetIfConditionKernel(
      ExecutionScopeId execution_scope_id, GraphConditional* if_conditional,
      DeviceMemory<bool> predicate) override;
  absl::Status LaunchSetIfElseConditionKernel(
      ExecutionScopeId execution_scope_id, GraphConditional* if_conditional,
      GraphConditional* else_conditional,
      DeviceMemory<bool> predicate) override;
  absl::Status LaunchSetCaseConditionKernel(
      ExecutionScopeId execution_scope_id, GraphConditionals conditionals,
      DeviceMemory<int32_t> index, int32_t batch_offset,
      bool enable_conditional_default) override;
  absl::Status LaunchSetForConditionKernel(ExecutionScopeId execution_scope_id,
                                           GraphConditional* conditional,
                                           DeviceMemory<int32_t> loop_counter,
                                           int32_t iterations) override;
  absl::Status LaunchSetWhileConditionKernel(
      ExecutionScopeId execution_scope_id, GraphConditional* conditional,
      DeviceMemory<bool> predicate) override;
  absl::StatusOr<NoOpKernel*> GetNoOpKernel() override;

  std::unique_ptr<GpuCommandBuffer> CreateNestedCommandBuffer(
      CUgraph graph) override;

  absl::StatusOr<GpuGraphNodeInfo*> CreateMemsetNode(
      const Dependencies& dependencies, DeviceMemoryBase destination,
      BitPattern bit_pattern, size_t num_elements) override;

  absl::StatusOr<GpuGraphNodeInfo*> CreateMemcpyD2DNode(
      const Dependencies& dependencies, DeviceMemoryBase destination,
      DeviceMemoryBase source, uint64_t size) override;

  absl::StatusOr<GpuGraphNodeInfo*> CreateChildNode(
      const Dependencies& dependencies, const CommandBuffer& nested) override;

  absl::StatusOr<GpuGraphNodeInfo*> CreateKernelNode(
      const Dependencies& dependencies, const ThreadDim& threads,
      const BlockDim& blocks, const Kernel& kernel,
      const KernelArgsPackedArrayBase& args) override;

  absl::StatusOr<GpuGraphNodeInfo*> CreateBarrierNode(
      const Dependencies& dependencies) override;

  absl::Status Trace(Stream* stream,
                     absl::AnyInvocable<absl::Status()> function) override;

  absl::Status LaunchGraph(Stream* stream) override;

  absl::StatusOr<size_t> GetNodeCount() const override;

  absl::StatusOr<GraphConditional*> CreateConditionalHandle() override;

  // A signature of a device kernels updating conditional handle(s).
  using SetIfConditionKernel =
      TypedKernel<CUgraphConditionalHandle, DeviceMemory<bool>>;

  using SetIfElseConditionKernel =
      TypedKernel<CUgraphConditionalHandle, CUgraphConditionalHandle,
                  DeviceMemory<bool>>;

  using SetCaseConditionKernel =
      TypedKernel<CUgraphConditionalHandle, CUgraphConditionalHandle,
                  CUgraphConditionalHandle, CUgraphConditionalHandle,
                  CUgraphConditionalHandle, CUgraphConditionalHandle,
                  CUgraphConditionalHandle, CUgraphConditionalHandle,
                  DeviceMemory<int32_t>, int32_t, int32_t, bool>;

  using SetForConditionKernel =
      TypedKernel<CUgraphConditionalHandle, DeviceMemory<int32_t>, int32_t>;

  using SetWhileConditionKernel =
      TypedKernel<CUgraphConditionalHandle, DeviceMemory<bool>>;

  // Lazy loaded auxiliary kernels required for building CUDA graphs (no-op
  // barriers, updating conditional handles, etc.).
  SetIfConditionKernel set_if_condition_kernel_;
  SetIfElseConditionKernel set_if_else_condition_kernel_;
  SetCaseConditionKernel set_case_condition_kernel_;
  SetForConditionKernel set_for_condition_kernel_;
  SetWhileConditionKernel set_while_condition_kernel_;
  NoOpKernel noop_kernel_;

  GpuExecutor* parent_;

  std::vector<std::unique_ptr<GpuGraphNodeInfo>> node_storage_;
  std::vector<std::unique_ptr<GraphConditional>> conditionals_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_H_
