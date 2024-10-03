/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_COMMAND_BUFFER_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_COMMAND_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "rocm/include/hip/driver_types.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"

namespace stream_executor::gpu {

// RocmCommandBuffer provides platform-specific CommandBuffer implementation
// (it's backed by CUDA or HIP graphs on NVIDIA and AMD devices).
class RocmCommandBuffer : public CommandBuffer {
 public:
  // A handle to a Gpu graph node and a metadata describing its properties. Each
  // command (launch, memcpy, etc.) creates one or more graph nodes.
  struct RocmGraphNodeInfo {
    // A handle to the gpu graph node corresponding to a command.
    hipGraphNode_t handle = nullptr;
  };

  // A handle to Gpu graph barrier and metadata describing its properties. Each
  // call to `Barrier` creates a new barrier record.
  struct RocmGraphBarrierInfo {
    // A handle to graph node acting as a barrier that defines execution order.
    // It can be a handle to a `RocmGraphNodeInfo` node or a handle to an empty
    // node created to be a barrier. We try to reuse existing nodes as barriers
    // if possible to reduce the size of constructed gpu graphs.
    hipGraphNode_t handle = nullptr;

    // If `true` it means `handle` corresponds to an empty node specifically
    // created to act as an execution barrier, otherwise `handle` points to one
    // of the nodes created for recorded commands.
    bool is_barrier_node = true;

    // Nodes with index smaller than `nodes_offset` are synchronized with this
    // barrier. We use this offset to find nodes added after the last barrier
    // that should be added as dependencies to the next barrier.
    size_t nodes_offset = 0;
  };

  RocmCommandBuffer(Mode mode, GpuExecutor* parent, hipGraph_t graph,
                    bool is_owned_graph = true);
  ~RocmCommandBuffer() override;

  absl::Status Barrier(ExecutionScopeId execution_scope_id) override;

  absl::Status Barrier(
      absl::Span<const ExecutionScopeId> execution_scope_ids) override;

  absl::Status Barrier(ExecutionScopeId from_execution_scope_id,
                       ExecutionScopeId to_execution_scope_id) override;

  absl::Status Launch(ExecutionScopeId execution_scope_id,
                      const ThreadDim& threads, const BlockDim& blocks,
                      const Kernel& kernel, const KernelArgs& args) override;

  absl::Status AddNestedCommandBuffer(ExecutionScopeId execution_scope_id,
                                      const CommandBuffer& nested) override;

  absl::Status MemcpyDeviceToDevice(ExecutionScopeId execution_scope_id,
                                    DeviceMemoryBase* dst,
                                    const DeviceMemoryBase& src,
                                    uint64_t size) override;

  absl::Status Memset(ExecutionScopeId execution_scope_id,
                      DeviceMemoryBase* dst, BitPattern bit_pattern,
                      size_t num_elements) override;

  absl::Status If(ExecutionScopeId execution_scope_id,
                  DeviceMemory<bool> predicate, Builder then_builder) override;

  absl::Status IfElse(ExecutionScopeId execution_scope_id,
                      DeviceMemory<bool> predicate, Builder then_builder,
                      Builder else_builder) override;

  absl::Status Case(ExecutionScopeId execution_scope_id,
                    DeviceMemory<int32_t> index,
                    std::vector<Builder> branches) override;

  absl::Status For(ExecutionScopeId execution_scope_id, int32_t num_iteration,
                   DeviceMemory<int32_t> loop_counter,
                   Builder body_builder) override;

  absl::Status While(ExecutionScopeId execution_scope_id,
                     DeviceMemory<bool> pred,
                     ExecutionScopeBuilder cond_builder,
                     Builder body_builder) override;

  absl::Status Finalize() override;
  absl::Status Update() override;
  absl::Status Submit(Stream* stream) override;

  hipGraphExec_t executable() const { return exec_; }
  hipGraph_t graph() const { return graph_; }

  Mode mode() const override { return mode_; }
  State state() const override { return state_; }

  static RocmCommandBuffer* Cast(CommandBuffer* command_buffer) {
    return static_cast<RocmCommandBuffer*>(command_buffer);
  }

  static const RocmCommandBuffer* Cast(const CommandBuffer* command_buffer) {
    return static_cast<const RocmCommandBuffer*>(command_buffer);
  }

  absl::Span<const RocmGraphNodeInfo> nodes(ExecutionScopeId id) const;
  absl::Span<const RocmGraphBarrierInfo> barriers(ExecutionScopeId id) const;

  absl::Span<const RocmGraphNodeInfo> nodes() const {
    return nodes(kDefaulExecutionScope);
  }

  absl::Span<const RocmGraphBarrierInfo> barriers() const {
    return barriers(kDefaulExecutionScope);
  }

 private:
  absl::Status Trace(Stream* stream,
                     absl::AnyInvocable<absl::Status()> function) override;

  // We track the total number of allocated and alive executable graphs in the
  // process to track the command buffers resource usage. Executable graph
  // allocates resources on a GPU devices (rule of thumb is ~8kb per node), so
  // we have to be careful not to keep too many of them alive for too long, or
  // we have a higher risk of OOM errors.
  static int64_t AliveExecs();

 private:
  using Dependencies = absl::InlinedVector<hipGraphNode_t, 1>;

  using NoOpKernel = TypedKernel<>;

  // Overwrites the `exec_` handle in a Gpu command buffer by `exec`, and
  // restores to the original handle when destroyed. This allows us updating
  // primary graph executable using nested command buffers (command buffers that
  // do not have their own executable), which is required for updating
  // conditional commands.
  struct ScopedGpuGraphExec {
    ScopedGpuGraphExec(RocmCommandBuffer* cmd_buffer, hipGraphExec_t exec);
    ~ScopedGpuGraphExec();

    RocmCommandBuffer* cmd_buffer;
    hipGraphExec_t restore;
    bool restore_is_owned;
  };

  using AllocationResult = std::pair<hipDeviceptr_t, uint64_t>;

  Dependencies GetBarrier(ExecutionScopeId execution_scope_id);

  // Recursively disable all nodes corresponding to barriers (including nested
  // conditional command buffers). This is work around the fact that we can't
  // use empty nodes inside conditional CUDA graphs and instead we add no-op
  // kernel nodes, however large number of no-op kernels impacts performance.
  absl::Status DisableBarriersExecution(hipGraphExec_t exec);

  // Launches CUDA kernels with packed arguments.
  absl::Status LaunchWithPackedArgs(
      ExecutionScopeId execution_scope_id, const ThreadDim& threads,
      const BlockDim& blocks, const Kernel& kernel,
      const KernelArgsPackedArrayBase& packed_args);

  // Returns OK status if command buffer is not finalized and it is still
  // possible to add new commands to it, otherwise returns internal error.
  absl::Status CheckNotFinalized();

  // Creates a new no-op node acting as a barrier.
  absl::StatusOr<hipGraphNode_t> CreateBarrierNode(
      const Dependencies& dependencies);

  // Collects a set of dependencies for a new barrier.
  Dependencies GetBarrierDependencies(ExecutionScopeId execution_scope_id);

  static_assert(std::is_pointer_v<hipGraph_t>, "hipGraph_t must be a pointer");
  static_assert(std::is_pointer_v<hipGraphExec_t>,
                "hipGraphExec_t must be a pointer");
  static_assert(std::is_pointer_v<hipGraphNode_t>,
                "hipGraphNode_t must be a pointer");

  Mode mode_;
  State state_ = State::kCreate;

  GpuExecutor* parent_;  // not owned, must outlive *this

  hipGraph_t graph_ = nullptr;  // owned if `is_owned_graph_`
  bool is_owned_graph_ = true;  // ownership of `graph_`

  hipGraphExec_t exec_ = nullptr;    // owned if `is_owned_graph_exec_`
  bool is_owned_graph_exec_ = true;  // ownership of `is_owned_graph_exec_`

  // ExecutionScope holds the state of an underlying CUDA graph (nodes an
  // barriers added to a graph) for a single execution scope.
  struct ExecutionScope {
    // Tracks indices into data structures during command buffer updates.
    struct UpdateState {
      // Index points to the graph node inside `nodes` that will be updated
      // next.
      int64_t node_idx = 0;

      // Index points to the barrier node inside `barriers` that will be updated
      // on a next call to `Barrier(...)`.
      int64_t barrier_idx = 0;

      // Index points to the conditional command buffers that will be updated
      // next when we'll be updating next conditional command (If, Case, While).
      int64_t conditional_idx = 0;
    };

    // Gpu graph nodes corresponding to recorded commands (launch, memcpy,
    // etc.).
    std::vector<RocmGraphNodeInfo> nodes;

    // Gpu graph barriers that define recorded commands execution order.
    std::vector<RocmGraphBarrierInfo> barriers;

    // Tracks execution scope update state.
    UpdateState update_state;
  };

  // Execution scopes recorded into the command buffer.
  absl::flat_hash_map<ExecutionScopeId, ExecutionScope> execution_scopes_;

  // Track the number of command buffer updates for debugging.
  int64_t num_updates_ = 0;

  // Lazy loaded auxiliary kernels required for building CUDA graphs (no-op
  // barriers, updating conditional handles, etc.).
  NoOpKernel noop_kernel_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_COMMAND_BUFFER_H_
