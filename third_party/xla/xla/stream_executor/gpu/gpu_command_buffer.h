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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_COMMAND_BUFFER_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_COMMAND_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor_internal.h"

namespace stream_executor::gpu {

// GpuCommandBuffer provides platform-specific CommandBuffer implementation
// (it's backed by CUDA or HIP graphs on NVIDIA and AMD devices).
class GpuCommandBuffer : public CommandBuffer {
 public:
  GpuCommandBuffer(Mode mode, GpuExecutor* parent, GpuGraphHandle graph,
                   bool is_owned_graph = true);
  ~GpuCommandBuffer() override;

  absl::Status Barrier(StreamExecutor* executor) override;

  absl::Status Launch(const ThreadDim& threads, const BlockDim& blocks,
                      const Kernel& kernel, const KernelArgs& args) override;

  absl::Status AddNestedCommandBuffer(const CommandBuffer& nested) override;

  absl::Status MemcpyDeviceToDevice(DeviceMemoryBase* dst,
                                    const DeviceMemoryBase& src,
                                    uint64_t size) override;

  absl::Status Memset(DeviceMemoryBase* dst, BitPattern bit_pattern,
                      size_t num_elements) override;

  absl::StatusOr<DeviceMemoryBase> Allocate(size_t bytes) override;

  absl::Status Free(DeviceMemoryBase dst) override;

  absl::Status If(StreamExecutor* executor, DeviceMemory<bool> predicate,
                  Builder then_builder) override;

  absl::Status IfElse(StreamExecutor* executor, DeviceMemory<bool> predicate,
                      Builder then_builder, Builder else_builder) override;

  absl::Status Case(StreamExecutor* executor, DeviceMemory<int32_t> index,
                    std::vector<Builder> branches) override;

  absl::Status For(StreamExecutor* executor, int32_t num_iteration,
                   DeviceMemory<int32_t> loop_counter,
                   Builder body_builder) override;

  absl::Status While(StreamExecutor* executor, DeviceMemory<bool> pred,
                     Builder cond_builder, Builder body_builder) override;

  absl::Status Finalize() override;
  absl::Status Update() override;

  GpuGraphExecHandle executable() const { return exec_; }
  GpuGraphHandle graph() const { return graph_; }

  Mode mode() const override { return mode_; }
  State state() const override { return state_; }

  static GpuCommandBuffer* Cast(CommandBuffer* command_buffer) {
    return static_cast<GpuCommandBuffer*>(command_buffer);
  }

  static const GpuCommandBuffer* Cast(const CommandBuffer* command_buffer) {
    return static_cast<const GpuCommandBuffer*>(command_buffer);
  }

 private:
  absl::Status Trace(Stream* stream,
                     absl::AnyInvocable<absl::Status()> function) override;

  // We track the total number of allocated and alive executable graphs in the
  // process to track the command buffers resource usage. Executable graph
  // allocates resources on a GPU devices (rule of thumb is ~8kb per node), so
  // we have to be careful not to keep too many of them alive for too long, or
  // we have a higher risk of OOM errors.
  //
  // TODO(ezhulenev): We need to have a policy for how to evict unused
  // executable graph instances from a device, currently lifetime of an
  // executable graph is tied to a parent command buffer, and we can have
  // thousands of command buffers alive at the same time.
  static int64_t AllocatedExecs();
  static int64_t AliveExecs();

 private:
  using Dependencies = absl::InlinedVector<GpuGraphNodeHandle, 1>;

  using NoOpKernel = TypedKernel<>;

  // A signature of a device kernels updating conditional handle(s).
  using SetIfConditionKernel =
      TypedKernel<GpuGraphConditionalHandle, DeviceMemory<bool>>;

  using SetIfElseConditionKernel =
      TypedKernel<GpuGraphConditionalHandle, GpuGraphConditionalHandle,
                  DeviceMemory<bool>>;

  using SetCaseConditionKernel =
      TypedKernel<GpuGraphConditionalHandle, GpuGraphConditionalHandle,
                  GpuGraphConditionalHandle, GpuGraphConditionalHandle,
                  GpuGraphConditionalHandle, GpuGraphConditionalHandle,
                  GpuGraphConditionalHandle, GpuGraphConditionalHandle,
                  DeviceMemory<int32_t>, int32_t>;

  using SetForConditionKernel =
      TypedKernel<GpuGraphConditionalHandle, DeviceMemory<int32_t>, int32_t>;

  using SetWhileConditionKernel =
      TypedKernel<GpuGraphConditionalHandle, DeviceMemory<bool>>;

  // A callback to launch a kernel that updates conditional handles state.
  using SetConditionFn =
      std::function<absl::Status(absl::Span<const GpuGraphConditionalHandle>)>;

  // An extension of `Builder` for building conditional command buffers tied to
  // conditional handles.
  using ConditionBuilder =
      std::function<absl::Status(CommandBuffer*, GpuGraphConditionalHandle)>;

  // Wraps a regular command buffer builder into condition builder.
  static ConditionBuilder ToConditionBuilder(Builder builder);

  using ConditionType = typename GpuDriver::GpuGraphConditionalNodeParams::Type;

  // Overwrites the `exec_` handle in a Gpu command buffer by `exec`, and
  // restores to the original handle when destroyed. This allows us updating
  // primary graph executable using nested command buffers (command buffers that
  // do not have their own executable), which is required for updating
  // conditional commands.
  struct ScopedGpuGraphExec {
    ScopedGpuGraphExec(GpuCommandBuffer* cmd_buffer, GpuGraphExecHandle exec);
    ~ScopedGpuGraphExec();

    GpuCommandBuffer* cmd_buffer;
    GpuGraphExecHandle restore;
    bool restore_is_owned;
  };

  // For each conditional node in the Gpu graph we keep a record of conditional
  // command buffers attached to a node, so we can apply updates to them.
  struct ConditionalCommandBuffers {
    ConditionalCommandBuffers(
        std::vector<GpuGraphConditionalHandle> handles,
        std::vector<std::unique_ptr<GpuCommandBuffer>> command_buffers)
        : handles(std::move(handles)),
          command_buffers(std::move(command_buffers)) {}

    std::vector<GpuGraphConditionalHandle> handles;
    std::vector<std::unique_ptr<GpuCommandBuffer>> command_buffers;
  };

  using AllocationResult = std::pair<GpuDevicePtr, uint64_t>;

  absl::StatusOr<std::vector<GpuGraphConditionalHandle>>
  CreateConditionalHandles(size_t num_handles);

  absl::StatusOr<std::vector<GpuGraphHandle>> CreateConditionalNodes(
      ConditionType type, absl::Span<const GpuGraphConditionalHandle> handles);

  absl::StatusOr<std::vector<std::unique_ptr<GpuCommandBuffer>>>
  CreateConditionalCommandBuffers(
      absl::Span<const GpuGraphConditionalHandle> handles,
      absl::Span<const GpuGraphHandle> graphs,
      absl::Span<const ConditionBuilder> builders);

  absl::Status UpdateConditionalCommandBuffers(
      absl::Span<const GpuGraphConditionalHandle> handles,
      absl::Span<const std::unique_ptr<GpuCommandBuffer>> command_buffers,
      absl::Span<const ConditionBuilder> builders);

  absl::Status CreateConditionalCommand(
      StreamExecutor* executor, ConditionType type,
      SetConditionFn set_condition,
      absl::Span<const ConditionBuilder> builders);

  Dependencies GetBarrier();

  // Returns loaded no-op kernel used as a barrier, or loads it on a given
  // stream executor. Loaded kernel owned by a current command buffer.
  absl::StatusOr<NoOpKernel*> GetNoOpKernel(StreamExecutor* executor);

  // Recursively disable all nodes corresponding to barriers (including nested
  // conditional command buffers). This is work around the fact that we can't
  // use empty nodes inside conditional CUDA graphs and instead we add no-op
  // kernel nodes, however large number of no-op kernels impacts performance.
  absl::Status DisableBarriersExecution(GpuGraphExecHandle exec);

  // Launches CUDA kernels with packed arguments.
  absl::Status LaunchWithPackedArgs(
      const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
      const KernelArgsPackedArrayBase& packed_args);

  // Returns OK status if command buffer is not finalized and it is still
  // possible to add new commands to it, otherwise returns internal error.
  absl::Status CheckNotFinalized();

  // Returns OK status if the number of command buffers is equal to the expected
  // one, otherwise returns internal error.
  absl::Status CheckNumCommandBuffers(
      const ConditionalCommandBuffers& cmd_buffers, size_t num_cmd_buffers);

  static_assert(std::is_pointer_v<GpuGraphHandle>,
                "GpuGraphHandle must be a pointer");
  static_assert(std::is_pointer_v<GpuGraphExecHandle>,
                "GpuGraphExecHandle must be a pointer");
  static_assert(std::is_pointer_v<GpuGraphNodeHandle>,
                "GpuGraphNodeHandle must be a pointer");

  Mode mode_;
  State state_ = State::kCreate;

  GpuExecutor* parent_;  // not owned, must outlive *this

  GpuGraphHandle graph_ = nullptr;  // owned if `is_owned_graph_`
  bool is_owned_graph_ = true;      // ownership of `graph_`

  GpuGraphExecHandle exec_ = nullptr;  // owned if `is_owned_graph_exec_`
  bool is_owned_graph_exec_ = true;    // ownership of `is_owned_graph_exec_`

  // Handle of a graph node that acts as a barrier for all newly added commands.
  GpuGraphNodeHandle barrier_ = nullptr;

  // A handle to a Gpu graph node and a metadata describing the node properties.
  struct GpuGraphNodeInfo {
    // Gpu graph node handle owned by `graph_` instance.
    GpuGraphNodeHandle handle = nullptr;
  };

  // Gpu graph nodes info for load bearing graph nodes (kernel, memcpy, etc.)
  // corresponding to command buffer commands and also to no-op nodes
  // corresponding to barriers (nodes defining DAG structure).
  std::vector<GpuGraphNodeInfo> nodes_;

  // Handles to no-op graph nodes corresponding to barriers that define nodes
  // execution order. Can be nullptr if regular node acts as a barrier.
  std::vector<GpuGraphNodeHandle> barriers_;

  // Command buffers for conditional nodes in the Gpu graph. Underlying Gpu
  // graphs owned by the `graph_` instance.
  std::vector<ConditionalCommandBuffers> conditional_command_buffers_;

  // Track the number of command buffer updates for debugging.
  int64_t num_updates_ = 0;

  // Tracks indices into internal data structures during command buffer updates.
  struct UpdateState {
    // Index points to the graph node inside `nodes_` that will be updated next.
    int64_t node_idx = 0;

    // Index points to the barrier node inside `barriers_` that will be updated
    // on a next call to `Barrier()`.
    int64_t barrier_idx = 0;

    // Index points to the conditional command buffers that will be updated next
    // when we'll be updating next conditional command (If, Case, While).
    int64_t conditional_idx = 0;
  };

  UpdateState update_state_;

  // Loaded instance of a no-op kernel used as command buffer barrier.
  NoOpKernel noop_kernel_;
};

//===----------------------------------------------------------------------===//
// Implementation details device kernels required by GpuCommandBuffer.
//===----------------------------------------------------------------------===//

// A no-op kernel required for creating barriers inside command buffers because
// empty nodes are not supported within conditional CUDA graphs (in CUDA 12.3).
void* GetNoOpKernel();

// See `cuda_conditional_kernels.cu.cc` for CUDA implementations. These are
// various kernels that update Gpu conditionals based on the device memory
// values, and allow implementing on-device control flow via conditional command
// buffers.

void* GetSetIfConditionKernel();
void* GetSetIfElseConditionKernel();
void* GetSetCaseConditionKernel();
void* GetSetForConditionKernel();
void* GetSetWhileConditionKernel();

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_COMMAND_BUFFER_H_
