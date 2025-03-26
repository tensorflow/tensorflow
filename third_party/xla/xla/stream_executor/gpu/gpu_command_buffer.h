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
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/scoped_update_mode.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/casts.h"

namespace stream_executor::gpu {

// GpuCommandBuffer provides platform-specific CommandBuffer implementation
// (it's backed by CUDA or HIP graphs on NVIDIA and AMD devices).
class GpuCommandBuffer : public CommandBuffer {
  // GraphNodeHandleOpaque is an opaque type that won't be ODR used, hence
  // doesn't need to fully defined. It's an implementation detail of the
  // GraphNodeHandle defined below.
  struct GraphNodeHandleOpaque;

  // GraphConditionalOpaque is an opaque type that won't be ODR used, hence
  // doesn't need to fully defined. It's an implementation detail of the
  // GraphConditionalHandle defined below.
  struct GraphConditionalOpaque;

 public:
  // A type of a conditional command support by the GPU command buffer.
  enum class ConditionType { kIf, kWhile };

  // A graph node handle is an opaque handle that identifies a graph node in the
  // graph associated with a command buffer. GraphNodeHandles are created by
  // node factory functions and can be referenced in node update functions.
  // The handle has the same properties as a pointer (can be constructed from a
  // nullptr, trivial copyable, POD, etc.), that's why we use a pointer to
  // define it.
  using GraphNodeHandle = GraphNodeHandleOpaque*;

  // A graph conditional handle is an opaque handle that is tied to a nested
  // command buffer. Its value determines whether the nested command buffer is
  // executed or not. Set condition functions will update the conditional
  // handles values. The handle has the same properties as a pointer (can be
  // constructed from a nullptr, trivially copyable, POD, etc.), that's why we
  // use a pointer to define it.
  using GraphConditionalHandle = GraphConditionalOpaque*;
  using GraphConditionalHandles = absl::Span<const GraphConditionalHandle>;

  // A GPU command recorded into a GPU command buffer.
  struct GpuCommand : public CommandBuffer::Command {
    explicit GpuCommand(GraphNodeHandle handle) : handle(handle) {}

    // A handle to the gpu graph node corresponding to a command.
    GraphNodeHandle handle = nullptr;
  };

  // A handle to Gpu graph barrier and metadata describing its properties. Each
  // call to `Barrier` creates a new barrier record.
  struct GpuGraphBarrierInfo {
    // A handle to graph node acting as a barrier that defines execution order.
    // It can be a handle to a `GpuGraphNodeInfo` node or a handle to an empty
    // node created to be a barrier. We try to reuse existing nodes as barriers
    // if possible to reduce the size of constructed gpu graphs.
    GraphNodeHandle handle{};

    // If `true` it means `handle` corresponds to an empty node specifically
    // created to act as an execution barrier, otherwise `handle` points to one
    // of the nodes created for recorded commands.
    bool is_barrier_node = true;

    // Nodes with index smaller than `nodes_offset` are synchronized with this
    // barrier. We use this offset to find nodes added after the last barrier
    // that should be added as dependencies to the next barrier.
    size_t nodes_offset = 0;
  };

  GpuCommandBuffer(Mode mode, StreamExecutor* parent);

  absl::Status Barrier() override;

  using CommandBuffer::Launch;
  absl::Status Launch(const ThreadDim& threads, const BlockDim& blocks,
                      const Kernel& kernel, const KernelArgs& args) override;

  absl::Status AddNestedCommandBuffer(const CommandBuffer& nested) override;

  absl::Status MemcpyDeviceToDevice(DeviceMemoryBase* dst,
                                    const DeviceMemoryBase& src,
                                    uint64_t size) override;

  absl::StatusOr<const Command*> Memset(
      DeviceMemoryBase* dst, BitPattern bit_pattern, size_t num_elements,
      absl::Span<const Command* const> dependencies) override;

  absl::Status Memset(const Command* command, DeviceMemoryBase* dst,
                      const BitPattern& bit_pattern,
                      size_t num_elements) override;

  absl::Status If(DeviceMemory<bool> predicate, Builder then_builder) override;

  absl::Status IfElse(DeviceMemory<bool> predicate, Builder then_builder,
                      Builder else_builder) override;

  // Case operation that uses bool value as branch index
  absl::Status Case(DeviceMemory<bool> index,
                    std::vector<Builder> branches) override;

  // Case operation that uses int32 value as branch index
  absl::Status Case(DeviceMemory<int32_t> index,
                    std::vector<Builder> branches) override;

  absl::Status For(int32_t num_iteration, DeviceMemory<int32_t> loop_counter,
                   Builder body_builder) override;

  absl::Status While(DeviceMemory<bool> pred, Builder cond_builder,
                     Builder body_builder) override;

  absl::Status Finalize() override;
  absl::Status Update() override;
  absl::Status Submit(Stream* stream) override;

  Mode mode() const override { return mode_; }
  State state() const override { return state_; }

  absl::Span<const std::unique_ptr<GpuCommand>> commands() const;
  absl::Span<const GpuGraphBarrierInfo> barriers() const;

  // Returns the list of dependencies for a given node. `node` must be a node
  // added to the current command buffer. The returned node pointer's lifetimes
  // are bound to the current command buffer.
  virtual absl::StatusOr<std::vector<GraphNodeHandle>> GetNodeDependencies(
      GraphNodeHandle node) = 0;

 protected:
  // We track the total number of allocated and alive executable graphs in the
  // process to track the command buffers resource usage. Executable graph
  // allocates resources on a GPU devices (rule of thumb is ~8kb per node), so
  // we have to be careful not to keep too many of them alive for too long, or
  // we have a higher risk of OOM errors.
  static int64_t AliveExecs();
  static int64_t NotifyExecCreated();
  static int64_t NotifyExecDestroyed();

  using Dependencies = absl::InlinedVector<GraphNodeHandle, 1>;

  using NoOpKernel = TypedKernel<>;

 private:
  // A callback to launch a kernel that updates conditional handles state.
  using SetConditionFn =
      std::function<absl::Status(absl::Span<const GraphConditionalHandle>)>;

  // An extension of `Builder` for building conditional command buffers tied to
  // conditional handles.
  using ConditionBuilder =
      std::function<absl::Status(GpuCommandBuffer*, GraphConditionalHandle)>;

  // Wraps a regular command buffer builder into condition builder.
  static ConditionBuilder ToConditionBuilder(Builder builder);

  // Prepares a nested command buffer for an update of the graph.
  // It's a prerequisite to a call to `Update` on a nested command buffer.
  // The return value needs to be kept alive until the update is finished. An
  // update finishes by a call to `Finalize`.
  virtual std::unique_ptr<ScopedUpdateMode> ActivateUpdateMode(
      GpuCommandBuffer* nested_cmd_buffer) = 0;

  // For each conditional node in the Gpu graph we keep a record of conditional
  // command buffers attached to a node, so we can apply updates to them.
  struct ConditionalCommandBuffers {
    std::vector<GraphConditionalHandle> conditionals;
    std::vector<std::unique_ptr<GpuCommandBuffer>> command_buffers;
  };

  absl::StatusOr<std::vector<GraphConditionalHandle>> CreateConditionalHandles(
      size_t num_handles);

  absl::StatusOr<std::vector<std::unique_ptr<GpuCommandBuffer>>>
  CreateConditionalCommandBuffers(
      ConditionType type, absl::Span<const GraphConditionalHandle> conditionals,
      absl::Span<const ConditionBuilder> builders);

  absl::Status UpdateConditionalCommandBuffers(
      absl::Span<const GraphConditionalHandle> handles,
      absl::Span<const std::unique_ptr<GpuCommandBuffer>> command_buffers,
      absl::Span<const ConditionBuilder> builders);

  absl::StatusOr<std::unique_ptr<GpuCommandBuffer>>
  CreateConditionalCommandBuffer(ConditionType type,
                                 GraphConditionalHandle conditional);

  // Adds a new conditional command (If, IfElse, Case, While, For) to the
  // command buffer.
  absl::Status AddConditionalCommandNode(
      ConditionType type, SetConditionFn set_condition,
      absl::Span<const ConditionBuilder> builders);

  Dependencies GetBarrier();

  // Launches a kernels that updates the state of the given graph conditional
  // based on the predicate. If the predicate is true, `if_conditional` is set
  // to 1, otherwise to 0.
  virtual absl::Status LaunchSetIfConditionKernel(
      GraphConditionalHandle if_conditional, DeviceMemory<bool> predicate) = 0;
  // Launches a kernels that updates the state of the given graph conditionals
  // based on the predicate. If the predicate is true, `if_conditional` is set
  // to 1 and `else_conditional` to 0. If the predicate is false,
  // `if_conditional` is set to 0 and `else_conditional` to 1.
  virtual absl::Status LaunchSetIfElseConditionKernel(

      GraphConditionalHandle if_conditional,
      GraphConditionalHandle else_conditional,
      DeviceMemory<bool> predicate) = 0;
  // Launches a kernel that updates the state of the given graph conditionals
  // based on the index and batch_offset. conditional[x] is set to 1 if index
  // == x + batch_offset and 0 otherwise. `conditionals` may contain up to 8
  // conditionals
  virtual absl::Status LaunchSetCaseConditionKernel(
      GraphConditionalHandles conditionals, DeviceMemory<uint8_t> index,
      bool index_is_bool, int32_t batch_offset,
      bool enable_conditional_default) = 0;
  // Launches a kernel that updates the state of the given graph conditional
  // based on the loop counter and the total number of iterations. If the loop
  // counter is less than the number of iterations, `conditional` is set to 1,
  // otherwise to 0. The loop counter is also incremented by 1.
  virtual absl::Status LaunchSetForConditionKernel(
      GraphConditionalHandle conditional, DeviceMemory<int32_t> loop_counter,
      int32_t iterations) = 0;
  // Launches a kernel that updates the state of the given graph conditional
  // based on the predicate. If the predicate is true, `conditional` is set to
  // 1, otherwise to 0.
  virtual absl::Status LaunchSetWhileConditionKernel(
      GraphConditionalHandle conditional, DeviceMemory<bool> predicate) = 0;

  // Recursively disable all nodes corresponding to barriers (including nested
  // conditional command buffers). This is work around the fact that we can't
  // use empty nodes inside conditional CUDA graphs and instead we add no-op
  // kernel nodes, however large number of no-op kernels impacts performance.
  // The function needs access to the root command buffer which holds the
  // executable graph.
  absl::Status DisableBarriersExecution(GpuCommandBuffer& root_command_buffer);

  // Launches CUDA kernels with packed arguments.
  absl::Status LaunchWithPackedArgs(
      const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
      const KernelArgsPackedArrayBase& packed_args);

 protected:
  // Creates a nested command buffer, associated with the same executor.
  // The given graph will not be owned by the created command buffer.
  struct ConditionalNodeResult {
    GraphNodeHandle node_handle;
    std::unique_ptr<GpuCommandBuffer> command_buffer;
  };

  // Returns OK status if command buffer is not finalized and it is still
  // possible to add new commands to it, otherwise returns internal error.
  absl::Status CheckNotFinalized();

  // Returns OK status if the command buffer can be updated.
  virtual absl::Status CheckCanBeUpdated() = 0;

 private:
  // Returns OK status if the number of command buffers is equal to the expected
  // one, otherwise returns internal error.
  absl::Status CheckNumCommandBuffers(
      const ConditionalCommandBuffers& cmd_buffers, size_t num_cmd_buffers);

  // Collects a set of dependencies for a new barrier.
  Dependencies GetBarrierDependencies();

  absl::Status Case(DeviceMemory<uint8_t> index, bool index_is_bool,
                    std::vector<Builder> branches);

  // Constructs a new command for the given graph node handle and appends it to
  // the command buffer.
  const Command* AppendCommand(GraphNodeHandle handle) {
    commands_.push_back(std::make_unique<GpuCommand>(handle));
    return commands_.back().get();
  }

  // Converts a list of command dependencies to a list of graph node handles.
  Dependencies ToGraphNodeDependencies(
      absl::Span<const Command* const> dependencies) {
    Dependencies handles;
    for (const Command* dependency : dependencies) {
      auto* gpu_command = tsl::down_cast<const GpuCommand*>(dependency);
      handles.push_back(gpu_command->handle);
    }
    return handles;
  }

  // Adds a new conditional node to the graph and creates a corresponding nested
  // command buffer.
  virtual absl::StatusOr<ConditionalNodeResult> CreateConditionalNode(
      absl::Span<const GraphNodeHandle> dependencies,
      GraphConditionalHandle conditional, ConditionType type) = 0;

  // Adds a new memset node to the underlying graph.
  virtual absl::StatusOr<GraphNodeHandle> CreateMemsetNode(
      absl::Span<const GraphNodeHandle> dependencies,
      DeviceMemoryBase destination, BitPattern bit_pattern,
      size_t num_elements) = 0;

  // Updates an existing memset node. Note that `node_handle` needs to be refer
  // to a node created by `CreateMemsetNode`.
  virtual absl::Status UpdateMemsetNode(GraphNodeHandle node_handle,
                                        DeviceMemoryBase destination,
                                        BitPattern bit_pattern,
                                        size_t num_elements) = 0;

  // Adds a new memcpy node to the graph.
  virtual absl::StatusOr<GraphNodeHandle> CreateMemcpyD2DNode(
      absl::Span<const GraphNodeHandle> dependencies,
      DeviceMemoryBase destination, DeviceMemoryBase source, uint64_t size) = 0;

  virtual absl::Status UpdateMemcpyD2DNode(GraphNodeHandle node_handle,
                                           DeviceMemoryBase destination,
                                           DeviceMemoryBase source,
                                           uint64_t size) = 0;

  // Adds a new nested command buffer node to the graph.
  virtual absl::StatusOr<GraphNodeHandle> CreateChildNode(
      absl::Span<const GraphNodeHandle> dependencies,
      const CommandBuffer& nested) = 0;

  // Associate another command buffer with this child node. Will return an
  // error if the given node has not been created as a child node.
  virtual absl::Status UpdateChildNode(GraphNodeHandle node_handle,
                                       const CommandBuffer& nested) = 0;

  // Adds a new kernel launch node to the graph.
  virtual absl::StatusOr<GraphNodeHandle> CreateKernelNode(
      absl::Span<const GraphNodeHandle> dependencies, const ThreadDim& threads,
      const BlockDim& blocks, const Kernel& kernel,
      const KernelArgsPackedArrayBase& args) = 0;

  // Updates the kernel launch node with the given parameters. Will return an
  // error if the given node has not been created as a kernel launch node.
  virtual absl::Status UpdateKernelNode(
      GraphNodeHandle node_handle, const ThreadDim& threads,
      const BlockDim& blocks, const Kernel& kernel,
      const KernelArgsPackedArrayBase& args) = 0;

  // Creates a new no-op node acting as a barrier and adds it to the graph.
  virtual absl::StatusOr<GraphNodeHandle> CreateBarrierNode(
      absl::Span<const GraphNodeHandle> dependencies) = 0;

  // Enables or disables the execution of the given node in the graph.
  virtual absl::Status SetNodeExecutionEnabled(GraphNodeHandle node_handle,
                                               bool enabled) = 0;

  // Launches an instantiated graph. Only supported on primary command buffers.
  virtual absl::Status LaunchGraph(Stream* stream) = 0;

  // Returns the number of nodes in the graph associated with this command
  // buffer.
  virtual absl::StatusOr<size_t> GetNodeCount() const = 0;

  // This gets called at the beginning of `Finalize` and allows subclasses to
  // perform any necessary preparation before the graph is finalized.
  virtual absl::Status PrepareFinalization() = 0;

  // Create a new conditional handle in the underlying graph.
  virtual absl::StatusOr<GraphConditionalHandle> CreateConditionalHandle() = 0;

  // Writes the underlying graph to a file in graphviz DOT format.
  virtual absl::Status WriteGraphToDotFile(absl::string_view path) = 0;

  // Instantiates the executable graph from the underlying graph.
  virtual absl::Status InstantiateGraph() = 0;

  Mode mode_;
  State state_ = State::kCreate;

  StreamExecutor* parent_;  // not owned, must outlive *this

  // Track the number of command buffer updates for debugging.
  int64_t num_updates_ = 0;

  // Gpu commands recorded into the command buffer.
  std::vector<std::unique_ptr<GpuCommand>> commands_;

  // Gpu graph barriers that define recorded commands execution order.
  std::vector<GpuGraphBarrierInfo> barriers_;

  // Command buffers for conditional nodes in the Gpu graph. Underlying Gpu
  // graphs owned by the `graph_` instance.
  std::vector<ConditionalCommandBuffers> conditional_command_buffers_;

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

  // Tracks execution scope update state.
  UpdateState update_state_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_COMMAND_BUFFER_H_
