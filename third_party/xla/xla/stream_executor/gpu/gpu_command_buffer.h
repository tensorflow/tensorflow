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
#include "xla/stream_executor/stream.h"
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

  // Conditional node handle with an associated command buffer. Command buffer
  // is owned by the conditional node, and we return it back to the caller so it
  // can record (or update) commands into it.
  struct GraphConditionalNodeHandle {
    GraphNodeHandle handle;
    std::unique_ptr<GpuCommandBuffer> command_buffer;
  };

  // A simple GPU command recorded into a GPU command buffer. Most of the GPU
  // commands have a single node in the GPU graph, i.e. memset or kernel launch.
  struct GpuCommand : public CommandBuffer::Command {
    explicit GpuCommand(GraphNodeHandle handle) : handle(handle) {}

    // A handle to the gpu graph node corresponding to a command.
    GraphNodeHandle handle = nullptr;
  };

  // A GPU command recorded for the If operation.
  struct GpuIfCommand : public CommandBuffer::Command {
    GraphConditionalHandle then_conditional;
    GraphNodeHandle set_condition_node;
    GraphConditionalNodeHandle then_conditional_node;
  };

  // A GPU command recorded for the IfElse operation.
  struct GpuIfElseCommand : public CommandBuffer::Command {
    GraphConditionalHandle then_conditional;
    GraphConditionalHandle else_conditional;
    GraphNodeHandle set_condition_node;
    GraphConditionalNodeHandle then_conditional_node;
    GraphConditionalNodeHandle else_conditional_node;
  };

  // A GPU command recorded for the Case operation.
  struct GpuCaseCommand : public CommandBuffer::Command {
    std::vector<GraphConditionalHandle> conditionals;
    std::vector<GraphNodeHandle> set_condition_nodes;
    std::vector<GraphConditionalNodeHandle> conditional_nodes;
    GraphNodeHandle barrier_node;
  };

  // A GPU command recorded for the While operation.
  struct GpuWhileCommand : public CommandBuffer::Command {
    GraphConditionalHandle conditional;
    GraphNodeHandle set_init_condition_node;
    GraphNodeHandle set_body_condition_node;
    GraphConditionalNodeHandle conditional_node;
  };

  GpuCommandBuffer(Mode mode, StreamExecutor* parent);

  using CommandBuffer::Launch;

  absl::StatusOr<const Command*> Launch(
      const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
      const KernelArgs& args,
      absl::Span<const Command* const> dependencies) override;

  absl::Status Launch(const Command* command, const ThreadDim& threads,
                      const BlockDim& blocks, const Kernel& kernel,
                      const KernelArgs& args) override;

  absl::StatusOr<const Command*> AddNestedCommandBuffer(
      const CommandBuffer& nested,
      absl::Span<const Command* const> dependencies) override;

  absl::Status AddNestedCommandBuffer(const Command* command,
                                      const CommandBuffer& nested) override;

  absl::StatusOr<const Command*> MemcpyDeviceToDevice(
      DeviceMemoryBase* dst, const DeviceMemoryBase& src, uint64_t size,
      absl::Span<const Command* const> dependencies) override;

  absl::Status MemcpyDeviceToDevice(const Command* command,
                                    DeviceMemoryBase* dst,
                                    const DeviceMemoryBase& src,
                                    uint64_t size) override;

  absl::StatusOr<const Command*> Memset(
      DeviceMemoryBase* dst, BitPattern bit_pattern, size_t num_elements,
      absl::Span<const Command* const> dependencies) override;

  absl::Status Memset(const Command* command, DeviceMemoryBase* dst,
                      const BitPattern& bit_pattern,
                      size_t num_elements) override;

  absl::StatusOr<const Command*> Case(
      DeviceMemory<int32_t> index, std::vector<Builder> branches,
      absl::Span<const Command* const> dependencies) override;

  absl::StatusOr<const Command*> DnnGraph(
      dnn::DnnGraph&, Stream&, absl::Span<DeviceMemoryBase> operands,
      absl::Span<const Command* const> dependencies) override;

  absl::Status DnnGraph(const Command*, dnn::DnnGraph&, Stream&,
                        absl::Span<DeviceMemoryBase> operands) override;

  absl::StatusOr<const Command*> Case(
      DeviceMemory<bool> index, std::vector<Builder> branches,
      absl::Span<const Command* const> dependencies) override;

  absl::Status Case(const Command* command, DeviceMemory<int32_t> index,
                    std::vector<Builder> branches) override;

  absl::Status Case(const Command* command, DeviceMemory<bool> index,
                    std::vector<Builder> branches) override;

  absl::StatusOr<const Command*> While(
      DeviceMemory<bool> pred, Builder cond_builder, Builder body_builder,
      absl::Span<const Command* const> dependencies) override;

  absl::Status While(const Command* command, DeviceMemory<bool> pred,
                     Builder cond_builder, Builder body_builder) override;

  absl::Status Finalize() override;
  absl::Status Update() override;
  absl::Status Submit(Stream* stream) override;

  Mode mode() const override { return mode_; }
  State state() const override { return state_; }

  absl::Span<const std::unique_ptr<Command>> commands() const;

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

 private:
  // Prepares a nested command buffer for an update of the graph.
  // It's a prerequisite to a call to `Update` on a nested command buffer.
  // The return value needs to be kept alive until the update is finished. An
  // update finishes by a call to `Finalize`.
  virtual std::unique_ptr<ScopedUpdateMode> ActivateUpdateMode(
      GpuCommandBuffer* nested_cmd_buffer) = 0;

  absl::StatusOr<std::vector<GraphConditionalHandle>> CreateConditionalHandles(
      size_t num_handles);

  Dependencies GetAutoDependencies() const;

  //===--------------------------------------------------------------------===//
  // APIs for launching kernels to update conditional handles.
  //===--------------------------------------------------------------------===//

  // Launches a kernel that updates the state of the given graph conditionals
  // based on the index and batch_offset. conditional[x] is set to 1 if index
  // equas `x + batch_offset` and `0` otherwise. `conditionals` may contain up
  // to 8 conditionals.
  virtual absl::StatusOr<GraphNodeHandle> CreateSetCaseConditionNode(
      absl::Span<const GraphConditionalHandle> conditionals,
      DeviceMemory<uint8_t> index, bool index_is_bool, int32_t batch_offset,
      bool enable_conditional_default,
      absl::Span<const GraphNodeHandle> dependencies) = 0;

  virtual absl::Status UpdateSetCaseConditionNode(
      GraphNodeHandle handle,
      absl::Span<const GraphConditionalHandle> conditionals,
      DeviceMemory<uint8_t> index, bool index_is_bool, int32_t batch_offset,
      bool enable_conditional_default) = 0;

  // Launches a kernel that updates the state of the given graph conditional
  // based on the predicate. If the predicate is true, `conditional` is set to
  // 1, otherwise to 0.
  virtual absl::StatusOr<GraphNodeHandle> CreateSetWhileConditionNode(
      GraphConditionalHandle conditional, DeviceMemory<bool> predicate,
      absl::Span<const GraphNodeHandle> dependencies) = 0;

  virtual absl::Status UpdateSetWhileConditionNode(
      GraphNodeHandle handle, GraphConditionalHandle conditional,
      DeviceMemory<bool> predicate) = 0;

  //===--------------------------------------------------------------------===//

  // Launches CUDA kernels with packed arguments.
  absl::StatusOr<const Command*> LaunchWithPackedArgs(
      const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
      const KernelArgsPackedArrayBase& packed_args,
      absl::Span<const Command* const> dependencies);

  // Updates a kernel launch command with packed arguments.
  absl::Status LaunchWithPackedArgs(
      const Command* command, const ThreadDim& threads, const BlockDim& blocks,
      const Kernel& kernel, const KernelArgsPackedArrayBase& packed_args);

 protected:
  // Returns OK status if command buffer is not finalized and it is still
  // possible to add new commands to it, otherwise returns internal error.
  absl::Status CheckNotFinalized();

  // Returns OK status if the command buffer can be updated.
  virtual absl::Status CheckCanBeUpdated() = 0;

 private:
  absl::StatusOr<const Command*> Case(
      DeviceMemory<uint8_t> index, bool index_is_bool,
      std::vector<Builder> branches,
      absl::Span<const Command* const> dependencies);

  absl::Status Case(const Command* command, DeviceMemory<uint8_t> index,
                    bool index_is_bool, std::vector<Builder> branches);

  // Constructs a new command for the given graph node handle and appends it to
  // the command buffer.
  const Command* AppendCommand(GraphNodeHandle handle) {
    commands_.push_back(std::make_unique<GpuCommand>(handle));
    return commands_.back().get();
  }

  // Appends a new command to the command buffer.
  template <typename T>
  const Command* AppendCommand(T command) {
    commands_.push_back(std::make_unique<T>(std::move(command)));
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

  //===--------------------------------------------------------------------===//
  // APIs for creating and updating underlying GPU graph nodes.
  //===--------------------------------------------------------------------===//

  // Adds a new conditional node to the graph and creates a corresponding nested
  // command buffer.
  virtual absl::StatusOr<GraphConditionalNodeHandle> CreateConditionalNode(
      absl::Span<const GraphNodeHandle> dependencies,
      GraphConditionalHandle conditional, ConditionType type) = 0;

  // Adds a new memset node to the underlying graph.
  virtual absl::StatusOr<GraphNodeHandle> CreateMemsetNode(
      absl::Span<const GraphNodeHandle> dependencies,
      DeviceMemoryBase destination, BitPattern bit_pattern,
      size_t num_elements) = 0;

  // Updates an existing memset node. Note that `node_handle` needs to refer
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

  virtual absl::Status PopulateDnnGraphNode(
      dnn::DnnGraph&, Stream&, absl::Span<DeviceMemoryBase> operands) = 0;

  virtual absl::Status UpdateDnnGraphNode(dnn::DnnGraph&, Stream&,
                                          absl::Span<DeviceMemoryBase> operands,
                                          GraphNodeHandle) = 0;

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

  //===--------------------------------------------------------------------===//

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
  std::vector<std::unique_ptr<Command>> commands_;

  // Tracks indices into data structures during command buffer updates.
  struct UpdateState {
    int64_t command_idx = 0;
  };

  // Tracks execution scope update state.
  UpdateState update_state_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_COMMAND_BUFFER_H_
