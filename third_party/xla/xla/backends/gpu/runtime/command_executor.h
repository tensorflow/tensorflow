/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COMMAND_EXECUTOR_H_
#define XLA_BACKENDS_GPU_RUNTIME_COMMAND_EXECUTOR_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/macros.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/execution_graph.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/command_buffer.h"

namespace xla::gpu {

// Command executor is responsible for recording commands sequence into the
// underlying command buffer and setting up dependencies between commands.
class CommandExecutor {
 public:
  CommandExecutor() = default;
  CommandExecutor(CommandExecutor&&) = default;
  CommandExecutor& operator=(CommandExecutor&&) = default;

  using RecordParams = Command::RecordParams;

  // Synchronization mode defines how much concurrency is allowed between
  // commands in the sequence.
  enum class SynchronizationMode {
    // Serializes execution of all commands recorded into the command buffer
    // by adding a dependency between them.
    kSerialize,

    // Relies on execution graph to insert dependencies between commands
    // that have buffer of resource conflicts, and building a DAG of commands.
    kConcurrent,

    // Uses the same latency hidden scheduling results used in the thunk
    // scheduling.
    kLHS,
  };

  template <typename Sink>
  friend void AbslStringify(Sink& sink, SynchronizationMode mode) {
    switch (mode) {
      case SynchronizationMode::kSerialize:
        sink.Append("serialize");
        break;
      case SynchronizationMode::kConcurrent:
        sink.Append("concurrent");
        break;
      case SynchronizationMode::kLHS:
        sink.Append("lhs");
        break;
    }
  }

  // Creates a command executor from a sequence of commands using given
  // synchronization mode.
  static absl::StatusOr<CommandExecutor> Create(
      CommandSequence commands, SynchronizationMode synchronization_mode);

  // Prepares all commands added to a sequence.
  absl::Status Prepare(const Thunk::PrepareParams& params);

  // Initializes all commands added to a sequence.
  absl::Status Initialize(const Thunk::InitializeParams& params,
                          CommandStateManager& state);

  // Records commands into the command buffer.
  //
  // This method automatically switches between `RecordCreate` or `RecordUpdate`
  // depending on the command buffer state. After recording all commands into
  // the underlying command buffer it finalizes it. This is a high level API for
  // the clients of a command executor, that wants to build and executable
  // command buffer. For a more fine grained recording and updating of command
  // buffers see `RecordCreate` and `RecordUpdate` APIs defined below.
  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer);

  // Records command creation into the command buffer. Command buffer must be
  // in create state. The next command sequence recorded into the same command
  // buffer must use returned commands as dependencies, to guarantee that it is
  // correctly ordered after this command sequence.
  //
  // This Record function allows multiple CommandExecutor to be recorded into a
  // single command buffer. e.g. we can have Executor A, B, C to be recorded
  // into the same command buffer in the order of A -> B -> C. In this pattern,
  // B's source commands will depend on A's sink commands, and C's source
  // commands will also depend on B's sink commands.
  absl::StatusOr<std::vector<const se::CommandBuffer::Command*>> RecordCreate(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, se::CommandBuffer* command_buffer,
      absl::Span<const se::CommandBuffer::Command* const> dependencies) const;

  // Records command updates into the command buffer. Command buffer must be
  // in update state. Command buffer update can't change the dependency
  // structure of the underlying command buffer and can only update attributes
  // of the individual commands.
  absl::Status RecordUpdate(const Thunk::ExecuteParams& execute_params,
                            const RecordParams& record_params,
                            se::CommandBuffer* command_buffer) const;

  // Returns buffers referenced by commands in this sequence.
  const absl::flat_hash_set<BufferUse>& buffers() const;

  // Returns buffer allocations indices referenced by commands in this sequence.
  absl::Span<const BufferAllocation::Index> allocs_indices() const;

  bool empty() const { return commands_.empty(); }
  size_t size() const { return commands_.size(); }

  bool requires_initialization() const {
    return absl::c_any_of(commands_, [](const auto& cmd) {
      return cmd->requires_initialization();
    });
  }

  bool force_update() const {
    return absl::c_any_of(commands_,
                          [](const auto& cmd) { return cmd->force_update(); });
  }

  bool support_loop_unroll() const {
    return absl::c_all_of(
        commands_, [](const auto& cmd) { return cmd->support_loop_unroll(); });
  }

  // Renders the execution graph using default renderer. Returns url of the
  // rendered graph, or an error if rendering failed.
  absl::StatusOr<std::string> RenderExecutionGraph();

 private:
  // We use index into the `commands_` vector as a command id.
  using CommandId = int64_t;

  // A state associated with commands in the sequence. We rely on this state to
  // efficiently update command recorded into the command buffer.
  struct RecordState : public CommandState {
    const se::CommandBuffer::Command* command;
  };

  CommandExecutor(SynchronizationMode synchronization_mode,
                  CommandSequence commands,
                  std::optional<ExecutionGraph> execution_graph);

  absl::Status CheckCommandBufferState(
      se::CommandBuffer* command_buffer,
      se::CommandBuffer::State expected_state) const;

  // Returns true if command has no dependencies.
  bool IsSource(CommandId id) const;

  // Returns true if command is not a dependency of any other commands.
  bool IsSink(CommandId id) const;

  // Returns dependencies of the command with the given id.
  std::vector<const se::CommandBuffer::Command*> Dependencies(
      const RecordParams& record_params, se::CommandBuffer* command_buffer,
      CommandId id) const;

  SynchronizationMode synchronization_mode_;
  CommandSequence commands_;

  // In automatic synchronization mode we build an execution graph for the
  // sequence of commands and use it to set up dependencies between commands.
  std::optional<ExecutionGraph> execution_graph_;

  // Buffers referenced by commands in this sequence.
  absl::flat_hash_set<BufferUse> buffers_;

  // Unique buffer allocations indices referenced by all commands in this
  // sequence (sorted by the buffer allocation index).
  std::vector<BufferAllocation::Index> allocs_indices_;

  // A mapping from command id to unique buffer allocations indices referenced
  // by the command (sorted by the buffer allocation index).
  std::vector<std::vector<BufferAllocation::Index>> cmd_allocs_indices_;
};

using CommandBufferCmdExecutor ABSL_DEPRECATE_AND_INLINE() = CommandExecutor;

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COMMAND_EXECUTOR_H_
