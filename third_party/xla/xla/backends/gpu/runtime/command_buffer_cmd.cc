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

#include "xla/backends/gpu/runtime/command_buffer_cmd.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/collective_broadcast_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_slice_thunk.h"
#include "xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/debug_options_flags.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi_api.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/execution_graph.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/util.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla::gpu {

using MemoryAccess = BufferUse::MemoryAccess;

std::string CommandBufferCmdString(CommandBufferCmdType type) {
  switch (type) {
#define CASE_CMD_STRING(enum_name, cmd_name, ...) \
  case CommandBufferCmdType::enum_name:           \
    return cmd_name;
    COMMAND_BUFFER_CMD_LIST(CASE_CMD_STRING)
#undef CASE_CMD_STRING
    default:
      return "UnknownCmd";
  }
}

static absl::string_view ReductionKindString(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::MAX:
      return "max";
    case ReductionKind::MIN:
      return "min";
    case ReductionKind::PRODUCT:
      return "product";
    case ReductionKind::SUM:
      return "sum";
  }
}

// Create a callback to create a command buffer from a command sequence.
static se::CommandBuffer::CreateCommands CreateCommands(
    const CommandBufferCmdExecutor* commands,
    const Thunk::ExecuteParams* execute_params,
    const CommandBufferCmd::RecordParams* record_params) {
  return [=](se::CommandBuffer* command_buffer,
             absl::Span<const se::CommandBuffer::Command* const> dependencies) {
    return commands->RecordCreate(*execute_params, *record_params,
                                  command_buffer, dependencies);
  };
}

// Create callbacks to create a command buffer from command sequences.
static std::vector<se::CommandBuffer::CreateCommands> CreateCommands(
    absl::Span<const CommandBufferCmdExecutor> commands,
    const Thunk::ExecuteParams* execute_params,
    const CommandBufferCmd::RecordParams* record_params) {
  std::vector<se::CommandBuffer::CreateCommands> create_commands;
  for (const CommandBufferCmdExecutor& cmd : commands) {
    create_commands.push_back(
        CreateCommands(&cmd, execute_params, record_params));
  }
  return create_commands;
}

// Create a callback to update a command buffer with command sequence.
static se::CommandBuffer::UpdateCommands UpdateCommands(
    const CommandBufferCmdExecutor* commands,
    const Thunk::ExecuteParams* execute_params,
    const CommandBufferCmd::RecordParams* record_params) {
  return [=](se::CommandBuffer* command_buffer) {
    return commands->RecordUpdate(*execute_params, *record_params,
                                  command_buffer);
  };
}

// Create callbacks to update a command buffer with command sequence.
static std::vector<se::CommandBuffer::UpdateCommands> UpdateCommands(
    absl::Span<const CommandBufferCmdExecutor> commands,
    const Thunk::ExecuteParams* execute_params,
    const CommandBufferCmd::RecordParams* record_params) {
  std::vector<se::CommandBuffer::UpdateCommands> update_commands;
  for (const CommandBufferCmdExecutor& cmd : commands) {
    update_commands.push_back(
        UpdateCommands(&cmd, execute_params, record_params));
  }
  return update_commands;
}

//===----------------------------------------------------------------------===//
// CommandBufferCmd::RecordAction helpers.
//===----------------------------------------------------------------------===//

using CreateCommand =
    absl::FunctionRef<absl::StatusOr<const se::CommandBuffer::Command*>(
        absl::Span<const se::CommandBuffer::Command* const> dependencies)>;

using UpdateCommand =
    absl::FunctionRef<absl::Status(const se::CommandBuffer::Command* command)>;

// Handles a record action by calling one of the user-provided functions.
static absl::StatusOr<const se::CommandBuffer::Command*> Handle(
    CommandBufferCmd::RecordAction action, CreateCommand create_command,
    UpdateCommand update_command) {
  if (auto* create = std::get_if<CommandBufferCmd::RecordCreate>(&action)) {
    return create_command(create->dependencies);
  }

  if (auto* update = std::get_if<CommandBufferCmd::RecordUpdate>(&action)) {
    TF_RETURN_IF_ERROR(update_command(update->command));
    return update->command;
  }

  return Internal("Invalid record action");
}

//===----------------------------------------------------------------------===//
// CommandBufferCmd
//===----------------------------------------------------------------------===//

CommandBufferCmd::StateManager::TypeId
CommandBufferCmd::StateManager::GetNextTypeId() {
  static auto* counter = new std::atomic<int64_t>(1);
  return TypeId(counter->fetch_add(1));
}

CommandBufferCmd::State* CommandBufferCmd::StateManager::GetOrNull(
    const CommandBufferCmd* cmd, const se::CommandBuffer* command_buffer,
    TypeId type_id) {
  Key key = {cmd, command_buffer, type_id};
  if (auto it = state_.find(key); it != state_.end()) {
    return it->second.get();
  }
  return nullptr;
}

CommandBufferCmd::State* CommandBufferCmd::StateManager::GetOrCreate(
    const CommandBufferCmd* cmd, const se::CommandBuffer* command_buffer,
    TypeId type_id, absl::FunctionRef<std::unique_ptr<State>()> create) {
  Key key = {cmd, command_buffer, type_id};
  if (auto it = state_.find(key); it != state_.end()) {
    return it->second.get();
  }
  return state_.try_emplace(key, create()).first->second.get();
}

//===----------------------------------------------------------------------===//
// CommandBufferCmdSequence
//===----------------------------------------------------------------------===//

namespace {
// An adaptor from CommandBufferCmd to ExecutionGraph::Operation for building an
// execution graph from a command sequence.
class CommandOperation : public ExecutionGraph::Operation {
 public:
  explicit CommandOperation(const CommandBufferCmd* cmd)
      : name_(cmd->ToString()), buffers_(cmd->buffers()) {}

  absl::string_view name() const final { return name_; }
  absl::Span<const BufferUse> BufferUses() const final { return buffers_; }
  absl::Span<const ResourceUse> ResourceUses() const final { return {}; }

 private:
  std::string name_;
  CommandBufferCmd::BufferUseVector buffers_;
};
}  // namespace

absl::StatusOr<CommandBufferCmdExecutor> CommandBufferCmdExecutor::Create(
    CommandBufferCmdSequence commands,
    SynchronizationMode synchronization_mode) {
  std::optional<ExecutionGraph> execution_graph = std::nullopt;

  // In automatic synchronization mode construct an execution graph for the
  // sequence of commands and derive the structure of command dependencies
  // from the buffer use conflicts.
  if (synchronization_mode == SynchronizationMode::kAutomatic) {
    std::vector<CommandOperation> operations;
    operations.reserve(commands.size());
    for (const std::unique_ptr<CommandBufferCmd>& cmd : commands) {
      operations.emplace_back(cmd.get());
    }
    TF_ASSIGN_OR_RETURN(execution_graph,
                        ExecutionGraph::Create<CommandOperation>(operations));
  }

  return CommandBufferCmdExecutor(synchronization_mode, std::move(commands),
                                  std::move(execution_graph));
}

CommandBufferCmdExecutor::CommandBufferCmdExecutor(
    SynchronizationMode synchronization_mode, CommandBufferCmdSequence commands,
    std::optional<ExecutionGraph> execution_graph)
    : synchronization_mode_(synchronization_mode),
      commands_(std::move(commands)),
      execution_graph_(std::move(execution_graph)) {
  // Buffer allocations referenced by commands in this sequence.
  absl::btree_set<BufferAllocation::Index> allocs_indices;

  for (const std::unique_ptr<CommandBufferCmd>& cmd : commands_) {
    absl::btree_set<BufferAllocation::Index> cmd_allocs_indices;

    for (const BufferUse& buffer : cmd->buffers()) {
      buffers_.insert(buffer);
      allocs_indices.insert(buffer.slice().index());
      cmd_allocs_indices.insert(buffer.slice().index());
    }

    // Record buffer allocations indices referenced by the `cmd`.
    cmd_allocs_indices_.emplace_back(cmd_allocs_indices.begin(),
                                     cmd_allocs_indices.end());
  }

  // Record all buffer allocations indices referenced by all commands in this
  // sequence.
  allocs_indices_.assign(allocs_indices.begin(), allocs_indices.end());
}

absl::Status CommandBufferCmdExecutor::Prepare(
    const Thunk::PrepareParams& params,
    Thunk::ResourceRequestsInterface& resource_requests) {
  for (auto& command : commands_) {
    TF_RETURN_IF_ERROR(command->Prepare(params, resource_requests));
  }
  return absl::OkStatus();
}

absl::Status CommandBufferCmdExecutor::Initialize(
    const Thunk::InitializeParams& params,
    CommandBufferCmd::StateManager& state) {
  for (auto& command : commands_) {
    TF_RETURN_IF_ERROR(command->Initialize(params, state));
  }
  return absl::OkStatus();
}

absl::Status CommandBufferCmdExecutor::Record(
    const Thunk::ExecuteParams& execute_params,
    const CommandBufferCmd::RecordParams& record_params,
    se::CommandBuffer* command_buffer) {
  VLOG(3) << "Record " << commands_.size() << " commands into command buffer";

  if (command_buffer->state() == se::CommandBuffer::State::kFinalized) {
    TF_RETURN_IF_ERROR(command_buffer->Update());
  }

  if (command_buffer->state() == se::CommandBuffer::State::kUpdate) {
    TF_RETURN_IF_ERROR(
        RecordUpdate(execute_params, record_params, command_buffer));
  } else {
    TF_RETURN_IF_ERROR(
        RecordCreate(execute_params, record_params, command_buffer, {})
            .status());
  }

  return command_buffer->Finalize();
}

absl::StatusOr<std::vector<const se::CommandBuffer::Command*>>
CommandBufferCmdExecutor::RecordCreate(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    absl::Span<const se::CommandBuffer::Command* const> dependencies) const {
  // Command buffer must be in create state.
  TF_RETURN_IF_ERROR(CheckCommandBufferState(
      command_buffer, se::CommandBuffer::State::kCreate));

  VLOG(1) << "Create " << commands_.size() << " commands";
  uint64_t start_micros = tsl::Env::Default()->NowMicros();

  // Short-circuit if there are no commands to record.
  if (commands_.empty()) {
    return std::vector<const se::CommandBuffer::Command*>{};
  }

  // Keep a state associated with commands in the sequence in the state manager.
  CommandBufferCmd::StateManager& state = record_params.state;

  // Collect sink commands while recording the command sequence.
  std::vector<const se::CommandBuffer::Command*> sink_commands;

  for (CommandId id = 0; id < commands_.size(); ++id) {
    CommandBufferCmd* command = commands_[id].get();

    std::optional<tsl::profiler::ScopedAnnotation> annotation =
        GetKernelAnnotation(command->profile_annotation());

    // Skip recording collective commands if mock collectives are enabled.
    if (execute_params.mock_collectives &&
        dynamic_cast<CollectiveCmd*>(command)) {
      continue;
    }

    // Create new commands by recording them into the command buffer.
    DCHECK(!state.GetOrNull<RecordState>(command, command_buffer))
        << "Record state must be null for " << command->ToString();
    auto* record_state =
        state.GetOrCreate<RecordState>(command, command_buffer);

    std::vector<const se::CommandBuffer::Command*> command_dependencies =
        Dependencies(record_params, command_buffer, id);

    // Source command must depend on external dependencies passed by the caller,
    // internal commands dependencies are defined by the command sequence
    // structure (buffer and resource dependencies).
    auto record_action =
        IsSource(id) ? CommandBufferCmd::RecordCreate{dependencies}
                     : CommandBufferCmd::RecordCreate{command_dependencies};

    TF_ASSIGN_OR_RETURN(
        record_state->command,
        command->Record(execute_params, record_params, std::move(record_action),
                        command_buffer));

    // Collect sink commands as external dependencies for the next command
    // sequence recorded into the same command buffer.
    if (IsSink(id)) {
      sink_commands.push_back(record_state->command);
    }
  }

  uint64_t end_micros = tsl::Env::Default()->NowMicros();
  VLOG(1) << absl::StrFormat(
      "Created %d commands in %d μs (num sink commands: %d)", commands_.size(),
      end_micros - start_micros, sink_commands.size());

  return sink_commands;
}

absl::Status CommandBufferCmdExecutor::RecordUpdate(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params,
    se::CommandBuffer* command_buffer) const {
  // Command buffer must be already prepared for recording updates.
  TF_RETURN_IF_ERROR(CheckCommandBufferState(
      command_buffer, se::CommandBuffer::State::kUpdate));

  VLOG(1) << "Update " << commands_.size() << " commands";
  uint64_t start_micros = tsl::Env::Default()->NowMicros();

  // Short-circuit if there are no commands to update.
  if (commands_.empty()) {
    return absl::OkStatus();
  }

  // Keep a state associated with commands in the sequence in the state manager.
  CommandBufferCmd::StateManager& state = record_params.state;

  // Check if command `id` has to be updated based on the buffer allocations
  // that changed since the last call to `Record`. We keep intersection vector
  // outside of a lambda to avoid repeated heap allocations on every call.
  std::vector<BufferAllocation::Index> alloc_intersection;
  auto skip_command_update = [&](CommandId id) {
    // If we don't know what allocations changed since the last call to
    // `Record` we must always update the command.
    if (!record_params.updated_allocs) {
      return false;
    }

    // We always update commands that require initialization, even if buffer
    // allocations didn't change.
    CommandBufferCmd* command = commands_[id].get();
    if (command->requires_initialization() && record_params.is_initialization) {
      return false;
    }

    DCHECK(absl::c_is_sorted(*record_params.updated_allocs))
        << "Updated allocs must be sorted: "
        << absl::StrJoin(*record_params.updated_allocs, ", ");

    DCHECK(absl::c_is_sorted(cmd_allocs_indices_[id]))
        << "Command allocs must be sorted: "
        << absl::StrJoin(cmd_allocs_indices_[id], ", ");

    alloc_intersection.clear();
    absl::c_set_intersection(cmd_allocs_indices_[id],
                             *record_params.updated_allocs,
                             std::back_inserter(alloc_intersection));
    return alloc_intersection.empty();
  };

  size_t num_skipped_command_updates = 0;

  for (CommandId id = 0; id < commands_.size(); ++id) {
    CommandBufferCmd* command = commands_[id].get();

    std::optional<tsl::profiler::ScopedAnnotation> annotation =
        GetKernelAnnotation(command->profile_annotation());

    // Skip updating collective commands if mock collectives are enabled.
    if (execute_params.mock_collectives &&
        dynamic_cast<CollectiveCmd*>(command)) {
      continue;
    }

    // Skip updating command if it doesn't use any of the updated allocations.
    if (skip_command_update(id)) {
      ++num_skipped_command_updates;
      continue;
    }

    // Update existing commands in the command buffer.
    auto* record_state = state.GetOrNull<RecordState>(command, command_buffer);
    DCHECK(record_state) << "Record state must be not null for "
                         << command->ToString();

    auto record_action = CommandBufferCmd::RecordUpdate{record_state->command};
    TF_ASSIGN_OR_RETURN(
        record_state->command,
        command->Record(execute_params, record_params, std::move(record_action),
                        command_buffer));
  }

  uint64_t end_micros = tsl::Env::Default()->NowMicros();
  VLOG(1) << "Updated " << commands_.size() << " commands in "
          << (end_micros - start_micros) << " μs (skipped "
          << num_skipped_command_updates << " command updates)";

  return absl::OkStatus();
}

absl::Status CommandBufferCmdExecutor::CheckCommandBufferState(
    se::CommandBuffer* command_buffer,
    se::CommandBuffer::State expected_state) const {
  if (command_buffer->state() != expected_state) {
    return Internal("Command buffer must be in %v state, got %v",
                    expected_state, command_buffer->state());
  }
  return absl::OkStatus();
}

bool CommandBufferCmdExecutor::IsSource(CommandId id) const {
  return execution_graph_ ? execution_graph_->is_source(id) : id == 0;
}

bool CommandBufferCmdExecutor::IsSink(CommandId id) const {
  return execution_graph_ ? execution_graph_->is_sink(id)
                          : id + 1 == commands_.size();
}

std::vector<const se::CommandBuffer::Command*>
CommandBufferCmdExecutor::Dependencies(const RecordParams& record_params,
                                       se::CommandBuffer* command_buffer,
                                       CommandId id) const {
  // Source commands have no dependencies.
  if (IsSource(id)) {
    return {};
  }

  // Collect commands that are dependencies of the command `id`.
  absl::InlinedVector<CommandId, 4> dependencies_ids;
  if (execution_graph_) {
    for (const ExecutionGraph::NodeEdge& in_edge :
         execution_graph_->in_edges(id)) {
      dependencies_ids.push_back(in_edge.id);
    }
  } else {
    dependencies_ids.push_back(id - 1);
  }

  // Collect dependencies from the recorded command state.
  std::vector<const se::CommandBuffer::Command*> dependencies;
  for (CommandId dependency_id : dependencies_ids) {
    auto* record_state = record_params.state.GetOrNull<RecordState>(
        commands_[dependency_id].get(), command_buffer);
    DCHECK(record_state) << "Record state must be not null for "
                         << commands_[dependency_id]->ToString();

    if (record_state->command == nullptr) {
      // Some commands might end up not recording anything into the command
      // buffer, e.g. memcpy commands where source and destination are the same.
      // We have to follow dependencies of such commands to find the real
      // dependencies, so we don't record a command that is immediately ready to
      // execute, as it will create data races.
      auto deps = Dependencies(record_params, command_buffer, dependency_id);
      dependencies.insert(dependencies.end(), deps.begin(), deps.end());
    } else {
      dependencies.push_back(record_state->command);
    }
  }

  return dependencies;
}

const absl::flat_hash_set<BufferUse>& CommandBufferCmdExecutor::buffers()
    const {
  return buffers_;
}

absl::Span<const BufferAllocation::Index>
CommandBufferCmdExecutor::allocs_indices() const {
  return allocs_indices_;
}

//===----------------------------------------------------------------------===//
// TracedCommandBuffer
//===----------------------------------------------------------------------===//

TracedCommandBuffer::TracedCommandBuffer(
    const CommandBufferCmd* trace_cmd,
    CommandBufferCmd::BufferUseVector buffers, int64_t capacity)
    : trace_cmd_(trace_cmd), capacity_(capacity), entries_(capacity) {
  CHECK_GT(capacity, 0) << "capacity must be larger than 0";  // NOLINT
  // Collect unique buffer allocation indices in a set first and convert to
  // vector as flat hash set iteration has measurable overheads.
  absl::flat_hash_set<BufferAllocation::Index> allocs_indices;
  for (auto& buffer : buffers) {
    allocs_indices.insert(buffer.slice().index());
  }
  allocs_indices_.assign(allocs_indices.begin(), allocs_indices.end());
}

absl::StatusOr<se::CommandBuffer*> TracedCommandBuffer::GetOrTraceCommandBuffer(
    const BufferAllocations* buffer_allocation, se::StreamExecutor* executor,
    se::Stream* stream, absl::FunctionRef<absl::Status(se::Stream*)> trace,
    se::StreamPriority priority) {
  // Collect memory addresses for relevant allocations.
  absl::InlinedVector<se::DeviceMemoryBase, 4> allocs;
  allocs.reserve(allocs_indices_.size());
  for (auto& index : allocs_indices_) {
    allocs.emplace_back(buffer_allocation->GetDeviceAddress(index));
  }

  // Moves entry at `i` position to front and moves entries in `[0, i)` range
  // one element to the right. Returns reference to the first entry.
  auto shift_right = [&](size_t i) -> Entry& {
    if (i == 0) {
      return entries_[0];
    }

    Entry entry = std::move(entries_[i]);
    do {
      entries_[i] = std::move(entries_[i - 1]);
    } while (--i > 0);

    return entries_[0] = std::move(entry);
  };

  for (size_t i = 0; i < capacity_; ++i) {
    // Found entry for a given allocations, move it to front and return a
    // pointer to cached command buffer.
    if (ABSL_PREDICT_TRUE(absl::c_equal(entries_[i].recorded_allocs, allocs) &&
                          entries_[i].command_buffer)) {
      VLOG(6) << "Command buffer trace cache hit for command "
              << trace_cmd_->ToString();
      return shift_right(i).command_buffer.get();
    }

    // Create a new entry by calling a user-provided tracing function, move it
    // to front and return a pointer to cached command buffer.
    if (entries_[i].command_buffer == nullptr) {
      TF_ASSIGN_OR_RETURN(
          entries_[i].command_buffer,
          se::TraceCommandBufferFactory::Create(executor, stream, trace));
      entries_[i].recorded_allocs.assign(allocs.begin(), allocs.end());
      if (priority != se::StreamPriority::Default) {
        TF_RETURN_IF_ERROR(entries_[i].command_buffer->SetPriority(priority));
      }
      VLOG(6) << "Command buffer trace cache create new item for command "
              << trace_cmd_->ToString();
      return shift_right(i).command_buffer.get();
    }
  }

  // Create a new entry by calling a user-provided tracing function, replace the
  // last entry with it, move it to front and return a pointer to cached command
  // buffer.
  TF_ASSIGN_OR_RETURN(
      entries_[capacity_ - 1].command_buffer,
      se::TraceCommandBufferFactory::Create(executor, stream, trace));
  entries_[capacity_ - 1].recorded_allocs.assign(allocs.begin(), allocs.end());
  VLOG(6) << "Command buffer trace cache does replacement for command "
          << trace_cmd_->ToString();
  return shift_right(capacity_ - 1).command_buffer.get();
}

//===----------------------------------------------------------------------===//
// TracedCommandBufferCmd
//===----------------------------------------------------------------------===//

TracedCommandBufferCmd::TracedCommandBufferCmd(
    CommandBufferCmdType cmd_type, ExecutionStreamId execution_stream_id)
    : CommandBufferCmd(cmd_type, execution_stream_id) {}

absl::StatusOr<const se::CommandBuffer::Command*>
TracedCommandBufferCmd::RecordTracedCommand(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer,
    absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  auto traced_cmd = record_params.state.GetOrCreate<TracedCommandBuffer>(
      this, command_buffer, [&] {
        const auto& debug_options = xla::GetDebugOptionsFromFlags();
        return std::make_unique<TracedCommandBuffer>(
            this, buffers(), debug_options.xla_cmd_buffer_trace_cache_size());
      });

  TF_ASSIGN_OR_RETURN(
      auto nested_cmd,
      traced_cmd->GetOrTraceCommandBuffer(
          execute_params.buffer_allocations, execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, trace, priority()));

  VLOG(5) << "Record traced command into command buffer: " << command_buffer;
  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateNestedCommand(*nested_cmd, dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateNestedCommand(command, *nested_cmd);
      });
}

//===----------------------------------------------------------------------===//
// ComputationId
//===----------------------------------------------------------------------===//

ComputationIdCmd::ComputationIdCmd(ExecutionStreamId execution_stream_id,
                                   BufferAllocation::Slice dest, Kind kind)
    : CommandBufferCmd(CommandBufferCmdType::kComputationIdCmd,
                       execution_stream_id),
      dest_(dest),
      kind_(kind) {}

CommandBufferCmd::BufferUseVector ComputationIdCmd::buffers() const {
  return {{dest_, MemoryAccess::kWrite}};
}

absl::StatusOr<const se::CommandBuffer::Command*> ComputationIdCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dest_);

  GlobalDeviceId global_device_id =
      execute_params.collective_params->global_device_id;
  TF_ASSIGN_OR_RETURN(
      const DeviceAssignment::LogicalID logical_id,
      execute_params.collective_params->device_assn->LogicalIdForDevice(
          global_device_id));

  uint32_t value = kind_ == Kind::kReplica ? logical_id.replica_id
                                           : logical_id.computation_id;

  VLOG(5) << "ComputationIdCmd"
          << ": kind=" << (kind_ == Kind::kReplica ? "replica" : "partition")
          << "; value=" << value;
  VLOG(5) << "  Id: " << dest_ << " (" << dst.opaque() << ")";

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateMemset(&dst, value, /*num_elements=*/1,
                                            dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateMemset(command, &dst, value,
                                            /*num_elements=*/1);
      });
}

//===----------------------------------------------------------------------===//
// LaunchCmd
//===----------------------------------------------------------------------===//

LaunchCmd::LaunchCmd(ExecutionStreamId execution_stream_id,
                     std::string kernel_name,
                     absl::Span<const BufferAllocation::Slice> args,
                     absl::Span<const MemoryAccess> args_access,
                     LaunchDimensions dims, int64_t shmem_bytes)
    : CommandBufferCmd(CommandBufferCmdType::kLaunchCmd, execution_stream_id),
      kernel_name_(std::move(kernel_name)),
      args_(args.begin(), args.end()),
      args_access_(args_access.begin(), args_access.end()),
      dims_(dims),
      shmem_bytes_(shmem_bytes) {}

absl::Status LaunchCmd::Initialize(const Thunk::InitializeParams& params,
                                   StateManager& state) {
  {
    absl::MutexLock lock(&mutex_);
    if (kernels_.contains(params.executor)) {
      return absl::OkStatus();
    }
  }

  std::unique_ptr<se::Kernel> kernel;
  if (!params.src.binary.empty()) {
    TF_ASSIGN_OR_RETURN(
        kernel, CreateKernel(kernel_name_, args_.size(), params.src.binary,
                             params.executor, shmem_bytes_));

  } else {
    TF_ASSIGN_OR_RETURN(
        kernel, CreateKernel(kernel_name_, args_.size(), params.src.text,
                             params.executor, shmem_bytes_));
  }

  absl::MutexLock lock(&mutex_);
  kernels_.emplace(params.executor, std::move(kernel));
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> LaunchCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  VLOG(5) << "LaunchCmd: kernel=" << kernel_name_
          << "; shmem_bytes=" << shmem_bytes_;

  se::Kernel* kernel = [&] {
    absl::MutexLock lock(&mutex_);
    return kernels_[execute_params.stream->parent()].get();
  }();

  if (kernel == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Kernel not loaded on a command buffer executor: ", kernel_name_));
  }

  absl::InlinedVector<se::DeviceMemoryBase, 4> buffers;
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceMemoryBase buf =
        execute_params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(5) << "  Arg: " << arg << ": " << buf.opaque();
    buffers.push_back(buf);
  }

  TF_ASSIGN_OR_RETURN(
      auto kernel_args,
      se::PackKernelArgs<se::DeviceMemoryBase>(buffers, shmem_bytes_));

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateLaunch(
            dims_.thread_counts_per_block(), dims_.block_counts(), *kernel,
            *kernel_args, dependencies, priority());
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateLaunch(
            command, dims_.thread_counts_per_block(), dims_.block_counts(),
            *kernel, *kernel_args);
      });
}

CommandBufferCmd::BufferUseVector LaunchCmd::buffers() const {
  BufferUseVector buffers;
  for (int32_t i = 0; i < args_.size(); ++i) {
    buffers.emplace_back(args_[i], args_access_[i]);
  }
  return buffers;
}

//===----------------------------------------------------------------------===//
// CustomKernelLaunchCmd
//===----------------------------------------------------------------------===//

CustomKernelLaunchCmd::CustomKernelLaunchCmd(
    ExecutionStreamId execution_stream_id,
    absl::Span<const BufferAllocation::Slice> args,
    absl::Span<const MemoryAccess> args_access, CustomKernel custom_kernel)
    : CommandBufferCmd(CommandBufferCmdType::kCustomKernelLaunchCmd,
                       execution_stream_id),
      args_(args.begin(), args.end()),
      args_access_(args_access.begin(), args_access.end()),
      custom_kernel_(std::move(custom_kernel)) {}

absl::Status CustomKernelLaunchCmd::Initialize(
    const Thunk::InitializeParams& params, StateManager& state) {
  {
    absl::MutexLock lock(&mutex_);
    if (kernels_.contains(params.executor)) {
      return absl::OkStatus();
    }
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::Kernel> kernel,
      params.executor->LoadKernel(custom_kernel_.kernel_spec()));

  absl::MutexLock lock(&mutex_);
  kernels_.emplace(params.executor, std::move(kernel));
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> CustomKernelLaunchCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  VLOG(5) << "CustomKernelLaunchCmd: custom_kernel=" << custom_kernel_.name();

  se::Kernel* kernel = [&] {
    absl::MutexLock lock(&mutex_);
    return kernels_[execute_params.stream->parent()].get();
  }();

  if (kernel == nullptr) {
    return absl::InternalError(
        absl::StrCat("Custom kernel not loaded on a command buffer executor: ",
                     custom_kernel_.name()));
  }

  absl::InlinedVector<se::DeviceMemoryBase, 4> buffers;
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceMemoryBase buf =
        execute_params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(5) << "  Arg: " << arg << ": " << buf.opaque();
    buffers.push_back(buf);
  }

  se::KernelArgsDeviceMemoryArray kernel_args(
      buffers, custom_kernel_.shared_memory_bytes());

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateLaunch(
            custom_kernel_.thread_dims(), custom_kernel_.block_dims(), *kernel,
            kernel_args, dependencies, priority());
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateLaunch(
            command, custom_kernel_.thread_dims(), custom_kernel_.block_dims(),
            *kernel, kernel_args);
      });
}

CommandBufferCmd::BufferUseVector CustomKernelLaunchCmd::buffers() const {
  BufferUseVector buffers;
  for (int32_t i = 0; i < args_.size(); ++i) {
    buffers.emplace_back(args_[i], args_access_[i]);
  }
  return buffers;
}

//===----------------------------------------------------------------------===//
// MemcpyDeviceToDeviceCmd
//===----------------------------------------------------------------------===//

MemcpyDeviceToDeviceCmd::MemcpyDeviceToDeviceCmd(
    ExecutionStreamId execution_stream_id, BufferAllocation::Slice dst,
    BufferAllocation::Slice src, int64_t num_bytes)
    : CommandBufferCmd(CommandBufferCmdType::kMemcpyDeviceToDeviceCmd,
                       execution_stream_id),
      dst_(dst),
      src_(src),
      num_bytes_(num_bytes) {}

absl::StatusOr<const se::CommandBuffer::Command*>
MemcpyDeviceToDeviceCmd::Record(const Thunk::ExecuteParams& execute_params,
                                const RecordParams& record_params,
                                RecordAction record_action,
                                se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_);
  se::DeviceMemoryBase src =
      execute_params.buffer_allocations->GetDeviceAddress(src_);

  VLOG(5) << "MemcpyDeviceToDeviceCmd: num_bytes = " << num_bytes_;
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";
  VLOG(5) << "  Src: " << src_ << " (" << src.opaque() << ")";

  if (num_bytes_ == 0) {
    VLOG(5) << "Skip recording MemcpyDeviceToDeviceCmd command of 0 bytes";
    return nullptr;
  }

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateMemcpyD2D(&dst, src, num_bytes_,
                                               dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateMemcpyD2D(command, &dst, src, num_bytes_);
      });
}

CommandBufferCmd::BufferUseVector MemcpyDeviceToDeviceCmd::buffers() const {
  return {{dst_, MemoryAccess::kWrite}, {src_, MemoryAccess::kRead}};
}

//===----------------------------------------------------------------------===//
// MemzeroCmd
//===----------------------------------------------------------------------===//

MemzeroCmd::MemzeroCmd(ExecutionStreamId execution_stream_id,
                       BufferAllocation::Slice dst)
    : CommandBufferCmd(CommandBufferCmdType::kMemzeroCmd, execution_stream_id),
      dst_(dst) {}

absl::StatusOr<const se::CommandBuffer::Command*> MemzeroCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_);

  VLOG(5) << "MemzeroCmd:";
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";

  if (dst_.size() == 0) {
    VLOG(5) << "Skip recording MemzeroCmd command of 0 bytes";
    return nullptr;
  }

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateMemset(&dst, uint8_t{0},
                                            /*num_elements=*/dst_.size(),
                                            dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateMemset(command, &dst, uint8_t{0},
                                            /*num_elements=*/dst_.size());
      });
}

CommandBufferCmd::BufferUseVector MemzeroCmd::buffers() const {
  return {{dst_, MemoryAccess::kWrite}};
}

//===----------------------------------------------------------------------===//
// Memset32Cmd
//===----------------------------------------------------------------------===//

Memset32Cmd::Memset32Cmd(ExecutionStreamId execution_stream_id,
                         BufferAllocation::Slice dst, uint32_t bit_pattern)
    : CommandBufferCmd(CommandBufferCmdType::kMemset32Cmd, execution_stream_id),
      dst_(dst),
      bit_pattern_(bit_pattern) {}

absl::StatusOr<const se::CommandBuffer::Command*> Memset32Cmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_);

  VLOG(5) << "Memset32Cmd: bit_pattern=" << bit_pattern_;
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";

  if (dst_.size() == 0) {
    VLOG(5) << "Skip recording Memset32Cmd command of 0 bytes";
    return nullptr;
  }

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateMemset(
            &dst, bit_pattern_,
            /*num_elements=*/dst_.size() / sizeof(uint32_t), dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateMemset(
            command, &dst, bit_pattern_,
            /*num_elements=*/dst_.size() / sizeof(uint32_t));
      });
}

CommandBufferCmd::BufferUseVector Memset32Cmd::buffers() const {
  return {{dst_, MemoryAccess::kWrite}};
}

//===----------------------------------------------------------------------===//
// CaseCmd
//===----------------------------------------------------------------------===//

CaseCmd::CaseCmd(ExecutionStreamId execution_stream_id,
                 BufferAllocation::Slice index, bool index_is_bool,
                 std::vector<CommandBufferCmdExecutor> branches)
    : CommandBufferCmd(CommandBufferCmdType::kCaseCmd, execution_stream_id),
      index_(index),
      index_is_bool_(index_is_bool),
      branches_(std::move(branches)) {}

absl::Status CaseCmd::Initialize(const Thunk::InitializeParams& params,
                                 StateManager& state) {
  for (auto& branch : branches_) {
    TF_RETURN_IF_ERROR(branch.Initialize(params, state));
  }
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> CaseCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase index =
      execute_params.buffer_allocations->GetDeviceAddress(index_);

  VLOG(5) << "CaseCmd:";
  VLOG(5) << "  index: " << index_ << " (" << index.opaque() << ")";

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        if (index_is_bool_) {
          return command_buffer->CreateCase(
              se::DeviceMemory<bool>(index),
              CreateCommands(branches_, &execute_params, &record_params),
              dependencies);
        }
        return command_buffer->CreateCase(
            se::DeviceMemory<int32_t>(index),
            CreateCommands(branches_, &execute_params, &record_params),
            dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        if (index_is_bool_) {
          return command_buffer->UpdateCase(
              command, se::DeviceMemory<bool>(index),
              UpdateCommands(branches_, &execute_params, &record_params));
        }
        return command_buffer->UpdateCase(
            command, se::DeviceMemory<int32_t>(index),
            UpdateCommands(branches_, &execute_params, &record_params));
      });
}

bool CaseCmd::requires_initialization() {
  return absl::c_any_of(
      branches_, [](const auto& seq) { return seq.requires_initialization(); });
}

CommandBufferCmd::BufferUseVector CaseCmd::buffers() const {
  absl::flat_hash_set<BufferUse> buffers;
  buffers.emplace(index_, MemoryAccess::kRead);
  for (auto& branch : branches_) {
    buffers.insert(branch.buffers().begin(), branch.buffers().end());
  }
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// WhileCmd
//===----------------------------------------------------------------------===//

WhileCmd::WhileCmd(ExecutionStreamId execution_stream_id,
                   BufferAllocation::Slice pred,
                   CommandBufferCmdExecutor cond_commands,
                   CommandBufferCmdExecutor body_commands)
    : CommandBufferCmd(CommandBufferCmdType::kWhileCmd, execution_stream_id),
      pred_(pred),
      cond_commands_(std::move(cond_commands)),
      body_commands_(std::move(body_commands)) {}

absl::Status WhileCmd::Initialize(const Thunk::InitializeParams& params,
                                  StateManager& state) {
  TF_RETURN_IF_ERROR(cond_commands_.Initialize(params, state));
  return body_commands_.Initialize(params, state);
}

absl::StatusOr<const se::CommandBuffer::Command*> WhileCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase pred =
      execute_params.buffer_allocations->GetDeviceAddress(pred_);

  VLOG(5) << "WhileCmd: cond_commands=" << cond_commands_.size()
          << " body_commands=" << body_commands_.size();
  VLOG(5) << "  pred: " << pred_ << " (" << pred.opaque() << ")";

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateWhile(
            se::DeviceMemory<bool>(pred),
            CreateCommands(&cond_commands_, &execute_params, &record_params),
            CreateCommands(&body_commands_, &execute_params, &record_params),
            dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateWhile(
            command, se::DeviceMemory<bool>(pred),
            UpdateCommands(&cond_commands_, &execute_params, &record_params),
            UpdateCommands(&body_commands_, &execute_params, &record_params));
      });
}

bool WhileCmd::requires_initialization() {
  return (cond_commands_.requires_initialization() ||
          body_commands_.requires_initialization());
}

CommandBufferCmd::BufferUseVector WhileCmd::buffers() const {
  absl::flat_hash_set<BufferUse> buffers;
  buffers.emplace(pred_, MemoryAccess::kWrite);
  buffers.insert(cond_commands_.buffers().begin(),
                 cond_commands_.buffers().end());
  buffers.insert(body_commands_.buffers().begin(),
                 body_commands_.buffers().end());
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// GemmCmd
//===----------------------------------------------------------------------===//

GemmCmd::GemmCmd(ExecutionStreamId execution_stream_id, GemmConfig config,
                 const BufferAllocation::Slice& lhs_buffer,
                 const BufferAllocation::Slice& rhs_buffer,
                 const BufferAllocation::Slice& output_buffer,
                 const BufferAllocation::Slice& workspace, bool deterministic)
    : TracedCommandBufferCmd(CommandBufferCmdType::kGemmCmd,
                             execution_stream_id),
      config_(std::move(config)),
      lhs_buffer_(lhs_buffer),
      rhs_buffer_(rhs_buffer),
      output_buffer_(output_buffer),
      workspace_(workspace),
      deterministic_(deterministic) {}

absl::Status GemmCmd::Initialize(const Thunk::InitializeParams& params,
                                 StateManager& state) {
  if (!params.stream->parent()->AsBlas()) {
    return absl::InternalError("Failed to initialize BLAS support for GemmCmd");
  }
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> GemmCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase lhs =
      execute_params.buffer_allocations->GetDeviceAddress(lhs_buffer_);
  se::DeviceMemoryBase rhs =
      execute_params.buffer_allocations->GetDeviceAddress(rhs_buffer_);
  se::DeviceMemoryBase out =
      execute_params.buffer_allocations->GetDeviceAddress(output_buffer_);
  se::DeviceMemoryBase workspace =
      execute_params.buffer_allocations->GetDeviceAddress(workspace_);

  VLOG(5) << "GemmCmd: deterministic=" << deterministic_;
  VLOG(5) << "  Lhs: " << lhs_buffer_ << " (" << lhs.opaque() << ")";
  VLOG(5) << "  Lhs: " << rhs_buffer_ << " (" << rhs.opaque() << ")";
  VLOG(5) << "  Out: " << output_buffer_ << " (" << out.opaque() << ")";
  VLOG(5) << "  Workspace: " << workspace_ << " (" << workspace.opaque() << ")";

  return RecordTracedCommand(execute_params, record_params,
                             std::move(record_action), command_buffer,
                             [&](se::Stream* stream) {
                               return RunGemm(config_, lhs, rhs, out, workspace,
                                              deterministic_, stream);
                             });
}

CommandBufferCmd::BufferUseVector GemmCmd::buffers() const {
  return {{lhs_buffer_, MemoryAccess::kRead},
          {rhs_buffer_, MemoryAccess::kRead},
          {output_buffer_, MemoryAccess::kWrite},
          {workspace_, MemoryAccess::kWrite}};
}

//===----------------------------------------------------------------------===//
// CublasLtCmd
//===----------------------------------------------------------------------===//

CublasLtCmd::CublasLtCmd(ExecutionStreamId execution_stream_id,
                         const CublasLtMatmulThunk& matmul_thunk)
    : TracedCommandBufferCmd(CommandBufferCmdType::kCublasLtCmd,
                             execution_stream_id),
      CublasLtMatmulThunk(matmul_thunk) {}

absl::Status CublasLtCmd::Initialize(const Thunk::InitializeParams& params,
                                     StateManager& state) {
  TF_RETURN_IF_ERROR(CublasLtMatmulThunk::Initialize(params));
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> CublasLtCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  // This call is required to make sure matmul plan is already created and
  // cached before recording the command buffer.
  TF_RETURN_IF_ERROR(GetCachedMatmulPlan(execute_params).status());

  VLOG(5) << "CublasLtCmd:";
  VLOG(5) << "  a_buffer: " << a_.ToString();
  VLOG(5) << "  b_buffer: " << b_.ToString();
  VLOG(5) << "  c_buffer: " << c_.ToString();
  VLOG(5) << "  d_buffer: " << d_.ToString();
  VLOG(5) << "  bias_buffer: " << bias_.ToString();
  VLOG(5) << "  aux_buffer: " << aux_.ToString();
  VLOG(5) << "  a_scale_buffer: " << a_scale_.ToString();
  VLOG(5) << "  b_scale_buffer: " << b_scale_.ToString();
  VLOG(5) << "  c_scale_buffer: " << c_scale_.ToString();
  VLOG(5) << "  d_scale_buffer: " << d_scale_.ToString();
  VLOG(5) << "  d_amax_buffer: " << d_amax_.ToString();
  // workspace buffer is guaranteed to be non-null here.
  VLOG(5) << "  workspace_buffer: " << workspace_->ToString();

  return RecordTracedCommand(
      execute_params, record_params, std::move(record_action), command_buffer,
      [&](se::Stream* stream) {
        return ExecuteOnStreamInternal(stream, execute_params);
      });
}

CommandBufferCmd::BufferUseVector CublasLtCmd::buffers() const {
  BufferUseVector buffer_usage;
  buffer_usage.reserve(13);
  buffer_usage.push_back({a_, MemoryAccess::kRead});
  buffer_usage.push_back({b_, MemoryAccess::kRead});
  buffer_usage.push_back({c_, MemoryAccess::kRead});
  buffer_usage.push_back({d_, MemoryAccess::kWrite});
  buffer_usage.push_back({*workspace_, MemoryAccess::kWrite});

  if (bias_.allocation() != nullptr) {
    buffer_usage.push_back({bias_, MemoryAccess::kRead});
  }
  if (a_scale_.allocation() != nullptr) {
    buffer_usage.push_back({a_scale_, MemoryAccess::kRead});
  }
  if (b_scale_.allocation() != nullptr) {
    buffer_usage.push_back({b_scale_, MemoryAccess::kRead});
  }
  if (c_scale_.allocation() != nullptr) {
    buffer_usage.push_back({c_scale_, MemoryAccess::kRead});
  }
  if (d_scale_.allocation() != nullptr) {
    buffer_usage.push_back({d_scale_, MemoryAccess::kRead});
  }
  if (aux_.allocation() != nullptr) {
    buffer_usage.push_back({aux_, MemoryAccess::kWrite});
  }
  if (d_amax_.allocation() != nullptr) {
    buffer_usage.push_back({d_amax_, MemoryAccess::kRead});
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CuDnnCmd
//===----------------------------------------------------------------------===//

CuDnnCmd::CuDnnCmd(ExecutionStreamId execution_stream_id,
                   absl::Span<const BufferAllocation::Slice> args,
                   const std::shared_ptr<se::dnn::LazyDnnGraph> graph)
    : TracedCommandBufferCmd(CommandBufferCmdType::kCuDnnCmd,
                             execution_stream_id),
      args_(args.cbegin(), args.cend()),
      graph_(graph) {}

absl::Status CuDnnCmd::Initialize(const Thunk::InitializeParams& params,
                                  StateManager&) {
  if (!params.stream->parent()->AsDnn()) {
    return absl::InternalError("Failed to initialize DNN support for CuDnnCmd");
  }
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> CuDnnCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  CHECK(graph_ != nullptr);
  std::vector<se::DeviceMemoryBase> operands;
  operands.reserve(args_.size());
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceMemoryBase buf =
        execute_params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(5) << "  Arg: " << arg << ": " << buf.opaque();
    operands.push_back(buf);
  }
  TF_ASSIGN_OR_RETURN(
      const bool supports_explicit,
      graph_->get()->SupportsExplicitCommandBufferConstruction());
  if (supports_explicit) {
    return Handle(
        std::move(record_action),
        [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
          return command_buffer->CreateDnnGraphCommand(
              *graph_->get(), *execute_params.stream,
              absl::Span<se::DeviceMemoryBase>(operands), dependencies);
        },
        [&](const se::CommandBuffer::Command* command) {
          return command_buffer->UpdateDnnGraphCommand(
              command, *graph_->get(), *execute_params.stream,
              absl::Span<se::DeviceMemoryBase>(operands));
        });
  }
  return RecordTracedCommand(
      execute_params, record_params, std::move(record_action), command_buffer,
      [&](se::Stream* stream) {
        return graph_->get()->Execute(
            *stream, absl::Span<se::DeviceMemoryBase>(operands),
            execute_params.collective_params->local_device_ordinal);
      });
}

CommandBufferCmd::BufferUseVector CuDnnCmd::buffers() const {
  CommandBufferCmd::BufferUseVector buffer_usage;
  buffer_usage.reserve(args_.size());
  for (int i = 0; i < args_.size() - 1; ++i) {
    buffer_usage.push_back({args_[i], MemoryAccess::kRead});
  }
  buffer_usage.push_back({args_.back(), MemoryAccess::kWrite});
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CustomCallCmd
//===----------------------------------------------------------------------===//

absl::StatusOr<const se::CommandBuffer::Command*> CustomCallCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  if (handler_ == nullptr) {
    return RecordLegacyCustomCall(execute_params, record_params,
                                  std::move(record_action), command_buffer);
  }
  return RecordXlaFfiCall(execute_params, record_params,
                          std::move(record_action), command_buffer);
}

namespace {
// Records each buffer associated with each slice into the provided vector.
// Returns an error if any of the slices is missing a buffer allocation.
absl::Status GetBuffers(
    const Thunk::ExecuteParams& execute_params,
    absl::Span<const std::optional<CustomCallCmd::Slice>> slices,
    std::vector<void*>& buffers, absl::string_view label) {
  for (int i = 0; i < slices.size(); ++i) {
    if (!slices[i].has_value()) {
      buffers.push_back(nullptr);
      VLOG(5) << label << i << ": null";
      continue;
    }

    if (!slices[i]->slice.allocation()) {
      return absl::InternalError("custom call input missing buffer allocation");
    }

    auto buffer =
        execute_params.buffer_allocations->GetDeviceAddress(slices[i]->slice)
            .opaque();
    VLOG(5) << label << i << ": " << slices[i]->slice << " (" << buffer << ")";
    buffers.push_back(buffer);
  }
  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<const se::CommandBuffer::Command*>
CustomCallCmd::RecordLegacyCustomCall(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  std::vector<void*> buffers;
  buffers.reserve(operands_.size() + results_.size());

  VLOG(5) << "CustomCallCmd: target_name=" << target_name_;
  TF_RETURN_IF_ERROR(
      GetBuffers(execute_params, operands_, buffers, "  Operand "));
  TF_RETURN_IF_ERROR(
      GetBuffers(execute_params, results_, buffers, "  Result "));

  TF_ASSIGN_OR_RETURN(
      auto nested_cmd,
      se::TraceCommandBufferFactory::Create(
          execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, [&](se::Stream* stream) {
            XlaCustomCallStatus custom_call_status;
            call_target_(stream, buffers.data(), opaque_.data(), opaque_.size(),
                         &custom_call_status);
            auto message = CustomCallStatusGetMessage(&custom_call_status);
            if (message) {
              return absl::InternalError(
                  absl::StrCat("CustomCall failed: ", *message));
            }
            return absl::OkStatus();
          }));

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateNestedCommand(*nested_cmd, dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateNestedCommand(command, *nested_cmd);
      });
}

absl::StatusOr<const se::CommandBuffer::Command*>
CustomCallCmd::RecordXlaFfiCall(const Thunk::ExecuteParams& execute_params,
                                const RecordParams& record_params,
                                RecordAction record_action,
                                se::CommandBuffer* command_buffer) {
  // TODO(ezhulenev): This is not the most optimal approach, as we'll be doing
  // a lot of extra allocation on every call. We have to keep attributes
  // separate from arguments, as they do not change after thunk is
  // constructed.
  ffi::CallFrameBuilder builder(operands_.size(), results_.size());

  VLOG(5) << "CustomCallCmd: target_name=" << target_name_;

  absl::InlinedVector<se::DeviceMemoryBase, 4> arguments;
  arguments.reserve(operands_.size());

  for (int i = 0; i < operands_.size(); ++i) {
    const std::optional<Slice>& slice = operands_[i];
    if (!slice.has_value()) {
      arguments.push_back(se::DeviceMemoryBase{});
      continue;
    }

    se::DeviceMemoryBase buffer =
        execute_params.buffer_allocations->GetDeviceAddress(slice->slice);
    VLOG(5) << "  Operand " << i << ": " << slice->slice << " ("
            << buffer.opaque() << ")";
    arguments.push_back(buffer);
  }

  absl::InlinedVector<se::DeviceMemoryBase, 4> results;
  results.reserve(results_.size());

  for (int i = 0; i < results_.size(); ++i) {
    const std::optional<Slice>& slice = results_[i];
    if (!slice.has_value()) {
      results.push_back(se::DeviceMemoryBase{});
      continue;
    }

    se::DeviceMemoryBase buffer =
        execute_params.buffer_allocations->GetDeviceAddress(slice->slice);
    VLOG(5) << "  Result " << i << ": " << slice->slice << " ("
            << buffer.opaque() << ")";
    results.push_back(buffer);
  }

  // Borrow the FFI call frame from the object pool and update with the actual
  // device memory addresses.
  TF_ASSIGN_OR_RETURN(auto call_frame, call_frames_->GetOrCreate());
  TF_RETURN_IF_ERROR(call_frame->UpdateWithBuffers(arguments, results));

  RunId run_id = execute_params.collective_params->run_id;

  TF_ASSIGN_OR_RETURN(
      auto nested_cmd,
      se::TraceCommandBufferFactory::Create(
          execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, [&](se::Stream* stream) {
            ffi::CallOptions options = {
                run_id, execute_params.buffer_allocations->device_ordinal(),
                ffi::CallOptions::GpuOptions{
                    stream,
                    execute_params.buffer_allocations->memory_allocator()},
                /*called_computation=*/nullptr,  // TODO(b/342285364)
                execute_params.ffi_execution_context};
            return ffi::Call(handler_, *call_frame, options);
          }));

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateNestedCommand(*nested_cmd, dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateNestedCommand(command, *nested_cmd);
      });
}

CommandBufferCmd::BufferUseVector CustomCallCmd::buffers() const {
  CommandBufferCmd::BufferUseVector buffer_usage;
  for (auto& slices : {operands_, results_}) {
    for (const std::optional<Slice>& slice : slices) {
      if (slice.has_value()) {
        buffer_usage.push_back({slice->slice, MemoryAccess::kWrite});
      }
    }
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CollectiveCmd
//===----------------------------------------------------------------------===//

CollectiveCmd::CollectiveCmd(CommandBufferCmdType cmd_type,
                             ExecutionStreamId execution_stream_id,
                             ExecutionStreamId async_from_stream_id,
                             CollectiveConfig config)
    : CommandBufferCmd(cmd_type, execution_stream_id,
                       se::StreamPriority::Highest),
      async_from_stream_id_(async_from_stream_id),
      config_(std::move(config)) {}

absl::Status CollectiveCmd::Prepare(
    const Thunk::PrepareParams& params,
    Thunk::ResourceRequestsInterface& resource_requests) {
  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives,
                      Thunk::GetGpuCollectives(params));
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(collectives, *params.collective_params,
                      config().replica_groups, config().group_mode,
                      GetAsyncStreamKind()));
  return resource_requests.AddClique(clique_key);
}

absl::StatusOr<const se::CommandBuffer::Command*>
CollectiveCmd::RecordTracedCommand(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer,
    absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<se::CommandBuffer> nested_cmd,
                      se::TraceCommandBufferFactory::Create(
                          execute_params.stream->parent(),
                          execute_params.command_buffer_trace_stream, trace));

  if (priority() != se::StreamPriority::Default) {
    TF_RETURN_IF_ERROR(nested_cmd->SetPriority(priority()));
  }

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateNestedCommand(*nested_cmd, dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateNestedCommand(command, *nested_cmd);
      });
}

//===----------------------------------------------------------------------===//
// AllReduceCmd
//===----------------------------------------------------------------------===//

AllReduceCmd::AllReduceCmd(ExecutionStreamId execution_stream_id,
                           ExecutionStreamId async_from_stream_id,
                           CollectiveConfig config,
                           ReductionKind reduction_kind,
                           absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandBufferCmdType::kAllReduceCmd, execution_stream_id,
                    async_from_stream_id, std::move(config)),
      reduction_kind_(reduction_kind),
      buffers_(buffers.begin(), buffers.end()) {}

absl::StatusOr<const se::CommandBuffer::Command*> AllReduceCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  VLOG(5) << "AllReduceCmd: reduction=" << ReductionKindString(reduction_kind_);

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    VLOG(5) << "  Src: " << buffers_[i].source_buffer << " ("
            << device_buffers[i].source_buffer.opaque() << ")";
    VLOG(5) << "  Dst: " << buffers_[i].destination_buffer << " ("
            << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "AllReduceCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives,
                      Thunk::GetGpuCollectives(execute_params));

  TF_ASSIGN_OR_RETURN(
      CommunicatorHandle comm_handle,
      GetComm(collectives, *execute_params.collective_params,
              *execute_params.collective_cliques, config().replica_groups,
              config().group_mode, GetAsyncStreamKind()));

  return RecordTracedCommand(
      execute_params, record_params, std::move(record_action), command_buffer,
      [&](se::Stream* stream) {
        return RunAllReduce(reduction_kind_, device_buffers, *stream,
                            comm_handle.comm);
      });
}

CommandBufferCmd::BufferUseVector AllReduceCmd::buffers() const {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(buffer.source_buffer, MemoryAccess::kRead);
    buffer_usage.emplace_back(buffer.destination_buffer, MemoryAccess::kWrite);
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// ReduceScatterCmd
//===----------------------------------------------------------------------===//

ReduceScatterCmd::ReduceScatterCmd(
    ExecutionStreamId execution_stream_id,
    ExecutionStreamId async_from_stream_id, CollectiveConfig config,
    ReductionKind reduction_kind,
    absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandBufferCmdType::kReduceScatter, execution_stream_id,
                    async_from_stream_id, std::move(config)),
      reduction_kind_(reduction_kind),
      buffers_(buffers.begin(), buffers.end()) {}

absl::StatusOr<const se::CommandBuffer::Command*> ReduceScatterCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  VLOG(5) << "ReduceScatterCmd: reduction="
          << ReductionKindString(reduction_kind_);

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    VLOG(5) << "  Src: " << buffers_[i].source_buffer << " ("
            << device_buffers[i].source_buffer.opaque() << ")";
    VLOG(5) << "  Dst: " << buffers_[i].destination_buffer << " ("
            << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "ReduceScatterCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives,
                      Thunk::GetGpuCollectives(execute_params));

  TF_ASSIGN_OR_RETURN(
      CommunicatorHandle comm_handle,
      GetComm(collectives, *execute_params.collective_params,
              *execute_params.collective_cliques, config().replica_groups,
              config().group_mode, GetAsyncStreamKind()));

  return RecordTracedCommand(execute_params, record_params, record_action,
                             command_buffer, [&](se::Stream* stream) {
                               return RunReduceScatter(reduction_kind_,
                                                       device_buffers, *stream,
                                                       comm_handle.comm);
                             });
}

CommandBufferCmd::BufferUseVector ReduceScatterCmd::buffers() const {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(buffer.source_buffer, MemoryAccess::kRead);
    buffer_usage.emplace_back(buffer.destination_buffer, MemoryAccess::kWrite);
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// AllToAllCmd
//===----------------------------------------------------------------------===//

AllToAllCmd::AllToAllCmd(ExecutionStreamId execution_stream_id,
                         ExecutionStreamId async_from_stream_id,
                         CollectiveConfig config, bool has_split_dimension,
                         absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandBufferCmdType::kAllToAll, execution_stream_id,
                    async_from_stream_id, std::move(config)),
      has_split_dimension_(has_split_dimension),
      buffers_(buffers.begin(), buffers.end()) {}

absl::StatusOr<const se::CommandBuffer::Command*> AllToAllCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  VLOG(5) << "AllToAllCmd, has_split_dimension=" << has_split_dimension_;

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    VLOG(5) << "  Src: " << buffers_[i].source_buffer << " ("
            << device_buffers[i].source_buffer.opaque() << ")";
    VLOG(5) << "  Dst: " << buffers_[i].destination_buffer << " ("
            << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "ReduceScatterCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives,
                      Thunk::GetGpuCollectives(execute_params));
  TF_ASSIGN_OR_RETURN(
      CommunicatorHandle comm_handle,
      GetComm(collectives, *execute_params.collective_params,
              *execute_params.collective_cliques, config().replica_groups,
              config().group_mode, GetAsyncStreamKind()));

  return RecordTracedCommand(
      execute_params, record_params, std::move(record_action), command_buffer,
      [&](se::Stream* stream) {
        return RunAllToAll(has_split_dimension_, device_buffers, *stream,
                           comm_handle.comm);
      });
}

CommandBufferCmd::BufferUseVector AllToAllCmd::buffers() const {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(buffer.source_buffer, MemoryAccess::kRead);
    buffer_usage.emplace_back(buffer.destination_buffer, MemoryAccess::kWrite);
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// AllGatherCmd
//===----------------------------------------------------------------------===//

AllGatherCmd::AllGatherCmd(ExecutionStreamId execution_stream_id,
                           ExecutionStreamId async_from_stream_id,
                           CollectiveConfig config,
                           absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandBufferCmdType::kAllGatherCmd, execution_stream_id,
                    async_from_stream_id, std::move(config)),
      buffers_(buffers.begin(), buffers.end()) {}

absl::StatusOr<const se::CommandBuffer::Command*> AllGatherCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  VLOG(5) << "AllGatherCmd:";

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    VLOG(5) << "  Src: " << buffers_[i].source_buffer << " ("
            << device_buffers[i].source_buffer.opaque() << ")";
    VLOG(5) << "  Dst: " << buffers_[i].destination_buffer << " ("
            << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "AllGatherCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives,
                      Thunk::GetGpuCollectives(execute_params));

  TF_ASSIGN_OR_RETURN(
      CommunicatorHandle comm_handle,
      GetComm(collectives, *execute_params.collective_params,
              *execute_params.collective_cliques, config().replica_groups,
              config().group_mode, GetAsyncStreamKind()));

  return RecordTracedCommand(
      execute_params, record_params, std::move(record_action), command_buffer,
      [&](se::Stream* stream) {
        return RunAllGather(device_buffers, *stream, comm_handle.comm);
      });
}

CommandBufferCmd::BufferUseVector AllGatherCmd::buffers() const {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(buffer.source_buffer, MemoryAccess::kRead);
    buffer_usage.emplace_back(buffer.destination_buffer, MemoryAccess::kWrite);
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CollectiveBroadcastCmd
//===----------------------------------------------------------------------===//

CollectiveBroadcastCmd::CollectiveBroadcastCmd(
    ExecutionStreamId execution_stream_id,
    ExecutionStreamId async_from_stream_id, CollectiveConfig config,
    absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandBufferCmdType::kCollectiveBroadcastCmd,
                    execution_stream_id, async_from_stream_id,
                    std::move(config)),
      buffers_(buffers.begin(), buffers.end()) {}

absl::StatusOr<const se::CommandBuffer::Command*>
CollectiveBroadcastCmd::Record(const Thunk::ExecuteParams& execute_params,
                               const RecordParams& record_params,
                               RecordAction record_action,
                               se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  VLOG(5) << "CollectiveBroadcastCmd:";

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    VLOG(5) << "  Src: " << buffers_[i].source_buffer << " ("
            << device_buffers[i].source_buffer.opaque() << ")";
    VLOG(5) << "  Dst: " << buffers_[i].destination_buffer << " ("
            << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "CollectiveBroadcastCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives,
                      Thunk::GetGpuCollectives(execute_params));

  TF_ASSIGN_OR_RETURN(
      CommunicatorHandle comm_handle,
      GetComm(collectives, *execute_params.collective_params,
              *execute_params.collective_cliques, config().replica_groups,
              config().group_mode, GetAsyncStreamKind()));

  return RecordTracedCommand(execute_params, record_params,
                             std::move(record_action), command_buffer,
                             [&](se::Stream* stream) {
                               return RunCollectiveBroadcast(
                                   device_buffers, *stream, comm_handle.comm);
                             });
}

CommandBufferCmd::BufferUseVector CollectiveBroadcastCmd::buffers() const {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(buffer.source_buffer, MemoryAccess::kRead);
    buffer_usage.emplace_back(buffer.destination_buffer, MemoryAccess::kWrite);
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// DynamicSliceFusionCmd
//===----------------------------------------------------------------------===//

DynamicSliceFusionCmd::DynamicSliceFusionCmd(
    ExecutionStreamId execution_stream_id,
    CommandBufferCmdExecutor embedded_commands,
    std::vector<std::optional<BufferAllocation::Slice>> arguments,
    std::vector<std::unique_ptr<BufferAllocation>> fake_allocations,
    std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>> offsets,
    std::vector<std::optional<Shape>> orig_shapes,
    std::vector<std::optional<Shape>> sliced_shapes,
    std::vector<std::optional<uint64_t>> offset_byte_sizes)
    : CommandBufferCmd(CommandBufferCmdType::kDynamicSliceFusionCmd,
                       execution_stream_id),
      embedded_commands_(std::move(embedded_commands)),
      fake_allocations_(std::move(fake_allocations)) {
  // Zip all arguments together to create a list of SliceDef.
  for (auto [arg, offset, orig_shape, sliced_shape, offset_byte_size] :
       llvm::zip_equal(arguments, offsets, orig_shapes, sliced_shapes,
                       offset_byte_sizes)) {
    slices_.push_back(DynamicSliceThunk::SliceDef{
        std::move(arg),
        std::move(offset),
        std::move(orig_shape),
        std::move(sliced_shape),
        std::move(offset_byte_size),
    });
  }

  for (auto [argument_idx, slice] : llvm::enumerate(slices_)) {
    embeded_to_origin_slice_map_[argument_idx] = slice.embedded_thunk_argument;
  }

  // Find how many offsets we might have to transfer from device to host and
  // pre-compute host allocation requirements.
  for (DynamicSliceThunk::SliceDef& slice : slices_) {
    offsets_allocs_base_.push_back(offsets_allocs_size_);
    if (slice.sliced_shape.has_value()) {
      offsets_allocs_size_ +=
          slice.sliced_shape->dimensions().size() * sizeof(int64_t);
    }
  }
}

// Force update the command when there is any non-constant value slice offset,
// because the memory address might changed if the offset is loop
// iterator or operator outputs even if the parent command's memory pointers
// do not change.
bool DynamicSliceFusionCmd::requires_initialization() {
  return !absl::c_all_of(slices_, [](const DynamicSliceThunk::SliceDef& slice) {
    if (!slice.offsets.has_value()) {
      return true;
    }
    return absl::c_all_of(slice.offsets.value(),
                          [](DynamicSliceThunk::Offset offset) {
                            return std::holds_alternative<int64_t>(offset);
                          });
  });
}

absl::Status DynamicSliceFusionCmd::Initialize(
    const Thunk::InitializeParams& params, StateManager& state) {
  TF_RETURN_IF_ERROR(embedded_commands_.Initialize(params, state));
  absl::MutexLock lock(&mutex_);
  if (offsets_allocs_.contains(params.executor)) {
    return absl::OkStatus();
  }

  VLOG(2) << "Allocate " << offsets_allocs_size_
          << " bytes for transferring offsets on executor: " << params.executor;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::MemoryAllocation> allocation,
      params.executor->HostMemoryAllocate(offsets_allocs_size_));
  offsets_allocs_.emplace(params.executor, std::move(allocation));
  return absl::OkStatus();
}

absl::Status DynamicSliceFusionCmd::Prepare(
    const Thunk::PrepareParams& params,
    Thunk::ResourceRequestsInterface& resource_requests) {
  for (DynamicSliceThunk::SliceDef& slice : slices_) {
    if (slice.offsets.has_value()) {
      TF_RET_CHECK(slice.embedded_thunk_argument.has_value());
      TF_RET_CHECK(slice.orig_shape.has_value());
      TF_RET_CHECK(slice.sliced_shape.has_value());
      TF_RET_CHECK(slice.offset_byte_size.has_value());

      TF_RET_CHECK(slice.orig_shape->IsArray());
      TF_RET_CHECK(slice.sliced_shape->IsArray());

      TF_RET_CHECK(slice.offsets->size() ==
                   slice.orig_shape->dimensions().size());
      TF_RET_CHECK(slice.sliced_shape->dimensions().size() ==
                   slice.orig_shape->dimensions().size());
    }
  }
  TF_RETURN_IF_ERROR(embedded_commands_.Prepare(params, resource_requests));
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> DynamicSliceFusionCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::Stream& stream = *execute_params.stream;

  const BufferAllocations& orig_allocations =
      *execute_params.buffer_allocations;
  absl::InlinedVector<se::DeviceMemoryBase, 8> slice_buffers(
      slices_.size(), se::DeviceMemoryBase());

  // Get memory allocation for copying offsets from device.
  int64_t* offsets_alloc = [&] {
    absl::MutexLock lock(&mutex_);
    return reinterpret_cast<int64_t*>(
        offsets_allocs_.at(stream.parent())->opaque());
  }();

  auto offset_value = [&](int64_t arg_idx, int64_t offset_idx) -> int64_t& {
    return offsets_alloc[offsets_allocs_base_.at(arg_idx) + offset_idx];
  };

  VLOG(2) << "Execute address computation thunk: slices=" << slices_.size();
  for (auto [argument_idx, slice] : llvm::enumerate(slices_)) {
    // Skip arguments that do not have buffer slices (tokens).
    if (!slice.embedded_thunk_argument.has_value()) {
      continue;
    }

    // `argument_buffer` will contain the original offset for slice
    // `argument_slice` within `orig_allocations`
    se::DeviceMemoryBase argument_buffer =
        orig_allocations.GetDeviceAddress(*slice.embedded_thunk_argument);

    // If argument is not sliced, just use the original buffer.
    if (!slice.offsets.has_value()) {
      slice_buffers[argument_idx] = argument_buffer;
      continue;
    }

    const Shape& src_shape = *slice.orig_shape;
    const Shape& dst_shape = *slice.sliced_shape;

    absl::InlinedVector<int64_t, 4> slice_starts;
    slice_starts.reserve(dst_shape.dimensions().size());

    // Number of issues d2h transfers to copy offset values from device to
    // host.
    int64_t num_transfers = 0;

    // Get offset for `argument_idx`-th argument, which has
    // `dst_shape.dimensions_size()` components.
    for (auto [offset_idx, values] : llvm::enumerate(llvm::zip(
             *slice.offsets, src_shape.dimensions(), dst_shape.dimensions()))) {
      auto [offset, src_dim, dst_dim] = values;
      if (int64_t* const_offset = std::get_if<int64_t>(&offset)) {
        // Forward slice offsets that are known constant values
        VLOG(2) << "  - arg " << argument_idx << "[" << offset_idx
                << "]: constant offset = " << *const_offset;
        offset_value(argument_idx, offset_idx) = *const_offset;

      } else {
        // Transfer slice offset value from device to host.
        auto alloc_slice = std::get<BufferAllocation::Slice>(offset);
        VLOG(2) << "  - arg " << argument_idx << "[" << offset_idx
                << "]: transfer offset from device " << alloc_slice.ToString();

        se::DeviceMemoryBase offset_src =
            orig_allocations.GetDeviceAddress(alloc_slice);
        int64_t* offset_dst = &offset_value(argument_idx, offset_idx);

        // Copy the `offset_idx`-th component of the offset for the
        // `argument_idx`-th argument from device to host.
        TF_RETURN_IF_ERROR(
            stream.Memcpy(offset_dst, offset_src, *slice.offset_byte_size));
        ++num_transfers;
      }
    }

    // Wait for the completion of all transfers.
    if (num_transfers > 0) {
      VLOG(2) << "Wait for completion of " << num_transfers << " transfer";
      TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());
    }

    // Clamp start indices:
    // start_indices[i] = min(max(start_indices[i], 0),
    //                        operand.dimension_size[i] - size_indices[i])
    for (auto [offset_idx, values] : llvm::enumerate(
             llvm::zip(src_shape.dimensions(), dst_shape.dimensions()))) {
      auto [src_dim, dst_dim] = values;
      int64_t start_index =
          std::min(std::max(offset_value(argument_idx, offset_idx), int64_t{0}),
                   src_dim - dst_dim);
      VLOG(2) << "arg idx: " << argument_idx << " offset_idx " << offset_idx
              << " with offset_value " << offset_value(argument_idx, offset_idx)
              << " start_idx: " << start_index << " src_dim: " << src_dim
              << " dst_dim:" << dst_dim;
      slice_starts.push_back(start_index);
    }

    // Compute new slice. No need to copy the content to new buffers as we can
    // reuse the original buffers since slices are contiguous.
    int64_t new_size = ShapeUtil::ByteSizeOf(dst_shape);

    int64_t new_offset = 0;
    for (auto [start, stride] :
         llvm::zip(slice_starts, *ShapeUtil::ByteStrides(src_shape))) {
      new_offset += start * stride;
    }

    VLOG(2) << "Create sliced argument " << argument_idx << " of shape "
            << slice.sliced_shape->ToString()
            << " by slicing argument of shape " << slice.orig_shape->ToString()
            << " at offset " << new_offset << " with " << new_size;
    slice_buffers[argument_idx] =
        argument_buffer.GetByteSlice(new_offset, new_size);
  }

  // Safe to create a local BufferAllocations here since buffers are only
  // slices of bigger ones allocated elsewhere.
  BufferAllocations slice_allocations(slice_buffers,
                                      orig_allocations.device_ordinal(),
                                      orig_allocations.memory_allocator());

  Thunk::ExecuteParams new_params =
      Thunk::ExecuteParams::CloneWithNewAllocations(execute_params,
                                                    slice_allocations);

  // TODO(b/406370928): Instead of creating a nested command buffer on every
  // call we should create it once and update it. CommandBufferThunk state
  // manager relies on command buffer pointer as an identity for command
  // buffers, and it means that command buffer commands sequence should not
  // create ephemeral command buffers at run time.
  auto nested_command_buffer =
      execute_params.stream->parent()
          ->CreateCommandBuffer(se::CommandBuffer::Mode::kNested)
          .value();
  TF_RETURN_IF_ERROR(embedded_commands_.Record(new_params, record_params,
                                               nested_command_buffer.get()));

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateNestedCommand(*nested_command_buffer,
                                                   dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateNestedCommand(command,
                                                   *nested_command_buffer);
      });
}

CommandBufferCmd::BufferUseVector DynamicSliceFusionCmd::buffers() const {
  CommandBufferCmd::BufferUseVector buffers;
  auto embed_buffers = embedded_commands_.buffers();
  for (auto buffer_usage : embed_buffers) {
    buffers.emplace_back(
        *embeded_to_origin_slice_map_.at(buffer_usage.slice().index()),
        buffer_usage.access());
  }
  return buffers;
}

}  // namespace xla::gpu
