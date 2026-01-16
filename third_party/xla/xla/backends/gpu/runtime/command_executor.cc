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

#include "xla/backends/gpu/runtime/command_executor.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/execution_graph.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla::gpu {

namespace {
// An adaptor from CommandBufferCmd to ExecutionGraph::Operation for building an
// execution graph from a command sequence.
class CommandOperation : public ExecutionGraph::Operation {
 public:
  explicit CommandOperation(Command::BufferUseVector buffers,
                            const Command* cmd)
      : name_(absl::StrFormat("cmd %s: %s", cmd->ToString(),
                              cmd->profile_annotation())),
        buffers_(std::move(buffers)),
        cmd_(cmd),
        resources_(cmd_->resources()) {}

  absl::string_view name() const final { return name_; }
  absl::Span<const BufferUse> BufferUses() const final { return buffers_; }
  absl::Span<const ResourceUse> ResourceUses() const final {
    return resources_;
  }
  void add_resource_use(ResourceUse resource_use) {
    resources_.push_back(resource_use);
  }

  const Command* cmd() const { return cmd_; }

  std::string ToString() const final {
    std::vector<std::string> resource_reprs;
    resource_reprs.reserve(resources_.size());
    for (const ResourceUse& use : resources_) {
      absl::string_view access =
          use.access() == ResourceUse::kRead ? "read" : "write";
      absl::string_view kind = Resource::ToString(use.resource()->kind());
      resource_reprs.push_back(
          absl::StrFormat("%s@%p(%s)", kind, use.resource().get(), access));
    }
    return absl::StrFormat("%s resources=[%s]", cmd_->ToString(),
                           absl::StrJoin(resource_reprs, ", "));
  }

 private:
  std::string name_;
  Command::BufferUseVector buffers_;
  const Command* cmd_;
  Command::ResourceUseVector resources_;

  // The token resource is used to specify dependency other than buffer data
  // flow, e.g, LHS topology will use token resource to specify dependency
  // across commands.
  std::shared_ptr<Resource> token_;
};
}  // namespace

static std::vector<CommandOperation> CreateCommandOperations(
    const CommandSequence& commands,
    CommandExecutor::SynchronizationMode synchronization_mode) {
  std::vector<CommandOperation> operations;
  operations.reserve(commands.size());
  VLOG(3) << "CreateCommandOperations with synchronization mode: "
          << (synchronization_mode ==
                      CommandExecutor::SynchronizationMode::kConcurrent
                  ? "Concurrent"
                  : "LHS");
  if (synchronization_mode ==
      CommandExecutor::SynchronizationMode::kConcurrent) {
    // For concurrent synchronization mode, pass in buffer and resources for
    // dependency inference.
    for (const std::unique_ptr<Command>& cmd : commands) {
      operations.emplace_back(cmd->buffers(), cmd.get());
    }
  }

  if (synchronization_mode == CommandExecutor::SynchronizationMode::kLHS) {
    // For LHS mode, don't pass in buffers.
    // Will use token resource to specify dependency across commands.
    for (const std::unique_ptr<Command>& cmd : commands) {
      operations.emplace_back(Command::BufferUseVector{}, cmd.get());
    }

    auto is_async_start = [](const CommandOperation& op) -> bool {
      auto* async_start = dynamic_cast<const AsyncStartCommand*>(op.cmd());
      return (async_start && async_start->IsAsync());
    };

    auto is_async_done = [](const CommandOperation& op) -> bool {
      auto* async_done = dynamic_cast<const AsyncDoneCommand*>(op.cmd());
      return (async_done && async_done->IsAsync());
    };

    auto find_async_start_cmd_id = [&](int64_t async_done_cmd_id) -> int64_t {
      auto* async_done = dynamic_cast<const AsyncDoneCommand*>(
          operations[async_done_cmd_id].cmd());
      CHECK(async_done);
      for (int64_t j = async_done_cmd_id - 1; j >= 0; --j) {
        if (is_async_start(operations[j])) {
          auto* async_start =
              dynamic_cast<const AsyncStartCommand*>(operations[j].cmd());
          if (async_start->IsAsync() &&
              async_done->async_start() == async_start) {
            return j;
          }
        }
      }
      return -1;
    };

    for (int64_t i = 0; i < operations.size(); ++i) {
      if (is_async_start(operations[i])) {
        for (int64_t j = i - 1; j >= 0; --j) {
          if (is_async_start(operations[j])) {
            continue;
          }
          operations[i].add_resource_use(
              ResourceUse::Read(commands[j]->token()));
          break;
        }
      } else if (is_async_done(operations[i])) {
        int64_t async_start_cmd_id = find_async_start_cmd_id(i);
        CHECK_NE(async_start_cmd_id, -1);
        operations[i].add_resource_use(
            ResourceUse::Read(commands[async_start_cmd_id]->token()));
        CHECK_GT(i, 0);
        if ((i - 1) != async_start_cmd_id) {
          operations[i].add_resource_use(
              ResourceUse::Read(commands[i - 1]->token()));
        }
      } else {
        for (int64_t j = i - 1; j >= 0; --j) {
          if (is_async_start(operations[j])) {
            // The first command in the async group does not depend on the async
            // command
            continue;
          }
          operations[i].add_resource_use(
              ResourceUse::Read(commands[j]->token()));
          break;
        }
      }
    }
  }

  if (VLOG_IS_ON(2)) {
    for (const CommandOperation& op : operations) {
      VLOG(2) << op.ToString();
    }
  }

  return operations;
}

absl::StatusOr<CommandExecutor> CommandExecutor::Create(
    CommandSequence commands, SynchronizationMode synchronization_mode) {
  std::optional<ExecutionGraph> execution_graph = std::nullopt;

  // In automatic synchronization mode construct an execution graph for the
  // sequence of commands and derive the structure of command dependencies
  // from the buffer use conflicts.
  if (synchronization_mode != SynchronizationMode::kSerialize) {
    auto operations = CreateCommandOperations(commands, synchronization_mode);
    TF_ASSIGN_OR_RETURN(execution_graph,
                        ExecutionGraph::Create<CommandOperation>(operations));
    VLOG(3) << "Execution graph: " << execution_graph->ToString();
  }

  return CommandExecutor(synchronization_mode, std::move(commands),
                         std::move(execution_graph));
}

CommandExecutor::CommandExecutor(SynchronizationMode synchronization_mode,
                                 CommandSequence commands,
                                 std::optional<ExecutionGraph> execution_graph)
    : synchronization_mode_(synchronization_mode),
      commands_(std::move(commands)),
      execution_graph_(std::move(execution_graph)) {
  // Buffer allocations referenced by commands in this sequence.
  absl::btree_set<BufferAllocation::Index> allocs_indices;

  for (const std::unique_ptr<Command>& cmd : commands_) {
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

absl::Status CommandExecutor::Prepare(const Thunk::PrepareParams& params) {
  for (auto& command : commands_) {
    TF_RETURN_IF_ERROR(command->Prepare(params));
  }
  return absl::OkStatus();
}

absl::Status CommandExecutor::Initialize(const Thunk::InitializeParams& params,
                                         CommandStateManager& state) {
  for (auto& command : commands_) {
    TF_RETURN_IF_ERROR(command->Initialize(params, state));
  }
  return absl::OkStatus();
}

// A helper function that logs the details of a given command sequence for
// debugging purposes only.
static void VlogCommandSequenceDetails(const CommandSequence& commands) {
  if (!VLOG_IS_ON(5)) {
    return;
  }

  int64_t input_count = 0;
  int64_t output_count = 0;
  int64_t temp_count = 0;
  int64_t input_temp_count = 0;
  int64_t output_temp_count = 0;
  int64_t input_output_count = 0;
  int64_t input_temp_output_count = 0;

  absl::flat_hash_map<std::string, int64_t> input_cmds;
  absl::flat_hash_map<std::string, int64_t> output_cmds;
  absl::flat_hash_map<std::string, int64_t> temp_cmds;
  absl::flat_hash_map<std::string, int64_t> input_temp_cmds;
  absl::flat_hash_map<std::string, int64_t> output_temp_cmds;
  absl::flat_hash_map<std::string, int64_t> input_output_cmds;
  absl::flat_hash_map<std::string, int64_t> input_temp_output_cmds;

  for (const auto& cmd : commands) {
    bool has_input = false;
    bool has_output = false;
    bool has_temp = false;

    for (const auto& buffer : cmd->buffers()) {
      if (buffer.slice().allocation()->IsPreallocatedTempBuffer()) {
        has_temp = true;
      }
      if (buffer.slice().allocation()->is_entry_computation_parameter()) {
        has_input = true;
      }
      if (buffer.slice().allocation()->maybe_live_out()) {
        has_output = true;
      }
    }

    std::string cmd_name = CommandTypeString(cmd->command_type());

    if (has_input && !has_output && !has_temp) {
      input_count++;
      input_cmds[cmd_name]++;
    }
    if (!has_input && has_output && !has_temp) {
      output_count++;
      output_cmds[cmd_name]++;
    }
    if (!has_input && !has_output && has_temp) {
      temp_count++;
      temp_cmds[cmd_name]++;
    }
    if (has_input && !has_output && has_temp) {
      input_temp_count++;
      input_temp_cmds[cmd_name]++;
    }
    if (!has_input && has_output && has_temp) {
      output_temp_count++;
      output_temp_cmds[cmd_name]++;
    }
    if (has_input && has_output && !has_temp) {
      input_output_count++;
      input_output_cmds[cmd_name]++;
    }
    if (has_input && has_output && has_temp) {
      input_temp_output_count++;
      input_temp_output_cmds[cmd_name]++;
    }
  }

  auto print_cmds = [](const absl::flat_hash_map<std::string, int64_t>& cmds) {
    std::string s;
    for (const auto& [name, count] : cmds) {
      absl::StrAppend(&s, "\n    ", name, ": ", count);
    }
    return s;
  };

  VLOG(5) << "CommandExecutor allocation summary:\n"
          << "  Total commands                                 : "
          << commands.size() << "\n"
          << "  ------------------------------------------------\n"
          << "  Commands consuming input buffer                : "
          << input_count << print_cmds(input_cmds) << "\n"
          << "  Commands consuming output buffer               : "
          << output_count << print_cmds(output_cmds) << "\n"
          << "  Commands consuming temp buffer                 : " << temp_count
          << print_cmds(temp_cmds) << "\n"
          << "  Commands consuming input, temp buffers         : "
          << input_temp_count << print_cmds(input_temp_cmds) << "\n"
          << "  Commands consuming output, temp buffers        : "
          << output_temp_count << print_cmds(output_temp_cmds) << "\n"
          << "  Commands consuming input, output buffers       : "
          << input_output_count << print_cmds(input_output_cmds) << "\n"
          << "  Commands consuming input, temp, output buffers : "
          << input_temp_output_count << print_cmds(input_temp_output_cmds);
}

absl::Status CommandExecutor::Record(const Thunk::ExecuteParams& execute_params,
                                     const RecordParams& record_params,
                                     se::CommandBuffer* command_buffer) {
  if (command_buffer->state() == se::CommandBuffer::State::kFinalized) {
    TF_RETURN_IF_ERROR(command_buffer->Update());
  }

  if (command_buffer->state() == se::CommandBuffer::State::kUpdate) {
    TF_RETURN_IF_ERROR(
        RecordUpdate(execute_params, record_params, command_buffer));
  } else {
    TF_RETURN_IF_ERROR(RecordCreate(execute_params, record_params,
                                    command_buffer, /*dependencies=*/{})
                           .status());
  }

  return command_buffer->Finalize();
}

absl::StatusOr<std::vector<const se::CommandBuffer::Command*>>
CommandExecutor::RecordCreate(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    absl::Span<const se::CommandBuffer::Command* const> dependencies) const {
  // Command buffer must be in create state.
  TF_RETURN_IF_ERROR(CheckCommandBufferState(
      command_buffer, se::CommandBuffer::State::kCreate));

  VLOG(1) << "Record create " << commands_.size()
          << " commands: dependencies=" << dependencies.size();
  uint64_t start_micros = tsl::Env::Default()->NowMicros();

  // Short-circuit if there are no commands to record.
  if (commands_.empty()) {
    return std::vector<const se::CommandBuffer::Command*>{};
  }

  // Keep a state associated with commands in the sequence in the state
  // manager.
  CommandStateManager& state = record_params.state;

  // Collect sink commands while recording the command sequence.
  std::vector<const se::CommandBuffer::Command*> sink_commands;

  for (CommandId id = 0; id < commands_.size(); ++id) {
    Command* command = commands_[id].get();

    std::optional<tsl::profiler::ScopedAnnotation> annotation =
        GetKernelAnnotation(command->profile_annotation());

    // Skip recording collective commands if mock collectives are enabled.
    if (execute_params.mock_collectives && IsCollectiveCommand(*command)) {
      continue;
    }

    // Create new commands by recording them into the command buffer.
    DCHECK(!state.GetOrNull<RecordState>(command, command_buffer))
        << "Record state must be null for " << command->ToString();
    auto* record_state =
        state.GetOrCreate<RecordState>(command, command_buffer);

    std::vector<const se::CommandBuffer::Command*> command_dependencies =
        Dependencies(record_params, command_buffer, id);

    // Source command must depend on external dependencies passed by the
    // caller, internal commands dependencies are defined by the command
    // sequence structure (buffer and resource dependencies).
    auto record_action = IsSource(id)
                             ? Command::RecordCreate{dependencies}
                             : Command::RecordCreate{command_dependencies};

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

  // Log the details of a created command buffer if it is the primary one.A
  if (command_buffer->mode() == se::CommandBuffer::Mode::kPrimary) {
    VlogCommandSequenceDetails(commands_);
  }

  return sink_commands;
}

absl::Status CommandExecutor::RecordUpdate(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params,
    se::CommandBuffer* command_buffer) const {
  VLOG(1) << "Record update " << commands_.size() << " commands";
  uint64_t start_micros = tsl::Env::Default()->NowMicros();

  // Command buffer must be already prepared for recording updates.
  TF_RETURN_IF_ERROR(CheckCommandBufferState(
      command_buffer, se::CommandBuffer::State::kUpdate));

  // Short-circuit if there are no commands to update.
  if (commands_.empty()) {
    return absl::OkStatus();
  }

  // Keep a state associated with commands in the sequence in the state
  // manager.
  CommandStateManager& state = record_params.state;

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
    Command* command = commands_[id].get();
    if (command->requires_initialization() && record_params.is_initialization) {
      return false;
    }

    if (command->force_update()) {
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
    Command* command = commands_[id].get();

    std::optional<tsl::profiler::ScopedAnnotation> annotation =
        GetKernelAnnotation(command->profile_annotation());

    // Skip updating collective commands if mock collectives are enabled.
    if (execute_params.mock_collectives && IsCollectiveCommand(*command)) {
      continue;
    }

    // Skip updating command if it doesn't use any of the updated allocations.
    if (skip_command_update(id)) {
      VLOG(3) << "Skip updating command " << command->ToString();
      ++num_skipped_command_updates;
      continue;
    }

    // Update existing commands in the command buffer.
    auto* record_state = state.GetOrNull<RecordState>(command, command_buffer);
    DCHECK(record_state) << "Record state must be not null for "
                         << command->ToString();

    Command::RecordUpdate record_action{record_state->command};
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

absl::Status CommandExecutor::CheckCommandBufferState(
    se::CommandBuffer* command_buffer,
    se::CommandBuffer::State expected_state) const {
  if (command_buffer->state() != expected_state) {
    return Internal("Command buffer must be in %v state, got %v",
                    expected_state, command_buffer->state());
  }
  return absl::OkStatus();
}

bool CommandExecutor::IsSource(CommandId id) const {
  return execution_graph_ ? execution_graph_->is_source(id) : id == 0;
}

bool CommandExecutor::IsSink(CommandId id) const {
  return execution_graph_ ? execution_graph_->is_sink(id)
                          : id + 1 == commands_.size();
}

std::vector<const se::CommandBuffer::Command*> CommandExecutor::Dependencies(
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    CommandId id) const {
  // Collect commands that are dependencies of the command `id`.
  absl::InlinedVector<CommandId, 4> dependencies_ids;

  if (IsSource(id)) {
    VLOG(2) << "Command ID " << id
            << " is a source command, empty dependencies";
    return {};
  }

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
      // buffer, e.g. memcpy commands where source and destination are the
      // same. We have to follow dependencies of such commands to find the
      // real dependencies, so we don't record a command that is immediately
      // ready to execute, as it will create data races.
      auto deps = Dependencies(record_params, command_buffer, dependency_id);
      dependencies.insert(dependencies.end(), deps.begin(), deps.end());
    } else {
      dependencies.push_back(record_state->command);
    }
  }

  return dependencies;
}

const absl::flat_hash_set<BufferUse>& CommandExecutor::buffers() const {
  return buffers_;
}

absl::Span<const BufferAllocation::Index> CommandExecutor::allocs_indices()
    const {
  return allocs_indices_;
}

absl::StatusOr<std::string> CommandExecutor::RenderExecutionGraph() {
  ExecutionGraph::Renderer* renderer = ExecutionGraph::GetRenderer();
  if (renderer == nullptr) {
    return Unimplemented("No execution graph renderer registered");
  }

  if (synchronization_mode_ == SynchronizationMode::kSerialize) {
    return Unimplemented(
        "Execution graph rendering is only supported for "
        "concurrent/LHS synchronization mode");
  }

  auto operations = CreateCommandOperations(commands_, synchronization_mode_);
  absl::InlinedVector<const ExecutionGraph::Operation*, 32> operations_ptrs;
  operations_ptrs.reserve(operations.size());
  for (const auto& operation : operations) {
    operations_ptrs.push_back(&operation);
  }

  auto graph_as_string = renderer->GenerateGraphAsString(operations_ptrs);
  return renderer->PublishGraph(graph_as_string);
}

}  // namespace xla::gpu
