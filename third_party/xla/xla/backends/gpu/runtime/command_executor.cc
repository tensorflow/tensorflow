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
// An adaptor from Command to ExecutionGraph::Operation for building an
// execution graph from a command sequence.
class CommandOperation : public ExecutionGraph::Operation {
 public:
  explicit CommandOperation(const Command* cmd,
                            absl::Span<const ResourceUse> extra_resources = {})
      : name_(absl::StrFormat("cmd %s: %s", cmd->ToString(0),
                              cmd->profile_annotation())),
        cmd_(cmd),
        buffers_(CollectBufferUses(cmd)),
        resources_({ResourceUse::Write(cmd->token())}) {
    resources_.insert(resources_.end(), extra_resources.begin(),
                      extra_resources.end());
  }

  absl::string_view name() const final { return name_; }
  absl::Span<const BufferUse> BufferUses() const final { return buffers_; }
  absl::Span<const ResourceUse> ResourceUses() const final {
    return resources_;
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
    return absl::StrFormat("%s resources=[%s]", cmd_->ToString(0),
                           absl::StrJoin(resource_reprs, ", "));
  }

 private:
  static Command::BufferUses CollectBufferUses(const Command* cmd) {
    absl::flat_hash_set<BufferUse> buffers;
    cmd->Walk([&](const Command* command) {
      auto command_buffers = command->buffer_uses();
      buffers.insert(command_buffers.begin(), command_buffers.end());
    });
    return {buffers.begin(), buffers.end()};
  }

  std::string name_;
  const Command* cmd_;
  Command::BufferUses buffers_;
  Command::ResourceUses resources_;
};

void VlogOperations(const std::vector<CommandOperation>& operations) {
  if (VLOG_IS_ON(2)) {
    for (const CommandOperation& op : operations) {
      VLOG(2) << op.ToString();
    }
  }
}

std::vector<CommandOperation> CreateCommandOperationsWithConcurrentMode(
    const CommandSequence& commands,
    absl::Span<const Command::ResourceUses> extra_resources) {
  VLOG(3) << "CreateCommandOperations with synchronization mode: Concurrent";
  std::vector<CommandOperation> operations;
  operations.reserve(commands.size());

  // For concurrent synchronization mode, pass in buffer and resources for
  // dependency inference.
  for (size_t i = 0; i < commands.size(); ++i) {
    operations.emplace_back(commands[i], extra_resources.empty()
                                             ? absl::Span<const ResourceUse>{}
                                             : extra_resources[i]);
  }

  VlogOperations(operations);
  return operations;
}

// Helper: Check if a command is an Async Start
bool IsAsyncStart(const Command* cmd) {
  const auto* async = dynamic_cast<const AsyncStartCommand*>(cmd);
  return async && async->IsAsync();
}

// Helper: Check if a command is an Async Done
bool IsAsyncDone(const Command* cmd) {
  const auto* async = dynamic_cast<const AsyncDoneCommand*>(cmd);
  return async && async->IsAsync();
}

// Helper: Find the corresponding Start command for a given Done command
int64_t FindMatchingStartId(const CommandSequence& commands, int64_t done_idx) {
  const auto* done_cmd =
      dynamic_cast<const AsyncDoneCommand*>(commands[done_idx]);
  CHECK(done_cmd);

  for (int64_t j = done_idx - 1; j >= 0; --j) {
    if (!IsAsyncStart(commands[j])) {
      continue;
    }

    const auto* start_cmd = dynamic_cast<const AsyncStartCommand*>(commands[j]);
    CHECK(start_cmd);

    if (start_cmd->IsAsync() && done_cmd->async_start() == start_cmd) {
      return j;
    }
  }
  return -1;
}

// Helper: Add dependency on the nearest previous command that is NOT an Async
// Start, by pushing into `extras`.
void AddDependencyOnPrevNonStart(const CommandSequence& commands,
                                 int64_t current_idx,
                                 Command::ResourceUses& extras) {
  for (int64_t j = current_idx - 1; j >= 0; --j) {
    if (IsAsyncStart(commands[j])) {
      // Skip other starts
      continue;
    }

    extras.push_back(ResourceUse::Read(commands[j]->token()));
    break;  // Found the dependency, stop scanning
  }
}

std::vector<CommandOperation> CreateCommandOperationsWithLHSMode(
    const CommandSequence& commands,
    absl::Span<const Command::ResourceUses> extra_resources) {
  VLOG(3) << "CreateCommandOperations with synchronization mode: LHS";

  // 1. Dependency Analysis Phase: pre-compute LHS resource uses per command.
  std::vector<Command::ResourceUses> lhs_extras(commands.size());
  for (int64_t i = 0; i < static_cast<int64_t>(commands.size()); ++i) {
    if (IsAsyncDone(commands[i])) {
      // CASE A: Async Done — depends on its matching Start command
      int64_t start_id = FindMatchingStartId(commands, i);
      CHECK_NE(start_id, -1);
      lhs_extras[i].push_back(ResourceUse::Read(commands[start_id]->token()));

      // Also depends on immediate predecessor (if it's not the start itself)
      CHECK_GT(i, 0);
      if ((i - 1) != start_id) {
        lhs_extras[i].push_back(ResourceUse::Read(commands[i - 1]->token()));
      }
    } else {
      // CASE B: Standard Command OR Async Start
      // Both share the same logic: depend on the previous non-async-start
      AddDependencyOnPrevNonStart(commands, i, lhs_extras[i]);
    }
  }

  // 2. Construction Phase: build operations with merged extra resources.
  std::vector<CommandOperation> operations;
  operations.reserve(commands.size());
  for (size_t i = 0; i < commands.size(); ++i) {
    Command::ResourceUses merged;
    if (!extra_resources.empty()) {
      merged.insert(merged.end(), extra_resources[i].begin(),
                    extra_resources[i].end());
    }
    merged.insert(merged.end(), lhs_extras[i].begin(), lhs_extras[i].end());
    operations.emplace_back(commands[i], merged);
  }

  VlogOperations(operations);
  return operations;
}

}  // namespace

static absl::StatusOr<std::vector<CommandOperation>> CreateCommandOperations(
    const CommandSequence& commands,
    CommandExecutor::SynchronizationMode synchronization_mode,
    absl::Span<const Command::ResourceUses> extra_resources) {
  using Mode = CommandExecutor::SynchronizationMode;
  switch (synchronization_mode) {
    // Building an execution graph works the same for kConcurrent and
    // kConcurrentRegions.
    case Mode::kConcurrent:
    case Mode::kConcurrentRegions: {
      return CreateCommandOperationsWithConcurrentMode(commands,
                                                       extra_resources);
    }

    case Mode::kLHS: {
      return CreateCommandOperationsWithLHSMode(commands, extra_resources);
    }

    case Mode::kSerialize:
    default: {
      return Unimplemented(
          "CreateCommandOperations is only supported for Concurrent/LHS "
          "synchronization mode");
    }
  }
}

absl::StatusOr<CommandExecutor> CommandExecutor::Create(
    CommandSequence commands, SynchronizationMode synchronization_mode,
    std::vector<Command::ResourceUses> extra_resources) {
  std::optional<ExecutionGraph> execution_graph = std::nullopt;

  // In automatic synchronization mode construct an execution graph for the
  // sequence of commands and derive the structure of command dependencies
  // from the buffer use conflicts.
  if (synchronization_mode != SynchronizationMode::kSerialize) {
    TF_ASSIGN_OR_RETURN(auto operations,
                        CreateCommandOperations(commands, synchronization_mode,
                                                extra_resources));
    TF_ASSIGN_OR_RETURN(execution_graph,
                        ExecutionGraph::Create<CommandOperation>(operations));
    VLOG(3) << "Execution graph: " << execution_graph->ToString();
  }

  return CommandExecutor(synchronization_mode, std::move(commands),
                         std::move(execution_graph),
                         std::move(extra_resources));
}

CommandExecutor::CommandExecutor(
    SynchronizationMode synchronization_mode, CommandSequence commands,
    std::optional<ExecutionGraph> execution_graph,
    std::vector<Command::ResourceUses> extra_resources)
    : synchronization_mode_(synchronization_mode),
      commands_(std::move(commands)),
      execution_graph_(std::move(execution_graph)),
      extra_resources_(std::move(extra_resources)) {
  // Walk all nested commands and collect all buffers used by this executor.
  commands_.Walk([&](const Command* command) {
    Command::BufferUses buffer_uses = command->buffer_uses();
    buffer_uses_.insert(buffer_uses.begin(), buffer_uses.end());
  });

  // Buffer allocations referenced by all buffer uses.
  absl::btree_set<BufferAllocation::Index> allocs_indices;
  for (const BufferUse& buffer_use : buffer_uses_) {
    allocs_indices.insert(buffer_use.slice().index());
  }
  allocs_indices_.assign(allocs_indices.begin(), allocs_indices.end());

  // Iterate over the commands in the top level command sequence to build a
  // mapping from command index to allocation indices.
  for (Command* cmd : commands_) {
    absl::btree_set<BufferAllocation::Index> cmd_allocs_indices;
    cmd->Walk([&](const Command* command) {
      for (const BufferUse& buffer_use : command->buffer_uses()) {
        cmd_allocs_indices.insert(buffer_use.slice().index());
      }
    });

    // Record buffer allocations indices referenced by the `cmd`.
    cmd_allocs_indices_.emplace_back(cmd_allocs_indices.begin(),
                                     cmd_allocs_indices.end());
  }
}

absl::Status CommandExecutor::Prepare(const Thunk::PrepareParams& params) {
  for (auto& command : commands_) {
    TF_RETURN_IF_ERROR(command->Prepare(params));
  }
  return absl::OkStatus();
}

absl::Status CommandExecutor::Initialize(
    const Thunk::InitializeParams& params) {
  for (auto& command : commands_) {
    TF_RETURN_IF_ERROR(command->Initialize(params));
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

    for (const auto& buffer : cmd->buffer_uses()) {
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
                                     const Command::RecordParams& record_params,
                                     se::CommandBuffer* command_buffer,
                                     RecordId record_id) {
  if (command_buffer->state() == se::CommandBuffer::State::kFinalized) {
    TF_RETURN_IF_ERROR(command_buffer->Update());
  }

  if (command_buffer->state() == se::CommandBuffer::State::kUpdate) {
    TF_RETURN_IF_ERROR(
        RecordUpdate(execute_params, record_params, command_buffer, record_id));
  } else {
    TF_RETURN_IF_ERROR(RecordCreate(execute_params, record_params,
                                    command_buffer, /*dependencies=*/{},
                                    record_id)
                           .status());
  }

  return command_buffer->Finalize();
}

absl::StatusOr<std::vector<const se::CommandBuffer::Command*>>
CommandExecutor::RecordCreate(
    const Thunk::ExecuteParams& execute_params,
    const Command::RecordParams& record_params,
    se::CommandBuffer* command_buffer,
    absl::Span<const se::CommandBuffer::Command* const> dependencies,
    RecordId record_id) const {
  // Command buffer must be in create state.
  TF_RETURN_IF_ERROR(CheckCommandBufferState(
      command_buffer, se::CommandBuffer::State::kCreate));

  VLOG(1) << absl::StreamFormat(
      "Record create %d commands into command buffer %p: dependencies=%d, "
      "record_id=%v",
      commands_.size(), command_buffer, dependencies.size(), record_id);
  uint64_t start_micros = tsl::Env::Default()->NowMicros();

  // Short-circuit if there are no commands to record.
  if (commands_.empty()) {
    return std::vector<const se::CommandBuffer::Command*>{};
  }

  auto* state = command_buffer->GetOrConstructResource<CommandExecutorsState>();
  RecordedCommands& recorded_commands =
      state->recorded_commands[std::make_pair(this, record_id)];

  // Check that this executor was not already recorded into the command buffer.
  if (!recorded_commands.empty()) {
    return Internal(
        "Recorded commands are not empty (size %d). Command executor can be "
        "recorded into the command buffer at most once. After it was recorded "
        "it can be only updated",
        recorded_commands.size());
  }

  // Collect sink commands while recording the command sequence.
  std::vector<const se::CommandBuffer::Command*> sink_commands;

  for (CommandId id = 0; id < commands_.size(); ++id) {
    Command* command = commands_[id];

    std::optional<tsl::profiler::ScopedAnnotation> annotation =
        GetKernelAnnotation(command->profile_annotation());

    // Skip recording collective commands if mock collectives are enabled.
    if (execute_params.mock_collectives && IsCollectiveCommand(*command)) {
      continue;
    }

    std::vector<const se::CommandBuffer::Command*> command_dependencies =
        Dependencies(record_params, command_buffer, id, record_id);

    // Source command must depend on external dependencies passed by the
    // caller, internal commands dependencies are defined by the command
    // sequence structure (buffer and resource dependencies).
    auto record_action = IsSource(id)
                             ? Command::RecordCreate{dependencies}
                             : Command::RecordCreate{command_dependencies};

    TF_ASSIGN_OR_RETURN(
        const se::CommandBuffer::Command* recorded_command,
        command->Record(execute_params, record_params, std::move(record_action),
                        command_buffer));

    // Collect sink commands as external dependencies for the next command
    // sequence recorded into the same command buffer.
    if (IsSink(id)) {
      sink_commands.push_back(recorded_command);
    }

    recorded_commands.push_back(recorded_command);
  }

  uint64_t end_micros = tsl::Env::Default()->NowMicros();
  VLOG(1) << absl::StreamFormat(
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
    const Command::RecordParams& record_params,
    se::CommandBuffer* command_buffer, RecordId record_id) const {
  VLOG(1) << absl::StreamFormat(
      "Record update %d commands into command buffer %p: record_id=%v",
      commands_.size(), command_buffer, record_id);
  uint64_t start_micros = tsl::Env::Default()->NowMicros();

  // Command buffer must be already prepared for recording updates.
  TF_RETURN_IF_ERROR(CheckCommandBufferState(
      command_buffer, se::CommandBuffer::State::kUpdate));

  // Short-circuit if there are no commands to update.
  if (commands_.empty()) {
    return absl::OkStatus();
  }

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

    Command* command = commands_[id];

    // For CAPTURE_CMD_NEVER_UPDATE mode, always skip updates for commands
    // implemented via tracing (TracedCommand subclasses) or collective
    // operations (CollectiveCmd subclasses). Their buffer allocations are
    // VA-remapped to fixed offsets within the reserved VA range, so their
    // recorded addresses remain valid across executions — no update is needed.
    //
    // Note: CollectiveCmd satisfies both IsTracedCommand() and
    // requires_initialization(), but the requires_initialization() check below
    // is intentionally unreachable for traced commands in this mode. Because
    // their buffer addresses are stable (VA-mapped), re-initialization is
    // unnecessary.
    if (record_params.command_buffer_update_mode ==
            DebugOptions::CAPTURE_CMD_NEVER_UPDATE &&
        command->IsTracedCommand()) {
      VLOG(3) << "Skipping update for traced command " << id
              << " (CAPTURE_CMD_NEVER_UPDATE mode)";
      return true;
    }

    // We always update commands that require initialization, even if buffer
    // allocations didn't change.
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

  auto* state = command_buffer->GetOrConstructResource<CommandExecutorsState>();
  RecordedCommands& recorded_commands =
      state->recorded_commands[std::make_pair(this, record_id)];

  // Check this this executor was correctly recorded into the command buffer.
  if (recorded_commands.size() != commands_.size()) {
    return Internal(
        "The number of recorded commands must match the size of command "
        "sequence: %d vs %d",
        recorded_commands.size(), commands_.size());
  }

  size_t num_skipped_command_updates = 0;

  for (CommandId id = 0; id < commands_.size(); ++id) {
    Command* command = commands_[id];

    std::optional<tsl::profiler::ScopedAnnotation> annotation =
        GetKernelAnnotation(command->profile_annotation());

    // Skip updating collective commands if mock collectives are enabled.
    if (execute_params.mock_collectives && IsCollectiveCommand(*command)) {
      continue;
    }

    // Skip updating command if it doesn't use any of the updated allocations.
    if (skip_command_update(id)) {
      VLOG(3) << "Skip updating command " << command->ToString(0);
      ++num_skipped_command_updates;
      continue;
    }

    Command::RecordUpdate record_action{recorded_commands[id]};
    TF_ASSIGN_OR_RETURN(
        recorded_commands[id],
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
    const Command::RecordParams& record_params,
    se::CommandBuffer* command_buffer, CommandId id, RecordId record_id) const {
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

  auto* state = command_buffer->GetOrConstructResource<CommandExecutorsState>();
  RecordedCommands& recorded_commands =
      state->recorded_commands[std::make_pair(this, record_id)];

  // Collect dependencies from the recorded command state.
  std::vector<const se::CommandBuffer::Command*> dependencies;
  for (CommandId dependency_id : dependencies_ids) {
    DCHECK_LT(dependency_id, recorded_commands.size());
    if (recorded_commands[dependency_id] == nullptr) {
      // Some commands might end up not recording anything into the command
      // buffer, e.g. memcpy commands where source and destination are the
      // same. We have to follow dependencies of such commands to find the
      // real dependencies, so we don't record a command that is immediately
      // ready to execute, as it will create data races.
      auto deps =
          Dependencies(record_params, command_buffer, dependency_id, record_id);
      dependencies.insert(dependencies.end(), deps.begin(), deps.end());
    } else {
      dependencies.push_back(recorded_commands[dependency_id]);
    }
  }

  return dependencies;
}

const absl::flat_hash_set<BufferUse>& CommandExecutor::buffer_uses() const {
  return buffer_uses_;
}

absl::Span<const BufferAllocation::Index> CommandExecutor::allocs_indices()
    const {
  return allocs_indices_;
}

absl::StatusOr<std::string> CommandExecutor::RenderExecutionGraph() const {
  ExecutionGraph::Renderer* renderer = ExecutionGraph::GetRenderer();
  if (renderer == nullptr) {
    return Unimplemented("No execution graph renderer registered");
  }

  if (synchronization_mode_ == SynchronizationMode::kSerialize) {
    return Unimplemented(
        "Execution graph rendering is only supported for "
        "concurrent/LHS synchronization mode");
  }

  TF_ASSIGN_OR_RETURN(auto operations,
                      CreateCommandOperations(commands_, synchronization_mode_,
                                              extra_resources_));
  absl::InlinedVector<const ExecutionGraph::Operation*, 32> operations_ptrs;
  operations_ptrs.reserve(operations.size());
  for (const auto& operation : operations) {
    operations_ptrs.push_back(&operation);
  }

  auto graph_as_string = renderer->GenerateGraphAsString(operations_ptrs);
  return renderer->PublishGraph(graph_as_string);
}

}  // namespace xla::gpu
