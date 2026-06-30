/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/conditional_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/functional/overload.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/host_memory_pool.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

// Create a callback to create a command buffer from a command sequence.
se::CommandBuffer::CreateCommands CreateCommands(
    const CommandExecutor* commands, const Thunk::ExecuteParams* execute_params,
    const Command::RecordParams* record_params) {
  return [=](se::CommandBuffer* command_buffer,
             absl::Span<const se::CommandBuffer::Command* const> dependencies) {
    return commands->RecordCreate(*execute_params, *record_params,
                                  command_buffer, dependencies);
  };
}

// Create callbacks to create command buffers from command sequences.
std::vector<se::CommandBuffer::CreateCommands> CreateCommands(
    absl::Span<const CommandExecutor> commands,
    const Thunk::ExecuteParams* execute_params,
    const Command::RecordParams* record_params) {
  std::vector<se::CommandBuffer::CreateCommands> create_commands;
  create_commands.reserve(commands.size());
  for (const CommandExecutor& cmd : commands) {
    create_commands.push_back(
        CreateCommands(&cmd, execute_params, record_params));
  }
  return create_commands;
}

// Create a callback to update a command buffer with a command sequence.
se::CommandBuffer::UpdateCommands UpdateCommands(
    const CommandExecutor* commands, const Thunk::ExecuteParams* execute_params,
    const Command::RecordParams* record_params) {
  return [=](se::CommandBuffer* command_buffer) {
    return commands->RecordUpdate(*execute_params, *record_params,
                                  command_buffer);
  };
}

// Create callbacks to update command buffers with command sequences.
std::vector<se::CommandBuffer::UpdateCommands> UpdateCommands(
    absl::Span<const CommandExecutor> commands,
    const Thunk::ExecuteParams* execute_params,
    const Command::RecordParams* record_params) {
  std::vector<se::CommandBuffer::UpdateCommands> update_commands;
  update_commands.reserve(commands.size());
  for (const CommandExecutor& cmd : commands) {
    update_commands.push_back(
        UpdateCommands(&cmd, execute_params, record_params));
  }
  return update_commands;
}

using CreateCommand =
    absl::FunctionRef<absl::StatusOr<const se::CommandBuffer::Command*>(
        absl::Span<const se::CommandBuffer::Command* const> dependencies)>;

using UpdateCommand =
    absl::FunctionRef<absl::Status(const se::CommandBuffer::Command* command)>;

absl::StatusOr<const se::CommandBuffer::Command*> HandleRecordAction(
    Command::RecordAction action, CreateCommand create_command,
    UpdateCommand update_command) {
  if (auto* create = std::get_if<Command::RecordCreate>(&action)) {
    return create_command(create->dependencies);
  }

  if (auto* update = std::get_if<Command::RecordUpdate>(&action)) {
    RETURN_IF_ERROR(update_command(update->command));
    return update->command;
  }

  return Internal("Invalid record action");
}

}  // namespace

ConditionalThunk::ConditionalThunk(ThunkInfo thunk_info,
                                   const ShapedSlice& branch_index_buffer_index,
                                   std::vector<ThunkSequence> branch_thunks)
    : Command(Kind::kConditional, std::move(thunk_info)),
      branch_index_buffer_index_(branch_index_buffer_index),
      branch_index_is_bool_(branch_index_buffer_index.shape.element_type() ==
                            PRED) {
  PrimitiveType element_type = branch_index_buffer_index.shape.element_type();
  CHECK(element_type == PRED || element_type == S32);
  CHECK_EQ(branch_index_buffer_index.shape.dimensions(),
           std::vector<int64_t>{});

  for (auto& branch_thunks : branch_thunks) {
    branch_executors_.push_back(ThunkExecutor(std::move(branch_thunks)));
  }
}

absl::Status ConditionalThunk::Prepare(const PrepareParams& params) {
  if (branch_index_is_bool_) {
    TF_RET_CHECK(branch_executors_.size() == 2);
  } else {
    TF_RET_CHECK(!branch_executors_.empty());
  }
  for (auto& branch_executor : branch_executors_) {
    RETURN_IF_ERROR(branch_executor.Prepare(params));
  }
  if (command_branch_executors_.has_value()) {
    for (CommandExecutor& branch_executor : *command_branch_executors_) {
      RETURN_IF_ERROR(branch_executor.Prepare(params));
    }
  }
  return absl::OkStatus();
}

absl::Status ConditionalThunk::Initialize(const InitializeParams& params) {
  if (branch_index_is_bool_) {
    TF_RET_CHECK(branch_executors_.size() == 2);
  } else {
    TF_RET_CHECK(!branch_executors_.empty());
  }
  for (auto& branch_executor : branch_executors_) {
    RETURN_IF_ERROR(branch_executor.Initialize(params));
  }
  if (command_branch_executors_.has_value()) {
    for (CommandExecutor& branch_executor : *command_branch_executors_) {
      RETURN_IF_ERROR(branch_executor.Initialize(params));
    }
  }

  absl::MutexLock lock(mutex_);

  if (!host_memory_pools_.contains(params.executor)) {
    PrimitiveType type =
        branch_index_is_bool_ ? PrimitiveType::PRED : PrimitiveType::S32;
    ASSIGN_OR_RETURN(std::unique_ptr<HostMemoryPool> pool,
                     HostMemoryPool::Create(params.executor, type));
    host_memory_pools_[params.executor] = std::move(pool);
  }

  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> ConditionalThunk::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  if (!command_branch_executors_.has_value()) {
    return FailedPrecondition(
        "ConditionalThunk command-buffer branches are not initialized");
  }

  se::DeviceAddressBase branch_index =
      execute_params.buffer_allocations->GetDeviceAddress(
          branch_index_buffer_index_.slice);

  VLOG(5) << "ConditionalThunk::Record:";
  VLOG(5) << "  branch_index: " << branch_index_buffer_index_ << " ("
          << branch_index.opaque() << ")";

  absl::Span<const CommandExecutor> command_branches =
      absl::MakeConstSpan(*command_branch_executors_);

  return HandleRecordAction(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        if (branch_index_is_bool_) {
          return command_buffer->CreateCase(
              se::DeviceAddress<bool>(branch_index),
              CreateCommands(command_branches, &execute_params, &record_params),
              dependencies);
        }
        return command_buffer->CreateCase(
            se::DeviceAddress<int32_t>(branch_index),
            CreateCommands(command_branches, &execute_params, &record_params),
            dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        if (branch_index_is_bool_) {
          return command_buffer->UpdateCase(
              command, se::DeviceAddress<bool>(branch_index),
              UpdateCommands(command_branches, &execute_params,
                             &record_params));
        }
        return command_buffer->UpdateCase(
            command, se::DeviceAddress<int32_t>(branch_index),
            UpdateCommands(command_branches, &execute_params, &record_params));
      });
}

absl::Status ConditionalThunk::SetOrUpdateCommandBufferBranchExecutors(
    std::vector<CommandExecutor> branch_executors) {
  if (branch_index_is_bool_) {
    TF_RET_CHECK(branch_executors.size() == 2);
  } else {
    TF_RET_CHECK(!branch_executors.empty());
  }
  command_branch_executors_ = std::move(branch_executors);
  return absl::OkStatus();
}

absl::Status ConditionalThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto& stream = *params.stream;

  HostMemoryPool* pool;
  {
    absl::MutexLock lock(mutex_);
    pool = host_memory_pools_.at(stream.parent()).get();
  }
  ASSIGN_OR_RETURN(HostMemoryPool::Handle handle, pool->Acquire());

  // Copy the predicate value from device.
  auto branch_index_or_pred = [&]() -> std::variant<int32_t*, bool*> {
    if (branch_index_is_bool_) {
      return handle.get<bool>();
    }
    return handle.get<int32_t>();
  }();

  se::DeviceAddressBase branch_index_address =
      params.buffer_allocations->GetDeviceAddress(
          branch_index_buffer_index_.slice);
  if (branch_index_is_bool_) {
    RETURN_IF_ERROR(stream.Memcpy(std::get<bool*>(branch_index_or_pred),
                                  branch_index_address, sizeof(bool)));
  } else {
    RETURN_IF_ERROR(stream.Memcpy(std::get<int32_t*>(branch_index_or_pred),
                                  branch_index_address, sizeof(int32_t)));
  }

  if (absl::Status blocked = stream.BlockHostUntilDone(); !blocked.ok()) {
    return Internal("Failed to retrieve branch_index value on stream %p: %s.",
                    &stream, blocked.message());
  }

  int32_t branch_index = std::visit(
      absl::Overload([](int32_t* branch_index) { return *branch_index; },
                     [](bool* pred) { return *pred ? 0 : 1; }),
      branch_index_or_pred);

  absl::string_view branch_kind =
      std::visit(absl::Overload([](int32_t*) { return "index"; },
                                [](bool*) { return "pred"; }),
                 branch_index_or_pred);

  VLOG(3) << "ConditionalThunk: branch_index=" << branch_index
          << " (kind: " << branch_kind << ")";

  // Handle default scenario for branch_index not in [0, num_branches).
  if (branch_index < 0 || branch_index >= branch_executors_.size()) {
    branch_index = static_cast<int32_t>(branch_executors_.size()) - 1;
  }

  // Execute the branch computation corresponding to the value of branch_index.
  RETURN_IF_ERROR(branch_executors_[branch_index].ExecuteOnStream(params));

  return absl::OkStatus();
}

absl::Status ConditionalThunk::WalkNested(Walker callback) {
  for (ThunkExecutor& branch_executor : branch_executors_) {
    RETURN_IF_ERROR(branch_executor.thunks().WalkNested(callback));
  }
  return absl::OkStatus();
}

absl::Status ConditionalThunk::WalkNestedCommands(CommandWalker callback) {
  if (!command_branch_executors_.has_value()) {
    return absl::OkStatus();
  }
  for (CommandExecutor& branch_executor : *command_branch_executors_) {
    RETURN_IF_ERROR(branch_executor.Walk(callback));
  }
  return absl::OkStatus();
}

absl::Status ConditionalThunk::TransformNested(Transformer callback) {
  for (ThunkExecutor& branch_executor : branch_executors_) {
    RETURN_IF_ERROR(branch_executor.thunks().TransformNested(callback));
  }
  return absl::OkStatus();
}

absl::StatusOr<ThunkProto> ConditionalThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  auto* conditional_thunk_proto = proto.mutable_conditional_thunk();
  ASSIGN_OR_RETURN(*conditional_thunk_proto->mutable_branch_index_buffer(),
                   branch_index_buffer_index_.ToProto());

  for (const ThunkExecutor& branch_executor : branch_executors_) {
    ThunkSequenceProto thunk_sequence_proto;
    for (const std::unique_ptr<Thunk>& thunk : branch_executor.thunks()) {
      ASSIGN_OR_RETURN(*thunk_sequence_proto.add_thunks(), thunk->ToProto());
    }
    *conditional_thunk_proto->add_branch_thunks() =
        std::move(thunk_sequence_proto);
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<ConditionalThunk>> ConditionalThunk::FromProto(
    ThunkInfo thunk_info, const ConditionalThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    const Deserializer& deserializer) {
  ASSIGN_OR_RETURN(ShapedSlice branch_index_buffer_index,
                   ShapedSlice::FromProto(thunk_proto.branch_index_buffer(),
                                          buffer_allocations));

  std::vector<ThunkSequence> branch_thunks;
  branch_thunks.reserve(thunk_proto.branch_thunks_size());
  for (const auto& thunk_sequence_proto : thunk_proto.branch_thunks()) {
    ThunkSequence thunks;
    for (const auto& proto : thunk_sequence_proto.thunks()) {
      ASSIGN_OR_RETURN(std::unique_ptr<Thunk> thunk, deserializer(proto));
      thunks.push_back(std::move(thunk));
    }
    branch_thunks.push_back(std::move(thunks));
  }
  return std::make_unique<ConditionalThunk>(std::move(thunk_info),
                                            branch_index_buffer_index,
                                            std::move(branch_thunks));
}

std::string ConditionalThunk::ToString(int indent) const {
  std::string indent_str(indent * 2, ' ');
  std::string result;
  absl::StrAppend(&result, indent_str, "\n");
  if (branch_index_is_bool_) {
    CHECK_EQ(branch_executors_.size(), 2);
    absl::StrAppend(&result, indent_str, "false_branch:\n",
                    branch_executors_[0].thunks().ToString(indent + 1));
    absl::StrAppend(&result, indent_str, "true_branch:\n",
                    branch_executors_[1].thunks().ToString(indent + 1));
  } else {
    for (size_t i = 0; i < branch_executors_.size(); ++i) {
      absl::StrAppend(&result, indent_str, "branch_", i, ":\n",
                      branch_executors_[i].thunks().ToString(indent + 1));
    }
  }
  return result;
}

}  // namespace xla::gpu
