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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/while_loop.h"
#include "xla/executable_run_options.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Create a callback to create a command buffer from a command sequence.
static se::CommandBuffer::CreateCommands CreateCommands(
    const CommandExecutor* commands, const Thunk::ExecuteParams* execute_params,
    const Command::RecordParams* record_params) {
  return [=](se::CommandBuffer* command_buffer,
             absl::Span<const se::CommandBuffer::Command* const> dependencies) {
    return commands->RecordCreate(*execute_params, *record_params,
                                  command_buffer, dependencies);
  };
}

// Create a callback to update a command buffer with command sequence.
static se::CommandBuffer::UpdateCommands UpdateCommands(
    const CommandExecutor* commands, const Thunk::ExecuteParams* execute_params,
    const Command::RecordParams* record_params) {
  return [=](se::CommandBuffer* command_buffer) {
    return commands->RecordUpdate(*execute_params, *record_params,
                                  command_buffer);
  };
}

//===----------------------------------------------------------------------===//
// Command::RecordAction helpers.
//===----------------------------------------------------------------------===//

using CreateCommand =
    absl::FunctionRef<absl::StatusOr<const se::CommandBuffer::Command*>(
        absl::Span<const se::CommandBuffer::Command* const> dependencies)>;

using UpdateCommand =
    absl::FunctionRef<absl::Status(const se::CommandBuffer::Command* command)>;

// Handles a record action by calling one of the user-provided functions.
static absl::StatusOr<const se::CommandBuffer::Command*> Handle(
    Command::RecordAction action, CreateCommand create_command,
    UpdateCommand update_command) {
  if (auto* create = std::get_if<Command::RecordCreate>(&action)) {
    return create_command(create->dependencies);
  }

  if (auto* update = std::get_if<Command::RecordUpdate>(&action)) {
    TF_RETURN_IF_ERROR(update_command(update->command));
    return update->command;
  }

  return Internal("Invalid record action");
}

//===----------------------------------------------------------------------===//
// WhileCmd
//===----------------------------------------------------------------------===//

WhileCmd::WhileCmd(BufferAllocation::Slice pred, CommandExecutor cond_commands,
                   CommandExecutor body_commands,
                   std::optional<int64_t> trip_count, bool enable_loop_unroll)
    : Command(CommandType::kWhileCmd),
      pred_(pred),
      cond_commands_(std::move(cond_commands)),
      body_commands_(std::move(body_commands)),
      trip_count_(trip_count),
      enable_loop_unroll_(enable_loop_unroll) {}

absl::Status WhileCmd::Initialize(const Thunk::InitializeParams& params) {
  TF_RETURN_IF_ERROR(cond_commands_.Initialize(params));
  TF_RETURN_IF_ERROR(body_commands_.Initialize(params));
  if (enable_loop_unroll_ && body_commands_.support_loop_unroll() &&
      cond_commands_.support_loop_unroll() && trip_count_.has_value()) {
    is_unrolled_loop_ = true;
  }
  VLOG(3) << "WhileCmd::Initialize: enable_loop_unroll_=" << enable_loop_unroll_
          << ", body_support=" << body_commands_.support_loop_unroll()
          << ", cond_support=" << cond_commands_.support_loop_unroll()
          << ", trip_count=" << trip_count_.value_or(-1)
          << ", is_unrolled_loop_=" << is_unrolled_loop_;
  return absl::OkStatus();
}

absl::Status WhileCmd::Prepare(const Thunk::PrepareParams& params) {
  TF_RETURN_IF_ERROR(cond_commands_.Prepare(params));
  TF_RETURN_IF_ERROR(body_commands_.Prepare(params));
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> WhileCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::DeviceAddressBase pred =
      execute_params.buffer_allocations->GetDeviceAddress(pred_);

  VLOG(5) << "WhileCmd: cond_commands=" << cond_commands_.size()
          << " body_commands=" << body_commands_.size();
  VLOG(5) << "  pred: " << pred_ << " (" << pred.opaque() << ")";
  VLOG(5) << "  trip_count: " << trip_count_.value_or(-1)
          << " (unroll: " << is_unrolled_loop_ << ")";
  if (is_unrolled_loop_) {
    // When the loop is unrolled, we need to record the body commands for
    // `trip_count` times into child_command_buffer, and implement the While
    // command as a child command.
    //
    // Unroll the while loop body for `trip_count` times.
    // Unrolled execution sequence: cond -> body -> cond -> body -> ...
    // In the unrolled pattern, we still need to run the cond commands because
    // body commands might depends on the value of index variable that is
    // updated by condition commands.

    auto record_fn =
        [&](se::CommandBuffer* child_command_buffer) -> absl::Status {
      VLOG(3) << "Recording unrolled loop with trip_count: "
              << trip_count_.value();

      Command::RecordParams new_record_params = record_params;
      std::vector<const se::CommandBuffer::Command*> dependencies;

      ScopedWhileLoop loop("record_fn", trip_count_);
      for (int64_t i = 0; i < *trip_count_; loop.IncLoopIteration(), ++i) {
        CommandExecutor::RecordId record_id(i);
        TF_ASSIGN_OR_RETURN(dependencies,
                            cond_commands_.RecordCreate(
                                execute_params, new_record_params,
                                child_command_buffer, dependencies, record_id));
        TF_ASSIGN_OR_RETURN(dependencies,
                            body_commands_.RecordCreate(
                                execute_params, new_record_params,
                                child_command_buffer, dependencies, record_id));
      }

      return absl::OkStatus();
    };

    auto update_fn =
        [&](se::CommandBuffer* child_command_buffer) -> absl::Status {
      VLOG(3) << "Updating unrolled loop with trip_count: "
              << trip_count_.value();

      Command::RecordParams new_record_params = record_params;

      ScopedWhileLoop loop("record_fn", trip_count_);
      for (int64_t i = 0; i < *trip_count_; loop.IncLoopIteration(), ++i) {
        CommandExecutor::RecordId record_id(i);
        TF_RETURN_IF_ERROR(
            cond_commands_.RecordUpdate(execute_params, new_record_params,
                                        child_command_buffer, record_id));
        TF_RETURN_IF_ERROR(
            body_commands_.RecordUpdate(execute_params, new_record_params,
                                        child_command_buffer, record_id));
      }

      return absl::OkStatus();
    };

    return Handle(
        std::move(record_action),
        [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
          return command_buffer->CreateChildCommand(record_fn, dependencies);
        },
        [&](const se::CommandBuffer::Command* command) {
          return command_buffer->UpdateChildCommand(command, update_fn);
        });
  }

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateWhile(
            se::DeviceAddress<bool>(pred),
            CreateCommands(&cond_commands_, &execute_params, &record_params),
            CreateCommands(&body_commands_, &execute_params, &record_params),
            dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateWhile(
            command, se::DeviceAddress<bool>(pred),
            UpdateCommands(&cond_commands_, &execute_params, &record_params),
            UpdateCommands(&body_commands_, &execute_params, &record_params));
      });
}

Command::BufferUses WhileCmd::buffer_uses() const {
  return {BufferUse::Read(pred_, ShapeUtil::MakeShape(PRED, {}))};
}

absl::Status WhileCmd::WalkNested(
    absl::FunctionRef<absl::Status(Thunk*)> callback) {
  RETURN_IF_ERROR(cond_commands_.Walk(
      [&](Command* cmd) -> absl::Status { return callback(cmd); }));
  return body_commands_.Walk(
      [&](Command* cmd) -> absl::Status { return callback(cmd); });
}

absl::Status WhileCmd::WalkNestedCommands(CommandWalker callback) {
  RETURN_IF_ERROR(cond_commands_.Walk(callback));
  return body_commands_.Walk(callback);
}

}  // namespace xla::gpu
