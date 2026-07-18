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

#include "xla/backends/gpu/runtime/while_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/host_memory_pool.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/gpu/runtime/while_loop.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::gpu {

using ::tsl::profiler::TraceMe;
using ::tsl::profiler::TraceMeEncode;

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

// Create a callback to update a command buffer with a command sequence.
se::CommandBuffer::UpdateCommands UpdateCommands(
    const CommandExecutor* commands, const Thunk::ExecuteParams* execute_params,
    const Command::RecordParams* record_params) {
  return [=](se::CommandBuffer* command_buffer) {
    return commands->RecordUpdate(*execute_params, *record_params,
                                  command_buffer);
  };
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

WhileThunk::WhileThunk(
    ThunkInfo thunk_info,
    const BufferAllocation::Slice& condition_result_buffer_index,
    ThunkSequence condition_thunks, ThunkSequence body_thunks,
    std::optional<int64_t> trip_count)
    : Command(Kind::kWhile, std::move(thunk_info)),
      condition_result_buffer_index_(condition_result_buffer_index),
      condition_executor_(std::move(condition_thunks)),
      body_executor_(std::move(body_thunks)),
      trip_count_(trip_count) {}

absl::Status WhileThunk::Prepare(const PrepareParams& params) {
  RETURN_IF_ERROR(condition_executor_.Prepare(params));
  RETURN_IF_ERROR(body_executor_.Prepare(params));
  if (command_condition_executor_.has_value()) {
    RETURN_IF_ERROR(command_condition_executor_->Prepare(params));
  }
  if (command_body_executor_.has_value()) {
    RETURN_IF_ERROR(command_body_executor_->Prepare(params));
  }
  return absl::OkStatus();
}

absl::Status WhileThunk::Initialize(const InitializeParams& params) {
  RETURN_IF_ERROR(condition_executor_.Initialize(params));
  RETURN_IF_ERROR(body_executor_.Initialize(params));
  if (command_condition_executor_.has_value()) {
    RETURN_IF_ERROR(command_condition_executor_->Initialize(params));
  }
  if (command_body_executor_.has_value()) {
    RETURN_IF_ERROR(command_body_executor_->Initialize(params));
  }

  is_unrolled_loop_ = false;
  if (command_condition_executor_.has_value() &&
      command_body_executor_.has_value() && enable_loop_unroll_ &&
      command_body_executor_->support_loop_unroll() &&
      command_condition_executor_->support_loop_unroll() &&
      trip_count_.has_value()) {
    is_unrolled_loop_ = true;
  }
  VLOG(3) << "WhileThunk::Initialize command buffer: enable_loop_unroll_="
          << enable_loop_unroll_ << ", body_support="
          << (command_body_executor_.has_value()
                  ? command_body_executor_->support_loop_unroll()
                  : false)
          << ", cond_support="
          << (command_condition_executor_.has_value()
                  ? command_condition_executor_->support_loop_unroll()
                  : false)
          << ", trip_count=" << trip_count_.value_or(-1)
          << ", is_unrolled_loop_=" << is_unrolled_loop_;

  absl::MutexLock lock(mutex_);
  if (!host_memory_pools_.contains(params.executor)) {
    ASSIGN_OR_RETURN(
        std::unique_ptr<HostMemoryPool> pool,
        HostMemoryPool::Create(params.executor, PrimitiveType::PRED));
    host_memory_pools_[params.executor] = std::move(pool);
  }
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> WhileThunk::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  if (!command_condition_executor_.has_value() ||
      !command_body_executor_.has_value()) {
    return FailedPrecondition(
        "WhileThunk command-buffer condition/body executors are not "
        "initialized");
  }

  se::DeviceAddressBase pred =
      execute_params.buffer_allocations->GetDeviceAddress(
          condition_result_buffer_index_);

  VLOG(5) << "WhileThunk::Record:";
  VLOG(5) << "  cond_commands=" << command_condition_executor_->size()
          << " body_commands=" << command_body_executor_->size();
  VLOG(5) << "  pred: " << condition_result_buffer_index_ << " ("
          << pred.opaque() << ")";
  VLOG(5) << "  trip_count: " << trip_count_.value_or(-1)
          << " (unroll: " << is_unrolled_loop_ << ")";

  if (is_unrolled_loop_) {
    // Unrolled execution sequence: cond -> body -> cond -> body -> ...
    auto record_fn =
        [&](se::CommandBuffer* child_command_buffer) -> absl::Status {
      VLOG(3) << "Recording unrolled loop with trip_count: "
              << trip_count_.value();

      Command::RecordParams new_record_params = record_params;
      std::vector<const se::CommandBuffer::Command*> dependencies;

      ScopedWhileLoop loop("record_fn", trip_count_);
      for (int64_t i = 0; i < *trip_count_; loop.IncLoopIteration(), ++i) {
        CommandExecutor::RecordId record_id(i);
        ASSIGN_OR_RETURN(dependencies,
                         command_condition_executor_->RecordCreate(
                             execute_params, new_record_params,
                             child_command_buffer, dependencies, record_id));
        ASSIGN_OR_RETURN(dependencies,
                         command_body_executor_->RecordCreate(
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
        RETURN_IF_ERROR(command_condition_executor_->RecordUpdate(
            execute_params, new_record_params, child_command_buffer,
            record_id));
        RETURN_IF_ERROR(command_body_executor_->RecordUpdate(
            execute_params, new_record_params, child_command_buffer,
            record_id));
      }

      return absl::OkStatus();
    };

    return HandleRecordAction(
        std::move(record_action),
        [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
          return command_buffer->CreateChildCommand(record_fn, dependencies);
        },
        [&](const se::CommandBuffer::Command* command) {
          return command_buffer->UpdateChildCommand(command, update_fn);
        });
  }

  return HandleRecordAction(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateWhile(
            se::DeviceAddress<bool>(pred),
            CreateCommands(&*command_condition_executor_, &execute_params,
                           &record_params),
            CreateCommands(&*command_body_executor_, &execute_params,
                           &record_params),
            dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateWhile(
            command, se::DeviceAddress<bool>(pred),
            UpdateCommands(&*command_condition_executor_, &execute_params,
                           &record_params),
            UpdateCommands(&*command_body_executor_, &execute_params,
                           &record_params));
      });
}

absl::Status WhileThunk::SetOrUpdateCommandBufferExecutors(
    CommandExecutor condition_executor, CommandExecutor body_executor,
    bool enable_loop_unroll) {
  command_condition_executor_ = std::move(condition_executor);
  command_body_executor_ = std::move(body_executor);
  enable_loop_unroll_ = enable_loop_unroll;
  is_unrolled_loop_ = false;
  return absl::OkStatus();
}

absl::Status WhileThunk::ExecuteOnStream(const ExecuteParams& params) {
  ScopedWhileLoop loop(profile_annotation(), trip_count_);
  se::Stream& stream = *params.stream;

  int device_ordinal = stream.parent()->device_ordinal();
  if (trip_count_.has_value()) {
    XLA_VLOG_DEVICE(2, device_ordinal)
        << "Executing WhileThunk for " << *trip_count_ << " iterations";
    for (size_t i = 0; i < trip_count_; loop.IncLoopIteration(), ++i) {
      TraceMe trace(
          [&] { return absl::StrFormat("[iter=%d]", loop.loop_iteration()); });
      XLA_VLOG_DEVICE(3, device_ordinal)
          << "Executing iteration # " << i
          << " (Device: " << stream.parent()->device_ordinal() << ")";
      RETURN_IF_ERROR(body_executor_.ExecuteOnStream(params));
    }
    return absl::OkStatus();
  }

  HostMemoryPool* pool;
  {
    absl::MutexLock lock(mutex_);
    pool = host_memory_pools_.at(stream.parent()).get();
  }
  ASSIGN_OR_RETURN(HostMemoryPool::Handle handle, pool->Acquire());
  bool* condition_result = handle.get<bool>();
  se::DeviceAddressBase condition_result_data =
      params.buffer_allocations->GetDeviceAddress(
          condition_result_buffer_index_);

  for (;; loop.IncLoopIteration()) {
    TraceMe trace(
        [&] { return absl::StrFormat("[iter=%d]", loop.loop_iteration()); });

    XLA_VLOG_DEVICE(3, device_ordinal)
        << "Executing WhileThunk condition computation; iter="
        << loop.loop_iteration();
    RETURN_IF_ERROR(condition_executor_.ExecuteOnStream(params));

    // Copy the result of condition computation and break the loop if 'false'.
    RETURN_IF_ERROR(
        stream.Memcpy(condition_result, condition_result_data, sizeof(bool)));

    if (absl::Status blocked = stream.BlockHostUntilDone(); !blocked.ok()) {
      return Internal(
          "Failed to complete all kernels launched on stream %p: %s", &stream,
          blocked.message());
    }

    XLA_VLOG_DEVICE(3, device_ordinal)
        << "condition_result = " << *condition_result;
    if (!*condition_result) {
      XLA_VLOG_DEVICE(3, device_ordinal)
          << "Break WhileThunk loop; iter=" << loop.loop_iteration();
      break;
    }

    XLA_VLOG_DEVICE(3, device_ordinal)
        << "Executing WhileThunk body computation; iter="
        << loop.loop_iteration();
    RETURN_IF_ERROR(body_executor_.ExecuteOnStream(params));
  }
  return absl::OkStatus();
}

absl::Status WhileThunk::WalkNested(Walker callback) {
  RETURN_IF_ERROR(condition_executor_.thunks().WalkNested(callback));
  return body_executor_.thunks().WalkNested(callback);
}

absl::Status WhileThunk::WalkNestedCommands(CommandWalker callback) {
  if (command_condition_executor_.has_value()) {
    RETURN_IF_ERROR(command_condition_executor_->Walk(callback));
  }
  if (command_body_executor_.has_value()) {
    RETURN_IF_ERROR(command_body_executor_->Walk(callback));
  }
  return absl::OkStatus();
}

absl::Status WhileThunk::TransformNested(Transformer callback) {
  RETURN_IF_ERROR(condition_executor_.thunks().TransformNested(callback));
  RETURN_IF_ERROR(body_executor_.thunks().TransformNested(callback));
  return absl::OkStatus();
}

std::string WhileThunk::ToString(int indent) const {
  std::string indent_str(indent * 2, ' ');
  std::string result;
  absl::StrAppend(&result, indent_str, "\ncondition:\n");
  absl::StrAppend(&result, condition_executor_.thunks().ToString(indent + 1));
  absl::StrAppend(&result, indent_str, "body:\n");
  absl::StrAppend(&result, body_executor_.thunks().ToString(indent + 1));
  return result;
}

absl::StatusOr<ThunkProto> WhileThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  auto* while_proto = proto.mutable_while_thunk();
  ASSIGN_OR_RETURN(*while_proto->mutable_condition_result_buffer_index(),
                   condition_result_buffer_index_.ToProto());

  {
    ThunkSequenceProto condition_proto;
    for (const std::unique_ptr<Thunk>& thunk : condition_executor_.thunks()) {
      ASSIGN_OR_RETURN(*condition_proto.add_thunks(), thunk->ToProto());
    }
    *while_proto->mutable_condition_thunk_sequence() =
        std::move(condition_proto);
  }

  {
    ThunkSequenceProto body_proto;
    for (const std::unique_ptr<Thunk>& thunk : body_executor_.thunks()) {
      ASSIGN_OR_RETURN(*body_proto.add_thunks(), thunk->ToProto());
    }
    *while_proto->mutable_body_thunk_sequence() = std::move(body_proto);
  }

  if (trip_count_.has_value()) {
    while_proto->set_trip_count(*trip_count_);
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<WhileThunk>> WhileThunk::FromProto(
    ThunkInfo thunk_info, const WhileThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    const Deserializer& deserializer) {
  ASSIGN_OR_RETURN(
      BufferAllocation::Slice condition_result_buffer_index,
      BufferAllocation::Slice::FromProto(
          thunk_proto.condition_result_buffer_index(), buffer_allocations));
  ThunkSequence condition_thunks;
  for (const auto& proto : thunk_proto.condition_thunk_sequence().thunks()) {
    ASSIGN_OR_RETURN(std::unique_ptr<Thunk> thunk, deserializer(proto));
    condition_thunks.push_back(std::move(thunk));
  }
  ThunkSequence body_thunks;
  for (const auto& proto : thunk_proto.body_thunk_sequence().thunks()) {
    ASSIGN_OR_RETURN(std::unique_ptr<Thunk> thunk, deserializer(proto));
    body_thunks.push_back(std::move(thunk));
  }
  std::optional<int64_t> trip_count;
  if (thunk_proto.has_trip_count()) {
    trip_count = thunk_proto.trip_count();
  }
  return std::make_unique<WhileThunk>(
      std::move(thunk_info), condition_result_buffer_index,
      std::move(condition_thunks), std::move(body_thunks), trip_count);
}

}  // namespace xla::gpu
