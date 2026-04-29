/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/traced_command.h"

#include <memory>
#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/traced_command_buffer.h"
#include "xla/debug_options_flags.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// TracedCommand
//===----------------------------------------------------------------------===//

TracedCommand::TracedCommand(CommandType cmd_type) : Command(cmd_type) {}

TracedCommand::TracedCommand(CommandType cmd_type, Thunk::Kind thunk_kind,
                             ThunkInfo thunk_info)
    : Command(cmd_type, thunk_kind, std::move(thunk_info)) {}

absl::StatusOr<const se::CommandBuffer::Command*> TracedCommand::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  return RecordTracedCommand(execute_params, record_params,
                             std::move(record_action), command_buffer,
                             [&](se::Stream* stream) {
                               ExecuteParams trace_params = execute_params;
                               trace_params.stream = stream;
                               return ExecuteOnStream(trace_params);
                             });
}

absl::StatusOr<const se::CommandBuffer::Command*>
TracedCommand::RecordTracedCommand(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer,
    absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  auto traced_cmd = record_params.state.GetOrCreate<TracedCommandBuffer>(
      this, command_buffer, [&] {
        const auto& debug_options = xla::GetDebugOptionsFromFlags();
        return std::make_unique<TracedCommandBuffer>(
            this, buffer_uses(),
            debug_options.xla_cmd_buffer_trace_cache_size());
      });

  TF_ASSIGN_OR_RETURN(
      auto nested_cmd,
      traced_cmd->GetOrTraceCommandBuffer(
          execute_params.buffer_allocations, execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, trace, priority()));

  VLOG(5) << "Record traced command into command buffer: " << command_buffer;

  if (auto* create = std::get_if<RecordCreate>(&record_action)) {
    return command_buffer->CreateChildCommand(*nested_cmd,
                                              create->dependencies);
  }

  if (auto* update = std::get_if<RecordUpdate>(&record_action)) {
    TF_RETURN_IF_ERROR(
        command_buffer->UpdateChildCommand(update->command, *nested_cmd));
    return update->command;
  }

  return Internal("Invalid record action");
}

}  // namespace xla::gpu
