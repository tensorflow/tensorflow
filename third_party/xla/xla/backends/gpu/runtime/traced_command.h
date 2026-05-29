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

#ifndef XLA_BACKENDS_GPU_RUNTIME_TRACED_COMMAND_H_
#define XLA_BACKENDS_GPU_RUNTIME_TRACED_COMMAND_H_

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// TracedCommand
//===----------------------------------------------------------------------===//

// A base class for commands implemented as tracing of stream activities.
// Subclasses may override Record() for custom behavior; the default
// implementation traces ExecuteOnStream() on the command_buffer_trace_stream.
class TracedCommand : public Command {
 public:
  bool IsTracedCommand() const override { return true; }

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

 protected:
  explicit TracedCommand(CommandType cmd_type);

  // Constructor for Thunk subclasses that are also TracedCommands.
  // Preserves the caller's Thunk::Kind and ThunkInfo.
  TracedCommand(CommandType cmd_type, Thunk::Kind thunk_kind,
                ThunkInfo thunk_info);

  // Creates a command buffer by calling a user-provided `trace` function and
  // adds it as a nested command to `command_buffer`. Traced command buffers
  // cached and reused in an instance of `TracedCommandBuffer` kept in `state`.
  absl::StatusOr<const se::CommandBuffer::Command*> RecordTracedCommand(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer,
      absl::FunctionRef<absl::Status(se::Stream*)> trace);
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_TRACED_COMMAND_H_
