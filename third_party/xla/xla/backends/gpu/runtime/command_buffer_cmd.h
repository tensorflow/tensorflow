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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_CMD_H_
#define XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_CMD_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/command_buffer.h"

namespace xla::gpu {

class WhileCmd : public Command {
 public:
  WhileCmd(BufferAllocation::Slice pred, CommandExecutor cond_commands,
           CommandExecutor body_commands,
           std::optional<int64_t> trip_count = std::nullopt,
           bool enable_loop_unroll = false);

  absl::Status Initialize(const Thunk::InitializeParams& params) override;

  absl::Status Prepare(const Thunk::PrepareParams& params) override;

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

  absl::Status WalkNested(
      absl::FunctionRef<absl::Status(Thunk*)> callback) override;

 private:
  absl::Status WalkNestedCommands(CommandWalker callback) override;

  BufferAllocation::Slice pred_;

  CommandExecutor cond_commands_;
  CommandExecutor body_commands_;

  std::optional<int64_t> trip_count_;
  bool enable_loop_unroll_ = false;
  bool is_unrolled_loop_ = false;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_CMD_H_
