/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/runtime3/command_buffer_thunk.h"

#include <memory>
#include <utility>

#include "absl/synchronization/mutex.h"
#include "xla/service/gpu/runtime3/command_buffer_cmd.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/stream_executor_pimpl.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

CommandBufferThunk::State::State(se::CommandBuffer command_buffer)
    : command_buffer(std::move(command_buffer)) {}

CommandBufferThunk::CommandBufferThunk(CommandBufferCmdSequence commands,
                                       ThunkInfo thunk_info)
    : Thunk(Thunk::kCommandBuffer, std::move(thunk_info)),
      commands_(std::move(commands)) {}

Status CommandBufferThunk::Initialize(se::StreamExecutor* executor,
                                      ExecutableSource executable_source) {
  return commands_.Initialize(executor, executable_source);
}

Status CommandBufferThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(State * state, GetOrCreateCommandBuffer(executor));

  absl::MutexLock lock(&state->mutex);

  CommandBufferCmd::RecordParams record_params = {params.buffer_allocations};
  TF_RETURN_IF_ERROR(commands_.Record(record_params, &state->command_buffer));

  return executor->Submit(params.stream, state->command_buffer);
}

StatusOr<CommandBufferThunk::State*>
CommandBufferThunk::GetOrCreateCommandBuffer(se::StreamExecutor* executor) {
  absl::MutexLock lock(&mutex_);

  // Check if command buffer already exists
  if (auto it = command_buffers_.find(executor); it != command_buffers_.end()) {
    return it->second.get();
  }

  // Create a new empty command buffer.
  TF_ASSIGN_OR_RETURN(auto command_buffer, se::CommandBuffer::Create(executor));
  auto emplaced = command_buffers_.emplace(
      executor, std::make_unique<State>(std::move(command_buffer)));

  return emplaced.first->second.get();
}

}  // namespace xla::gpu
