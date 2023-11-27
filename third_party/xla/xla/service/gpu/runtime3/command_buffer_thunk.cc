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

#include <utility>

#include "absl/synchronization/mutex.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/runtime3/command_buffer_cmd.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

CommandBufferThunk::ExecutorCommandBuffer::ExecutorCommandBuffer(
    se::CommandBuffer command_buffer)
    : command_buffer(std::move(command_buffer)) {}

CommandBufferThunk::CommandBufferThunk(CommandBufferCmdSequence commands,
                                       ThunkInfo thunk_info)
    : Thunk(Thunk::kCommandBuffer, std::move(thunk_info)),
      commands_(std::move(commands)) {}

Status CommandBufferThunk::Initialize(se::StreamExecutor* executor,
                                      ExecutableSource executable_source) {
  return commands_.Initialize(executor, executable_source);
}

bool CommandBufferThunk::ExecutorCommandBuffer::ShouldUpdateCommandBuffer(
    const CommandBufferCmdSequence& commands,
    const CommandBufferCmd::RecordParams& params) {
  bool should_update = false;
  const BufferAllocations* allocs = params.buffer_allocations;

  // We check only allocations referenced by commands in a cmd sequence, and
  // leave every other entry default initialized (nullptr device memory).
  for (BufferAllocation::Index index : commands.allocs_indices()) {
    se::DeviceMemoryBase alloc = allocs->GetDeviceAddress(index);

    if (recorded_allocs.size() <= index) {
      recorded_allocs.resize(index + 1);
    }

    if (!recorded_allocs[index].IsSameAs(alloc)) {
      recorded_allocs[index] = alloc;
      should_update = true;
    }
  }

  return should_update;
}

Status CommandBufferThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(ExecutorCommandBuffer * cmd_buffer,
                      GetOrCreateCommandBuffer(executor));

  CommandBufferCmd::RecordParams record_params = {executor,
                                                  params.buffer_allocations};

  absl::MutexLock lock(&cmd_buffer->mutex);

  if (cmd_buffer->ShouldUpdateCommandBuffer(commands_, record_params)) {
    TF_RETURN_IF_ERROR(
        commands_.Record(record_params, &cmd_buffer->command_buffer));
  }

  return executor->Submit(params.stream, cmd_buffer->command_buffer);
}

StatusOr<CommandBufferThunk::ExecutorCommandBuffer*>
CommandBufferThunk::GetOrCreateCommandBuffer(se::StreamExecutor* executor) {
  absl::MutexLock lock(&mutex_);

  // Check if command buffer already exists
  if (auto it = command_buffers_.find(executor); it != command_buffers_.end()) {
    return &it->second;
  }

  // Create a new empty command buffer.
  TF_ASSIGN_OR_RETURN(auto command_buffer, se::CommandBuffer::Create(executor));
  auto emplaced = command_buffers_.emplace(executor, std::move(command_buffer));

  return &emplaced.first->second;
}

}  // namespace xla::gpu
