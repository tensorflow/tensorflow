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

#include "xla/stream_executor/command_buffer.h"

#include <memory>
#include <utility>

#include "xla/stream_executor/stream_executor_internal.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {

/*static*/ tsl::StatusOr<CommandBuffer> CommandBuffer::Create(
    StreamExecutor* executor) {
  // TODO(ezhulenev): Construct command buffer from platform-specific command
  // buffer implementation. It requires cleaning up build files first.
  std::unique_ptr<internal::CommandBufferInterface> command_buffer = nullptr;
  return tsl::StatusOr<CommandBuffer>(std::move(command_buffer));
}

CommandBuffer::CommandBuffer(
    std::unique_ptr<internal::CommandBufferInterface> implementation)
    : implementation_(std::move(implementation)) {}

}  // namespace stream_executor
