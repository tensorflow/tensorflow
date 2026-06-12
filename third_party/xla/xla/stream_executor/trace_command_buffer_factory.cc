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

#include "xla/stream_executor/trace_command_buffer_factory.h"

#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {

absl::StatusOr<std::unique_ptr<CommandBuffer>>
TraceCommandBufferFactory::Create(
    StreamExecutor* executor,
    absl::AnyInvocable<absl::Status(Stream*)> function,
    CommandBuffer::Mode mode) {
  ASSIGN_OR_RETURN(auto stream, executor->CreateStream());
  stream->SetName("Command buffer tracer");
  return TraceCommandBufferFactory::Create(executor, stream.get(),
                                           std::move(function), mode);
}

absl::StatusOr<std::unique_ptr<CommandBuffer>>
TraceCommandBufferFactory::Create(
    StreamExecutor* executor, Stream* stream,
    absl::AnyInvocable<absl::Status(Stream*)> function,
    CommandBuffer::Mode mode) {
  if (stream == nullptr) {
    LOG(ERROR) << "Can't trace command buffer on a null stream";
    return absl::InternalError("Can't trace command buffer on a null stream");
  }
  // Prepare an empty command buffer instance.
  auto command_buffer_or = executor->CreateCommandBuffer(mode);
  if (!command_buffer_or.ok()) {
    LOG(ERROR) << "Failed to create command buffer: "
               << command_buffer_or.status();
    return command_buffer_or.status();
  }
  ASSIGN_OR_RETURN(std::unique_ptr<CommandBuffer> command_buffer,
                   std::move(command_buffer_or));

  // Trace and finalize the command buffer.
  auto trace_status = command_buffer->Trace(
      stream, [&](Stream* capture_stream) { return function(capture_stream); });
  if (!trace_status.ok()) {
    LOG(ERROR) << "Failed to trace command buffer: " << trace_status;
    return trace_status;
  }
  auto finalize_status = command_buffer->Finalize();
  if (!finalize_status.ok()) {
    LOG(ERROR) << "Failed to finalize command buffer: " << finalize_status;
    return finalize_status;
  }

  return command_buffer;
}

}  // namespace stream_executor
