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
#ifndef XLA_STREAM_EXECUTOR_TRACE_COMMAND_BUFFER_FACTORY_H_
#define XLA_STREAM_EXECUTOR_TRACE_COMMAND_BUFFER_FACTORY_H_

#include <memory>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_interface.h"

namespace stream_executor {

class TraceCommandBufferFactory {
 public:
  // Creates a new command buffer on the given executor by tracing `function`
  // invocation. All StreamExecutor operations on a Stream argument will be
  // recorded into the command buffer. Returned command buffer is finalized, and
  // can't be updated.
  //
  // Command buffer tracing should be used only when it is impossible to use
  // explicit construction APIs, e.g. when calling external libraries. By
  // default we construct traced command buffers in nested mode because the
  // primary use case for traced command buffers is to be inserted into primary
  // command buffers constructed with explicit APIs.
  static absl::StatusOr<std::unique_ptr<CommandBuffer>> Create(
      StreamExecutorInterface* executor,
      absl::AnyInvocable<absl::Status(Stream*)> function,
      CommandBuffer::Mode mode = CommandBuffer::Mode::kNested);

  // Creates a new command buffer on the given executor by tracing `function`
  // invocation using a user provided stream that will be passed to `function`.
  static absl::StatusOr<std::unique_ptr<CommandBuffer>> Create(
      StreamExecutorInterface* executor, Stream* stream,
      absl::AnyInvocable<absl::Status(Stream*)> function,
      CommandBuffer::Mode mode = CommandBuffer::Mode::kNested);
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_TRACE_COMMAND_BUFFER_FACTORY_H_
