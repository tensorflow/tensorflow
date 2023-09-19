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

#ifndef XLA_STREAM_EXECUTOR_COMMAND_BUFFER_H_
#define XLA_STREAM_EXECUTOR_COMMAND_BUFFER_H_

#include <memory>

#include "xla/stream_executor/platform/port.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {

class StreamExecutor;

namespace internal {
class CommandBufferInterface;
}

// Command buffer represent a "bundle of work items" for StreamExecutor device
// that can be submitted with one API call, e.g. command buffer might have
// multiple device kernels and synchronization barriers between them. Command
// buffers allow to amortize the cost of launching "work" on device by building
// it on the host ahead of time without expensive interaction with underlying
// device.
class CommandBuffer {
 public:
  explicit CommandBuffer(
      std::unique_ptr<internal::CommandBufferInterface> implementation);

  static tsl::StatusOr<CommandBuffer> Create(StreamExecutor* executor);

 private:
  std::unique_ptr<internal::CommandBufferInterface> implementation_;

  SE_DISALLOW_COPY_AND_ASSIGN(CommandBuffer);
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_COMMAND_BUFFER_H_
