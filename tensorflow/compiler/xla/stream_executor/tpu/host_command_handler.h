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

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_TPU_HOST_COMMAND_HANDLER_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_TPU_HOST_COMMAND_HANDLER_H_

#include <functional>

namespace tensorflow::tpu {
// Handler to invoke when a host command is received.
// `program_stack_byte_offset` is the byte address of the current program stack.
// This value is always 0 prior to TPU v4, but may vary per run on TPU v4+.
using HostCommandHandler =
    std::function<void(uint32_t command, int64_t program_stack_byte_offset)>;

}  // namespace tensorflow::tpu

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_TPU_HOST_COMMAND_HANDLER_H_
