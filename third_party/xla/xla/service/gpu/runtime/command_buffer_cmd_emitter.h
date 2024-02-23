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

#ifndef XLA_SERVICE_GPU_RUNTIME_COMMAND_BUFFER_CMD_EMITTER_H_
#define XLA_SERVICE_GPU_RUNTIME_COMMAND_BUFFER_CMD_EMITTER_H_

#include "absl/status/statusor.h"
#include "xla/service/gpu/runtime/command_buffer_cmd.h"
#include "xla/service/gpu/thunk.h"

namespace xla::gpu {

// Converts thunk sequence to a command buffer cmd sequence. If `force_barrier`
// is true we automatically insert barriers between all commands in a sequence.
// Otherwise we use buffer usage aliasing to allow commands to run concurrently
// and insert barriers only when needed for correctness.
absl::StatusOr<CommandBufferCmdSequence> ConvertToCommands(
    const ThunkSequence& sequence,
    CommandBufferCmdSequence::SynchronizationMode synchronization_mode);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_COMMAND_BUFFER_CMD_EMITTER_H_
