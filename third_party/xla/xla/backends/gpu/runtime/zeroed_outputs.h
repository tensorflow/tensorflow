/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_ZEROED_OUTPUTS_H_
#define XLA_BACKENDS_GPU_RUNTIME_ZEROED_OUTPUTS_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"

namespace xla::gpu {

// Zeroing of the buffers a command must clear before it runs.
//
// Thunks that zero some of their outputs first (Triton/Pallas kernels declaring
// `zeroed_outputs`, cuDNN fMHA) do it with `stream->MemZero` when they execute
// eagerly. Inside a command buffer the equivalent is a memset command that the
// thunk's own command depends on, created once and then pointed at the new
// destinations whenever buffer addresses change.
//
// The memsets take the dependencies the caller was given and the caller's own
// command takes the memsets, so anything downstream still transitively depends
// on whatever the caller was told to wait for.
//
// Typical use from `Command::Record`:
//
//   ABSL_ASSIGN_OR_RETURN(
//       ZeroedOutputs zeroed,
//       RecordZeroedOutputs(this, record_params, record_action,
//                           command_buffer, buffers));
//
//   if (auto* create = std::get_if<RecordCreate>(&record_action)) {
//     ABSL_ASSIGN_OR_RETURN(
//         const se::CommandBuffer::Command* command,
//         command_buffer->CreateLaunch(
//             ..., zeroed.dependencies(create->dependencies), priority()));
//     ABSL_RETURN_IF_ERROR(AttachZeroedOutputs(
//         this, record_params, command_buffer, command, std::move(zeroed)));
//     return command;
//   }
//
// The update branch needs no extra code: by the time `RecordZeroedOutputs`
// returns it has already pointed the memsets at the new addresses.
struct ZeroedOutputs {
  // Memset commands created by `RecordZeroedOutputs`. Populated only on
  // `RecordCreate`, and empty when there is nothing to zero.
  std::vector<const se::CommandBuffer::Command*> memsets;

  // Dependencies for the command the caller is about to create: the memsets, so
  // that it runs after the zeroing, or `fallback` when there is nothing to
  // zero.
  absl::Span<const se::CommandBuffer::Command* const> dependencies(
      absl::Span<const se::CommandBuffer::Command* const> fallback) const {
    return memsets.empty() ? fallback : absl::MakeConstSpan(memsets);
  }
};

// Creates (on `RecordCreate`) or updates (on `RecordUpdate`) the memset
// commands that zero `buffers`.
//
// Buffers of zero bytes are skipped, because a zero-length memset is not a
// useful command to record. Which buffers those are is fixed by buffer
// assignment rather than by the addresses of the day, so the set of memsets is
// the same on create and on every later update.
absl::StatusOr<ZeroedOutputs> RecordZeroedOutputs(
    const Command* command, const Command::RecordParams& record_params,
    const Command::RecordAction& record_action,
    se::CommandBuffer* command_buffer,
    absl::Span<const se::DeviceAddressBase> buffers);

// Associates the memsets in `zeroed` with `recorded`, the command the caller
// created for itself, so that the next `RecordUpdate` can find them again.
//
// Call it after a `RecordCreate`. It is a no-op when there was nothing to zero.
absl::Status AttachZeroedOutputs(const Command* command,
                                 const Command::RecordParams& record_params,
                                 se::CommandBuffer* command_buffer,
                                 const se::CommandBuffer::Command* recorded,
                                 ZeroedOutputs zeroed);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_ZEROED_OUTPUTS_H_
