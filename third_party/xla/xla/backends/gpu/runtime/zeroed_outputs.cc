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

#include "xla/backends/gpu/runtime/zeroed_outputs.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

// The memsets recorded for a command, keyed by the command node the caller
// recorded for itself.
//
// Keyed by node rather than kept as a single vector because one command can be
// recorded into the same command buffer more than once under different record
// ids, which is what an unrolled while loop does. All of those recordings share
// a single state object, because the state manager keys only by command and
// command buffer. `CustomCallThunk` and `RaggedAllToAllThunk` key their
// recorded nodes the same way.
struct ZeroedOutputsState : public CommandState {
  absl::flat_hash_map<const se::CommandBuffer::Command*,
                      std::vector<const se::CommandBuffer::Command*>>
      memsets;
};

}  // namespace

absl::StatusOr<ZeroedOutputs> RecordZeroedOutputs(
    const Command* command, const Command::RecordParams& record_params,
    const Command::RecordAction& record_action,
    se::CommandBuffer* command_buffer,
    absl::Span<const se::DeviceAddressBase> buffers) {
  ZeroedOutputs zeroed;

  if (const auto* create = std::get_if<Command::RecordCreate>(&record_action)) {
    zeroed.memsets.reserve(buffers.size());
    for (se::DeviceAddressBase buffer : buffers) {
      if (buffer.size() == 0) {
        continue;
      }
      ABSL_ASSIGN_OR_RETURN(
          const se::CommandBuffer::Command* memset,
          command_buffer->CreateMemset(&buffer, uint8_t{0},
                                       /*num_elements=*/buffer.size(),
                                       create->dependencies));
      zeroed.memsets.push_back(memset);
    }
    return zeroed;
  }

  if (const auto* update = std::get_if<Command::RecordUpdate>(&record_action)) {
    size_t num_to_zero = 0;
    for (const se::DeviceAddressBase& buffer : buffers) {
      num_to_zero += buffer.size() != 0;
    }
    if (num_to_zero == 0) {
      return zeroed;
    }

    auto* state = record_params.state.GetOrNull<ZeroedOutputsState>(
        command, command_buffer);
    TF_RET_CHECK(state != nullptr)
        << "No zeroing memsets were recorded into this command buffer";

    auto it = state->memsets.find(update->command);
    TF_RET_CHECK(it != state->memsets.end())
        << "No zeroing memsets were recorded alongside command node "
        << update->command;

    const std::vector<const se::CommandBuffer::Command*>& memsets = it->second;
    TF_RET_CHECK(memsets.size() == num_to_zero)
        << "Recorded " << memsets.size() << " zeroing memsets but have "
        << num_to_zero << " buffers to zero; the set of non-empty buffers must "
        << "not change between create and update";

    size_t i = 0;
    for (se::DeviceAddressBase buffer : buffers) {
      if (buffer.size() == 0) {
        continue;
      }
      ABSL_RETURN_IF_ERROR(command_buffer->UpdateMemset(
          memsets[i++], &buffer, uint8_t{0}, /*num_elements=*/buffer.size()));
    }
    return zeroed;
  }

  return Internal("Invalid record action");
}

absl::Status AttachZeroedOutputs(const Command* command,
                                 const Command::RecordParams& record_params,
                                 se::CommandBuffer* command_buffer,
                                 const se::CommandBuffer::Command* recorded,
                                 ZeroedOutputs zeroed) {
  if (zeroed.memsets.empty()) {
    return absl::OkStatus();
  }

  TF_RET_CHECK(recorded != nullptr)
      << "Zeroing memsets were recorded but the command that depends on them "
         "was not, so they could never be updated";

  auto* state = record_params.state.GetOrCreate<ZeroedOutputsState>(
      command, command_buffer);
  state->memsets[recorded] = std::move(zeroed.memsets);
  return absl::OkStatus();
}

}  // namespace xla::gpu
