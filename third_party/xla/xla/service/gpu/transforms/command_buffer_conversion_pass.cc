/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/command_buffer_conversion_pass.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/overload.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/platform.h"

namespace xla {
namespace gpu {

using CommandBufferConfig = CommandBufferConversionPass::CommandBufferConfig;

namespace {

CommandBufferConfig GetCommandBufferConfig(
    const DebugOptions& debug_options,
    const se::DeviceDescription& device_info) {
  absl::flat_hash_set<DebugOptions::CommandBufferCmdType> commands;
  for (auto cmd_type : debug_options.xla_gpu_enable_command_buffer()) {
    commands.insert(static_cast<DebugOptions::CommandBufferCmdType>(cmd_type));
  }

  absl::flat_hash_set<std::string> legacy_custom_call_targets;
  for (const auto& target :
       debug_options.legacy_command_buffer_custom_call_targets()) {
    legacy_custom_call_targets.insert(target);
  }

  CommandBufferConfig config{
      std::move(commands), std::move(legacy_custom_call_targets), device_info};

  // Erase command buffer cmd types that are not supported by the gpu runtime.
  static constexpr auto kRequireConditionals = {DebugOptions::CONDITIONAL,
                                                DebugOptions::WHILE};
  static constexpr auto kRequireTracing = {
      DebugOptions::CUBLAS, DebugOptions::CUBLASLT, DebugOptions::CUDNN,
      DebugOptions::CUSTOM_CALL, DebugOptions::COLLECTIVES};

  auto erase = [&](absl::Span<const DebugOptions::CommandBufferCmdType> cmds) {
    for (auto cmd : cmds) {
      if (config.enabled_commands.erase(cmd)) {
        VLOG(1) << "Removed command buffer support for "
                << DebugOptions::CommandBufferCmdType_Name(cmd)
                << " as it's not supported with gpu toolkit version "
                << device_info.runtime_version() << " and driver version "
                << device_info.driver_version()
                << ". This might negatively impact peformance. To enable "
                << DebugOptions::CommandBufferCmdType_Name(cmd)
                << " support in command buffers use cuda-compat package: "
                << (tsl::kIsOpenSource
                        ? "https://docs.nvidia.com/deploy/cuda-compatibility/."
                        : "set CUDA_COMPAT_LOAD=1 env variable.");
      }
    }
  };

  // Check if CUDA/ROCM driver supports required features.
  auto erase_cuda = [&](const se::CudaComputeCapability& cuda_comp) {
    if (std::min(device_info.runtime_version(), device_info.driver_version()) <
        se::SemanticVersion{12, 3, 0}) {
      erase(kRequireTracing);       // cuStreamBeginCaptureToGraph
      erase(kRequireConditionals);  // on-device control flow
    }
  };
  auto erase_rocm = [&](const se::RocmComputeCapability& rocm_comp) {
    erase(kRequireConditionals);  // on-device control flow
  };

  std::visit(absl::Overload(erase_cuda, erase_rocm),
             device_info.gpu_compute_capability());

  return config;
}

// TODO(aliia): Add support for other command buffer types.
std::optional<DebugOptions::CommandBufferCmdType> GetCommandBufferCmdType(
    Thunk::Kind kind) {
  switch (kind) {
    case Thunk::kCopy:
    case Thunk::kKernel:
      return DebugOptions::FUSION;
    case Thunk::kGemm:
      return DebugOptions::CUBLAS;
    case Thunk::kAllGatherStart:
    case Thunk::kAllReduceStart:
    case Thunk::kReduceScatterStart:
    case Thunk::kAllToAllStart:
    case Thunk::kCollectiveBroadcastStart:
    case Thunk::kCollectivePermuteStart:
    case Thunk::kRaggedAllToAllStart:
    case Thunk::kRecv:
    case Thunk::kSend:
      return DebugOptions::COLLECTIVES;
    default:
      return std::nullopt;
  }
}

bool IsConvertible(Thunk* thunk, const CommandBufferConfig& config) {
  if (thunk->IsAsyncDone()) {
    return true;
  }
  auto cmd_type = GetCommandBufferCmdType(thunk->kind());
  if (!cmd_type.has_value()) {
    return false;  // Thunk kind is not supported for command buffer conversion.
  }
  return config.enabled_commands.contains(*cmd_type);
}

// Collect the sequence of thunks that contains the async start and its
// corresponding done thunk. If there is another start thunk
// between the original start and done, we may potentially extend the sequence
// to include its corresponding done thunk. For example, if we call this
// function on async-start_a in the following sequence:
//
// async_start_a
// async_start_b
// async_done_a
// async_done_b
//
// The returned sequence will contain async_done_b. So that all async pairs
// are captured by the same command buffer.

// Returns the shortest non-empty sequence of thunks that form a valid async
// region as a span. If no such region is found, an empty span is returned.
absl::Span<std::unique_ptr<Thunk>> CollectAndCheckAsyncRegion(
    absl::Span<std::unique_ptr<Thunk>> thunks,
    const CommandBufferConfig& config) {
  absl::flat_hash_set<AsyncEventsUniqueId> unpaired_ids_of_async_starts;

  for (size_t i = 0; i < thunks.size(); ++i) {
    auto& thunk = thunks[i];

    // Check if thunk is convertible
    if (!IsConvertible(thunk.get(), config)) {
      return {};  // All thunks in the region must be convertible.
    }

    // Check if it is async start thunk
    if (thunk->IsAsyncStart() && thunk->GetAsyncEventsUniqueId().has_value()) {
      unpaired_ids_of_async_starts.insert(
          thunk->GetAsyncEventsUniqueId().value());
    }

    // Check if it is async done thunk
    if (thunk->IsAsyncDone() && thunk->GetAsyncEventsUniqueId().has_value()) {
      auto it = unpaired_ids_of_async_starts.find(
          thunk->GetAsyncEventsUniqueId().value());
      if (it == unpaired_ids_of_async_starts.end()) {
        return {};  // We found an async end for an event, whose async
                    // start is not part of the region
      }
      unpaired_ids_of_async_starts.erase(it);
    }

    if (unpaired_ids_of_async_starts.empty()) {
      return thunks.subspan(
          0, i + 1);  // We found pairs to all open async events and all thunks
                      // in between are convertible
    }
  }
  return {};  // error didn't find an end for some start
}

}  // namespace

absl::StatusOr<bool> CommandBufferConversionPass::Run(
    SequentialThunk* root_thunk_ptr, const DebugOptions& debug_options,
    const se::DeviceDescription& device_info) {
  CommandBufferConfig config =
      GetCommandBufferConfig(debug_options, device_info);

  CommandBufferCmdExecutor::SynchronizationMode synchronization_mode =
      debug_options.xla_gpu_graph_enable_concurrent_region()
          ? CommandBufferCmdExecutor::SynchronizationMode::kAutomatic
          : CommandBufferCmdExecutor::SynchronizationMode::kSerialize;

  bool changed = false;

  auto convert_thunks_to_command_buffer =
      [&](std::vector<std::unique_ptr<Thunk>>& thunks_to_convert)
      -> absl::StatusOr<std::unique_ptr<CommandBufferThunk>> {
    TF_ASSIGN_OR_RETURN(
        CommandBufferCmdExecutor cmd_executor,
        ConvertToCommands(thunks_to_convert,
                          ConvertToCommandsOptions{synchronization_mode}));

    return std::make_unique<CommandBufferThunk>(
        std::move(cmd_executor), Thunk::ThunkInfo(),
        std::make_unique<SequentialThunk>(Thunk::ThunkInfo(),
                                          std::move(thunks_to_convert)),
        debug_options.xla_enable_command_buffers_during_profiling());
  };

  std::vector<std::unique_ptr<Thunk>> current_command_buffer_thunks;

  std::vector<std::unique_ptr<Thunk>> new_thunks;

  auto flush_command_buffer = [&]() -> absl::Status {
    // If we don't have enough thunks to form a command buffer, we just add
    // them to the new thunks sequence as is.
    if (current_command_buffer_thunks.size() <
        std::max(1, debug_options.xla_gpu_graph_min_graph_size())) {
      new_thunks.insert(
          new_thunks.end(),
          std::make_move_iterator(current_command_buffer_thunks.begin()),
          std::make_move_iterator(current_command_buffer_thunks.end()));
      current_command_buffer_thunks.clear();
      return absl::OkStatus();
    }

    TF_ASSIGN_OR_RETURN(
        auto cmd_buffer_thunk,
        convert_thunks_to_command_buffer(current_command_buffer_thunks));
    // Check that the command buffer thunk is not empty
    assert(cmd_buffer_thunk->thunks() != nullptr &&
           !cmd_buffer_thunk->thunks()->thunks().empty());
    new_thunks.push_back(std::move(cmd_buffer_thunk));
    changed = true;
    current_command_buffer_thunks.clear();
    return absl::OkStatus();
  };

  // TODO(aliia): use post order here
  auto& original_thunks = root_thunk_ptr->thunks();

  for (size_t i = 0; i < original_thunks.size(); ++i) {
    auto& thunk = original_thunks[i];

    // We always have to capture both corresponding start and done events in the
    // same command buffer.
    if (thunk->IsAsyncStart()) {
      // Collect and check async region
      absl::Span<std::unique_ptr<Thunk>> region = CollectAndCheckAsyncRegion(
          absl::MakeSpan(original_thunks).subspan(i), config);

      if (!region.empty()) {
        // If a valid region is found, add the whole region to the current
        // sequence and continue processing.
        current_command_buffer_thunks.insert(
            current_command_buffer_thunks.end(),
            std::make_move_iterator(region.begin()),
            std::make_move_iterator(region.end()));
        i += region.size() - 1;
        continue;
      }
    } else if (IsConvertible(thunk.get(), config) && !thunk->IsAsyncDone()) {
      // Check if thunk is convertible and not an async done: async done thunks
      // can be only added to the current_command_buffer_thunks as part of a
      // valid async regions.

      current_command_buffer_thunks.push_back(std::move(thunk));
      continue;
    }

    // If the current thunk is not convertible, flush collected eligible thunk
    // to a command buffer thunk and add it to the processed sequence. Then add
    // non-convertible thunk to the sequence.
    TF_RETURN_IF_ERROR(flush_command_buffer());
    new_thunks.push_back(std::move(thunk));
  }

  TF_RETURN_IF_ERROR(flush_command_buffer());

  root_thunk_ptr->thunks() = std::move(new_thunks);

  return changed;
}

}  // namespace gpu
}  // namespace xla
