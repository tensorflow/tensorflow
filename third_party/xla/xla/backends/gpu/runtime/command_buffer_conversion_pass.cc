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

#include "xla/backends/gpu/runtime/command_buffer_conversion_pass.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_memcpy_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_slice_thunk.h"
#include "xla/backends/gpu/runtime/ragged_all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/platform.h"
#include "tsl/profiler/lib/profiler_lock.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {
namespace {

using CommandBufferConfig = CommandBufferConversionPass::CommandBufferConfig;

CommandBufferConfig GetCommandBufferConfig(
    const DebugOptions& debug_options, const se::DeviceDescription& device_info,
    const HloModule* hlo_module) {
  absl::flat_hash_set<DebugOptions::CommandBufferCmdType> commands;
  for (auto cmd_type : debug_options.xla_gpu_enable_command_buffer()) {
    commands.insert(static_cast<DebugOptions::CommandBufferCmdType>(cmd_type));
  }

  // Extract slice_size (partition_size) from HloModule config if available.
  int64_t num_local_devices = 0;
  if (hlo_module != nullptr) {
    num_local_devices = hlo_module->config().partition_size();
  }

  CommandBufferConfig config{std::move(commands), device_info,
                             num_local_devices};

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
                << ". This might negatively impact performance. To enable "
                << DebugOptions::CommandBufferCmdType_Name(cmd)
                << " support in command buffers use cuda-compat package: "
                << (tsl::kIsOpenSource
                        ? "https://docs.nvidia.com/deploy/cuda-compatibility/."
                        : "set CUDA_COMPAT_LOAD=1 env variable.");
      }
    }
  };

  // Check if CUDA/ROCM driver supports required features.
  if (device_info.gpu_compute_capability().IsCuda()) {
    if (std::min(device_info.runtime_version(), device_info.driver_version()) <
        se::SemanticVersion{12, 3, 0}) {
      erase(kRequireTracing);       // cuStreamBeginCaptureToGraph
      erase(kRequireConditionals);  // on-device control flow
    }
  }
  if (device_info.gpu_compute_capability().IsRocm()) {
    erase(kRequireConditionals);  // on-device control flow
  }

  return config;
}

// Maps Thunk::Kind to DebugOptions::CommandBufferCmdType for checking
// command buffer eligibility against the `--xla_gpu_enable_command_buffer`
// flag. Returns std::nullopt if a thunk is not supported.
std::optional<DebugOptions::CommandBufferCmdType> GetCommandBufferCmdType(
    const Thunk& thunk) {
  auto kind = thunk.kind();
  switch (kind) {
    case Thunk::kCopy:
      if (dynamic_cast<const DynamicMemcpyThunk*>(&thunk)) {
        return DebugOptions::DYNAMIC_SLICE_COPY_FUSION;
      } else if (dynamic_cast<const DeviceToDeviceCopyThunk*>(&thunk)) {
        return DebugOptions::FUSION;
      } else {
        // Only copy within the same device can be converted to command buffers.
        VLOG(2) << "Unsupported thunk kind: " << Thunk::KindToString(kind);
        return std::nullopt;
      }
    case Thunk::kCustomKernel:
    case Thunk::kKernel:
    case Thunk::kPartitionId:
    case Thunk::kReplicaId:
      return DebugOptions::FUSION;
    case Thunk::kWhile:
      return DebugOptions::WHILE;
    case Thunk::kConditional:
      return DebugOptions::CONDITIONAL;
    case Thunk::kGemm:
      return DebugOptions::CUBLAS;
    case Thunk::kAllGather:
    case Thunk::kAllReduce:
    case Thunk::kAllToAll:
    case Thunk::kCollectiveBroadcast:
    case Thunk::kCollectivePermute:
    case Thunk::kRaggedAllToAll:
    case Thunk::kReduceScatter:
    case Thunk::kRecv:
    case Thunk::kSend:
      return DebugOptions::COLLECTIVES;
    case Thunk::kCuDnn:
      return DebugOptions::CUDNN;
    case Thunk::kCustomCall:
      return DebugOptions::CUSTOM_CALL;
    case Thunk::kCublasLtMatmul:
      return DebugOptions::CUBLASLT;
    case Thunk::kDynamicSlice:
      return DebugOptions::DYNAMIC_SLICE_FUSION;
    default:
      VLOG(2) << "Unsupported thunk kind: " << Thunk::KindToString(kind);
      return std::nullopt;
  }
}

bool ThunkSequenceIsConvertible(const ThunkSequence& thunks,
                                const CommandBufferConfig& config);
size_t CheckAsyncRegion(absl::Span<const std::unique_ptr<Thunk>> thunks,
                        const CommandBufferConfig& config);

// Returns true if the WhileThunk is convertible to a command buffer operation.
// This requires that all thunks in both the condition and body sequences are
// convertible.
bool IsConvertible(const WhileThunk& while_thunk,
                   const CommandBufferConfig& config) {
  return ThunkSequenceIsConvertible(while_thunk.body_executor().thunks(),
                                    config) &&
         ThunkSequenceIsConvertible(while_thunk.condition_executor().thunks(),
                                    config);
}

// Returns true if the ConditionalThunk is convertible to a command buffer
// operation. This requires that all thunks in all branch sequences are
// convertible.
bool IsConvertible(const ConditionalThunk& conditional_thunk,
                   const CommandBufferConfig& config) {
  return absl::c_all_of(
      conditional_thunk.branch_executors(), [&](const auto& branch) {
        return ThunkSequenceIsConvertible(branch.thunks(), config);
      });
}

// Returns true if the CustomCallThunk is convertible to a command buffer
// operation. Checks if the registered FFI handler is compatible with command
// buffers.
bool IsConvertible(const CustomCallThunk& custom_call_thunk,
                   const CommandBufferConfig& config) {
  const std::string& target_name = custom_call_thunk.target_name();

  // Check if FFI handler is compatible with command buffers.
  absl::StatusOr<ffi::HandlerRegistration> registration =
      ffi::FindHandler(target_name, "gpu");
  return registration.ok()
             ? ffi::IsCommandBufferCompatible(registration->metadata)
             : false;
}

// Returns true if the RaggedAllToAllThunk is convertible to a command
// buffer operation.
// This requires the one-shot barrier kernel to be explicitly enabled.
bool IsConvertible(const RaggedAllToAllThunk& ra2a_thunk,
                   const CommandBufferConfig& config) {
  // 1. Check flags
  bool flags_enabled = ra2a_thunk.is_one_shot_kernel_enabled();
  if (!flags_enabled) {
    return false;
  }

  // 2. Check kernel support (types, limits)
  if (!ra2a_thunk.IsOneShotKernelSupported()) {
    return false;
  }

  // 3. Check Topology
  // If we know the slice size (devices per node), we MUST verify that
  // the collective is local. If it spans across nodes, the one-shot
  // kernel (and P2P) cannot be used.
  if (config.num_local_devices > 0) {
    bool is_local = IsAllReplicasLocal(
        config.num_local_devices, ra2a_thunk.ragged_all_to_all_config().config);
    if (!is_local) {
      VLOG(2) << "Skipping RaggedAllToAll Command Buffer conversion: "
                 "Operation requires multi-host communication "
              << "(num_local_devices=" << config.num_local_devices
              << "), but one-shot kernel only supports "
                 "local (intra-node) cliques.";
      return false;
    }
  }

  // If slice_size is unknown (0), we optimistically assume it's convertible
  // and let the runtime Initialize/Record safety checks fail if we are wrong.
  return true;
}

// Returns true if the DynamicSliceThunk is convertible to a command buffer
// operation. This requires that all embedded thunks are also convertible,
// e.g. a DynamicSliceThunk wrapping a collective is not convertible if
// collectives are not enabled for command buffer capture.
static bool IsConvertible(const DynamicSliceThunk& dynamic_slice_thunk,
                          const CommandBufferConfig& config) {
  return ThunkSequenceIsConvertible(
      dynamic_slice_thunk.get_embedded_executor().thunks(), config);
}

// Returns true if the AsyncStartThunk is convertible to a command buffer
// operation. This requires that all nested thunks are convertible.
static bool IsConvertible(const AsyncStartThunk& async_start_thunk,
                          const CommandBufferConfig& config) {
  return ThunkSequenceIsConvertible(async_start_thunk.thunks(), config);
}

// Returns true if the given Thunk is convertible to a command buffer operation
// based on the provided `config`.
bool IsConvertible(const Thunk& thunk, const CommandBufferConfig& config) {
  // Async start thunks are convertible if all nested thunks are convertible.
  if (thunk.kind() == Thunk::kAsyncStart) {
    return IsConvertible(static_cast<const AsyncStartThunk&>(thunk), config);
  }

  // Async done thunks are no-op from command buffer perspective.
  if (thunk.kind() == Thunk::kAsyncDone) {
    return true;
  }

  auto cmd_type = GetCommandBufferCmdType(thunk);
  if (!cmd_type.has_value()) {
    return false;  // Thunk kind is not supported for command buffer conversion.
  }

  if (!config.enabled_commands.contains(*cmd_type)) {
    VLOG(2) << "Thunk kind " << Thunk::KindToString(thunk.kind())
            << " lowering is not enabled by the user for type "
            << DebugOptions::CommandBufferCmdType_Name(*cmd_type);
    return false;  // Thunk kind is not supported for command buffer conversion.
  }

  if (thunk.kind() == Thunk::kWhile) {
    return IsConvertible(static_cast<const WhileThunk&>(thunk), config);
  }

  if (thunk.kind() == Thunk::kConditional) {
    return IsConvertible(static_cast<const ConditionalThunk&>(thunk), config);
  }

  if (thunk.kind() == Thunk::kCustomCall) {
    if (auto* ffi_thunk = dynamic_cast<const CustomCallThunk*>(&thunk)) {
      return IsConvertible(*ffi_thunk, config);
    }
    // Legacy custom calls are not command-buffer compatible.
    return false;
  }

  if (thunk.kind() == Thunk::kDynamicSlice) {
    return IsConvertible(static_cast<const DynamicSliceThunk&>(thunk), config);
  }

  if (thunk.kind() == Thunk::kRaggedAllToAll) {
    return IsConvertible(static_cast<const RaggedAllToAllThunk&>(thunk),
                         config);
  }
  return true;
}

bool ThunkSequenceIsConvertible(const ThunkSequence& thunks,
                                const CommandBufferConfig& config) {
  for (size_t i = 0; i < thunks.size(); ++i) {
    auto& thunk = thunks[i];
    if (!IsConvertible(*thunk.get(), config)) {
      return false;
    }
    if (thunk->kind() == Thunk::kAsyncStart) {
      size_t region_size =
          CheckAsyncRegion(absl::MakeSpan(thunks).subspan(i), config);
      if (region_size == 0) {
        return false;
      }
      i += region_size - 1;
    }
  }
  return true;
}

// Collects and returns the size of the shortest non-empty sequence of thunks
// that form a valid async region.
// The sequence considered as a valid async region if each start thunk has a
// corresponding done thunk and vice versa, and all thunks in between are
// convertible. If there is another start thunk between the original start and
// done, we may potentially extend the sequence to include its corresponding
// done thunk. For example, if we call this function on async-start_a in the
// following sequence:
//
// async_start_a
// async_start_b
// async_done_a
// async_done_b
//
// The returned sequence will contain async_done_b. So that all async pairs
// are captured by the same command buffer.
size_t CheckAsyncRegion(absl::Span<const std::unique_ptr<Thunk>> thunks,
                        const CommandBufferConfig& config) {
  absl::flat_hash_set<uint64_t> unpaired_ids;

  for (size_t i = 0; i < thunks.size(); ++i) {
    auto& thunk = thunks[i];

    // Check if thunk is convertible
    if (!IsConvertible(*thunk, config)) {
      return 0;  // All thunks in the region must be convertible.
    }

    // Track AsyncStartThunk/AsyncDoneThunk pairs via AsyncExecutionId.
    if (thunk->kind() == Thunk::kAsyncStart) {
      unpaired_ids.insert(static_cast<const AsyncStartThunk&>(*thunk)
                              .async_execution_id()
                              .value());
    }
    if (thunk->kind() == Thunk::kAsyncDone) {
      auto id = static_cast<const AsyncDoneThunk&>(*thunk)
                    .async_execution_id()
                    .value();
      auto it = unpaired_ids.find(id);
      if (it == unpaired_ids.end()) {
        return 0;  // Done without matching start in the region.
      }
      unpaired_ids.erase(it);
    }

    if (unpaired_ids.empty()) {
      return i + 1;  // All start/done pairs are matched.
    }
  }
  return 0;  // error didn't find an end for some start
}

// Returns the shortest non-empty sequence of thunks that form a valid async
// region as a span. If no such region is found, an empty span is returned.
absl::Span<std::unique_ptr<Thunk>> CollectAndCheckAsyncRegion(
    absl::Span<std::unique_ptr<Thunk>> thunks,
    const CommandBufferConfig& config) {
  return thunks.subspan(0, CheckAsyncRegion(thunks, config));
}

absl::StatusOr<CommandExecutor::SynchronizationMode> GetSynchronizationMode(
    DebugOptions::CommandBufferSchedulingMode scheduling_mode) {
  switch (scheduling_mode) {
    case DebugOptions::SERIALIZE:
      return CommandExecutor::SynchronizationMode::kSerialize;
    case DebugOptions::CONCURRENT:
      return CommandExecutor::SynchronizationMode::kConcurrent;
    case DebugOptions::LHS:
      return CommandExecutor::SynchronizationMode::kLHS;
    case DebugOptions::CONCURRENT_REGIONS:
      return CommandExecutor::SynchronizationMode::kConcurrentRegions;
    default:
      return Internal("Unsupported command buffer scheduling mode: %d",
                      scheduling_mode);
  }
}

absl::StatusOr<std::unique_ptr<CommandBufferThunk>>
ConvertThunksToCommandBuffer(
    ThunkSequence thunks_to_convert,
    CommandExecutor::SynchronizationMode synchronization_mode,
    const DebugOptions& debug_options) {
  bool enable_loop_unroll = debug_options.xla_gpu_command_buffer_unroll_loops();
  DebugOptions::CommandBufferUpdateMode update_mode =
      debug_options.xla_gpu_command_buffer_update_mode();
  ASSIGN_OR_RETURN(CommandExecutor cmd_executor,
                   ConvertToCommands(thunks_to_convert,
                                     ConvertToCommandsOptions{
                                         synchronization_mode,
                                         enable_loop_unroll, update_mode}));

  std::string command_buffer_profile_annotation = absl::StrCat(
      "command_buffer",
      !thunks_to_convert.empty()
          ? absl::StrCat("_", thunks_to_convert.front()->thunk_info().thunk_id)
          : "");

  if (VLOG_IS_ON(2)) {
    auto graph = cmd_executor.RenderExecutionGraph();
    if (graph.ok()) {
      VLOG(2) << command_buffer_profile_annotation << " graph: " << *graph;
    }
  }

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = command_buffer_profile_annotation;
  if (tsl::profiler::ProfilerLock::HasActiveSession() &&
      !debug_options.xla_enable_command_buffers_during_profiling()) {
    thunk_info.profile_annotation += " (disabled for profiling)";
  }
  VLOG(2) << "Creating command buffer thunk with the following thunks: "
          << absl::StrJoin(
                 thunks_to_convert, ", ",
                 [](std::string* out, const std::unique_ptr<Thunk>& thunk) {
                   absl::StrAppend(out, thunk->thunk_info().profile_annotation);
                 });
  return std::make_unique<CommandBufferThunk>(
      std::move(cmd_executor), std::move(thunk_info),
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(),
                                        std::move(thunks_to_convert)),
      debug_options.xla_enable_command_buffers_during_profiling(), update_mode);
}

absl::Status FlushCommandBuffer(
    CommandExecutor::SynchronizationMode synchronization_mode,
    const DebugOptions& debug_options,
    ThunkSequence& current_command_buffer_thunks, ThunkSequence& new_thunks,
    bool& changed) {
  // If we don't have enough thunks to form a command buffer, we just add
  // them to the new thunks sequence as is.
  if (current_command_buffer_thunks.size() <
      std::max(1, debug_options.xla_gpu_graph_min_graph_size())) {
    if (VLOG_IS_ON(2)) {
      for (const auto& thunk : current_command_buffer_thunks) {
        VLOG(2) << "Thunk kind " << Thunk::KindToString(thunk->kind())
                << " is not lowered to command buffer because command size is "
                   "less than the min graph size";
      }
    }
    new_thunks.insert(
        new_thunks.end(),
        std::make_move_iterator(current_command_buffer_thunks.begin()),
        std::make_move_iterator(current_command_buffer_thunks.end()));
    current_command_buffer_thunks.clear();
    return absl::OkStatus();
  }

  ASSIGN_OR_RETURN(
      auto cmd_buffer_thunk,
      ConvertThunksToCommandBuffer(std::move(current_command_buffer_thunks),
                                   synchronization_mode, debug_options));
  current_command_buffer_thunks.clear();

  // Check that the command buffer thunk is not empty
  assert(cmd_buffer_thunk->thunks() != nullptr &&
         !cmd_buffer_thunk->thunks()->thunks().empty());
  new_thunks.push_back(std::move(cmd_buffer_thunk));
  changed = true;
  return absl::OkStatus();
}

}  // namespace

std::string CommandBufferConversionPass::CommandBufferConfig::ToString() const {
  auto formatter = [](std::string* out,
                      DebugOptions::CommandBufferCmdType cmd) {
    absl::StrAppend(out, DebugOptions::CommandBufferCmdType_Name(cmd));
  };
  std::string cmd_names = absl::StrJoin(enabled_commands, ", ", formatter);
  return absl::StrCat("enabled_commands: [", cmd_names, "]");
}

absl::StatusOr<bool> CommandBufferConversionPass::Run(
    ThunkSequence* thunk_sequence, const DebugOptions& debug_options,
    const HloModule* absl_nullable hlo_module,
    const se::DeviceDescription& device_info,
    ThunkPassBufferAllocator& allocator) {
  tsl::profiler::TraceMe traceme("CommandBufferConversionPass");

  CommandBufferConfig config =
      GetCommandBufferConfig(debug_options, device_info, hlo_module);
  VLOG(1) << "Module " << module_name_
          << " CommandBufferConfig: " << config.ToString();
  ASSIGN_OR_RETURN(CommandExecutor::SynchronizationMode synchronization_mode,
                   GetSynchronizationMode(
                       debug_options.xla_gpu_command_buffer_scheduling_mode()));

  bool changed = false;

  ThunkSequence current_command_buffer_thunks;
  ThunkSequence new_thunks;

  auto flush_command_buffer = [&]() -> absl::Status {
    return FlushCommandBuffer(synchronization_mode, debug_options,
                              current_command_buffer_thunks, new_thunks,
                              changed);
  };

  auto& original_thunks = *thunk_sequence;

  for (size_t i = 0; i < original_thunks.size(); ++i) {
    auto& thunk = original_thunks[i];

    // We always have to capture both corresponding start and done events in the
    // same command buffer.
    if (thunk->kind() == Thunk::kAsyncStart) {
      // Collect and check async region
      absl::Span<std::unique_ptr<Thunk>> region = CollectAndCheckAsyncRegion(
          absl::MakeSpan(original_thunks).subspan(i), config);

      if (!region.empty()) {
        // If a valid region is found, add the whole region to the current
        // sequence and continue processing.
        i += region.size() - 1;
        absl::c_move(region, std::back_inserter(current_command_buffer_thunks));
        continue;
      }
    } else if (IsConvertible(*thunk.get(), config) &&
               thunk->kind() != Thunk::kAsyncDone) {
      // Check if thunk is convertible and not an async done: async done thunks
      // can be only added to the current_command_buffer_thunks as part of a
      // valid async regions.
      current_command_buffer_thunks.push_back(std::move(thunk));
      continue;
    }
    if (thunk->kind() == Thunk::kWhile) {
      // If a `WhileThunk` itself is not eligible for conversion into a
      // command buffer, we attempt to convert thunks within its body
      auto while_thunk = static_cast<WhileThunk*>(thunk.get());
      ASSIGN_OR_RETURN(bool changed_in_body,
                       Run(&while_thunk->body_executor().thunks(),
                           debug_options, hlo_module, device_info, allocator));
      changed |= changed_in_body;
    } else if (thunk->kind() == Thunk::kConditional) {
      // If a `ConditionalThunk` itself is not eligible for conversion into a
      // command buffer, we attempt to convert thunks within its branches.
      auto conditional_thunk = static_cast<ConditionalThunk*>(thunk.get());
      for (auto& branch_executor : conditional_thunk->branch_executors()) {
        ASSIGN_OR_RETURN(bool changed_in_branch,
                         Run(&branch_executor.thunks(), debug_options,
                             hlo_module, device_info, allocator));
        changed |= changed_in_branch;
      }
    }

    // If the current thunk is not convertible, flush collected eligible thunk
    // to a command buffer thunk and add it to the processed sequence. Then add
    // non-convertible thunk to the sequence.
    RETURN_IF_ERROR(flush_command_buffer());
    new_thunks.push_back(std::move(thunk));
  }

  // Flush the last command buffer.
  RETURN_IF_ERROR(flush_command_buffer());

  *thunk_sequence = std::move(new_thunks);
  return changed;
}

}  // namespace gpu
}  // namespace xla
