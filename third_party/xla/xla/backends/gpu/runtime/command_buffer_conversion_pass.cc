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
#include <bitset>
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
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/async_execution.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/collective_group_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_slice_fusion_v2_thunk.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/ragged_all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/platform.h"
#include "tsl/profiler/lib/profiler_lock.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {
namespace {

using CommandBufferConfig = CommandBufferConversionPass::CommandBufferConfig;

std::optional<DebugOptions::CollectiveOpType> GetCollectiveOpType(
    Thunk::Kind kind) {
  switch (kind) {
    case Thunk::kAllGather:
      return DebugOptions::ALLGATHER;
    case Thunk::kAllReduce:
      return DebugOptions::ALLREDUCE;
    case Thunk::kAllToAll:
      return DebugOptions::ALLTOALL;
    case Thunk::kCollectiveBroadcast:
      return DebugOptions::COLLECTIVEBROADCAST;
    case Thunk::kCollectivePermute:
      return DebugOptions::COLLECTIVEPERMUTE;
    case Thunk::kRaggedAllToAll:
      return DebugOptions::RAGGEDALLTOALL;
    case Thunk::kReduceScatter:
      return DebugOptions::REDUCESCATTER;
    default:
      return std::nullopt;
  }
}

CommandBufferConfig GetCommandBufferConfig(
    const DebugOptions& debug_options, const se::DeviceDescription& device_info,
    const HloModule* hlo_module) {
  absl::flat_hash_set<DebugOptions::CommandBufferCmdType> commands;
  for (auto cmd_type : debug_options.xla_gpu_enable_command_buffer()) {
    commands.insert(static_cast<DebugOptions::CommandBufferCmdType>(cmd_type));
  }

  std::bitset<CommandBufferConfig::kMaxCollectiveOps> enabled_collectives;
  const auto& filter =
      debug_options.xla_gpu_enable_collectives_command_buffer_filter();
  if (!filter.empty()) {
    for (auto op_type : filter) {
      if (op_type >= 0 && op_type < CommandBufferConfig::kMaxCollectiveOps) {
        enabled_collectives.set(op_type);
      } else {
        LOG(WARNING) << "Invalid collective op type: " << op_type;
      }
    }
  } else {
    enabled_collectives.set(DebugOptions::ALLCOLLECTIVES);
  }

  // Extract slice_size (partition_size) from HloModule config if available.
  int64_t num_local_devices = 0;
  if (hlo_module != nullptr) {
    num_local_devices = hlo_module->config().partition_size();
  }

  CommandBufferConfig config{
      std::move(commands), std::move(enabled_collectives), device_info,
      debug_options.xla_gpu_command_buffer_unroll_loops(), num_local_devices};

  // oneAPI command buffers are not implemented yet. Hence, disable command
  // buffer conversion for the oneAPI backend.
  // TODO(intel-tf): Remove this fallback once oneAPI command buffers are
  // implemented.
  if (device_info.gpu_compute_capability().IsOneAPI()) {
    config.enabled_commands.clear();
    return config;
  }

  // Erase command buffer cmd types that are not supported by the gpu runtime.
  static constexpr auto kRequireConditionals = {DebugOptions::CONDITIONAL,
                                                DebugOptions::WHILE};
  static constexpr auto kRequireTracing = {
      DebugOptions::CUBLAS,      DebugOptions::CUBLASLT,
      DebugOptions::CUDNN,       DebugOptions::CUSTOM_CALL,
      DebugOptions::COLLECTIVES, DebugOptions::CONVOLUTION};

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
      if (dynamic_cast<const DeviceToDeviceCopyThunk*>(&thunk)) {
        return DebugOptions::FUSION;
      } else {
        // Only copy within the same device can be converted to command buffers.
        VLOG(2) << "Unsupported thunk kind: " << Thunk::KindToString(kind);
        return std::nullopt;
      }
    case Thunk::Kind::kHostExecuteStart:
    case Thunk::Kind::kHostExecuteDone:
      return DebugOptions::HOST_EXECUTE;
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
    case Thunk::kGroup:
    case Thunk::kRaggedAllToAll:
    case Thunk::kReduceScatter:
    case Thunk::kRecv:
    case Thunk::kSend:
      return DebugOptions::COLLECTIVES;
    case Thunk::kCuDnn:
      return DebugOptions::CUDNN;
    case Thunk::kConvolution:
      return DebugOptions::CONVOLUTION;
    case Thunk::kCustomCall:
    case Thunk::kSelectK:
      return DebugOptions::CUSTOM_CALL;
    case Thunk::kCublasLtMatmul:
      return DebugOptions::CUBLASLT;
    case Thunk::kDynamicSliceFusion:
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

bool SupportsMovedChildCommands(const se::DeviceDescription& device_info) {
  const auto* cuda_cc =
      device_info.gpu_compute_capability().cuda_compute_capability();
  if (cuda_cc == nullptr) {
    return false;
  }
  return std::min({device_info.runtime_version(), device_info.driver_version(),
                   device_info.compile_time_toolkit_version()}) >=
         se::SemanticVersion{12, 9, 0};
}

bool HasLoopDependentDynamicSliceFusionV2(const ThunkSequence& thunks) {
  bool has_loop_dependent_dsf = false;
  for (const std::unique_ptr<Thunk>& thunk : thunks) {
    if (has_loop_dependent_dsf) {
      break;
    }
    thunk->Walk([&](const Thunk* nested) {
      if (const auto* dsf =
              dynamic_cast<const DynamicSliceFusionV2Thunk*>(nested);
          dsf != nullptr && dsf->HasLoopDependentOffsets()) {
        has_loop_dependent_dsf = true;
      }
    });
  }
  return has_loop_dependent_dsf;
}

// Returns true if the WhileThunk is convertible to a command buffer operation.
// This requires that all thunks in both the condition and body sequences are
// convertible.
bool IsConvertible(const WhileThunk& while_thunk,
                   const CommandBufferConfig& config) {
  if (HasLoopDependentDynamicSliceFusionV2(
          while_thunk.body_executor().thunks()) &&
      !(config.enable_loop_unroll && while_thunk.trip_count().has_value())) {
    VLOG(2) << "WhileThunk is not convertible to command buffers because its "
               "body contains loop-dependent DynamicSliceFusionV2Thunk and "
               "the while loop will not be unrolled";
    return false;
  }
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
        config.num_local_devices,
        ra2a_thunk.ragged_all_to_all_config().config.replica_groups,
        ra2a_thunk.ragged_all_to_all_config().config.group_mode);
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

// Returns true if the DynamicSliceFusionV2Thunk is convertible to a command
// buffer operation. Runtime offset verification performs synchronous D2H copies
// and is intentionally unsupported for command buffer lowering.
static bool IsConvertible(
    const DynamicSliceFusionV2Thunk& dynamic_slice_fusion_thunk,
    const CommandBufferConfig& config) {
  if (!SupportsMovedChildCommands(config.device_description)) {
    VLOG(2) << "DynamicSliceFusionV2Thunk is not convertible to command "
               "buffers because child command nodes require CUDA 12.9+";
    return false;
  }
  if (dynamic_slice_fusion_thunk.verify_offsets()) {
    VLOG(2) << "DynamicSliceFusionV2Thunk is not convertible to command "
               "buffers because runtime offset verification is enabled";
    return false;
  }
  return ThunkSequenceIsConvertible(dynamic_slice_fusion_thunk.thunks(),
                                    config);
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

  if (*cmd_type == DebugOptions::COLLECTIVES) {
    if (!config.enabled_collectives.test(DebugOptions::ALLCOLLECTIVES)) {
      auto op_type = GetCollectiveOpType(thunk.kind());
      if (op_type.has_value() && !config.enabled_collectives.test(*op_type)) {
        VLOG(2) << "Collective thunk kind " << Thunk::KindToString(thunk.kind())
                << " is not enabled by the collectives filter";
        return false;
      }
    }
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

  if (thunk.kind() == Thunk::kDynamicSliceFusion) {
    return IsConvertible(static_cast<const DynamicSliceFusionV2Thunk&>(thunk),
                         config);
  }

  if (thunk.kind() == Thunk::kRaggedAllToAll) {
    return IsConvertible(static_cast<const RaggedAllToAllThunk&>(thunk),
                         config);
  }

  if (thunk.kind() == Thunk::kGroup) {
    return ThunkSequenceIsConvertible(
        static_cast<const CollectiveGroupThunk&>(thunk).thunks(), config);
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
    } else if (thunk->kind() == Thunk::kAsyncDone) {
      // Every done that belongs to a start in this sequence was consumed above
      // as part of its region. This done joins an operation started outside
      // the sequence; capturing it would drop the join (the top-level loop in
      // `RunImpl` keeps such thunks in place for the same reason).
      return false;
    }
  }
  return true;
}

// Collects and returns the size of the shortest non-empty sequence of thunks
// that form a closed async region: each start thunk has a corresponding done
// thunk and vice versa. If there is another start thunk between the original
// start and done, we may potentially extend the sequence to include its
// corresponding done thunk. For example, if we call this function on
// async-start_a in the following sequence:
//
// async_start_a
// async_start_b
// async_done_a
// async_done_b
//
// The returned sequence will contain async_done_b. So that all async pairs
// are captured by the same command buffer.
// Find the boundary independently of command-buffer eligibility. If any thunk
// is unsupported, the whole region must remain outside the command buffer:
// capturing an inner region could lose ordering with an outstanding operation
// on the same async stream.
//
// Returns 0 when no closed region starts here: either a start in the scanned
// range has no done in this sequence, or a done in the range joins a start
// from outside it. The caller handles both the same way, by leaving the start
// as a thunk and keeping its stream open until it is joined.
size_t AsyncRegionSize(absl::Span<const std::unique_ptr<Thunk>> thunks) {
  absl::flat_hash_set<const AsyncExecution*> unpaired_executions;

  for (size_t i = 0; i < thunks.size(); ++i) {
    auto& thunk = thunks[i];

    if (thunk->kind() == Thunk::kAsyncStart) {
      // Pipelined starts can share the canonical start's execution state, but
      // AsyncExecution::Start rejects a second start before the matching done,
      // so a valid sequence never inserts the same execution twice. Optimized
      // builds fall back to leaving such a region uncaptured.
      bool inserted = unpaired_executions
                          .insert(static_cast<const AsyncStartThunk&>(*thunk)
                                      .async_execution()
                                      .get())
                          .second;
      DCHECK(inserted) << "Async execution started twice before its done: "
                       << thunk->profile_annotation();
      if (!inserted) {
        return 0;
      }
    }
    if (thunk->kind() == Thunk::kAsyncDone) {
      auto* execution =
          static_cast<const AsyncDoneThunk&>(*thunk).async_execution().get();
      auto it = unpaired_executions.find(execution);
      if (it == unpaired_executions.end()) {
        return 0;  // Done without matching start in the region.
      }
      unpaired_executions.erase(it);
    }

    if (unpaired_executions.empty()) {
      return i + 1;  // All start/done pairs are matched.
    }
  }
  return 0;  // At least one start has no matching done in this sequence.
}

// Returns the size of a closed region only if every thunk can be converted.
size_t CheckAsyncRegion(absl::Span<const std::unique_ptr<Thunk>> thunks,
                        const CommandBufferConfig& config) {
  size_t size = AsyncRegionSize(thunks);
  for (const std::unique_ptr<Thunk>& thunk : thunks.first(size)) {
    if (!IsConvertible(*thunk, config)) {
      return 0;
    }
  }
  return size;
}

// Returns true if `thunk` or any thunk nested in it starts an async region on
// one of `streams`. DynamicSliceFusionV2Thunk hides its embedded thunks from
// Thunk::Walk, so they are inspected explicitly.
bool ContainsAsyncStartOnStreams(
    const Thunk& thunk, const absl::flat_hash_set<ExecutionStreamId>& streams) {
  if (streams.empty()) {
    return false;
  }
  // A non-OK status stops the walk at the first match.
  absl::Status walk = thunk.Walk([&](const Thunk* nested) -> absl::Status {
    if (nested->kind() == Thunk::kAsyncStart) {
      if (streams.contains(static_cast<const AsyncStartThunk&>(*nested)
                               .execution_stream_id())) {
        return absl::CancelledError();
      }
    } else if (nested->kind() == Thunk::kDynamicSliceFusion) {
      const auto& fusion =
          static_cast<const DynamicSliceFusionV2Thunk&>(*nested);
      for (const std::unique_ptr<Thunk>& embedded : fusion.thunks()) {
        if (ContainsAsyncStartOnStreams(*embedded, streams)) {
          return absl::CancelledError();
        }
      }
    }
    return absl::OkStatus();
  });
  return !walk.ok();
}

// Returns the index one past the done that joins the start at
// `thunks[start_index]`, or `thunks.size()` if no thunk in this sequence joins
// it. Until that index, work started on the start's stream is outstanding.
size_t AsyncJoinEnd(absl::Span<const std::unique_ptr<Thunk>> thunks,
                    size_t start_index) {
  const AsyncExecution* execution =
      static_cast<const AsyncStartThunk&>(*thunks[start_index])
          .async_execution()
          .get();
  for (size_t i = start_index + 1; i < thunks.size(); ++i) {
    if (thunks[i]->kind() == Thunk::kAsyncDone &&
        static_cast<const AsyncDoneThunk&>(*thunks[i])
                .async_execution()
                .get() == execution) {
      return i + 1;
    }
  }
  return thunks.size();
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
  ABSL_ASSIGN_OR_RETURN(
      CommandExecutor cmd_executor,
      ConvertToCommands(
          thunks_to_convert,
          ConvertToCommandsOptions{synchronization_mode, enable_loop_unroll}));

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
  VLOG(2) << "Creating command buffer thunk "
          << command_buffer_profile_annotation << " with the following thunks: "
          << absl::StrJoin(
                 thunks_to_convert, ", ",
                 [](std::string* out, const std::unique_ptr<Thunk>& thunk) {
                   absl::StrAppend(out, thunk->thunk_info().profile_annotation);
                 });
  return std::make_unique<CommandBufferThunk>(
      std::move(cmd_executor), std::move(thunk_info),
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(),
                                        std::move(thunks_to_convert)),
      debug_options.xla_enable_command_buffers_during_profiling());
}

int64_t CountCommandBufferSize(ThunkSequence& thunks) {
  int64_t count = 0;
  (void)thunks.WalkNested([&](Thunk* nested) -> absl::Status {
    if (nested->kind() != Thunk::kAsyncDone &&
        nested->kind() != Thunk::kAsyncStart &&
        nested->kind() != Thunk::kGroup &&
        nested->kind() != Thunk::kSequential) {
      ++count;
    }
    return absl::OkStatus();
  });
  return std::max<int64_t>(thunks.size(), count);
}

absl::Status FlushCommandBuffer(
    CommandExecutor::SynchronizationMode synchronization_mode,
    const DebugOptions& debug_options,
    ThunkSequence& current_command_buffer_thunks, ThunkSequence& new_thunks,
    bool& changed) {
  // If we don't have enough thunks to form a command buffer, we just add
  // them to the new thunks sequence as is.
  if (CountCommandBufferSize(current_command_buffer_thunks) <
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

  ABSL_ASSIGN_OR_RETURN(
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

  std::string collectives_filter;
  if (enabled_collectives.test(DebugOptions::ALLCOLLECTIVES)) {
    collectives_filter = "ALLCOLLECTIVES";
  } else {
    std::vector<std::string> enabled_names;
    for (int i = 0; i < kMaxCollectiveOps; ++i) {
      if (enabled_collectives.test(i) &&
          DebugOptions::CollectiveOpType_IsValid(i)) {
        enabled_names.push_back(DebugOptions::CollectiveOpType_Name(
            static_cast<DebugOptions::CollectiveOpType>(i)));
      }
    }
    collectives_filter = absl::StrJoin(enabled_names, ", ");
  }

  return absl::StrCat("enabled_commands: [", cmd_names,
                      "], enabled_collectives: [", collectives_filter, "]");
}

absl::StatusOr<bool> CommandBufferConversionPass::Run(
    ThunkSequence* thunk_sequence, const DebugOptions& debug_options,
    const HloModule* absl_nullable hlo_module,
    const se::DeviceDescription& device_info,
    ThunkPassBufferAllocator& allocator) {
  return RunImpl(thunk_sequence, debug_options, hlo_module, device_info,
                 allocator, /*open_async_streams=*/{});
}

absl::StatusOr<bool> CommandBufferConversionPass::RunImpl(
    ThunkSequence* thunk_sequence, const DebugOptions& debug_options,
    const HloModule* absl_nullable hlo_module,
    const se::DeviceDescription& device_info,
    ThunkPassBufferAllocator& allocator,
    const absl::flat_hash_set<ExecutionStreamId>& open_async_streams) {
  tsl::profiler::TraceMe traceme("CommandBufferConversionPass");

  CommandBufferConfig config =
      GetCommandBufferConfig(debug_options, device_info, hlo_module);
  VLOG(1) << "Module " << module_name_
          << " CommandBufferConfig: " << config.ToString();
  ABSL_ASSIGN_OR_RETURN(CommandExecutor::SynchronizationMode synchronization_mode,
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

  // An async region is "open" when it cannot be captured as a whole: some
  // thunk in it is not convertible, or its start has no matching done in this
  // sequence. Its start stays a thunk and enqueues work on its async stream
  // that a command buffer launched on the main stream cannot observe. That
  // stream is "open" until the done that joins it, so while it is open:
  //  * plain commands still execute on the main stream and can be captured,
  //    exactly as their thunks never observed the async stream either;
  //  * async regions on other streams can still be captured: their bodies
  //    only ever waited on the main stream, so a graph loses no ordering;
  //  * async starts on the open stream stay in place, because capturing them
  //    would move their bodies into a graph on the main stream and lose
  //    stream order with the outstanding operation;
  //  * control flow containing such starts is not captured whole; its bodies
  //    are converted recursively with the same streams open throughout.
  // `open_stream_ends` maps each open stream to the index one past the thunk
  // that joins it, or the end of the sequence for an unmatched start.
  absl::flat_hash_map<ExecutionStreamId, size_t> open_stream_ends;
  for (ExecutionStreamId stream : open_async_streams) {
    open_stream_ends[stream] = original_thunks.size();
  }
  auto open_streams_at = [&](size_t index) {
    absl::flat_hash_set<ExecutionStreamId> streams;
    for (const auto& [stream, end] : open_stream_ends) {
      if (index < end) {
        streams.insert(stream);
      }
    }
    return streams;
  };

  for (size_t i = 0; i < original_thunks.size(); ++i) {
    auto& thunk = original_thunks[i];
    const absl::flat_hash_set<ExecutionStreamId> open_streams =
        open_streams_at(i);

    if (thunk->kind() == Thunk::kAsyncStart) {
      const auto& start = static_cast<const AsyncStartThunk&>(*thunk);
      // We always have to capture both corresponding start and done events in
      // the same command buffer.
      if (!open_streams.contains(start.execution_stream_id())) {
        absl::Span<std::unique_ptr<Thunk>> tail =
            absl::MakeSpan(original_thunks).subspan(i);
        absl::Span<std::unique_ptr<Thunk>> region =
            tail.first(CheckAsyncRegion(tail, config));
        if (!region.empty() &&
            absl::c_none_of(region, [&](const std::unique_ptr<Thunk>& nested) {
              return ContainsAsyncStartOnStreams(*nested, open_streams);
            })) {
          // If a valid region is found, add the whole region to the current
          // sequence and continue processing.
          i += region.size() - 1;
          absl::c_move(region,
                       std::back_inserter(current_command_buffer_thunks));
          continue;
        }
      }
      // This start stays a thunk so that its body keeps executing on its async
      // stream; the stream is open until the done that joins it.
      size_t& end = open_stream_ends[start.execution_stream_id()];
      end = std::max(end, AsyncJoinEnd(original_thunks, i));
    }

    // Async start and done thunks are only captured as part of a valid async
    // region above; on their own they stay in place.
    const bool is_async_boundary = thunk->kind() == Thunk::kAsyncStart ||
                                   thunk->kind() == Thunk::kAsyncDone;
    if (!is_async_boundary && IsConvertible(*thunk, config) &&
        !ContainsAsyncStartOnStreams(*thunk, open_streams)) {
      current_command_buffer_thunks.push_back(std::move(thunk));
      continue;
    }

    if (thunk->kind() == Thunk::kWhile) {
      // If a `WhileThunk` itself is not captured into a command buffer, we
      // attempt to convert thunks within its body.
      auto while_thunk = static_cast<WhileThunk*>(thunk.get());
      ABSL_ASSIGN_OR_RETURN(
          bool changed_in_body,
          RunImpl(&while_thunk->body_executor().thunks(), debug_options,
                  hlo_module, device_info, allocator, open_streams));
      changed |= changed_in_body;
    } else if (thunk->kind() == Thunk::kConditional) {
      // If a `ConditionalThunk` itself is not captured into a command buffer,
      // we attempt to convert thunks within its branches.
      auto conditional_thunk = static_cast<ConditionalThunk*>(thunk.get());
      for (auto& branch_executor : conditional_thunk->branch_executors()) {
        ABSL_ASSIGN_OR_RETURN(
            bool changed_in_branch,
            RunImpl(&branch_executor.thunks(), debug_options, hlo_module,
                    device_info, allocator, open_streams));
        changed |= changed_in_branch;
      }
    }

    // If the current thunk is not captured, flush collected eligible thunks
    // to a command buffer thunk and add it to the processed sequence. Then add
    // the thunk itself to the sequence.
    ABSL_RETURN_IF_ERROR(flush_command_buffer());
    new_thunks.push_back(std::move(thunk));
  }

  // Flush the last command buffer.
  ABSL_RETURN_IF_ERROR(flush_command_buffer());

  *thunk_sequence = std::move(new_thunks);
  return changed;
}

}  // namespace gpu
}  // namespace xla
