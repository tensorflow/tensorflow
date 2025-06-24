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
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/overload.h"
#include "absl/log/log.h"
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

  TF_ASSIGN_OR_RETURN(
      CommandBufferCmdExecutor cmd_executor,
      ConvertToCommands(root_thunk_ptr->thunks(),
                        ConvertToCommandsOptions{synchronization_mode}));

  auto cmd_buffer_thunk = std::make_unique<CommandBufferThunk>(
      std::move(cmd_executor), Thunk::ThunkInfo(),
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(),
                                        std::move(root_thunk_ptr->thunks())),
      debug_options.xla_enable_command_buffers_during_profiling());

  ThunkSequence new_thunk_sequence;
  new_thunk_sequence.push_back(std::move(cmd_buffer_thunk));
  root_thunk_ptr->thunks() = std::move(new_thunk_sequence);

  return true;
}

}  // namespace gpu
}  // namespace xla
