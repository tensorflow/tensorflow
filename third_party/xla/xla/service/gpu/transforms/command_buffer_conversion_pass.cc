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
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/overload.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

CommandBufferConversionPass::CommandBufferConfig
CommandBufferConversionPass::GetCommandBufferConfig(
    const HloModuleConfig& hlo_module_config,
    const se::DeviceDescription& device_info) {
  absl::flat_hash_set<DebugOptions::CommandBufferCmdType> commands;
  for (auto cmd_type :
       hlo_module_config.debug_options().xla_gpu_enable_command_buffer()) {
    commands.insert(static_cast<DebugOptions::CommandBufferCmdType>(cmd_type));
  }

  absl::flat_hash_set<std::string> legacy_custom_call_targets;
  for (const auto& target : hlo_module_config.debug_options()
                                .legacy_command_buffer_custom_call_targets()) {
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
#if defined(PLATFORM_GOOGLE)
                << "set CUDA_COMPAT_LOAD=1 env variable.";
#else
                << "https://docs.nvidia.com/deploy/cuda-compatibility/.";
#endif
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

  std::visit(Overload{erase_cuda, erase_rocm},
             device_info.gpu_compute_capability());

  return config;
}

void CommandBufferConversionPass::GetThunksInPostOrder(
    const Thunk* thunk, absl::flat_hash_set<const Thunk*>& visited,
    std::vector<const Thunk*>& post_order) {
  if (thunk == nullptr || !visited.insert(thunk).second) {
    return;
  }

  if (thunk->kind() == Thunk::kSequential) {
    const auto* sequential_thunk = static_cast<const SequentialThunk*>(thunk);
    for (const auto& nested_thunk : sequential_thunk->thunks()) {
      GetThunksInPostOrder(nested_thunk.get(), visited, post_order);
    }
  } else if (thunk->kind() == Thunk::kConditional) {
    const auto* conditional_thunk = static_cast<const ConditionalThunk*>(thunk);
    for (const auto& branch_thunk : conditional_thunk->branch_thunks()) {
      GetThunksInPostOrder(branch_thunk.get(), visited, post_order);
    }
  } else if (thunk->kind() == Thunk::kWhile) {
    const auto* while_thunk = static_cast<const WhileThunk*>(thunk);
    GetThunksInPostOrder(while_thunk->condition_thunk_sequence(), visited,
                         post_order);
    GetThunksInPostOrder(while_thunk->body_thunk_sequence(), visited,
                         post_order);
  }

  post_order.push_back(thunk);
}

std::vector<const Thunk*> CommandBufferConversionPass::GetThunksInPostOrder(
    const SequentialThunk* root_thunk) {
  absl::flat_hash_set<const Thunk*> visited;
  std::vector<const Thunk*> post_order;
  GetThunksInPostOrder(root_thunk, visited, post_order);
  return post_order;
}

absl::StatusOr<bool> CommandBufferConversionPass::Run(
    std::unique_ptr<SequentialThunk>& root_thunk_ptr,
    const HloModuleConfig& hlo_module_config,
    const se::DeviceDescription& device_info) {
  CommandBufferConfig config =
      GetCommandBufferConfig(hlo_module_config, device_info);

  CommandBufferCmdExecutor::SynchronizationMode synchronization_mode =
      hlo_module_config.debug_options().xla_gpu_graph_enable_concurrent_region()
          ? CommandBufferCmdExecutor::SynchronizationMode::kAutomatic
          : CommandBufferCmdExecutor::SynchronizationMode::kSerialize;

  TF_ASSIGN_OR_RETURN(
      CommandBufferCmdExecutor cmd_executor,
      ConvertToCommands(root_thunk_ptr->thunks(),
                        ConvertToCommandsOptions{synchronization_mode}));

  auto cmd_buffer_thunk = std::make_unique<CommandBufferThunk>(
      std::move(cmd_executor), Thunk::ThunkInfo(), std::move(root_thunk_ptr),
      hlo_module_config.debug_options()
          .xla_enable_command_buffers_during_profiling());

  root_thunk_ptr = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo(), std::vector<std::unique_ptr<Thunk>>());
  root_thunk_ptr->thunks().push_back(std::move(cmd_buffer_thunk));

  return true;
}

}  // namespace gpu
}  // namespace xla
