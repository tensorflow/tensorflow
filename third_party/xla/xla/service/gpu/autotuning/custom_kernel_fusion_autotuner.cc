/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/autotuning/custom_kernel_fusion_autotuner.h"

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuning/autotuner_compile_util.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/autotuning/redzone_buffers.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

namespace {
absl::StatusOr<std::unique_ptr<HloModule>> ExtractFusionModule(
    HloInstruction* fusion_instruction, int64_t kernel_index) {
  std::unique_ptr<HloModule> hlo_module =
      ExtractInstructionIntoNewModule(*fusion_instruction);

  HloInstruction* instruction =
      hlo_module->entry_computation()->root_instruction();
  GpuBackendConfig gpu_config =
      instruction->backend_config<GpuBackendConfig>().value();
  gpu_config.mutable_fusion_backend_config()
      ->mutable_custom_fusion_config()
      ->set_kernel_index(kernel_index);
  TF_RETURN_IF_ERROR(instruction->set_backend_config(gpu_config));

  return hlo_module;
}

absl::StatusOr<std::vector<std::tuple<int, absl::Duration>>> ProfileKernels(
    std::vector<CustomKernel>& kernels, HloInstruction* fusion_instruction,
    AutotunerCompileUtil& compile_util, const AutotuneConfig& autotune_config,
    const DebugOptions& debug_options) {
  se::StreamExecutor* stream_exec = autotune_config.GetExecutor();
  std::vector<std::tuple<int, absl::Duration>> results;
  for (int i = 0; i < kernels.size(); ++i) {
    TF_ASSIGN_OR_RETURN(absl::StatusOr<std::unique_ptr<Executable>> executable,
                        compile_util.Compile([&](const DebugOptions& opt) {
                          return ExtractFusionModule(fusion_instruction, i);
                        }));

    se::DeviceMemoryAllocator* allocator = autotune_config.GetAllocator();
    std::unique_ptr<se::DeviceMemoryAllocator> owned_allocator;
    if (allocator == nullptr) {
      owned_allocator =
          std::make_unique<se::StreamExecutorMemoryAllocator>(stream_exec);
      allocator = owned_allocator.get();
    }

    bool should_init_buffers = autotune_config.should_init_buffers();
    bool should_check_correctness = autotune_config.should_check_correctness();
    int redzone_padding_bytes = debug_options.xla_gpu_redzone_padding_bytes();
    TF_ASSIGN_OR_RETURN(se::Stream* const stream, autotune_config.GetStream());
    TF_ASSIGN_OR_RETURN(auto rz_buffers,
                        RedzoneBuffers::FromInstruction(
                            *fusion_instruction, allocator, stream,
                            RedzoneBuffers::kAllInputs, should_init_buffers,
                            should_check_correctness, redzone_padding_bytes));

    TF_ASSIGN_OR_RETURN(
        AutotunerCompileUtil::ProfilingOutput profiling_output,
        compile_util.ProfileExecutable(executable->get(), stream,
                                       rz_buffers.input_buffers(),
                                       rz_buffers.input_shapes()));
    results.push_back({i, profiling_output.duration});
  }
  return results;
}

absl::StatusOr<int> FindFastestKernel(
    const std::vector<std::tuple<int, absl::Duration>>& results) {
  auto iter = absl::c_min_element(
      results, [](const std::tuple<int, absl::Duration>& lhs,
                  const std::tuple<int, absl::Duration>& rhs) {
        return std::get<1>(lhs) < std::get<1>(rhs);
      });
  if (iter == results.end()) {
    return absl::InternalError("Failed to find fastest kernel.");
  }
  return std::get<0>(*iter);
}

absl::Status UpdateFusionInstructionKernelIndex(
    HloInstruction* fusion_instruction, int kernel_index) {
  GpuBackendConfig gpu_config =
      fusion_instruction->backend_config<GpuBackendConfig>().value();
  gpu_config.mutable_fusion_backend_config()
      ->mutable_custom_fusion_config()
      ->set_kernel_index(kernel_index);
  TF_RETURN_IF_ERROR(fusion_instruction->set_backend_config(gpu_config));

  return absl::OkStatus();
}

absl::StatusOr<std::vector<CustomKernel>> LoadKernels(
    const HloInstruction* fusion_instruction,
    const AutotuneConfig& autotune_config) {
  auto config = fusion_instruction->backend_config<GpuBackendConfig>()
                    ->fusion_backend_config()
                    .custom_fusion_config();
  auto* registry = CustomKernelFusionRegistry::Default();
  auto* custom_kernel_fusion = registry->Lookup(config.name());

  // If custom fusion is not found it means that some of the build targets might
  // not be statically linked into the binary.
  if (custom_kernel_fusion == nullptr) {
    return absl::InternalError(
        absl::StrCat("Custom kernel fusion ", config.name(),
                     " not found in a default registry."));
  }

  se::StreamExecutor* stream_exec = autotune_config.GetExecutor();
  if (!stream_exec->SynchronizeAllActivity()) {
    return Internal("Failed to synchronize GPU for autotuning.");
  }
  se::DeviceDescription device_description =
      stream_exec->GetDeviceDescription();

  // Load custom kernels that can implement a fusion computation.
  TF_ASSIGN_OR_RETURN(
      std::vector<CustomKernel> kernels,
      custom_kernel_fusion->LoadKernels(
          device_description,
          fusion_instruction->fused_instructions_computation()));

  return kernels;
}

absl::StatusOr<bool> AutotuneCustomKernelFusion(
    HloInstruction* fusion_instruction, const AutotuneConfig& autotune_config,
    AutotunerCompileUtil& compile_util, const DebugOptions& debug_options) {
  int previous_kernel_index =
      fusion_instruction->backend_config<GpuBackendConfig>()
          ->fusion_backend_config()
          .custom_fusion_config()
          .kernel_index();

  TF_ASSIGN_OR_RETURN(std::vector<CustomKernel> kernels,
                      LoadKernels(fusion_instruction, autotune_config));

  std::vector<std::tuple<int, absl::Duration>> results;
  TF_ASSIGN_OR_RETURN(results,
                      ProfileKernels(kernels, fusion_instruction, compile_util,
                                     autotune_config, debug_options));

  TF_ASSIGN_OR_RETURN(int fastest_kernel_index, FindFastestKernel(results));

  TF_RETURN_IF_ERROR(UpdateFusionInstructionKernelIndex(fusion_instruction,
                                                        fastest_kernel_index));

  return previous_kernel_index != fastest_kernel_index;
}

bool IsCustomFusion(const HloComputation* computation) {
  if (!computation->IsFusionComputation()) {
    return false;
  }

  HloInstruction* instruction = computation->FusionInstruction();
  absl::StatusOr<GpuBackendConfig> gpu_backend_config =
      instruction->backend_config<GpuBackendConfig>();
  if (!gpu_backend_config.ok()) {
    return false;
  }

  if (instruction->fusion_kind() != HloInstruction::FusionKind::kCustom) {
    return false;
  }

  if (!gpu_backend_config->has_fusion_backend_config()) {
    return false;
  }

  return gpu_backend_config->fusion_backend_config().kind() ==
         kCustomFusionKind;
}
}  // namespace

absl::StatusOr<bool> CustomKernelFusionAutotuner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (config_.IsDeviceless()) {
    return false;
  }

  const DebugOptions& debug_options = module->config().debug_options();
  TF_ASSIGN_OR_RETURN(
      AutotunerCompileUtil compile_util,
      AutotunerCompileUtil::Create(config_.DeviceConfig(), debug_options));

  bool hlo_changed = false;
  for (const HloComputation* computation : module->computations()) {
    if (IsCustomFusion(computation)) {
      TF_ASSIGN_OR_RETURN(
          bool instruction_changed,
          AutotuneCustomKernelFusion(computation->FusionInstruction(), config_,
                                     compile_util, debug_options));
      if (instruction_changed) {
        hlo_changed = true;
      }
    }
  }

  return hlo_changed;
}

}  // namespace gpu
}  // namespace xla
