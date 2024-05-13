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

#include "xla/service/gpu/triton_fusion_numerics_verifier.h"

#include <memory>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuner_compile_util.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/buffer_comparator.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

namespace {

using ProfilingOutput = AutotunerCompileUtil::ProfilingOutput;

// Returns the input instruction as a fusion instruction, if it represents a
// Triton fusion. Otherwise, returns nullptr.
//
// TODO(b/326274248): Visit all Triton fusions. Now it's Softmax only.
absl::StatusOr<const HloFusionInstruction*> AsTritonFusion(
    const HloInstruction* hlo) {
  if (hlo->opcode() != HloOpcode::kFusion) {
    return nullptr;
  }
  const HloFusionInstruction* fusion = Cast<HloFusionInstruction>(hlo);
  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      fusion->backend_config<GpuBackendConfig>());
  const FusionBackendConfig& backend_config =
      gpu_config.fusion_backend_config();
  if (backend_config.kind() == kTritonSoftmaxFusionKind) {
    return fusion;
  }
  return nullptr;
}

std::unique_ptr<HloModule> NewHloModuleFromFusion(
    const HloFusionInstruction& fusion, const DebugOptions& debug_opts,
    bool clear_backend_config) {
  std::unique_ptr<HloModule> new_module =
      ExtractInstructionIntoNewModule(fusion);
  if (clear_backend_config) {
    new_module->entry_computation()->root_instruction()->clear_backend_config();
  }
  new_module->mutable_config().set_debug_options(debug_opts);

  return new_module;
}

}  // namespace

namespace triton_fusion_numerics_pass_internal {

absl::StatusOr<ScopedShapedBuffer> CompileAndRunFusion(
    AutotunerCompileUtil& util, const HloFusionInstruction& fusion,
    const AutotuneConfig& config, const DebugOptions& debug_opts,
    bool clear_backend_config) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                      util.Compile([&](const DebugOptions& opts) {
                        return NewHloModuleFromFusion(fusion, opts,
                                                      clear_backend_config);
                      }));
  TF_ASSIGN_OR_RETURN(auto rz_buffers, RedzoneBuffers::FromInstruction(
                                           fusion, config, debug_opts,
                                           RedzoneBuffers::kAllInputs));
  TF_ASSIGN_OR_RETURN(auto stream, config.GetStream());
  TF_ASSIGN_OR_RETURN(std::optional<ProfilingOutput> profiling_output,
                      util.ProfileExecutable(executable.get(), stream,
                                             rz_buffers.input_buffers(),
                                             rz_buffers.input_shapes()));
  if (!profiling_output.has_value()) {
    return Internal("No output after a successful verification run.");
  }

  return std::move(profiling_output->output);
}

absl::Status CompareBuffers(const ScopedShapedBuffer& current,
                            const ScopedShapedBuffer& expected,
                            const Shape& shape, const HloModuleConfig& config,
                            se::Stream* stream) {
  BufferComparator comparator(shape, config);
  TF_ASSIGN_OR_RETURN(bool outputs_match,
                      comparator.CompareEqual(stream, current.root_buffer(),
                                              expected.root_buffer()));

  if (!outputs_match) {
    return Internal("Triton fusion output does not match emitters output.");
  }
  return absl::OkStatus();
}

absl::Status ForAllTritonFusions(
    const HloModule& module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    absl::AnyInvocable<absl::Status(const HloFusionInstruction&)> fn) {
  for (HloComputation* computation :
       module.MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      TF_ASSIGN_OR_RETURN(auto triton_fusion, AsTritonFusion(instruction));
      if (triton_fusion != nullptr) {
        TF_RETURN_IF_ERROR(fn(*triton_fusion));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace triton_fusion_numerics_pass_internal

namespace {
absl::Status VerifyTritonFusion(AutotunerCompileUtil& util,
                                const HloFusionInstruction& fusion,
                                const AutotuneConfig& config,
                                const DebugOptions& debug_opts) {
  TF_ASSIGN_OR_RETURN(auto triton_result,
                      triton_fusion_numerics_pass_internal::CompileAndRunFusion(
                          util, fusion, config, debug_opts,
                          /*clear_backend_config=*/false));
  TF_ASSIGN_OR_RETURN(auto emitters_result,
                      triton_fusion_numerics_pass_internal::CompileAndRunFusion(
                          util, fusion, config, debug_opts,
                          /*clear_backend_config=*/true));

  TF_ASSIGN_OR_RETURN(auto stream, config.GetStream());
  return triton_fusion_numerics_pass_internal::CompareBuffers(
      triton_result, emitters_result, fusion.shape(),
      fusion.GetModule()->config(), stream);
}

}  // namespace

absl::StatusOr<bool> TritonFusionNumericsVerifier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (config_.IsDeviceless()) {
    return absl::InternalError(
        "Cannot run TritonFusionNumericsVerifier on a deviceless compilation.");
  }

  const DebugOptions& debug_options = module->config().debug_options();
  TF_ASSIGN_OR_RETURN(std::optional<AutotunerCompileUtil> opt_compile_util,
                      AutotunerCompileUtil::Create(config_, debug_options));
  TF_RET_CHECK(opt_compile_util.has_value());

  TF_RETURN_IF_ERROR(triton_fusion_numerics_pass_internal::ForAllTritonFusions(
      *module, execution_threads, [&](const HloFusionInstruction& fusion) {
        return VerifyTritonFusion(*opt_compile_util, fusion, config_,
                                  debug_options);
      }));
  return false;
}

}  // namespace xla::gpu
