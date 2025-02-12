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

#include "xla/service/gpu/transforms/triton_fusion_numerics_verifier.h"

#include <memory>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuning/autotuner_compile_util.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/buffer_comparator.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/transforms/fusion_wrapper.h"
#include "xla/service/gpu/transforms/priority_fusion.h"
#include "xla/service/gpu/transforms/tree_reduction_rewriter.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

namespace {

using ProfilingOutput = AutotunerCompileUtil::ProfilingOutput;

// Returns the input instruction as a fusion instruction, if it represents a
// Triton fusion. Otherwise, returns nullptr.
absl::StatusOr<const HloFusionInstruction*> AsTritonFusion(
    const HloInstruction* hlo) {
  if (HloPredicateIsNotOp<HloOpcode::kFusion>(hlo)) {
    return nullptr;
  }
  const HloFusionInstruction* fusion = Cast<HloFusionInstruction>(hlo);
  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      fusion->backend_config<GpuBackendConfig>());
  const FusionBackendConfig& backend_config =
      gpu_config.fusion_backend_config();
  if (backend_config.kind() == kTritonFusionKind) {
    return fusion;
  }
  return nullptr;
}

// Extracts the fusion, disables Triton, and re-runs the fusion pass in order
// to make sure that the fusions are suitable for the MLIR emitters and will be
// reasonably fast. Without this the generated code can be extremely slow (e.g.
// days instead of milliseconds).
absl::StatusOr<std::unique_ptr<HloModule>> NewHloModuleWithoutTritonFromFusion(
    const HloFusionInstruction& fusion, const DebugOptions& debug_opts,
    const se::DeviceDescription& gpu_device_info) {
  std::unique_ptr<HloModule> new_module =
      ExtractComputationIntoNewModule(*fusion.fused_instructions_computation());
  new_module->mutable_config().set_debug_options(debug_opts);
  new_module->mutable_config()
      .mutable_debug_options()
      .add_xla_disable_hlo_passes("triton-softmax-rewriter");

  TreeReductionRewriter tree_reduction_rewriter(gpu_device_info);
  TF_RETURN_IF_ERROR(tree_reduction_rewriter.Run(new_module.get()).status());

  PriorityFusion fusion_pass(
      /*thread_pool=*/nullptr, gpu_device_info, HloCostAnalysis::Options{});
  TF_RETURN_IF_ERROR(fusion_pass.Run(new_module.get()).status());

  // If the priority fusion pass above skipped some instructions, turn them
  // into fusions.
  FusionWrapper fusion_wrapper(gpu_device_info);
  TF_RETURN_IF_ERROR(fusion_wrapper.Run(new_module.get()).status());

  return new_module;
}

std::unique_ptr<HloModule> NewHloModuleWithTritonFromFusion(
    const HloFusionInstruction& fusion, const DebugOptions& debug_opts) {
  std::unique_ptr<HloModule> new_module =
      ExtractInstructionIntoNewModule(fusion);
  new_module->mutable_config().set_debug_options(debug_opts);
  return new_module;
}

}  // namespace

namespace triton_fusion_numerics_pass_internal {

absl::StatusOr<ScopedShapedBuffer> CompileAndRunFusion(
    AutotunerCompileUtil& util, const HloFusionInstruction& fusion,
    const AutotuneConfig& config, const DebugOptions& debug_opts,
    bool disable_triton) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      util.Compile([&](const DebugOptions& opts) {
        return disable_triton ? NewHloModuleWithoutTritonFromFusion(
                                    fusion, opts, config.GetDeviceDescription())
                              : NewHloModuleWithTritonFromFusion(fusion, opts);
      }));
  if (executable == nullptr) {
    return Internal("Failed to compile Triton fusion.");
  }

  TF_ASSIGN_OR_RETURN(auto rz_buffers, RedzoneBuffers::FromInstruction(
                                           fusion, config, debug_opts,
                                           RedzoneBuffers::kAllInputs));
  TF_ASSIGN_OR_RETURN(auto stream, config.GetStream());
  TF_ASSIGN_OR_RETURN(ProfilingOutput profiling_output,
                      util.ProfileExecutable(executable.get(), stream,
                                             rz_buffers.input_buffers(),
                                             rz_buffers.input_shapes()));

  return std::move(profiling_output).output;
}

absl::Status CompareBuffers(const ScopedShapedBuffer& current,
                            const ScopedShapedBuffer& expected,
                            const Shape& shape, const HloModuleConfig& config,
                            se::Stream* stream) {
  BufferComparator comparator(
      shape, config.debug_options().xla_gpu_autotune_gemm_rtol());
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
                          /*disable_triton=*/false));
  TF_ASSIGN_OR_RETURN(auto emitters_result,
                      triton_fusion_numerics_pass_internal::CompileAndRunFusion(
                          util, fusion, config, debug_opts,
                          /*disable_triton=*/true));

  TF_ASSIGN_OR_RETURN(auto stream, config.GetStream());
  auto status = triton_fusion_numerics_pass_internal::CompareBuffers(
      triton_result, emitters_result, fusion.shape(),
      fusion.GetModule()->config(), stream);

  if (!status.ok()) {
    LOG(ERROR) << "Triton numerics verification failed with: "
               << status.message();

    DumpToFileInDirOrStdout(
        *fusion.GetModule(),
        /*file_prefix=*/"",
        /*file_suffix=*/
        absl::StrCat("triton_fusion_numerics_verifier_failed_",
                     fusion.unique_id(), ".hlo"),
        /*contents=*/
        ExtractInstructionIntoNewModule(fusion)->ToString());
  }

  return status;
}

TritonFusionNumericsVerifier::FusionCacheKey CacheKeyForFusion(
    const HloFusionInstruction& fusion) {
  std::unique_ptr<HloModule> module = ExtractInstructionIntoNewModule(fusion);
  HloPrintOptions print_options = HloPrintOptions::ModuleFingerprint()
                                      .set_print_only_essential_constants(false)
                                      .set_print_backend_config(true)
                                      .set_sort_backend_config(true);
  return module->ToString(print_options);
}

}  // namespace

absl::StatusOr<bool> TritonFusionNumericsVerifier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (config_.IsDeviceless()) {
    return absl::InternalError(
        "Cannot run TritonFusionNumericsVerifier on a deviceless compilation.");
  }

  DebugOptions debug_options = module->config().debug_options();

  // We don't want to filter out kernels that spill registers on autotuning,
  // because we want to verify the numerics of those kernels as well.
  debug_options.set_xla_gpu_filter_kernels_spilling_registers_on_autotuning(
      false);

  TF_ASSIGN_OR_RETURN(AutotunerCompileUtil compile_util,
                      AutotunerCompileUtil::Create(config_, debug_options));

  TF_RETURN_IF_ERROR(triton_fusion_numerics_pass_internal::ForAllTritonFusions(
      *module, execution_threads, [&](const HloFusionInstruction& fusion) {
        auto key = CacheKeyForFusion(fusion);
        if (auto it = fusion_result_cache_.find(key);
            it != fusion_result_cache_.end()) {
          ++cache_hits_;
          return it->second;
        }
        auto result =
            VerifyTritonFusion(compile_util, fusion, config_, debug_options);
        fusion_result_cache_[key] = result;
        return result;
      }));
  return false;
}

}  // namespace xla::gpu
