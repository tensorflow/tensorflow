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

#include "xla/backends/gpu/transforms/triton_fusion_numerics_verifier.h"

#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"
#include "xla/backends/gpu/autotuner/gpu_profiler.h"
#include "xla/backends/gpu/runtime/buffer_comparator.h"
#include "xla/backends/gpu/transforms/fusion_wrapper.h"
#include "xla/backends/gpu/transforms/priority_fusion.h"
#include "xla/backends/gpu/transforms/tree_reduction_rewriter.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/call_inliner.h"
#include "xla/service/compiler.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using ::mlir::MLIRContext;

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
  if (backend_config.kind() == kTritonFusionKind ||
      backend_config.kind() == kTritonNestedGemmFusionKind) {
    return fusion;
  }
  return nullptr;
}

class FusionToCallVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleFusion(HloInstruction* hlo) override {
    auto* fusion = Cast<HloFusionInstruction>(hlo);

    std::unique_ptr<HloInstruction> new_call =
        HloInstruction::CreateCall(fusion->shape(), fusion->operands(),
                                   fusion->fused_instructions_computation());
    return ReplaceWithNewInstruction(fusion, std::move(new_call));
  }
};

absl::Status InlineModuleFusions(HloModule* hlo_module) {
  // HLO module for the triton emitter might contain multiple nested fusions.
  // Other emitters might not support them, thus we need to inline all fusions.
  while (true) {
    FusionToCallVisitor visitor;
    TF_RETURN_IF_ERROR(hlo_module->entry_computation()->Accept(&visitor));
    if (!visitor.changed()) {
      return absl::OkStatus();
    }
    HloPassPipeline pipeline("inline-fusions");
    pipeline.AddPass<CallInliner>();
    pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
    pipeline.AddPass<HloDCE>();
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
    VLOG(2) << "After inline call: " << hlo_module->ToString();
  }
  return absl::OkStatus();
}

// Extracts the fusion computation and re-runs the fusion pass in order to make
// sure that the fusions are suitable for the MLIR emitters and will be
// reasonably fast. Without this the generated code can be extremely slow (e.g.
// days instead of milliseconds).
absl::StatusOr<std::unique_ptr<HloModule>> NewHloModuleFromFusionComputation(
    const HloFusionInstruction& fusion, const DebugOptions& debug_opts,
    const se::DeviceDescription& gpu_device_info, const AliasInfo* alias_info,
    MLIRContext* mlir_context) {
  std::unique_ptr<HloModule> new_module =
      ExtractComputationIntoNewModule(*fusion.fused_instructions_computation());
  new_module->mutable_config().set_debug_options(debug_opts);
  TF_RETURN_IF_ERROR(InlineModuleFusions(new_module.get()));
  TreeReductionRewriter tree_reduction_rewriter(gpu_device_info);
  TF_RETURN_IF_ERROR(tree_reduction_rewriter.Run(new_module.get()).status());
  PriorityFusion fusion_pass(
      /*thread_pool=*/nullptr, gpu_device_info, alias_info,
      HloCostAnalysis::Options{}, mlir_context);
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

absl::Status ForAllTritonFusions(
    const HloModule& module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    absl::AnyInvocable<absl::Status(const HloFusionInstruction&)> fn) {
  for (HloComputation* computation :
       module.MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      TF_ASSIGN_OR_RETURN(auto triton_fusion, AsTritonFusion(instruction));
      if (triton_fusion != nullptr) {
        VLOG(2) << "processing fusion " << triton_fusion->name();
        TF_RETURN_IF_ERROR(fn(*triton_fusion));
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<ScopedShapedBuffer> CompileAndRunFusion(
    GpuProfiler& util, const HloFusionInstruction& fusion,
    const DebugOptions& debug_opts, bool disable_triton,
    se::StreamExecutor& stream_executor, se::DeviceAddressAllocator* allocator,
    const AliasInfo* alias_info, mlir::MLIRContext* mlir_context) {
  auto extractor = [&](const DebugOptions& opts)
      -> absl::StatusOr<std::unique_ptr<HloModule>> {
    return disable_triton
               ? NewHloModuleFromFusionComputation(
                     fusion, opts, stream_executor.GetDeviceDescription(),
                     alias_info, mlir_context)
               : NewHloModuleWithTritonFromFusion(fusion, opts);
  };
  DebugOptions adjusted_debug_opts = debug_opts;
  GpuCodegenBackend::AdjustDebugOptionsForAutotuning(adjusted_debug_opts);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> new_module,
                      extractor(adjusted_debug_opts));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Compiler> compiler,
      Compiler::GetForPlatform(stream_executor.GetPlatform()->id()));

  Compiler::CompileOptions compile_options;
  compile_options.device_allocator = allocator;
  compile_options.embed_hlo_module = false;

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                      compiler->RunBackend(std::move(new_module),
                                           &stream_executor, compile_options));

  if (executable == nullptr) {
    return absl::InternalError("Failed to compile Triton fusion.");
  }

  if (debug_opts.xla_gpu_filter_kernels_spilling_registers_on_autotuning()) {
    const auto spills_registers = [](const auto& pair) {
      return pair.second.store_bytes_spilled > 0 ||
             pair.second.load_bytes_spilled > 0;
    };
    if (absl::c_any_of(executable->module_stats(), spills_registers)) {
      return absl::InternalError(
          "Failed to compile Triton fusion (kernels spill registers).");
    }
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<InputBuffers> input_buffers,
                      util.CreateInputBuffers(executable.get()));

  TF_ASSIGN_OR_RETURN(ProfileResult profile_result,
                      util.Profile(executable.get(), *input_buffers));

  if (!profile_result.output_buffer.has_value()) {
    return Internal("Profiling did not return output buffer.");
  }

  return std::move(profile_result.output_buffer).value();
}

absl::Status CompareBuffers(const ScopedShapedBuffer& current,
                            const ScopedShapedBuffer& expected,
                            const Shape& shape, const DebugOptions& debug_opts,
                            se::Stream* stream) {
  return ShapeUtil::ForEachLeafShapeWithStatus(
      shape,
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        BufferComparator comparator(subshape,
                                    debug_opts.xla_gpu_autotune_gemm_rtol());
        TF_ASSIGN_OR_RETURN(
            bool outputs_match,
            comparator.CompareEqual(stream, current.buffer(index),
                                    expected.buffer(index)));

        if (!outputs_match) {
          return Internal(
              "Triton fusion output does not match emitters output.");
        }
        return absl::OkStatus();
      });
}

}  // namespace triton_fusion_numerics_pass_internal

namespace {

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

absl::Status TritonFusionNumericsVerifier::VerifyTritonFusion(
    GpuProfiler& util, const HloFusionInstruction& fusion,
    const DebugOptions& debug_opts) {
  TF_ASSIGN_OR_RETURN(auto triton_result,
                      triton_fusion_numerics_pass_internal::CompileAndRunFusion(
                          util, fusion, debug_opts,
                          /*disable_triton=*/false, stream_executor_,
                          allocator_, alias_info_, mlir_context_));
  TF_ASSIGN_OR_RETURN(auto emitters_result,
                      triton_fusion_numerics_pass_internal::CompileAndRunFusion(
                          util, fusion, debug_opts,
                          /*disable_triton=*/true, stream_executor_, allocator_,
                          alias_info_, mlir_context_));

  auto status = util.CheckOutputBuffer(triton_result, emitters_result,
                                       debug_opts.xla_gpu_autotune_gemm_rtol());
  VLOG(2) << "CompareBuffers result: " << status;
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

absl::StatusOr<bool> TritonFusionNumericsVerifier::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(3) << "TritonFusionNumericsVerifier::RunImpl";
  if (allocator_ == nullptr) {
    owned_allocator_ =
        std::make_unique<se::StreamExecutorAddressAllocator>(&stream_executor_);
    allocator_ = owned_allocator_.get();
  }

  DebugOptions debug_options = module->config().debug_options();
  // We don't want to filter out kernels that spill registers on autotuning,
  // because we want to verify the numerics of those kernels as well.
  debug_options.set_xla_gpu_filter_kernels_spilling_registers_on_autotuning(
      false);

  ProfileOptions profile_options;
  profile_options.redzone_padding_bytes =
      debug_options.xla_gpu_redzone_padding_bytes();
  profile_options.should_init_buffers = true;

  auto profile_util =
      GpuProfiler::Create(&stream_executor_, profile_options, allocator_);
  if (profile_util == nullptr) {
    return Internal("Failed to create GpuProfiler.");
  }

  TF_RETURN_IF_ERROR(triton_fusion_numerics_pass_internal::ForAllTritonFusions(
      *module, execution_threads,
      [&](const HloFusionInstruction& fusion) -> absl::Status {
        auto key = CacheKeyForFusion(fusion);
        if (auto it = fusion_result_cache_.find(key);
            it != fusion_result_cache_.end()) {
          ++cache_hits_;
          if (!it->second.ok() && it->second.message() == kUncompilableFusion) {
            VLOG(2) << "Skipping uncompilable fusion " << fusion.name()
                    << " from cache.";
            return absl::OkStatus();
          }
          return it->second;
        }
        auto result = VerifyTritonFusion(*profile_util, fusion, debug_options);
        fusion_result_cache_[key] = result;
        if (!result.ok() && result.message() == kUncompilableFusion) {
          VLOG(2) << "Skipping uncompilable fusion " << fusion.name() << ".";
          return absl::OkStatus();
        }
        return result;
      }));
  return false;
}

}  // namespace xla::gpu
