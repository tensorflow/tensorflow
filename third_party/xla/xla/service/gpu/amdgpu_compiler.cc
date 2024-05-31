/* Copyright 2017 The OpenXLA Authors.

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
#include "xla/service/gpu/amdgpu_compiler.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/IR/Module.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/call_inliner.h"
#include "xla/service/convert_mover.h"
#include "xla/service/dot_dimension_merger.h"
#include "xla/service/float_normalization.h"
#include "xla/service/float_support.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/conv_algorithm_picker.h"
#include "xla/service/gpu/cublas_pad_for_gemms.h"
#include "xla/service/gpu/cublas_padding_requirements.h"
#include "xla/service/gpu/cudnn_fused_conv_rewriter.h"
#include "xla/service/gpu/cusolver_rewriter.h"
#include "xla/service/gpu/gemm_algorithm_picker.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/gpu_conv_padding_legalization.h"
#include "xla/service/gpu/gpu_conv_rewriter.h"
#include "xla/service/gpu/gpu_sort_rewriter.h"
#include "xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/service/gpu/triangular_solve_rewriter.h"
#include "xla/service/hlo_constant_folding.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/reshape_mover.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif

namespace xla {
namespace gpu {

namespace {

class ConvBfloat16Support : public FloatSupport {
 public:
  explicit ConvBfloat16Support(const se::RocmComputeCapability& rocm)
      : FloatSupport(BF16),
        // TODO: MIOpen does not support bf16 convolutions yet
        is_conv_bf16_supported_(rocm.has_bf16_dtype_support()) {}

  bool SupportsLowPrecisionOperand(const HloInstruction& hlo,
                                   int64_t operand_index) const override {
    return (hlo.opcode() != HloOpcode::kConvolution) || is_conv_bf16_supported_;
  }

  bool SupportsLowPrecisionOutput(const HloInstruction& hlo) const override {
    return (hlo.opcode() != HloOpcode::kConvolution) || is_conv_bf16_supported_;
  }

  bool SupportsMixedPrecisions(const HloInstruction& hlo) const override {
    // Skip all HLOs other than convolutions.
    return (hlo.opcode() != HloOpcode::kConvolution);
  }

 private:
  bool is_conv_bf16_supported_;
};

}  // namespace

int32_t AMDGPUCompiler::GetToolkitVersion() const {
#if TENSORFLOW_USE_ROCM
  return TF_ROCM_VERSION;
#endif
  LOG(FATAL) << "Failed to get ROCm version.";
}

absl::Status AMDGPUCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, se::GpuComputeCapability gpu_version,
    se::dnn::VersionInfo dnn_version,
    se::DeviceMemoryAllocator* device_allocator) {
  // Convert convolutions into CustomCalls to MIOpen, then canonicalize them
  // (PadInsertion).
  HloPassPipeline pipeline("conv_canonicalization");
  pipeline.AddInvariantCheckerDebug<HloVerifier>(
      /*layout_sensitive=*/false,
      /*allow_mixed_precision=*/false);

  // Convert unsupported bf16 convolutions to f32.
  ConvBfloat16Support conv_bf16_support(
      std::get<se::RocmComputeCapability>(gpu_version));
  pipeline.AddPass<FloatNormalization>(&conv_bf16_support);

  pipeline.AddPass<GpusolverRewriter>();
  pipeline.AddPass<GpuConvRewriter>(gpu_version);
  pipeline.AddPass<GpuConvPaddingLegalization>();
  auto rcc = std::get<se::RocmComputeCapability>(gpu_version);
  pipeline.AddPass<CudnnFusedConvRewriter>(rcc, dnn_version,
                                           GetToolkitVersion());

  // The conv padding/vectorization passes which we need to get rid of.  They
  // also leave behind unnecessary tuple/get-tuple-element pairs that
  // TupleSimplifier fixes.
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<TupleSimplifier>();

  // tf2xla bridge, DepthwiseConvolutionConverter and GpuConvRewriter
  // introduces reshapes and transposes that can be eliminated using
  // AlgebraicSimplifier  We run algsimp to a fixed point.
  AlgebraicSimplifierOptions options =
      GetAlgebraicSimplifierOptions(hlo_module->config());
  options.set_enable_conv_operand_swap(false);
  options.set_enable_unconditional_reduce_of_concat_replacement(false);
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(options);

  // tf2xla bridge, DepthwiseConvolutionConverter, GpuConvRewriter, and
  // CudnnSimplifyPadding introduce reshapes and transposes.  Run ReshapeMover
  // to a fixed point.  Include algsimp because ReshapeMover relies on it.
  [&, &pipeline = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
          "reshape_mover_after_conv_canonicalization")] {
    ReshapeMoverOptions reshape_mover_options;
    reshape_mover_options.reshape_of_1d_broadcast_is_cheap = true;
    pipeline.AddPass<ReshapeMover>(reshape_mover_options);
    pipeline.AddPass<AlgebraicSimplifier>(options);
  }();

  // The reshapes and transposes can possibly be eliminated using
  // AlgebraicSimplifier. ConvertMover and ReshapeMover fight with each other.
  // ConvertMover wants to move some converts down the graph, but ReshapeMover
  // wants to move them up the graph. We run ConvertMover and algsimp to a fixed
  // point.
  [&, &pipeline = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
          "simplify_after_conv_canonicalization")] {
    pipeline.AddPass<ConvertMover>();
    pipeline.AddPass<AlgebraicSimplifier>(options);
  }();

  // GpuConvRewriter, GpuConvPaddingLegalization and
  // CudnnConvPadForTensorCores may add instructions which can be simplified
  // by constant folding.
  pipeline.AddPass<HloConstantFolding>();
  TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());

  return absl::OkStatus();
}

absl::Status AMDGPUCompiler::OptimizeHloPostLayoutAssignment(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& options, const TargetConfig& gpu_target_config,
    tsl::thread::ThreadPool* thread_pool) {
  HloPassPipeline pre_pipeline("AMDGPU post-layout_assignment part 1");

  auto rocm_compute_capability = std::get<se::RocmComputeCapability>(
      gpu_target_config.device_description.gpu_compute_capability());

  pre_pipeline.AddPass<DotDimensionMerger>();

  for (const auto& req : HipblasPaddingRequirements) {
    pre_pipeline.AddPass<CublasPadForGemms>(rocm_compute_capability,
                                            req.data_type, req.multiple_of);
  }
  // Padding a gemm operand that's a constant results in pad(constant).  Run
  // constant-folding to simplify this into a new constant.
  pre_pipeline.AddPass<HloConstantFolding>();
  TF_RETURN_IF_ERROR(pre_pipeline.Run(hlo_module).status());

  TF_RETURN_IF_ERROR(GpuCompiler::OptimizeHloPostLayoutAssignment(
      hlo_module, stream_exec, options, gpu_target_config, thread_pool));

  HloPassPipeline post_pipeline("AMDGPU post-layout_assignment part 2");

  // Transform TriangularSolve ops into custom-calls, so we can add temp
  // memory.
  post_pipeline.AddPass<TriangularSolveRewriter>();

  TF_RETURN_IF_ERROR(post_pipeline.Run(hlo_module).status());

  return absl::OkStatus();
}

// Linearize collective schedule under if online autotuning of convolutions is
// enabled.
bool AMDGPUCompiler::RequiresCollectiveScheduleLinearizer(
    const HloModule* module, se::StreamExecutor* stream_exec) {
  if (stream_exec == nullptr || !GpuConvAlgorithmPicker::IsEnabled(module)) {
    return false;
  }
  for (const HloComputation* comp : module->MakeNonfusionComputations()) {
    for (const HloInstruction* inst : comp->instructions()) {
      if (GpuConvAlgorithmPicker::IsCandidate(inst)) {
        return true;
      }
    }
  }
  // No convolution auto-tuning candidates found in the module.
  return false;
}

absl::Status AMDGPUCompiler::AddConvAndGemmAutotuningPasses(
    HloPassPipeline* pipeline, HloModule* hlo_module,
    AutotuneConfig& autotune_config, tsl::thread::ThreadPool* thread_pool) {
  if (GpuConvAlgorithmPicker::IsEnabled(hlo_module)) {
    pipeline->AddPass<GpuConvAlgorithmPicker>(autotune_config);
  }
  pipeline->AddPass<GemmAlgorithmPicker>(autotune_config);
  return absl::OkStatus();
}

absl::Status AMDGPUCompiler::AddCustomKernelReplacementPasses(
    HloPassPipeline* pipeline, const DebugOptions& debug_options) {
  if (debug_options.xla_gpu_enable_cub_radix_sort()) {
    pipeline->AddPass<GpuSortRewriter>();
  }
  return absl::OkStatus();
}

AMDGPUCompiler::AMDGPUCompiler()
    : GpuCompiler(stream_executor::rocm::kROCmPlatformId,
                  amdgpu::TargetTriple(), amdgpu::DataLayout()) {}

absl::StatusOr<GpuCompiler::BackendCompileResult>
AMDGPUCompiler::CompileTargetBinary(const HloModuleConfig& module_config,
                                    llvm::Module* llvm_module,
                                    se::GpuComputeCapability gpu_version,
                                    bool relocatable,
                                    const HloModule* debug_module,
                                    const CompileOptions& options) {
  if (relocatable) {
    return Unimplemented("relocatable target binary is not implemented");
  }

  std::vector<uint8_t> hsaco;
  {
    // This may print multiple lines per HLO compilation because of the
    // parallelized compilation of LLVM modules.
    XLA_SCOPED_LOGGING_TIMER_IF(
        "AMDGPUCompiler::CompileTargetBinary - CompileToHsaco",
        !options.is_autotuning_compilation);
    TF_ASSIGN_OR_RETURN(
        hsaco, amdgpu::CompileToHsaco(llvm_module, gpu_version,
                                      module_config.debug_options(),
                                      module_config.compilation_cache_key()));
  }

  return BackendCompileResult{"", std::move(hsaco)};
}

}  // namespace gpu
}  // namespace xla
