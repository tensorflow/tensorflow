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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Module.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/transforms/algebraic_simplifier.h"
#include "xla/backends/gpu/transforms/conv_padding_legalization.h"
#include "xla/backends/gpu/transforms/conv_rewriter.h"
#include "xla/backends/gpu/transforms/cublas_pad_for_gemms.h"
#include "xla/backends/gpu/transforms/cudnn_fused_conv_rewriter.h"
#include "xla/backends/gpu/transforms/triangular_solve_rewriter.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/convert_mover.h"
#include "xla/hlo/transforms/simplifiers/dot_dimension_merger.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/hlo/transforms/simplifiers/hlo_constant_folding.h"
#include "xla/hlo/transforms/simplifiers/reshape_mover.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/service/call_inliner.h"
#include "xla/service/compilation_stats.h"
#include "xla/service/compiler.h"
#include "xla/service/float_support.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/cublas_padding_requirements.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/llvm_gpu_backend/amdgpu_backend.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

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

absl::Status AMDGPUCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, const se::GpuComputeCapability& gpu_version,
    se::dnn::VersionInfo dnn_version,
    const se::SemanticVersion& toolkit_version,
    CompilationStats* compilation_stats) {
  // Convert convolutions into CustomCalls to MIOpen, then canonicalize them
  // (PadInsertion).
  HloPassPipeline pipeline("conv_canonicalization", compilation_stats);
  pipeline.AddInvariantCheckerDebug<HloVerifier>(
      /*layout_sensitive=*/false,
      /*allow_mixed_precision=*/false);

  // Convert unsupported bf16 convolutions to f32.
  ConvBfloat16Support conv_bf16_support(*gpu_version.rocm_compute_capability());
  pipeline.AddPass<FloatNormalization>(&conv_bf16_support);

  if (!hlo_module->config()
           .debug_options()
           .xla_gpu_experimental_disable_binary_libraries()) {
    pipeline.AddPass<ConvRewriter>(gpu_version);
    pipeline.AddPass<ConvPaddingLegalization>();
    auto rcc = gpu_version.rocm_compute_capability();
    pipeline.AddPass<CudnnFusedConvRewriter>(*rcc, dnn_version,
                                             toolkit_version);
  }

  // The conv padding/vectorization passes which we need to get rid of.  They
  // also leave behind unnecessary tuple/get-tuple-element pairs that
  // TupleSimplifier fixes.
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<TupleSimplifier>();

  // tf2xla bridge, DepthwiseConvolutionConverter and ConvRewriter
  // introduces reshapes and transposes that can be eliminated using
  // AlgebraicSimplifier  We run algsimp to a fixed point.
  AlgebraicSimplifierOptions algsimp_options = GetAlgebraicSimplifierOptions(
      AlgebraicSimplifierMode::kGpuConvoluationCanonicalization,
      hlo_module->config().debug_options(),
      /*is_rocm=*/true);
  pipeline.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(algsimp_options,
                                                       gpu_version);

  // tf2xla bridge, DepthwiseConvolutionConverter, ConvRewriter, and
  // CudnnSimplifyPadding introduce reshapes and transposes.  Run ReshapeMover
  // to a fixed point.  Include algsimp because ReshapeMover relies on it.
  [&, &pipeline = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
          "reshape_mover_after_conv_canonicalization")] {
    ReshapeMoverOptions reshape_mover_options;
    reshape_mover_options.reshape_of_1d_broadcast_is_cheap = true;
    pipeline.AddPass<ReshapeMover>(reshape_mover_options);
    pipeline.AddPass<GpuAlgebraicSimplifier>(algsimp_options, gpu_version);
  }();

  // The reshapes and transposes can possibly be eliminated using
  // AlgebraicSimplifier. ConvertMover and ReshapeMover fight with each other.
  // ConvertMover wants to move some converts down the graph, but ReshapeMover
  // wants to move them up the graph. We run ConvertMover and algsimp to a fixed
  // point.
  [&, &pipeline = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
          "simplify_after_conv_canonicalization")] {
    pipeline.AddPass<ConvertMover>();
    pipeline.AddPass<GpuAlgebraicSimplifier>(algsimp_options, gpu_version);
  }();

  // ConvRewriter, ConvPaddingLegalization and
  // CudnnConvPadForTensorCores may add instructions which can be simplified
  // by constant folding.
  pipeline.AddPass<HloConstantFolding>();
  TF_RETURN_IF_ERROR(
      pipeline
          .Run(hlo_module,
               /*execution_threads=*/{HloInstruction::kMainExecutionThread})
          .status());

  return absl::OkStatus();
}

absl::Status AMDGPUCompiler::OptimizeHloPostLayoutAssignment(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& options, const GpuTargetConfig& gpu_target_config,
    const GpuAliasInfo* alias_info, tsl::thread::ThreadPool* thread_pool,
    CompilationStats* compilation_stats) {
  HloPassPipeline pre_pipeline("AMDGPU post-layout_assignment part 1",
                               compilation_stats);

  pre_pipeline.AddPass<DotDimensionMerger>();

  for (const auto& req : HipblasPaddingRequirements) {
    pre_pipeline.AddPass<CublasPadForGemms>(
        gpu_target_config.device_description.gpu_compute_capability(),
        req.data_type, req.multiple_of);
  }
  // Padding a gemm operand that's a constant results in pad(constant).  Run
  // constant-folding to simplify this into a new constant.
  pre_pipeline.AddPass<HloConstantFolding>();
  TF_RETURN_IF_ERROR(
      pre_pipeline
          .Run(hlo_module,
               /*execution_threads=*/{HloInstruction::kMainExecutionThread})
          .status());

  TF_RETURN_IF_ERROR(GpuCompiler::OptimizeHloPostLayoutAssignment(
      hlo_module, stream_exec, options, gpu_target_config, alias_info,
      thread_pool, compilation_stats));

  HloPassPipeline post_pipeline("AMDGPU post-layout_assignment part 2",
                                compilation_stats);

  // Transform TriangularSolve ops into custom-calls, so we can add temp
  // memory.
  post_pipeline.AddPass<TriangularSolveRewriter>();

  TF_RETURN_IF_ERROR(
      post_pipeline
          .Run(hlo_module,
               /*execution_threads=*/{HloInstruction::kMainExecutionThread})
          .status());

  return absl::OkStatus();
}

AMDGPUCompiler::AMDGPUCompiler()
    : GpuCompiler(stream_executor::rocm::kROCmPlatformId,
                  amdgpu::TargetTriple(), amdgpu::DataLayout()) {}

absl::StatusOr<GpuCompiler::BackendCompileResult>
AMDGPUCompiler::CompileTargetBinary(
    const HloModuleConfig& module_config, llvm::Module* llvm_module,
    const se::DeviceDescription& device_description, bool relocatable,
    const HloModule* debug_module, const CompileOptions& options,
    std::optional<int> shard_number) {
  if (relocatable) {
    return Unimplemented("relocatable target binary is not implemented");
  }

  amdgpu::HsacoResult hsaco_result;
  {
    // This may print multiple lines per HLO compilation because of the
    // parallelized compilation of LLVM modules.
    XLA_SCOPED_LOGGING_TIMER_IF(
        "AMDGPUCompiler::CompileTargetBinary - CompileToHsaco",
        module_config.debug_options().xla_enable_scoped_logging_timers());
    TF_ASSIGN_OR_RETURN(
        hsaco_result,
        amdgpu::CompileToHsaco(llvm_module,
                               device_description.gpu_compute_capability(),
                               module_config.debug_options(),
                               module_config.compilation_cache_key()));
  }

  return BackendCompileResult{"", std::move(hsaco_result.hsaco),
                              /*dnn_compiled_graphs=*/{},
                              std::move(hsaco_result.module_stats)};
}

std::vector<std::string> AMDGPUCompiler::GetLLVMCommandLineOptions(
    const DebugOptions& debug_options) const {
  return amdgpu::GetAMDGPUBackendOptions(debug_options);
}
}  // namespace gpu
}  // namespace xla
