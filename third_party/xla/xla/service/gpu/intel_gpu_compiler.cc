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

#include "xla/service/gpu/intel_gpu_compiler.h"

#include "absl/status/status_macros.h"
#include "xla/backends/gpu/transforms/algebraic_simplifier.h"
#include "xla/backends/gpu/transforms/conv_padding_legalization.h"
#include "xla/backends/gpu/transforms/conv_rewriter.h"
#include "xla/backends/gpu/transforms/cudnn_fused_conv_rewriter.h"
#include "xla/backends/gpu/transforms/sycl_gemm_workspace_pass.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/convert_mover.h"
#include "xla/hlo/transforms/simplifiers/hlo_constant_folding.h"
#include "xla/hlo/transforms/simplifiers/reshape_mover.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/service/call_inliner.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/llvm_gpu_backend/spirv_backend.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo_verifier.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

namespace xla {
namespace gpu {

IntelGpuCompiler::IntelGpuCompiler()
    : GpuCompiler(stream_executor::sycl::kSyclPlatformId, spir::TargetTriple(),
                  spir::DataLayout()) {}

absl::Status IntelGpuCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, const se::GpuComputeCapability& gpu_version,
    se::dnn::VersionInfo dnn_version,
    const se::SemanticVersion& toolkit_version,
    CompilationStats* compilation_stats) {
  HloPassPipeline pipeline("conv_canonicalization", compilation_stats);
  pipeline.AddInvariantCheckerDebug<HloVerifier>(
      /*layout_sensitive=*/false,
      /*allow_mixed_precision=*/false);

  if (!hlo_module->config()
           .debug_options()
           .xla_gpu_experimental_disable_binary_libraries()) {
    pipeline.AddPass<ConvRewriter>(gpu_version);
    pipeline.AddPass<CudnnFusedConvRewriter>(gpu_version, dnn_version,
                                             toolkit_version);
    // ConvPaddingLegalization converts base dilation into explicit padding
    // operations, which is required for backends that don't support base
    // dilation natively.
    pipeline.AddPass<ConvPaddingLegalization>();
  }

  // ConvPaddingLegalization leaves behind unnecessary tuple/get-tuple-element
  // pairs that TupleSimplifier fixes.
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<TupleSimplifier>();

  // ConvRewriter introduces reshapes and transposes that can be eliminated
  // using AlgebraicSimplifier. We run algsimp to a fixed point.
  AlgebraicSimplifierOptions algsimp_options = GetAlgebraicSimplifierOptions(
      AlgebraicSimplifierMode::kGpuConvoluationCanonicalization,
      hlo_module->config().debug_options(),
      /*is_non_cuda_platform=*/true);
  pipeline.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(algsimp_options,
                                                       gpu_version);

  // Run ReshapeMover to a fixed point. Include algsimp because ReshapeMover
  // relies on it.
  [&, &pipeline = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
          "reshape_mover_after_conv_canonicalization")] {
    ReshapeMoverOptions reshape_mover_options;
    reshape_mover_options.reshape_of_1d_broadcast_is_cheap = true;
    pipeline.AddPass<ReshapeMover>(reshape_mover_options);
    pipeline.AddPass<GpuAlgebraicSimplifier>(algsimp_options, gpu_version);
  }();

  // ConvertMover and ReshapeMover fight with each other, so we run ConvertMover
  // and algsimp to a fixed point.
  [&, &pipeline = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
          "simplify_after_conv_canonicalization")] {
    pipeline.AddPass<ConvertMover>();
    pipeline.AddPass<GpuAlgebraicSimplifier>(algsimp_options, gpu_version);
  }();

  pipeline.AddPass<HloConstantFolding>();
  ABSL_RETURN_IF_ERROR(
      pipeline
          .Run(hlo_module,
               /*execution_threads=*/{HloInstruction::kMainExecutionThread})
          .status());
  return absl::OkStatus();
}

absl::Status IntelGpuCompiler::OptimizeHloPostLayoutAssignment(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& options, const GpuTopology& gpu_topology,
    const GpuAliasInfo* alias_info, tsl::thread::ThreadPool* thread_pool,
    CompilationStats* compilation_stats, mlir::MLIRContext* mlir_context) {
  ABSL_RETURN_IF_ERROR(GpuCompiler::OptimizeHloPostLayoutAssignment(
      hlo_module, stream_exec, options, gpu_topology, alias_info, thread_pool,
      compilation_stats, mlir_context));

  HloPassPipeline post_pipeline("Intel GPU post-layout_assignment",
                                compilation_stats);
  post_pipeline.AddPass<SyclGemmWorkspacePass>(
      gpu_topology.gpu_target_config()
          .device_description.gpu_compute_capability());
  return post_pipeline.Run(hlo_module).status();
}

absl::Status IntelGpuCompiler::AddConfigAssignerPass(
    HloPassPipeline* pipeline, HloModule* hlo_module,
    const se::GpuComputeCapability& gpu_version, const CompileOptions& options,
    tsl::thread::ThreadPool* thread_pool,
    stream_executor::StreamExecutor* stream_executor,
    const GpuTargetConfig* target_config, const AliasInfo* alias_info,
    mlir::MLIRContext* mlir_context,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn,
    const MultiProcessKeyValueStore& key_value_store) {
  // Return OkStatus as a stub.
  return absl::OkStatus();
}

absl::StatusOr<GpuCompiler::BackendCompileResult>
IntelGpuCompiler::CompileTargetBinary(
    const HloModuleConfig& module_config, llvm::Module* llvm_module,
    const stream_executor::DeviceDescription& device_description,
    bool relocatable, const HloModule* debug_module,
    std::optional<int> shard_number) {
  ABSL_ASSIGN_OR_RETURN(auto spirv_str,
                   spirv::CompileToSPIRV(
                       llvm_module, device_description.gpu_compute_capability(),
                       module_config.debug_options()));
  if (DumpingEnabledForHloModule(debug_module ? debug_module->name() : "",
                                 module_config.debug_options()) &&
      DumpingEnabledForEmitter("spirv", module_config.debug_options())) {
    if (debug_module) {
      DumpToFileInDirOrStdout(*debug_module, "",
                              shard_number.has_value()
                                  ? (std::to_string(*shard_number) + ".spv")
                                  : "spv",
                              spirv_str);
    } else {
      LOG(ERROR) << "Dumping is not implemented since the file name cannot be "
                    "inferred. Please implement (potentially MLIR) module -> "
                    "filename heuristic.";
    }
  }
  std::vector<uint8_t> spirv_bin(spirv_str.begin(), spirv_str.end());
  return BackendCompileResult{/*asm_text=*/"", std::move(spirv_bin)};
}

std::vector<std::string> IntelGpuCompiler::GetLLVMCommandLineOptions(
    const DebugOptions& debug_options) const {
  return spirv::GetSPIRVBackendOptions(debug_options);
}

}  // namespace gpu
}  // namespace xla
