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

#include "xla/service/gpu/nvptx_compiler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/call_inliner.h"
#include "xla/service/convert_mover.h"
#include "xla/service/dot_dimension_merger.h"
#include "xla/service/dump.h"
#include "xla/service/float_normalization.h"
#include "xla/service/float_support.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/autotuning/conv_algorithm_picker.h"
#include "xla/service/gpu/autotuning/gemm_algorithm_picker.h"
#include "xla/service/gpu/autotuning/gemm_fusion_autotuner.h"
#include "xla/service/gpu/buffer_sharing.h"
#include "xla/service/gpu/cublas_padding_requirements.h"
#include "xla/service/gpu/gpu_asm_opts_util.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "xla/service/gpu/metrics.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/service/gpu/transforms/algebraic_simplifier.h"
#include "xla/service/gpu/transforms/conv_padding_legalization.h"
#include "xla/service/gpu/transforms/conv_rewriter.h"
#include "xla/service/gpu/transforms/cublas_pad_for_gemms.h"
#include "xla/service/gpu/transforms/cudnn_custom_call_compiler.h"
#include "xla/service/gpu/transforms/cudnn_fused_conv_rewriter.h"
#include "xla/service/gpu/transforms/cudnn_fused_mha_rewriter.h"
#include "xla/service/gpu/transforms/cudnn_fused_mha_transpose_fusion.h"
#include "xla/service/gpu/transforms/cudnn_fusion_compiler.h"
#include "xla/service/gpu/transforms/cudnn_norm_rewriter.h"
#include "xla/service/gpu/transforms/cudnn_pad_for_convolutions.h"
#include "xla/service/gpu/transforms/cudnn_simplify_padding.h"
#include "xla/service/gpu/transforms/cudnn_vectorize_convolutions.h"
#include "xla/service/gpu/transforms/dot_sparsity_rewriter.h"
#include "xla/service/gpu/transforms/gpusolver_rewriter.h"
#include "xla/service/gpu/transforms/sort_rewriter.h"
#include "xla/service/gpu/transforms/triangular_solve_rewriter.h"
#include "xla/service/hlo_constant_folding.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/reshape_mover.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/stream_executor/cuda/cuda_asm_compiler.h"
#include "xla/stream_executor/cuda/cuda_diagnostics.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/cuda/nvjitlink.h"
#include "xla/stream_executor/cuda/nvjitlink_support.h"
#include "xla/stream_executor/cuda/ptx_compilation_method.h"
#include "xla/stream_executor/cuda/ptx_compiler.h"
#include "xla/stream_executor/cuda/ptx_compiler_support.h"
#include "xla/stream_executor/cuda/ptx_linking_method.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/util/env_var.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {
namespace {

class ConvBfloat16Support : public FloatSupport {
 public:
  explicit ConvBfloat16Support(
      se::dnn::VersionInfo cudnn_version,
      se::CudaComputeCapability cuda_compute_capability)
      : FloatSupport(BF16),
        is_conv_bf16_supported_((cudnn_version.major_version() > 8 ||
                                 (cudnn_version.major_version() == 8 &&
                                  cudnn_version.minor_version() >= 2)) &&
                                cuda_compute_capability.IsAtLeast(
                                    se::CudaComputeCapability::AMPERE)) {}

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

class MatmulBfloat16Support : public FloatSupport {
 public:
  explicit MatmulBfloat16Support(
      se::CudaComputeCapability cuda_compute_capability)
      : FloatSupport(BF16),
        is_matmul_bf16_supported_(cuda_compute_capability.IsAtLeast(
            se::CudaComputeCapability::AMPERE)) {}

  bool SupportsLowPrecisionOperand(const HloInstruction& hlo,
                                   int64_t operand_index) const override {
    return (hlo.opcode() != HloOpcode::kDot) || is_matmul_bf16_supported_;
  }

  bool SupportsLowPrecisionOutput(const HloInstruction& hlo) const override {
    return (hlo.opcode() != HloOpcode::kDot) || is_matmul_bf16_supported_;
  }

  bool SupportsMixedPrecisions(const HloInstruction& hlo) const override {
    return true;
  }

 private:
  bool is_matmul_bf16_supported_;
};

}  // namespace

absl::Status NVPTXCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, se::GpuComputeCapability gpu_version,
    se::dnn::VersionInfo dnn_version,
    se::DeviceMemoryAllocator* device_allocator,
    const se::SemanticVersion& toolkit_version) {
  auto cuda_compute_capability =
      std::get<se::CudaComputeCapability>(gpu_version);
  // Convert convolutions into CustomCalls to cudnn, then canonicalize them
  // (ConvPaddingLegalization). Also expand cuSolver calls.
  HloPassPipeline pipeline("conv_canonicalization");
  pipeline.AddInvariantCheckerDebug<HloVerifier>(
      /*layout_sensitive=*/false,
      /*allow_mixed_precision=*/false);

  // Convert unsupported bf16 convolutions to f32.
  ConvBfloat16Support conv_bf16_support(dnn_version, cuda_compute_capability);
  pipeline.AddPass<FloatNormalization>(&conv_bf16_support);

  // Convert unsupported bf16 matmuls to f32.
  MatmulBfloat16Support matmul_bf16_support(cuda_compute_capability);
  pipeline.AddPass<FloatNormalization>(&matmul_bf16_support);

  pipeline.AddPass<GpusolverRewriter>();
  if (!hlo_module->config()
           .debug_options()
           .xla_gpu_experimental_disable_binary_libraries()) {
    pipeline.AddPass<ConvRewriter>(cuda_compute_capability);
    pipeline.AddPass<CudnnFusedConvRewriter>(cuda_compute_capability,
                                             dnn_version, toolkit_version);
    pipeline.AddPass<ConvPaddingLegalization>();
    pipeline.AddPass<CudnnPadForConvolutions>(cuda_compute_capability);
    pipeline.AddPass<CudnnVectorizeConvolutions>(cuda_compute_capability,
                                                 dnn_version);
  }
  // The conv padding/vectorization passes which we need to get rid of.  They
  // also leave behind unnecessary tuple/get-tuple-element pairs that
  // TupleSimplifier fixes.
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<TupleSimplifier>();

  AlgebraicSimplifierOptions algsimp_options =
      GetAlgebraicSimplifierOptions(hlo_module->config());
  algsimp_options.set_supports_non_canonical_dots(false);
  algsimp_options.set_enable_conv_operand_swap(false);
  algsimp_options.set_enable_conv_add_multiply_reorder(true);
  algsimp_options.set_enable_unconditional_reduce_of_concat_replacement(false);
  pipeline.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(algsimp_options,
                                                       gpu_version);

  if (!hlo_module->config()
           .debug_options()
           .xla_gpu_experimental_disable_binary_libraries()) {
    // CudnnSimplifyPadding gets rid of some padding introduced by
    // CudnnPadForConvolutions and used by CudnnVectorizeConvolutions.  The
    // pattern-matches in this pass need to be run after inlining and
    // simplifying tuples from CudnnVectorizeConvolutions.  We also need to run
    // algsimp to e.g. clean up unnecessary nop `convert`s.
    pipeline.AddPass<CudnnSimplifyPadding>();
  }

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
  TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());

  return absl::OkStatus();
}

absl::Status NVPTXCompiler::OptimizeHloPostLayoutAssignment(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& options, const TargetConfig& gpu_target_config,
    tsl::thread::ThreadPool* thread_pool) {
  // This needs to run before GemmRewriter, which is part of
  // OptimizeHloPostLayoutAssignment().
  auto cuda_compute_capability = std::get<se::CudaComputeCapability>(
      gpu_target_config.device_description.gpu_compute_capability());

  if (hlo_module->config().debug_options().xla_gpu_enable_cudnn_fmha() &&
      !hlo_module->config()
           .debug_options()
           .xla_gpu_experimental_disable_binary_libraries()) {
    HloPassPipeline mha_fusion_pipeline(
        "nvptx cudnn multi-headed attention fusion");
    // The LayoutAssignment pass may leave behind kCopy instructions which are
    // duplicate or NOPs, so remove them with algebraic simplification and CSE.
    AlgebraicSimplifierOptions alg_sim_options =
        GetAlgebraicSimplifierOptions(hlo_module->config());
    alg_sim_options.set_supports_non_canonical_dots(false);
    alg_sim_options.set_is_layout_sensitive(true);
    alg_sim_options.set_enable_conv_operand_swap(false);
    alg_sim_options.set_enable_conv_add_multiply_reorder(true);
    // "slow" minmax means we propagate nan.
    alg_sim_options.set_minmax_propagate_nan(
        !hlo_module->config().debug_options().xla_gpu_enable_fast_min_max());
    alg_sim_options.set_enable_unconditional_reduce_of_concat_replacement(
        false);

    mha_fusion_pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true);
    se::GpuComputeCapability gpu_version =
        gpu_target_config.device_description.gpu_compute_capability();
    mha_fusion_pipeline.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(
        alg_sim_options, gpu_version);
    mha_fusion_pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true);
    // Rewrite Multi-Headed Attention modules to Fused MHA custom-calls.
    if (stream_exec) {
      mha_fusion_pipeline.AddPass<CudnnFusedMHARewriter>(
          cuda_compute_capability, stream_exec);
    } else {
      mha_fusion_pipeline.AddPass<CudnnFusedMHARewriter>(
          cuda_compute_capability, gpu_target_config.dnn_version_info);
    }
    mha_fusion_pipeline.AddPass<GpuAlgebraicSimplifier>(alg_sim_options,
                                                        gpu_version);
    mha_fusion_pipeline.AddPass<CudnnFusedMHATransposeFusion>();
    mha_fusion_pipeline.AddPass<HloDCE>();
    mha_fusion_pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true);
    TF_RETURN_IF_ERROR(mha_fusion_pipeline.Run(hlo_module).status());
  }

  HloPassPipeline pre_pipeline("nvptx post-layout_assignment part 1");
  if (hlo_module->config().debug_options().xla_gpu_enable_cudnn_layer_norm() &&
      !hlo_module->config()
           .debug_options()
           .xla_gpu_experimental_disable_binary_libraries()) {
    // Rewrite normalization patterns into cuDNN Custom Calls.
    pre_pipeline.AddPass<CudnnNormRewriter>(cuda_compute_capability);
  }

  pre_pipeline.AddPass<DotDimensionMerger>();
  pre_pipeline.AddPass<DotSparsityRewriter>();

  if (!hlo_module->config()
           .debug_options()
           .xla_gpu_experimental_disable_binary_libraries()) {
    for (const CublasPaddingRequirement& requirement :
         CublasPaddingRequirements) {
      if (cuda_compute_capability.IsAtLeast(
              requirement.min_compute_capability)) {
        pre_pipeline.AddPass<CublasPadForGemms>(cuda_compute_capability,
                                                requirement.data_type,
                                                requirement.multiple_of);
      }
    }
  }
  // Padding a gemm operand that's a constant results in pad(constant).  Run
  // constant-folding to simplify this into a new constant.
  pre_pipeline.AddPass<HloConstantFolding>();
  TF_RETURN_IF_ERROR(pre_pipeline.Run(hlo_module).status());

  TF_RETURN_IF_ERROR(GpuCompiler::OptimizeHloPostLayoutAssignment(
      hlo_module, stream_exec, options, gpu_target_config, thread_pool));

  HloPassPipeline post_pipeline("nvptx post-layout_assignment part 2");

  // Transform TriangularSolve ops into custom-calls, so we can add temp
  // memory.
  post_pipeline.AddPass<TriangularSolveRewriter>();
  TF_RETURN_IF_ERROR(post_pipeline.Run(hlo_module).status());

  return absl::OkStatus();
}

// Linearize collective schedule under if online autotuning of convolutions is
// enabled.
bool NVPTXCompiler::RequiresCollectiveScheduleLinearizer(
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

absl::Status NVPTXCompiler::AddConvAndGemmAutotuningPasses(
    HloPassPipeline* pipeline, const se::GpuComputeCapability& gpu_version,
    const CompileOptions& options, HloModule* hlo_module,
    AutotuneConfig& autotune_config, tsl::thread::ThreadPool* thread_pool) {
  if (hlo_module->config()
          .debug_options()
          .xla_gpu_experimental_disable_binary_libraries()) {
    return absl::OkStatus();
  }
  if (GpuConvAlgorithmPicker::IsEnabled(hlo_module)) {
    pipeline->AddPass<GpuConvAlgorithmPicker>(autotune_config);
  }
  // On Ampere or later, GemmAlgorithmPicker just provides a way to "warmup" the
  // execution. But we already do that during GemmFusionAutotuner pass. In that
  // case, we do a recursive compilation call that has
  // 'is_autotuning_compilation' set to true.
  if (!std::get<se::CudaComputeCapability>(gpu_version).IsAtLeastAmpere() ||
      options.is_autotuning_compilation) {
    pipeline->AddPass<GemmAlgorithmPicker>(autotune_config);
  }
  return absl::OkStatus();
}

absl::Status NVPTXCompiler::AddGemmFusionAutotuningPasses(
    HloPassPipeline* pipeline, HloModule* hlo_module,
    AutotuneConfig& autotune_config, tsl::thread::ThreadPool* thread_pool,
    const MultiProcessKeyValueStore& key_value_store,
    const se::SemanticVersion& toolkit_version) {
  pipeline->AddPass<GemmFusionAutotuner>(autotune_config, toolkit_version,
                                         thread_pool, key_value_store);
  return absl::OkStatus();
}

absl::Status NVPTXCompiler::AddCustomKernelReplacementPasses(
    HloPassPipeline* pipeline, const DebugOptions& debug_options) {
  if (debug_options.xla_gpu_enable_cub_radix_sort()) {
    pipeline->AddPass<SortRewriter>();
  }
  return absl::OkStatus();
}

absl::Status NVPTXCompiler::RunCudnnCompilerPasses(
    HloModule* module, se::StreamExecutor* stream_exec,
    BinaryMap* dnn_compiled_graphs) {
  if (module->config()
          .debug_options()
          .xla_gpu_experimental_disable_binary_libraries()) {
    return absl::OkStatus();
  }
  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("XlaCompileCudnnFusion:#module=%s,program_id=%d#",
                           module->name(), module->unique_id());
  });
  CuDnnFusionCompiler fusion_compiler(*stream_exec, *dnn_compiled_graphs);
  TF_RETURN_IF_ERROR(fusion_compiler.Run(module).status());
  CuDnnCustomCallCompiler call_compiler(*stream_exec, *dnn_compiled_graphs);
  return call_compiler.Run(module).status();
}

namespace {
// Try to load ptx from files defined in the FLAGS. If successful, return true.
bool MaybeLoadPtxFromFile(const HloModuleConfig module_config,
                          const HloModule* module, std::string* ptx) {
  // If the xla_gpu_ptx_file option is set, be explicit if a file is used
  // and warn when a file is not used to ease catching typo in filename.
  std::string prefix = xla::FilenameFor(*module, "", *ptx);
  std::string matched_filename;
  for (const std::string& full_filename :
       module_config.debug_options().xla_gpu_ptx_file()) {
    // To ease comparing many PTX versions, accept different suffixes then
    // the original filename.
    auto filename = tsl::io::Basename(full_filename);
    if (absl::StartsWith(filename, prefix)) {
      matched_filename = full_filename;
      VLOG(1) << "RunBackend() - Will load PTX from file: " << full_filename;
      break;
    }
  }
  if (!module_config.debug_options().xla_gpu_ptx_file().empty() &&
      matched_filename.empty()) {
    VLOG(1) << "RunBackend() - For module with prefix '" << prefix
            << "', we did not found a PTX file to load.";
  }

  if (!matched_filename.empty()) {
    std::ifstream ifs(matched_filename, std::ifstream::in);
    *ptx = std::string(std::istreambuf_iterator<char>(ifs),
                       std::istreambuf_iterator<char>());
    CHECK(!ptx->empty()) << "Empty or non existing PTX file: "
                         << matched_filename;
    return true;
  }
  return false;
}

// Try to load textual LLVM IR from files defined in the FLAGS. If
// successful, return the llvm::Module, otherwise return nullptr.
std::unique_ptr<llvm::Module> MaybeLoadLLVMFromFile(const HloModule* module,
                                                    llvm::Module* llvm_module) {
  // If the xla_gpu_llvm_ir_file option is set, be explicit if a file is used
  // and warn when a file is not used to ease catching typo in filename.
  if (module == nullptr) {
    return nullptr;
  }

  std::string prefix = xla::FilenameFor(*module, "", "");
  auto xla_gpu_llvm_ir_file =
      module->config().debug_options().xla_gpu_llvm_ir_file();
  auto matched_filename = absl::c_find_if(
      xla_gpu_llvm_ir_file, [prefix](const std::string& full_filename) {
        // To ease comparing many LLVM versions, accept different suffixes then
        // the original filename.
        return absl::StartsWith(tsl::io::Basename(full_filename), prefix);
      });
  if (!xla_gpu_llvm_ir_file.empty() &&
      matched_filename == std::end(xla_gpu_llvm_ir_file)) {
    VLOG(1) << "RunBackend() - For module with prefix '" << prefix
            << "', we did not found a LLVM file to load.";
  }

  if (matched_filename != std::end(xla_gpu_llvm_ir_file)) {
    VLOG(1) << "RunBackend() - Will load LLVM from file: " << *matched_filename;
    llvm::LLVMContext& context = llvm_module->getContext();
    llvm::SMDiagnostic err;
    std::unique_ptr<llvm::Module> loaded_module =
        llvm::parseIRFile(*matched_filename, err, context);

    if (!loaded_module) {
      err.print("ERR", llvm::errs());
      LOG(FATAL) << "Failed to load an LLVM file. It is probably invalid LLVM.";
    }
    // Overwrite the dumped not optimized LLVM to show which one will be used.
    llvm_ir::DumpIrIfEnabled(*module, *loaded_module, /*optimized=*/false);
    return loaded_module;
  }
  return nullptr;
}

}  // namespace

// Prints a warning if the ptx->sass JIT in the driver has known bugs.
//
// Using such a driver only a problem if we fail to use ptxas to compile our ptx
// and have to use the driver instead, so you should only call this function if
// we're going to use the driver JIT.
//
// Only prints a warning the first time it's called.
void WarnIfBadDriverJITVersion() {
  static absl::once_flag run_once;
  absl::call_once(run_once, [] {
    auto version_or_status = se::cuda::Diagnostician::FindKernelDriverVersion();
    if (!version_or_status.ok()) {
      LOG(WARNING) << "Couldn't read CUDA driver version.";
      return;
    }
    se::cuda::DriverVersion version = version_or_status.value();

    // The following versions of the driver JIT miscompile some address
    // calculations with large offsets (e.g. "load ptr + large_constant"),
    // b/70245379:
    //
    //  - 384.x before 384.108
    //  - 387.x before 387.40
    //  - 390.x before 390.10.
    //
    // In addition, only >= 396.20 contains ptxas >= 9.2.88, which contains the
    // fix for the "large multioutput fusions" miscompile, b/111107644.
    if (version < std::make_tuple(396, 20, 0)) {
      LOG(WARNING)
          << "*** WARNING *** Invoking the PTX->SASS JIT from driver version "
          << se::cuda::DriverVersionToString(version)
          << ", which is older than 396.20.0. These versions are known to "
             "miscompile XLA code, leading to incorrect results or "
             "invalid-address errors.\nXLA only uses the driver JIT if it "
             "cannot find ptxas; you don't need to update your driver if "
             "you can point XLA to ptxas 9.2.88 or newer.";
    }
  });
}

NVPTXCompiler::NVPTXCompiler()
    : GpuCompiler(stream_executor::cuda::kCudaPlatformId, nvptx::TargetTriple(),
                  nvptx::DataLayout()) {}

HloDataflowAnalysis::CanShareBuffer NVPTXCompiler::GetCanShareBuffer() const {
  return &CanShareBufferHint;
}

constexpr const uint8_t kPtxPrefix[] = {'P', 'T', 'X', ':', ' '};

absl::StatusOr<GpuCompiler::BackendCompileResult>
NVPTXCompiler::CompileTargetBinary(const HloModuleConfig& module_config,
                                   llvm::Module* llvm_module,
                                   se::GpuComputeCapability gpu_version,
                                   bool relocatable,
                                   const HloModule* debug_module,
                                   const CompileOptions& options) {
  std::unique_ptr<llvm::Module> loaded_module =
      MaybeLoadLLVMFromFile(debug_module, llvm_module);
  llvm::Module* selected_module = nullptr;
  if (loaded_module) {
    selected_module = loaded_module.get();
  } else {
    selected_module = llvm_module;
  }

  std::string ptx;
  if (!(debug_module &&
        MaybeLoadPtxFromFile(module_config, debug_module, &ptx))) {
    // This may print multiple lines per HLO compilation because of the
    // parallelized compilation of LLVM modules.
    XLA_SCOPED_LOGGING_TIMER_IF(
        absl::StrCat(
            "NVPTXCompiler::CompileTargetBinary - CompileToPtx for ",
            (debug_module != nullptr ? debug_module->name() : "(unknown")),
        !options.is_autotuning_compilation);
    uint64_t start_usecs = tsl::Env::Default()->NowMicros();
    TF_ASSIGN_OR_RETURN(ptx,
                        nvptx::CompileToPtx(selected_module, gpu_version,
                                            module_config.debug_options()));

    uint64_t end_usecs = tsl::Env::Default()->NowMicros();
    // This won't record values for calls that error out (because if they error
    // out we have no way of telling how far through the process we got).
    RecordLlvmPassesAndLlvmToPtxDuration(end_usecs - start_usecs);
  }

  TF_ASSIGN_OR_RETURN(se::PtxLinkingMethod linking_method,
                      ChooseLinkingMethod(module_config.debug_options()));

  if (linking_method == se::PtxLinkingMethod::kNvJitLink && relocatable) {
    VLOG(2) << "Deferring the PTX to CUBIN compilation of the relocatable "
               "module to the linking step.";
    std::vector<uint8_t> binary;
    if (!ptx.empty()) {
      binary.reserve(sizeof(kPtxPrefix) + ptx.size() + 1);
      binary.insert(binary.end(), kPtxPrefix, kPtxPrefix + sizeof(kPtxPrefix));
      binary.insert(binary.end(), ptx.begin(), ptx.end());
      binary.emplace_back('\0');
    }
    return BackendCompileResult{std::move(ptx), std::move(binary)};
  }

  absl::StatusOr<std::vector<uint8_t>> maybe_cubin =
      CompileGpuAsmOrGetCachedResult(
          ptx, std::get<se::CudaComputeCapability>(gpu_version), module_config,
          (debug_module != nullptr ? debug_module->name() : "(unknown)"),
          relocatable, options);

  if (!maybe_cubin.ok()) {
    return maybe_cubin.status();
  }
  return BackendCompileResult{std::move(ptx), std::move(maybe_cubin.value())};
}

using stream_executor::PtxCompilationMethod;

// Returns the supported compilation methods in the order of priority.
std::vector<PtxCompilationMethod> GetSupportedCompilationMethods() {
  std::vector<PtxCompilationMethod> methods;
  if (se::IsLibNvPtxCompilerSupported()) {
    methods.emplace_back(PtxCompilationMethod::kNvPtxCompiler);
  }
  if (se::IsLibNvJitLinkSupported()) {
    methods.emplace_back(PtxCompilationMethod::kNvJitLink);
  }
  methods.emplace_back(PtxCompilationMethod::kPtxas);
  return methods;
}

absl::StatusOr<PtxCompilationMethod> ChooseCompilationMethod(
    absl::Span<const PtxCompilationMethod> available_compilation_methods,
    const DebugOptions& debug_options, bool relocatable) {
  std::vector<PtxCompilationMethod> compilation_methods(
      available_compilation_methods.begin(),
      available_compilation_methods.end());
  VLOG(2) << "Available compilation methods: "
          << absl::StrJoin(compilation_methods, ", ");

  auto remove_compilation_method = [&](PtxCompilationMethod method) {
    auto it = absl::c_find(compilation_methods, method);
    if (it != compilation_methods.end()) {
      compilation_methods.erase(it);
    }
  };

  if (!debug_options.xla_gpu_enable_libnvjitlink()) {
    VLOG(3) << "Discarding NvJitLink since it is disabled.";
    remove_compilation_method(PtxCompilationMethod::kNvJitLink);
  }
  if (!debug_options.xla_gpu_enable_libnvptxcompiler()) {
    VLOG(3) << "Discarding NvPtxCompiler since it is disabled.";
    remove_compilation_method(PtxCompilationMethod::kNvPtxCompiler);
  }

  VLOG(2) << "Supported and enabled compilation methods: "
          << absl::StrJoin(compilation_methods, ", ");

  if (relocatable && absl::c_linear_search(compilation_methods,
                                           PtxCompilationMethod::kNvJitLink)) {
    // NvJitLink can't produce relocatable CUBINs.
    VLOG(3) << "Discarding NvJitLink since it can't produce the requested "
               "relocatable CUBIN.";
    remove_compilation_method(PtxCompilationMethod::kNvJitLink);
  }

  VLOG(2) << "Considered compilation methods: "
          << absl::StrJoin(compilation_methods, ", ");

  if (compilation_methods.empty()) {
    return absl::UnavailableError(
        "No supported compilation method is available.");
  }

  return compilation_methods.front();
}

static absl::StatusOr<std::vector<uint8_t>> AssembleOptionsAndCompile(
    const std::string& ptx, se::CudaComputeCapability cc,
    const HloModuleConfig& hlo_module_config,
    GpuCompiler::CompileOptions options, bool relocatable) {
  if (ptx.empty()) {
    return std::vector<uint8_t>();
  }

  se::GpuAsmOpts ptxas_config =
      PtxOptsFromDebugOptions(hlo_module_config.debug_options());
  if (relocatable) {
    ptxas_config.extra_flags.push_back("-c");
  }
  uint64_t start_usecs = tsl::Env::Default()->NowMicros();

  bool cancel_if_reg_spill =
      hlo_module_config.debug_options()
          .xla_gpu_filter_kernels_spilling_registers_on_autotuning() &&
      options.is_autotuning_compilation;

  std::vector<PtxCompilationMethod> supported_compilation_methods =
      GetSupportedCompilationMethods();
  TF_ASSIGN_OR_RETURN(
      PtxCompilationMethod compilation_method,
      ChooseCompilationMethod(supported_compilation_methods,
                              hlo_module_config.debug_options(), relocatable));

  VLOG(2) << "Using compilation method: " << compilation_method;

  absl::StatusOr<std::vector<uint8_t>> maybe_cubin = [&] {
    switch (compilation_method) {
      case PtxCompilationMethod::kNvJitLink:
        return se::CompileAndLinkUsingLibNvJitLink(
            cc.major, cc.minor,
            {se::NvJitLinkInput{
                se::NvJitLinkInput::Type::kPtx,
                absl::Span<const uint8_t>{
                    reinterpret_cast<const uint8_t*>(ptx.c_str()),
                    ptx.size() + 1 /* We need the null terminator. */}}},
            ptxas_config, cancel_if_reg_spill);

      case PtxCompilationMethod::kNvPtxCompiler:
        return se::CompileGpuAsmUsingLibNvPtxCompiler(
            cc.major, cc.minor, ptx.c_str(), ptxas_config, cancel_if_reg_spill);
      case PtxCompilationMethod::kPtxas:
        return se::CompileGpuAsmUsingPtxAs(cc.major, cc.minor, ptx.c_str(),
                                           ptxas_config, cancel_if_reg_spill);
    }
  }();

  if (maybe_cubin.ok()) {
    uint64_t end_usecs = tsl::Env::Default()->NowMicros();
    // This won't record values for calls that error out (because if they
    // error out we have no way of telling how far through the process we
    // got).
    RecordPtxToCubinDuration(end_usecs - start_usecs);

    VLOG(1) << "Compiled PTX size: " << ptx.size()
            << "bytes. CUBIN size: " << maybe_cubin.value().size() << "bytes.";

    return maybe_cubin;
  }

  if (maybe_cubin.status().code() == absl::StatusCode::kNotFound) {
    if (!hlo_module_config.debug_options()
             .xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found()) {
      LOG(WARNING) << nvptx::CantFindCudaMessage(
          "Can't find ptxas binary in ${CUDA_DIR}/bin.  Custom ptxas "
          "location can be specified using $PATH.",
          hlo_module_config.debug_options().xla_gpu_cuda_data_dir());
      LOG(FATAL) << "Can't find ptxas binary.  You can pass the flag "
                    "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found "
                    "to use the GPU driver for compiling ptx instead. However "
                    "this option is discouraged and can lead to increased "
                    "memory consumptions and other subtle runtime issues.";
    }

    // Missing ptxas is expected in some environments where CUDA SDK
    // binaries are not available. We don't want to spam logs with
    // identical warnings in this case.
    LOG_FIRST_N(WARNING, 1) << nvptx::CantFindCudaMessage(
        "Can't find ptxas binary in ${CUDA_DIR}/bin.  Will back to "
        "the GPU driver for PTX -> sass compilation.  This is OK so "
        "long as you don't see a warning below about an out-of-date "
        "driver version. Custom ptxas location can be specified "
        "using $PATH.",
        hlo_module_config.debug_options().xla_gpu_cuda_data_dir());

    // We're going to use the driver to JIT our PTX->SASS, so warn if
    // the JIT in the driver has known bugs.
    WarnIfBadDriverJITVersion();
    return maybe_cubin;
  }

  if (maybe_cubin.status().code() == absl::StatusCode::kCancelled) {
    return maybe_cubin;
  }

  if (maybe_cubin.status().code() == absl::StatusCode::kResourceExhausted) {
    return maybe_cubin;
  }

  if (maybe_cubin.status().code() != absl::StatusCode::kUnimplemented) {
    return AppendStatus(
        maybe_cubin.status(),
        "If the error message indicates that a file could not be written, "
        "please verify that sufficient filesystem space is provided.");
  }

  return maybe_cubin;
}

absl::StatusOr<std::vector<uint8_t>>
NVPTXCompiler::CompileGpuAsmOrGetCachedResult(
    const std::string& ptx, se::CudaComputeCapability cc,
    const HloModuleConfig& hlo_module_config, absl::string_view module_name,
    bool relocatable, const CompileOptions& options) {
  // This may print multiple lines per HLO compilation because of the
  // parallelized compilation of LLVM modules.
  XLA_SCOPED_LOGGING_TIMER_IF(
      absl::StrCat("NVPTXCompiler::CompileGpuAsmOrGetCachedResult for ",
                   module_name),
      !options.is_autotuning_compilation);
  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("XlaCompileGpuAsm:#module=%s#", module_name);
  });
  tsl::profiler::TraceMe activity("PTX->CUBIN",
                                  tsl::profiler::TraceMeLevel::kInfo);
  CompilationCacheValue* cache_value = nullptr;
  bool inserted = [&] {
    auto flags = CompilationCacheFlags{
        hlo_module_config.debug_options()
            .xla_gpu_filter_kernels_spilling_registers_on_autotuning()};
    absl::MutexLock lock(&mutex_);
    auto [iter, inserted] = compilation_cache_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(ptx, cc.major, cc.minor, relocatable, flags),
        std::forward_as_tuple());
    // Do not move this assignment outside of the critical section. There is
    // a TOCTOU if `compilation_cache_` is rehashed before the iterator is used.
    cache_value = &iter->second;
    return inserted;
  }();

  // Compile the ptx if it wasn't in the cache before we called this function.
  // Other threads asking for the same compilation key will block on
  // cache_value->mutex_ until compilation is done.
  absl::MutexLock lock(&cache_value->mutex);
  if (inserted) {
    CHECK(!cache_value->compilation_done);
    absl::Cleanup mark_compilation_as_done = [cache_value] {
      // Note that we will set this to true also in the error case, so that we
      // don't retry this compilation.
      cache_value->compilation_done = true;
      cache_value->compilation_done_cv.SignalAll();
    };

    cache_value->maybe_cubin = AssembleOptionsAndCompile(
        ptx, cc, hlo_module_config, options, relocatable);
    return cache_value->maybe_cubin;
  }

  while (!cache_value->compilation_done) {
    cache_value->compilation_done_cv.Wait(&cache_value->mutex);
  }

  return cache_value->maybe_cubin;
}

static bool IsNvlinkEnabled() {
  const bool use_nvlink_by_default =
#ifdef TF_DISABLE_NVLINK_BY_DEFAULT
      false;
#else
      true;
#endif
  bool use_nvlink;
  TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_USE_NVLINK_FOR_PARALLEL_COMPILATION",
                                      /*default_val=*/
                                      use_nvlink_by_default, &use_nvlink));
  return use_nvlink;
}

// Returns the version of the PTX compiler that will be used for the given
// debug options and preferred CUDA directory (Either libnvptxcompiler or ptxas)
static absl::StatusOr<stream_executor::SemanticVersion> GetAsmCompilerVersion(
    const DebugOptions& debug_options, const std::string& preferred_cuda_dir) {
  if (debug_options.xla_gpu_enable_libnvptxcompiler() &&
      se::IsLibNvPtxCompilerSupported()) {
    return stream_executor::GetLibNvPtxCompilerVersion();
  }

  return se::GetAsmCompilerVersion(preferred_cuda_dir);
}

absl::StatusOr<se::PtxLinkingMethod> NVPTXCompiler::ChooseLinkingMethod(
    const DebugOptions& debug_options) {
  se::GpuAsmOpts ptxas_config = PtxOptsFromDebugOptions(debug_options);
  std::string& preferred_cuda_dir = ptxas_config.preferred_cuda_dir;

  using LinkingMethod = se::PtxLinkingMethod;

  if (stream_executor::IsLibNvJitLinkSupported() &&
      debug_options.xla_gpu_enable_libnvjitlink()) {
    return se::PtxLinkingMethod::kNvJitLink;
  }

  TF_ASSIGN_OR_RETURN(auto asm_compiler_version,
                      GetAsmCompilerVersion(debug_options, preferred_cuda_dir));

  auto nvlink_version = stream_executor::GetNvLinkVersion(preferred_cuda_dir);
  if (IsNvlinkEnabled() && nvlink_version.ok() &&
      nvlink_version.value() >= asm_compiler_version) {
    return LinkingMethod::kNvLink;
  }

  int ptxas_version =
      asm_compiler_version.major() * 1000 + asm_compiler_version.minor() * 10;
  TF_ASSIGN_OR_RETURN(int driver_version,
                      se::gpu::GpuDriver::GetDriverVersion());

  if (driver_version >= ptxas_version) {
    return LinkingMethod::kDriver;
  }

  LOG_FIRST_N(WARNING, 1)
      << "The NVIDIA driver's CUDA version is "
      << absl::StrFormat("%d.%d", driver_version / 1000,
                         (driver_version % 1000) / 10)
      << " which is older than the PTX compiler version "
      << asm_compiler_version
      << ". Because the driver is older than the PTX compiler version, XLA is "
         "disabling parallel compilation, which may slow down compilation. "
         "You should update your NVIDIA driver or use the NVIDIA-provided "
         "CUDA forward compatibility packages.";

  return se::PtxLinkingMethod::kNone;
}

absl::StatusOr<bool> NVPTXCompiler::CanUseLinkModules(
    const HloModuleConfig& hlo_module_config) {
  // TODO(phawkins): rather than comparing version numbers, it might be more
  // robust if we simply tried to link something the first time we compile.
  TF_ASSIGN_OR_RETURN(se::PtxLinkingMethod linking_method,
                      ChooseLinkingMethod(hlo_module_config.debug_options()));
  return linking_method != se::PtxLinkingMethod::kNone;
}

absl::StatusOr<std::vector<uint8_t>> NVPTXCompiler::LinkModules(
    se::GpuComputeCapability compute_capability,
    se::StreamExecutor* stream_exec, std::vector<std::vector<uint8_t>> modules,
    const DebugOptions& debug_options) {
  if (modules.empty()) return std::vector<uint8_t>{};

  auto cc =
      std::get<stream_executor::CudaComputeCapability>(compute_capability);

  TF_ASSIGN_OR_RETURN(se::PtxLinkingMethod linking_method,
                      ChooseLinkingMethod(debug_options));
  VLOG(1) << "Linking " << modules.size()
          << " modules with linking method: " << linking_method;

  if (linking_method == se::PtxLinkingMethod::kNvJitLink) {
    const auto module_contains_ptx =
        [](const std::vector<uint8_t>& module) -> bool {
      return module.size() >= sizeof(kPtxPrefix) &&
             std::equal(std::begin(kPtxPrefix), std::end(kPtxPrefix),
                        std::begin(module));
    };

    std::vector<stream_executor::NvJitLinkInput> nvjitlink_inputs;
    nvjitlink_inputs.reserve(modules.size());
    for (std::vector<uint8_t>& module : modules) {
      if (module_contains_ptx(module)) {
        nvjitlink_inputs.push_back(
            {se::NvJitLinkInput::Type::kPtx,
             absl::Span<const uint8_t>(module).subspan(sizeof(kPtxPrefix))});
      } else {
        nvjitlink_inputs.push_back({se::NvJitLinkInput::Type::kCubin, module});
      }
    }

    se::GpuAsmOpts ptxas_config = PtxOptsFromDebugOptions(debug_options);
    return stream_executor::CompileAndLinkUsingLibNvJitLink(
        cc.major, cc.minor, nvjitlink_inputs, ptxas_config,
        /*cancel_if_reg_spill=*/false);
  }

  std::vector<stream_executor::CubinOrPTXImage> cubin_images;
  cubin_images.reserve(modules.size());
  for (std::vector<uint8_t>& module : modules) {
    {
      std::string profile = absl::StrCat("sm_", cc.major, cc.minor);
      cubin_images.push_back({std::move(profile), std::move(module)});
    }
  }

  if (linking_method == se::PtxLinkingMethod::kNvLink) {
    return LinkUsingNvlink(cc, debug_options.xla_gpu_cuda_data_dir(),
                           cubin_images);
  }
  return LinkGpuAsm(cc, se::gpu::ExtractGpuExecutor(stream_exec)->gpu_context(),
                    cubin_images);
}

}  // namespace gpu
}  // namespace xla
