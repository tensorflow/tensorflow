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

#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/convert_mover.h"
#include "xla/hlo/transforms/simplifiers/dot_dimension_merger.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/hlo/transforms/simplifiers/hlo_constant_folding.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/reshape_mover.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/call_inliner.h"
#include "xla/service/dump.h"
#include "xla/service/float_support.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/autotuning/conv_algorithm_picker.h"
#include "xla/service/gpu/autotuning/gemm_algorithm_picker.h"
#include "xla/service/gpu/autotuning/gemm_fusion_autotuner.h"
#include "xla/service/gpu/buffer_sharing.h"
#include "xla/service/gpu/cublas_padding_requirements.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/llvm_gpu_backend/nvptx_backend.h"
#include "xla/service/gpu/llvm_gpu_backend/nvptx_utils.h"
#include "xla/service/gpu/metrics.h"
#include "xla/service/gpu/ptx_compile_options_from_debug_options.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/service/gpu/transforms/algebraic_simplifier.h"
#include "xla/service/gpu/transforms/block_scaling_rewriter.h"
#include "xla/service/gpu/transforms/conv_padding_legalization.h"
#include "xla/service/gpu/transforms/conv_rewriter.h"
#include "xla/service/gpu/transforms/cublas_pad_for_gemms.h"
#include "xla/service/gpu/transforms/cudnn_custom_call_compiler.h"
#include "xla/service/gpu/transforms/cudnn_fused_conv_rewriter.h"
#include "xla/service/gpu/transforms/cudnn_fusion_compiler.h"
#include "xla/service/gpu/transforms/cudnn_norm_rewriter.h"
#include "xla/service/gpu/transforms/cudnn_pad_for_convolutions.h"
#include "xla/service/gpu/transforms/cudnn_simplify_padding.h"
#include "xla/service/gpu/transforms/cudnn_vectorize_convolutions.h"
#include "xla/service/gpu/transforms/dot_sparsity_rewriter.h"
#include "xla/service/gpu/transforms/gpusolver_rewriter.h"
#include "xla/service/gpu/transforms/triangular_solve_rewriter.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/stream_executor/cuda/assemble_compilation_provider.h"
#include "xla/stream_executor/cuda/caching_compilation_provider.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/compilation_provider_options.h"
#include "xla/stream_executor/cuda/cuda_diagnostics.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/cuda/cuda_solver_context.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
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
        is_conv_bf16_supported_(cuda_compute_capability.IsAtLeast(
            se::CudaComputeCapability::kAmpere)) {}

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
            se::CudaComputeCapability::kAmpere)) {}

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

  pipeline.AddPass<GpusolverRewriter>(
      stream_executor::CudaSolverContext::Create);
  if (!hlo_module->config()
           .debug_options()
           .xla_gpu_experimental_disable_binary_libraries()) {
    pipeline.AddPass<ConvRewriter>(cuda_compute_capability, dnn_version);
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

  HloPassPipeline pre_pipeline("nvptx post-layout_assignment part 1");
  if (hlo_module->config().debug_options().xla_gpu_enable_cudnn_layer_norm() &&
      !hlo_module->config()
           .debug_options()
           .xla_gpu_experimental_disable_binary_libraries()) {
    // Rewrite normalization patterns into cuDNN Custom Calls.
    pre_pipeline.AddPass<CudnnNormRewriter>(cuda_compute_capability);
  }

  pre_pipeline.AddPass<BlockScalingRewriter>(
      /*allow_cudnn=*/cuda_compute_capability.IsAtLeastBlackwell() &&
      gpu_target_config.dnn_version_info >= se::dnn::VersionInfo(9, 7));
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

absl::StatusOr<const se::cuda::CompilationProvider*>
NVPTXCompiler::GetCompilationProvider(const DebugOptions& debug_options) {
  absl::MutexLock lock(&compilation_providers_mutex_);
  std::unique_ptr<se::cuda::CompilationProvider>& compilation_provider =
      compilation_providers_[se::cuda::CompilationProviderOptions::
                                 FromDebugOptions(debug_options)];
  if (compilation_provider == nullptr) {
    TF_ASSIGN_OR_RETURN(
        compilation_provider,
        se::cuda::AssembleCompilationProvider(
            se::cuda::CompilationProviderOptions::FromDebugOptions(
                debug_options)));
  }
  return compilation_provider.get();
}

NVPTXCompiler::NVPTXCompiler()
    : GpuCompiler(stream_executor::cuda::kCudaPlatformId, nvptx::TargetTriple(),
                  nvptx::DataLayout()) {}

HloDataflowAnalysis::CanShareBuffer NVPTXCompiler::GetCanShareBuffer(
    const se::DeviceDescription& device_description) const {
  return [&](const HloInstruction* user, const HloInstruction* operand,
             const ShapeIndex& user_index) {
    return CanShareBufferHint(user, operand, user_index, device_description);
  };
}

absl::StatusOr<GpuCompiler::BackendCompileResult>
NVPTXCompiler::CompileTargetBinary(
    const HloModuleConfig& module_config, llvm::Module* llvm_module,
    const stream_executor::DeviceDescription& device_description,
    bool relocatable, const HloModule* debug_module,
    const CompileOptions& options, std::optional<int> shard_number) {
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
    TF_ASSIGN_OR_RETURN(
        ptx, nvptx::CompileToPtx(selected_module,
                                 device_description.gpu_compute_capability(),
                                 module_config.debug_options()));

    uint64_t end_usecs = tsl::Env::Default()->NowMicros();
    // This won't record values for calls that error out (because if they error
    // out we have no way of telling how far through the process we got).
    RecordLlvmPassesAndLlvmToPtxDuration(end_usecs - start_usecs);

    if (DumpingEnabledForHloModule(debug_module ? debug_module->name() : "",
                                   module_config.debug_options())) {
      if (debug_module) {
        DumpToFileInDirOrStdout(*debug_module, "",
                                shard_number.has_value()
                                    ? (std::to_string(*shard_number) + ".ptx")
                                    : "ptx",
                                ptx);
      } else {
        LOG(ERROR)
            << "Dumping is not implemented since the file name cannot be "
               "inferred. Please implement (potentially MLIR) module -> "
               "filename heuristic.";
      }
    }
  }

  if (ptx.empty()) {
    return BackendCompileResult{};
  }

  TF_ASSIGN_OR_RETURN(const se::cuda::CompilationProvider* compilation_provider,
                      GetCompilationProvider(module_config.debug_options()));

  se::cuda::CompilationOptions compilation_options =
      PtxCompileOptionsFromDebugOptions(
          module_config.debug_options(),
          /*is_autotuning_compilation=*/options.is_autotuning_compilation);

  se::CudaComputeCapability cc = std::get<se::CudaComputeCapability>(
      device_description.gpu_compute_capability());

  // This may print multiple lines per HLO compilation because of the
  // parallelized compilation of LLVM modules.
  std::string module_name =
      debug_module != nullptr ? debug_module->name() : "(unknown)";
  XLA_SCOPED_LOGGING_TIMER_IF(
      absl::StrCat("NVPTXCompiler::CompileTargetBinary - PtxToCubin for ",
                   module_name),
      !options.is_autotuning_compilation);
  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("XlaCompileGpuAsm:#module=%s#", module_name);
  });
  tsl::profiler::TraceMe activity("PTX->CUBIN",
                                  tsl::profiler::TraceMeLevel::kInfo);

  uint64_t start_usecs = tsl::Env::Default()->NowMicros();
  const auto record_ptx_to_cubin_metric = [&]() {
    uint64_t end_usecs = tsl::Env::Default()->NowMicros();
    // This won't record values for calls that error out (because if they
    // error out we have no way of telling how far through the process we
    // got).
    RecordPtxToCubinDuration(end_usecs - start_usecs);
  };

  if (relocatable) {
    TF_ASSIGN_OR_RETURN(se::cuda::RelocatableModule relocatable_module,
                        compilation_provider->CompileToRelocatableModule(
                            cc, ptx, compilation_options));
    record_ptx_to_cubin_metric();
    return BackendCompileResult{std::move(ptx),
                                std::move(relocatable_module.cubin)};
  }

  TF_ASSIGN_OR_RETURN(
      se::cuda::Assembly assembly,
      compilation_provider->Compile(cc, ptx, compilation_options));
  record_ptx_to_cubin_metric();
  return BackendCompileResult{std::move(ptx), std::move(assembly.cubin)};
}

absl::StatusOr<bool> NVPTXCompiler::CanUseLinkModules(
    const HloModuleConfig& hlo_module_config,
    const stream_executor::DeviceDescription& device_description) {
  TF_ASSIGN_OR_RETURN(
      const se::cuda::CompilationProvider* compilation_provider,
      GetCompilationProvider(hlo_module_config.debug_options()));
  return compilation_provider->SupportsCompileAndLink() &&
         compilation_provider->SupportsCompileToRelocatableModule();
}

absl::StatusOr<std::vector<uint8_t>> NVPTXCompiler::LinkModules(
    const stream_executor::DeviceDescription& device_description,
    se::StreamExecutor* stream_exec, std::vector<std::vector<uint8_t>> modules,
    const DebugOptions& debug_options) {
  if (modules.empty()) return std::vector<uint8_t>{};

  auto cc = std::get<stream_executor::CudaComputeCapability>(
      device_description.gpu_compute_capability());

  TF_ASSIGN_OR_RETURN(const se::cuda::CompilationProvider* compilation_provider,
                      GetCompilationProvider(debug_options));

  std::vector<se::cuda::CompilationProvider::RelocatableModuleOrPtx> inputs;
  inputs.reserve(modules.size());
  for (std::vector<uint8_t>& module : modules) {
    inputs.push_back(se::cuda::RelocatableModule{std::move(module)});
  }

  se::cuda::CompilationOptions compilation_options =
      PtxCompileOptionsFromDebugOptions(debug_options,
                                        /*is_autotuning_compilation=*/false);

  VLOG(1) << "Linking " << modules.size()
          << " modules with compilation provider "
          << compilation_provider->name();
  TF_ASSIGN_OR_RETURN(
      se::cuda::Assembly assembly,
      compilation_provider->CompileAndLink(cc, inputs, compilation_options));

  return std::move(assembly.cubin);
}

std::vector<std::string> NVPTXCompiler::GetLLVMCommandLineOptions(
    const DebugOptions& debug_options) const {
  return nvptx::GetNVPTXBackendOptions(debug_options);
}
}  // namespace gpu
}  // namespace xla
