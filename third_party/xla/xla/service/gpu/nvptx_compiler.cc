/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/call_inliner.h"
#include "xla/service/convert_mover.h"
#include "xla/service/dot_dimension_merger.h"
#include "xla/service/dump.h"
#include "xla/service/float_normalization.h"
#include "xla/service/float_support.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/buffer_sharing.h"
#include "xla/service/gpu/conv_algorithm_picker.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/cublas_pad_for_gemms.h"
#include "xla/service/gpu/cublas_padding_requirements.h"
#include "xla/service/gpu/cudnn_fused_conv_rewriter.h"
#include "xla/service/gpu/cudnn_fused_mha_rewriter.h"
#include "xla/service/gpu/cudnn_fused_mha_transpose_fusion.h"
#include "xla/service/gpu/cudnn_pad_for_convolutions.h"
#include "xla/service/gpu/cudnn_simplify_padding.h"
#include "xla/service/gpu/cudnn_vectorize_convolutions.h"
#include "xla/service/gpu/cusolver_rewriter.h"
#include "xla/service/gpu/gemm_algorithm_picker.h"
#include "xla/service/gpu/gpu_asm_opts_util.h"
#include "xla/service/gpu/gpu_conv_padding_legalization.h"
#include "xla/service/gpu/gpu_conv_rewriter.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "xla/service/gpu/metrics.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/service/gpu/triangular_solve_rewriter.h"
#include "xla/service/gpu/triton_autotuner.h"
#include "xla/service/hlo_constant_folding.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/layout_normalization.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/reshape_decomposer.h"
#include "xla/service/reshape_mover.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/stream_executor/cuda/cuda_diagnostics.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/asm_compiler.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_internal.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/util/env_var.h"

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

}  // namespace

Status NVPTXCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, se::GpuComputeCapability gpu_version,
    se::dnn::VersionInfo dnn_version,
    se::DeviceMemoryAllocator* device_allocator) {
  auto cuda_compute_capability =
      std::get<se::CudaComputeCapability>(gpu_version);
  // Convert convolutions into CustomCalls to cudnn, then canonicalize them
  // (GpuConvPaddingLegalization). Also expand cuSolver calls.
  HloPassPipeline pipeline("conv_canonicalization");
  pipeline.AddInvariantCheckerDebug<HloVerifier>(
      /*layout_sensitive=*/false,
      /*allow_mixed_precision=*/false);

  // Convert upsupported bf16 convolutions to f32.
  ConvBfloat16Support conv_bf16_support(dnn_version, cuda_compute_capability);
  pipeline.AddPass<FloatNormalization>(&conv_bf16_support);

  pipeline.AddPass<GpusolverRewriter>();
  pipeline.AddPass<GpuConvRewriter>();
  pipeline.AddPass<CudnnFusedConvRewriter>(cuda_compute_capability);
  pipeline.AddPass<GpuConvPaddingLegalization>();
  pipeline.AddPass<CudnnPadForConvolutions>(cuda_compute_capability);
  pipeline.AddPass<CudnnVectorizeConvolutions>(cuda_compute_capability,
                                               dnn_version);
  // The conv padding/vectorization passes which we need to get rid of.  They
  // also leave behind unnecessary tuple/get-tuple-element pairs that
  // TupleSimplifier fixes.
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<TupleSimplifier>();

  AlgebraicSimplifierOptions algsimp_options;
  algsimp_options.set_enable_conv_operand_swap(false);
  algsimp_options.set_enable_unconditional_reduce_of_concat_replacement(false);
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(algsimp_options);

  // CudnnSimplifyPadding gets rid of some padding introduced by
  // CudnnPadForConvolutions and used by CudnnVectorizeConvolutions.  The
  // pattern-matches in this pass need to be run after inlining and simplifying
  // tuples from CudnnVectorizeConvolutions.  We also need to run algsimp to
  // e.g. clean up unnecessary nop `convert`s.
  pipeline.AddPass<CudnnSimplifyPadding>();

  // tf2xla bridge, DepthwiseConvolutionConverter, GpuConvRewriter, and
  // CudnnSimplifyPadding introduce reshapes and transposes.  Run ReshapeMover
  // to a fixed point.  Include algsimp because ReshapeMover relies on it.
  [&, &pipeline = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
          "reshape_mover_after_conv_canonicalization")] {
    ReshapeMoverOptions reshape_mover_options;
    reshape_mover_options.reshape_of_1d_broadcast_is_cheap = true;
    pipeline.AddPass<HloPassFix<ReshapeMover>>(reshape_mover_options);
    pipeline.AddPass<AlgebraicSimplifier>(algsimp_options);
  }();

  // The reshapes and transposes can possibly be eliminated using
  // AlgebraicSimplifier. ConvertMover and ReshapeMover fight with each other.
  // ConvertMover wants to move some converts down the graph, but ReshapeMover
  // wants to move them up the graph. We run ConvertMover and algsimp to a fixed
  // point.
  [&, &pipeline = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
          "simplify_after_conv_canonicalization")] {
    pipeline.AddPass<ConvertMover>();
    pipeline.AddPass<AlgebraicSimplifier>(algsimp_options);
  }();

  // GpuConvRewriter, GpuConvPaddingLegalization and
  // CudnnConvPadForTensorCores may add instructions which can be simplified
  // by constant folding.
  pipeline.AddPass<HloConstantFolding>();
  TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());

  return OkStatus();
}

Status NVPTXCompiler::OptimizeHloPostLayoutAssignment(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& options, const TargetConfig& gpu_target_config,
    tsl::thread::ThreadPool* thread_pool) {
  HloPassPipeline pre_pipeline("nvptx post-layout_assignment part 1");

  // This needs to run before GemmRewriter, which is part of
  // OptimizeHloPostLayoutAssignment().
  auto cuda_compute_capability = std::get<se::CudaComputeCapability>(
      gpu_target_config.device_description.gpu_compute_capability());

  if (hlo_module->config().debug_options().xla_gpu_enable_cudnn_fmha()) {
    HloPassPipeline mha_fusion_pipeline(
        "nvptx cudnn multi-headed attention fusion");
    const DebugOptions& debug_options = hlo_module->config().debug_options();
    // The LayoutAssignment pass may leave behind kCopy instructions which are
    // duplicate or NOPs, so remove them with algebraic simplification and CSE.
    AlgebraicSimplifierOptions alg_sim_options;
    alg_sim_options.set_supports_non_canonical_dots(false);
    alg_sim_options.set_is_layout_sensitive(true);
    alg_sim_options.set_enable_conv_operand_swap(false);
    // "slow" minmax means we propagate nan.
    alg_sim_options.set_minmax_propagate_nan(
        !hlo_module->config().debug_options().xla_gpu_enable_fast_min_max());
    alg_sim_options.set_enable_unconditional_reduce_of_concat_replacement(
        false);
    if (debug_options.xla_gpu_normalize_layouts()) {
      mha_fusion_pipeline.AddPass<ReshapeDecomposer>();
      mha_fusion_pipeline.AddPass<LayoutNormalization>();
    }
    mha_fusion_pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true);
    mha_fusion_pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(
        alg_sim_options);
    mha_fusion_pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true);

    // Rewrite Multi-Headed Attention modules to Fused MHA custom-calls.
    if (stream_exec) {
      mha_fusion_pipeline.AddPass<CudnnFusedMHARewriter>(
          cuda_compute_capability, stream_exec);
    } else {
      mha_fusion_pipeline.AddPass<CudnnFusedMHARewriter>(
          cuda_compute_capability, gpu_target_config.dnn_version_info);
    }
    mha_fusion_pipeline.AddPass<AlgebraicSimplifier>(alg_sim_options);
    mha_fusion_pipeline.AddPass<CudnnFusedMHATransposeFusion>();
    mha_fusion_pipeline.AddPass<HloDCE>();
    mha_fusion_pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true);
    TF_RETURN_IF_ERROR(mha_fusion_pipeline.Run(hlo_module).status());
  }

  pre_pipeline.AddPass<DotDimensionMerger>();

  for (const CublasPaddingRequirement& requirement :
       CublasPaddingRequirements) {
    if (cuda_compute_capability.IsAtLeast(requirement.min_compute_capability)) {
      pre_pipeline.AddPass<CublasPadForGemms>(cuda_compute_capability,
                                              requirement.data_type,
                                              requirement.multiple_of);
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

  return OkStatus();
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

Status NVPTXCompiler::AddConvAndGemmAutotuningPasses(
    HloPassPipeline* pipeline, HloModule* hlo_module,
    AutotuneConfig& autotune_config, tsl::thread::ThreadPool* thread_pool) {
  if (GpuConvAlgorithmPicker::IsEnabled(hlo_module)) {
    pipeline->AddPass<GpuConvAlgorithmPicker>(autotune_config);
  }
  pipeline->AddPass<GemmAlgorithmPicker>(autotune_config);
  return OkStatus();
}

Status NVPTXCompiler::AddTritonGemmAutotuningPasses(
    HloPassPipeline* pipeline, HloModule* hlo_module,
    AutotuneConfig& autotune_config, tsl::thread::ThreadPool* thread_pool) {
  pipeline->AddPass<TritonAutotuner>(autotune_config, thread_pool);
  return OkStatus();
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

StatusOr<std::pair<std::string, std::vector<uint8_t>>>
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

  StatusOr<std::vector<uint8_t>> maybe_cubin = CompileGpuAsmOrGetCachedResult(
      ptx, std::get<se::CudaComputeCapability>(gpu_version), module_config,
      (debug_module != nullptr ? debug_module->name() : "(unknown)"),
      relocatable, options);

  if (maybe_cubin.status().code() == absl::StatusCode::kCancelled) {
    return maybe_cubin.status();
  }
  return std::pair<std::string, std::vector<uint8_t>>(
      std::move(ptx), std::move(maybe_cubin.value()));
}

StatusOr<std::vector<uint8_t>> NVPTXCompiler::CompileGpuAsmOrGetCachedResult(
    const std::string& ptx, se::CudaComputeCapability cc,
    const HloModuleConfig& hlo_module_config, absl::string_view module_name,
    bool relocatable, const CompileOptions& options) {
  // This may print multiple lines per HLO compilation because of the
  // parallelized compilation of LLVM modules.
  XLA_SCOPED_LOGGING_TIMER_IF(
      absl::StrCat("NVPTXCompiler::CompileGpuAsmOrGetCachedResult for ",
                   module_name),
      !options.is_autotuning_compilation);
  tsl::profiler::TraceMe activity("PTX->CUBIN",
                                  tsl::profiler::TraceMeLevel::kInfo);
  bool inserted;
  decltype(compilation_cache_.begin()) iter;
  // Pointers into compilation_cache_ where the ptx and (optional) cubin are
  // stored.
  const std::string* cache_ptx = nullptr;
  CompilationCacheValue* cache_value = nullptr;

  {
    absl::MutexLock lock(&mutex_);
    std::tie(iter, inserted) = compilation_cache_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(ptx, cc.major, cc.minor, relocatable),
        std::forward_as_tuple());
    cache_ptx = &iter->first.ptx;
    cache_value = &iter->second;
  }

  // Compile the ptx if it wasn't in the cache before we called this function.
  // Other threads asking for the same compilation key will block on
  // cache_value->mutex_ until compilation is done.
  {
    absl::MutexLock lock(&cache_value->mutex);
    if (inserted) {
      CHECK(!cache_value->compilation_done);
      if (!ptx.empty()) {
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
        StatusOr<std::vector<uint8_t>> maybe_cubin =
            se::CompileGpuAsm(cc.major, cc.minor, cache_ptx->c_str(),
                              ptxas_config, cancel_if_reg_spill);

        if (maybe_cubin.ok()) {
          uint64_t end_usecs = tsl::Env::Default()->NowMicros();
          // This won't record values for calls that error out (because if they
          // error out we have no way of telling how far through the process we
          // got).
          RecordPtxToCubinDuration(end_usecs - start_usecs);
          cache_value->cubin_data = std::move(maybe_cubin).value();
          VLOG(1) << "Compiled PTX size:" << ptx.size()
                  << " CUBIN size: " << cache_value->cubin_data.size();
        } else {
          if (maybe_cubin.status().code() == absl::StatusCode::kNotFound) {
            if (!hlo_module_config.debug_options()
                     .xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found()) {
              LOG(WARNING) << nvptx::CantFindCudaMessage(
                  "Can't find ptxas binary in ${CUDA_DIR}/bin.  Custom ptxas "
                  "location can be specified using $PATH.",
                  hlo_module_config.debug_options().xla_gpu_cuda_data_dir());
              LOG(FATAL)
                  << "Can't find ptxas binary.  You can pass the flag "
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
          } else if (maybe_cubin.status().code() ==
                     absl::StatusCode::kCancelled) {
            // Register spilling has occurred during autotuning, this config
            // should not be tried further.
            CHECK(options.is_autotuning_compilation);
            cache_value->compilation_done = true;
            cache_value->compilation_done_cv.SignalAll();
            return maybe_cubin;
          } else if (maybe_cubin.status().code() !=
                     absl::StatusCode::kUnimplemented) {
            // If unimplemented is returned, we fallback to the driver.
            LOG(FATAL) << "ptxas returned an error during compilation of ptx "
                          "to sass: '"
                       << maybe_cubin.status() << "'  "
                       << "If the error message indicates that a file could "
                          "not be written, please verify that sufficient "
                          "filesystem space is provided.";
          }

          // We're going to use the driver to JIT our PTX->SASS, so warn if
          // the JIT in the driver has known bugs.
          WarnIfBadDriverJITVersion();
        }
      }
      cache_value->compilation_done = true;
      cache_value->compilation_done_cv.SignalAll();
    } else {
      while (!cache_value->compilation_done) {
        cache_value->compilation_done_cv.Wait(&cache_value->mutex);
      }
    }
  }

  CHECK(cache_value != nullptr);
  CHECK(cache_value->compilation_done);
  return cache_value->cubin_data;
}

static std::optional<std::array<int64_t, 3>> GetNvLinkVersion(
    const std::string& preferred_cuda_dir) {
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

  if (!use_nvlink) {
    return std::nullopt;
  }

  // Make sure nvlink exists and is executable.
  const std::string bin_path =
      se::FindCudaExecutable("nvlink", preferred_cuda_dir);
  auto version = se::GetToolVersion(bin_path);
  if (!version.ok()) {
    return std::nullopt;
  }
  return *version;
}

StatusOr<NVPTXCompiler::LinkingMethod> NVPTXCompiler::ChooseLinkingMethod(
    const std::string& preferred_cuda_dir) {
  {
    absl::MutexLock lock(&mutex_);
    auto it = linking_methods_.find(preferred_cuda_dir);
    if (it != linking_methods_.end()) {
      return it->second;
    }
  }

  LinkingMethod linking_method = LinkingMethod::kNone;
  TF_ASSIGN_OR_RETURN(auto ptxas_version_tuple,
                      se::GetAsmCompilerVersion(preferred_cuda_dir));

  static const std::optional<std::array<int64_t, 3>> nvlink_version =
      GetNvLinkVersion(preferred_cuda_dir);
  if (nvlink_version && *nvlink_version >= ptxas_version_tuple) {
    linking_method = LinkingMethod::kNvLink;
  } else {
    int ptxas_version = std::get<0>(ptxas_version_tuple) * 1000 +
                        std::get<1>(ptxas_version_tuple) * 10;
    int driver_version;
    if (!se::gpu::GpuDriver::GetDriverVersion(&driver_version)) {
      return FailedPrecondition("Unable to get CUDA driver version");
    }

    if (driver_version >= ptxas_version) {
      linking_method = LinkingMethod::kDriver;
    } else {
      LOG_FIRST_N(WARNING, 1)
          << "The NVIDIA driver's CUDA version is "
          << absl::StrFormat("%d.%d", driver_version / 1000,
                             (driver_version % 1000) / 10)
          << " which is older than the ptxas CUDA version "
          << absl::StrFormat("(%d.%d.%d)", std::get<0>(ptxas_version_tuple),
                             std::get<1>(ptxas_version_tuple),
                             std::get<2>(ptxas_version_tuple))
          << ". Because the driver is older than the ptxas version, XLA is "
             "disabling parallel compilation, which may slow down compilation. "
             "You should update your NVIDIA driver or use the NVIDIA-provided "
             "CUDA forward compatibility packages.";
    }
  }
  {
    absl::MutexLock lock(&mutex_);
    linking_methods_[preferred_cuda_dir] = linking_method;
  }
  return linking_method;
}

StatusOr<bool> NVPTXCompiler::CanUseLinkModules(
    const HloModuleConfig& hlo_module_config) {
  // TODO(phawkins): rather than comparing version numbers, it might be more
  // robust if we simply tried to link something the first time we compile.
  auto ptxas_config =
      PtxOptsFromDebugOptions(hlo_module_config.debug_options());
  TF_ASSIGN_OR_RETURN(LinkingMethod linking_method,
                      ChooseLinkingMethod(ptxas_config.preferred_cuda_dir));
  return linking_method != LinkingMethod::kNone;
}

StatusOr<std::vector<uint8_t>> NVPTXCompiler::LinkModules(
    se::StreamExecutor* stream_exec, std::vector<std::vector<uint8_t>> modules,
    const DebugOptions& debug_options) {
  auto ptxas_config = PtxOptsFromDebugOptions(debug_options);

  std::vector<stream_executor::CubinOrPTXImage> images;
  images.reserve(modules.size());
  for (std::vector<uint8_t>& module : modules) {
    images.push_back({"", std::move(module)});
  }
  auto context = static_cast<se::gpu::GpuContext*>(
      stream_exec->platform_specific_handle().context);

  TF_ASSIGN_OR_RETURN(LinkingMethod linking_method,
                      ChooseLinkingMethod(ptxas_config.preferred_cuda_dir));
  if (linking_method == LinkingMethod::kNvLink) {
    return LinkUsingNvlink(debug_options.xla_gpu_cuda_data_dir(), context,
                           images);
  }
  return LinkGpuAsm(context, images);
}

}  // namespace gpu
}  // namespace xla
