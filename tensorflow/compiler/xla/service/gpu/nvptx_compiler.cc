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

#include "tensorflow/compiler/xla/service/gpu/nvptx_compiler.h"

#include <stdlib.h>

#include <fstream>

#include "absl/base/call_once.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_gemm_pad_for_tensor_cores.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_fused_conv_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_pad_for_convolutions.h"
#include "tensorflow/compiler/xla/service/gpu/cusolver_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_algorithm_picker.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_padding_legalization.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_layout_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/cuda_libdevice_path.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/stream_executor/cuda/cuda_diagnostics.h"
#include "tensorflow/stream_executor/gpu/asm_compiler.h"

namespace xla {
namespace gpu {

namespace {

namespace tracing = tensorflow::tracing;

static std::vector<std::string> CandidateCudaRoots(
    const HloModuleConfig& config) {
  return tensorflow::CandidateCudaRoots(
      config.debug_options().xla_gpu_cuda_data_dir());
}

void PrintCantFindCudaMessage(absl::string_view msg,
                              const HloModuleConfig& hlo_module_config) {
  LOG(WARNING) << msg;
  LOG(WARNING) << "Searched for CUDA in the following directories:";

  for (const auto& dir : CandidateCudaRoots(hlo_module_config)) {
    LOG(WARNING) << "  " << dir;
  }
  LOG(WARNING)
      << "You can choose the search directory by setting xla_gpu_cuda_data_dir "
         "in HloModule's DebugOptions.  For most apps, setting the environment "
         "variable XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda will work.";
}

// Returns the directory containing nvvm libdevice files.
string GetLibdeviceDir(const HloModuleConfig& hlo_module_config) {
  for (const string& cuda_root : CandidateCudaRoots(hlo_module_config)) {
    string libdevice_dir =
        tensorflow::io::JoinPath(cuda_root, "nvvm", "libdevice");
    VLOG(2) << "Looking for libdevice at " << libdevice_dir;
    if (tensorflow::Env::Default()->IsDirectory(libdevice_dir).ok()) {
      VLOG(2) << "Found libdevice dir " << libdevice_dir;
      return libdevice_dir;
    }
  }
  PrintCantFindCudaMessage(
      "Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice. This may "
      "result in compilation or runtime failures, if the program we try to run "
      "uses routines from libdevice.",
      hlo_module_config);

  // GetCudaRootCandidates always includes ".", but but if everything fails, we
  // return it anyway.  Better than returning the empty string.
  return ".";
}

}  // namespace

Status NVPTXCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  // Convert convolutions into CustomCalls to cudnn, then canonicalize them
  // (GpuConvPaddingLegalization). Also expand cuSolver calls.
  HloPassPipeline pipeline("conv_canonicalization");
  pipeline.AddInvariantCheckerDebug<HloVerifier>(
      /*layout_sensitive=*/false,
      /*allow_mixed_precision=*/false);
  pipeline.AddPass<CusolverRewriter>();
  pipeline.AddPass<GpuConvRewriter>();
  pipeline.AddPass<CudnnFusedConvRewriter>();
  pipeline.AddPass<GpuConvPaddingLegalization>();
  pipeline.AddPass<CudnnPadForConvolutions>(IsVoltaOrLater(*stream_exec));
  // CudnnConvPadForIntegerConvolutions and CudnnConvPadForTensorCores leaves
  // behind unnecessary tuple/get-tuple-element pairs that TupleSimplifier
  // fixes.
  pipeline.AddPass<TupleSimplifier>();

  // tf2xla bridge, DepthwiseConvolutionConverter and GpuConvRewriter
  // introduces reshapes and transposes that can be eliminated using
  // AlgebraicSimplifier
  {
    auto& pass = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
        "algebraic_simplification_post_conv_rewriter");
    pass.AddInvariantCheckerDebug<HloVerifier>(/*layout_sensitive=*/false,
                                               /*allow_mixed_precision=*/false);

    AlgebraicSimplifierOptions options;
    // When transposes appear in a fusion node, we can easily adjust the
    // multi-dimensional index to create the one needed for the operand. This
    // is not as easy with bitcasts, because we don't have the information
    // readily available which dimensions are permuted. In addition to that,
    // if we have a transpose and a reshape next to each other, they will both
    // be replaced by a bitcast, and we replace bitcast(bitcast) with one
    // bitcast. This leads to having to linearize and then delinearize the
    // index.
    options.set_replace_transpose_with_bitcast(false);
    options.set_enable_conv_operand_swap(false);
    options.set_cudnn_batchnorm_forward_training_metadata(
        kCudnnBatchNormForwardTrainingCallTarget);
    pass.AddPass<AlgebraicSimplifier>(options);
  }

  // GpuConvRewriter, GpuConvPaddingLegalization and
  // CudnnConvPadForTensorCores may add instructions which can be simplified
  // by constant folding.
  pipeline.AddPass<HloConstantFolding>();
  TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());

  return Status::OK();
}

Status NVPTXCompiler::OptimizeHloPostLayoutAssignment(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  HloPassPipeline pre_pipeline("nvptx post-layout_assignment part 1");
  // Pad the dimensions of matrices in dot operations to multiples of 8.
  // This needs to run before GemmRewriter, which is part of
  // OptimizeHloPostLayoutAssignment().
  if (IsVoltaOrLater(*stream_exec)) {
    pre_pipeline.AddPass<CublasGemmPadForTensorCores>();
  }
  TF_RETURN_IF_ERROR(pre_pipeline.Run(hlo_module).status());

  TF_RETURN_IF_ERROR(GpuCompiler::OptimizeHloPostLayoutAssignment(
      hlo_module, stream_exec, device_allocator));

  HloPassPipeline post_pipeline("nvptx post-layout_assignment part 2");

  // Find the fastest algorithm for GEMMs.
  post_pipeline.AddPass<GemmAlgorithmPicker>(stream_exec, device_allocator);
  TF_RETURN_IF_ERROR(post_pipeline.Run(hlo_module).status());

  return Status::OK();
}

namespace {
absl::optional<bool> CanShareBufferHint(const HloInstruction* user,
                                        const HloInstruction* operand,
                                        const ShapeIndex& user_index) {
  // Share the bias buffer with the parent instruction.
  if (IsCublasGemm(*user)) {
    if (user->operand_count() == 3 && user->operand(2) == operand) {
      return true;
    }
  }
  // The operand of cholesky can be shared with the first output.
  if (user->opcode() == HloOpcode::kCustomCall &&
      user->custom_call_target() == kCusolverCholeskyCallTarget) {
    return user_index.size() == 1 && user_index[0] == 0;
  }
  return absl::nullopt;
}

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
    se::cuda::DriverVersion version = version_or_status.ValueOrDie();

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

// Try to load ptx from files defined in the FLAGS. If successful, return true.
bool MaybeLoadPtxFromFile(const HloModule* module, std::string* ptx) {
  // If the xla_gpu_ptx_file options is set, be explicit when a file is used
  // and warn when a file is not used to ease catching typo in filename.
  std::string prefix = xla::FilenameFor(*module, "", *ptx);
  std::string matched_filename;
  for (const string& full_filename :
       module->config().debug_options().xla_gpu_ptx_file()) {
    // To ease comparing many PTX versions, accept different suffixes then
    // the original filename.
    auto filename = tensorflow::io::Basename(full_filename);
    if (absl::StartsWith(filename, prefix)) {
      matched_filename = full_filename;
      VLOG(0) << "RunBackend() - Will load PTX from file: " << full_filename;
      break;
    }
  }
  if (module->config().debug_options().xla_gpu_ptx_file().size() > 0 &&
      matched_filename.empty()) {
    VLOG(0) << "RunBackend() - For module with prefix '" << prefix
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

}  // namespace

NVPTXCompiler::NVPTXCompiler()
    : GpuCompiler(stream_executor::cuda::kCudaPlatformId, nvptx::kTargetTriple,
                  nvptx::kDataLayout) {}

HloDataflowAnalysis::CanShareBuffer NVPTXCompiler::GetCanShareBuffer() {
  return &CanShareBufferHint;
}

GpuVersion NVPTXCompiler::GetGpuVersion(se::StreamExecutor* stream_exec) {
  int cc_major, cc_minor;
  if (!stream_exec->GetDeviceDescription().cuda_compute_capability(&cc_major,
                                                                   &cc_minor)) {
    LOG(WARNING)
        << "Couldn't get compute capability for device; assuming sm_20.";
    cc_major = 2;
    cc_minor = 0;
  }

  return std::make_pair(cc_major, cc_minor);
}

StatusOr<std::pair<std::string, std::vector<uint8>>>
NVPTXCompiler::CompileTargetBinary(const HloModule* module,
                                   llvm::Module* llvm_module,
                                   GpuVersion gpu_version,
                                   se::StreamExecutor* stream_exec) {
  std::pair<int, int> compute_capability =
      absl::get<std::pair<int, int>>(gpu_version);

  std::string libdevice_dir;
  {
    tensorflow::mutex_lock lock(mutex_);

    // Find the directory containing libdevice.  To avoid searching for it every
    // time, we have a one-element cache, keyed on the module's config's
    // cuda_data_dir.
    if (cached_libdevice_dir_.empty()) {
      cached_libdevice_dir_ = GetLibdeviceDir(module->config());
    }
    libdevice_dir = cached_libdevice_dir_;
  }
  VLOG(2) << "Libdevice dir = " << libdevice_dir << "\n";

  string ptx;
  if (!MaybeLoadPtxFromFile(module, &ptx)) {
    XLA_SCOPED_LOGGING_TIMER(
        "NVPTXCompiler::CompileTargetBinary - CompileToPtx");
    TF_ASSIGN_OR_RETURN(
        ptx, nvptx::CompileToPtx(llvm_module, gpu_version, module->config(),
                                 libdevice_dir));
  }

  llvm_ir::DumpIrIfEnabled(*module, *llvm_module, /*optimized=*/true);

  if (user_post_optimization_hook_) {
    user_post_optimization_hook_(*llvm_module);
  }
  // Write PTX to IR dump directory, if IR dumping was requested.
  if (DumpingEnabledForHloModule(*module)) {
    DumpToFileInDirOrStdout(*module, "", "ptx", ptx);
  }

  std::vector<uint8> cubin = CompileGpuAsmOrGetCachedResult(
      stream_exec, ptx, compute_capability.first, compute_capability.second,
      module->config());

  return std::pair<std::string, std::vector<uint8>>(std::move(ptx),
                                                    std::move(cubin));
}

std::vector<uint8> NVPTXCompiler::CompileGpuAsmOrGetCachedResult(
    se::StreamExecutor* stream_exec, const string& ptx, int cc_major,
    int cc_minor, const HloModuleConfig& hlo_module_config) {
  XLA_SCOPED_LOGGING_TIMER("NVPTXCompiler::CompileGpuAsmOrGetCachedResult");
  tensorflow::profiler::TraceMe activity(
      "PTX->CUBIN", tensorflow::profiler::TraceMeLevel::kInfo);
  bool inserted;
  decltype(compilation_cache_.begin()) iter;
  // Pointers into compilation_cache_ where the ptx and (optional) cubin are
  // stored.
  const string* cache_ptx = nullptr;
  CompilationCacheValue* cache_value = nullptr;

  {
    tensorflow::mutex_lock lock(mutex_);
    std::tie(iter, inserted) = compilation_cache_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(ptx, cc_major, cc_minor),
        std::forward_as_tuple());
    cache_ptx = &iter->first.ptx;
    cache_value = &iter->second;
  }

  // Compile the ptx if it wasn't in the cache before we called this function.
  // Other threads asking for the same compilation key will block on
  // cache_value->mutex_ until compilation is done.
  {
    tensorflow::mutex_lock lock(cache_value->mutex_);
    if (inserted) {
      CHECK(!cache_value->compilation_done);
      if (!ptx.empty()) {
        StatusOr<std::vector<uint8>> maybe_cubin =
            se::CompileGpuAsm(stream_exec->device_ordinal(), cache_ptx->c_str(),
                              PtxOptsFromConfig(hlo_module_config));
        if (maybe_cubin.ok()) {
          cache_value->cubin_data = std::move(maybe_cubin).ValueOrDie();
          VLOG(2) << "Compiled PTX size:" << ptx.size()
                  << " CUBIN size: " << cache_value->cubin_data.size();
        } else {
          if (maybe_cubin.status().code() ==
              tensorflow::error::Code::NOT_FOUND) {
            // Missing ptxas is expected in some environments where CUDA SDK
            // binaries are not available. We don't want to spam logs with
            // identical warnings in this case.

            // TODO(jlebar): we should implement a LOG_FIRST_N and LOG_EVERY_N
            // for more general usage.
            static std::atomic<bool> warning_done(false);
            bool log_warning = !warning_done.exchange(true);
            if (log_warning) {
              PrintCantFindCudaMessage(
                  "Can't find ptxas binary in ${CUDA_DIR}/bin.  Will back to "
                  "the GPU driver for PTX -> sass compilation.  This is OK so "
                  "long as you don't see a warning below about an out-of-date "
                  "driver version. Custom ptxas location can be specified "
                  "using $PATH.",
                  hlo_module_config);
            }
            CHECK(hlo_module_config.debug_options()
                      .xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found())
                << "There was an error when trying to compile ptx into sass "
                   "code. If you want to try falling back to the GPU driver to "
                   "jit compile ptx, you can use the flag "
                   "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found."
                   " Use at your own risk though, it has known drawbacks like "
                   "increased memory consumption.";
          } else {
            LOG(ERROR) << "Error during compilation of ptx to sass: "
                       << maybe_cubin.status();
            CHECK(hlo_module_config.debug_options()
                      .xla_gpu_unsafe_fallback_to_driver_on_ptxas_error())
                << "There was an error when trying to compile ptx into sass "
                   "code. Up until May 14 2020, XLA silently ignored such "
                   "errors and fell back to the GPU driver. This is likely to "
                   "trigger subtle runtime issues and is hence discouraged. "
                   "If you want to temporarily restore this behavior use the "
                   "flag --xla_gpu_unsafe_fallback_to_driver_on_ptxas_error "
                   "and file a bug in b/components/366096.";
          }

          // We're going to use the driver to JIT our PTX->SASS, so warn if
          // the JIT in the driver has known bugs.
          WarnIfBadDriverJITVersion();
        }
      }
      cache_value->compilation_done = true;
      cache_value->compilation_done_cv_.notify_all();
    } else {
      while (!cache_value->compilation_done) {
        cache_value->compilation_done_cv_.wait(lock);
      }
    }
  }

  CHECK(cache_value != nullptr);
  CHECK(cache_value->compilation_done);
  return cache_value->cubin_data;
}

}  // namespace gpu
}  // namespace xla
