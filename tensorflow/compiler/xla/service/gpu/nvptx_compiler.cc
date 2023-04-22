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
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_pad_for_gemms.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_fused_conv_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_pad_for_convolutions.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_vectorize_convolutions.h"
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
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
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
#include "tensorflow/stream_executor/gpu/gpu_driver.h"

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

}  // namespace

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

  // GetCudaRootCandidates always includes ".", but if everything fails, we
  // return it anyway.  Better than returning the empty string.
  return ".";
}

Status NVPTXCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  // Convert convolutions into CustomCalls to cudnn, then canonicalize them
  // (GpuConvPaddingLegalization). Also expand cuSolver calls.
  HloPassPipeline pipeline("conv_canonicalization");
  pipeline.AddInvariantCheckerDebug<HloVerifier>(
      /*layout_sensitive=*/false,
      /*allow_mixed_precision=*/false);
  pipeline.AddPass<GpusolverRewriter>();
  pipeline.AddPass<GpuConvRewriter>();
  pipeline.AddPass<CudnnFusedConvRewriter>();
  pipeline.AddPass<GpuConvPaddingLegalization>();
  pipeline.AddPass<CudnnPadForConvolutions>(
      stream_exec->GetDeviceDescription().cuda_compute_capability());
  pipeline.AddPass<CudnnVectorizeConvolutions>(
      stream_exec->GetDeviceDescription().cuda_compute_capability());
  // CudnnConvPadForIntegerConvolutions and CudnnConvPadForTensorCores leave
  // behind unnecessary tuple/get-tuple-element pairs that TupleSimplifier
  // fixes.
  pipeline.AddPass<TupleSimplifier>();

  // tf2xla bridge, DepthwiseConvolutionConverter and GpuConvRewriter
  // introduces reshapes and transposes that can be eliminated using
  // AlgebraicSimplifier  We run algsimp to a fixed point.
  //
  // When transposes appear in a fusion node, we can easily adjust the
  // multi-dimensional index to create the one needed for the operand. This
  // is not as easy with bitcasts, because we don't have the information
  // readily available which dimensions are permuted. In addition to that,
  // if we have a transpose and a reshape next to each other, they will both
  // be replaced by a bitcast, and we replace bitcast(bitcast) with one
  // bitcast. This leads to having to linearize and then delinearize the
  // index.
  AlgebraicSimplifierOptions options;
  options.set_replace_transpose_with_bitcast(false);
  options.set_enable_conv_operand_swap(false);
  options.set_cudnn_batchnorm_forward_training_metadata(
      kCudnnBatchNormForwardTrainingCallTarget);
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(options);

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

  // This needs to run before GemmRewriter, which is part of
  // OptimizeHloPostLayoutAssignment().
  if (stream_exec->GetDeviceDescription().cuda_compute_capability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    pre_pipeline.AddPass<CublasPadForGemms>(PrimitiveType::BF16,
                                            /*pad_to_multiple_of=*/8);
  }
  if (stream_exec->GetDeviceDescription().cuda_compute_capability().IsAtLeast(
          se::CudaComputeCapability::VOLTA)) {
    // Pad gemms over S8 to multiples of 4 so cuBLAS can run them.
    pre_pipeline.AddPass<CublasPadForGemms>(PrimitiveType::S8,
                                            /*pad_to_multiple_of=*/4);

    // Pad the dimensions of matrices in dot operations to multiples of 8.
    pre_pipeline.AddPass<CublasPadForGemms>(PrimitiveType::F16,
                                            /*pad_to_multiple_of=*/8);
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
  switch (user->opcode()) {
    case HloOpcode::kAllReduce:
      // NCCL all-reduce can be performed in-place.
      return user->operand_count() == 1 ||
             (user_index.size() == 1 &&
              user->operand(user_index[0]) == operand);
    case HloOpcode::kCustomCall:
      // Share the bias buffer with the parent instruction.
      if (user->custom_call_target() == kGemmCallTarget) {
        return user->operand_count() == 3 && user->operand(2) == operand;
      }
      // The operand of cholesky can be shared with the first output.
      if (user->custom_call_target() == kCusolverCholeskyCallTarget) {
        return user_index.size() == 1 && user_index[0] == 0;
      }
      return false;
    default:
      return absl::nullopt;
  }
}

// Try to load ptx from files defined in the FLAGS. If successful, return true.
bool MaybeLoadPtxFromFile(const HloModuleConfig module_config,
                          const HloModule* module, std::string* ptx) {
  // If the xla_gpu_ptx_file option is set, be explicit if a file is used
  // and warn when a file is not used to ease catching typo in filename.
  std::string prefix = xla::FilenameFor(*module, "", *ptx);
  std::string matched_filename;
  for (const string& full_filename :
       module_config.debug_options().xla_gpu_ptx_file()) {
    // To ease comparing many PTX versions, accept different suffixes then
    // the original filename.
    auto filename = tensorflow::io::Basename(full_filename);
    if (absl::StartsWith(filename, prefix)) {
      matched_filename = full_filename;
      VLOG(0) << "RunBackend() - Will load PTX from file: " << full_filename;
      break;
    }
  }
  if (!module_config.debug_options().xla_gpu_ptx_file().empty() &&
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
      xla_gpu_llvm_ir_file, [prefix](const string& full_filename) {
        // To ease comparing many LLVM versions, accept different suffixes then
        // the original filename.
        return absl::StartsWith(tensorflow::io::Basename(full_filename),
                                prefix);
      });
  if (!xla_gpu_llvm_ir_file.empty() &&
      matched_filename == std::end(xla_gpu_llvm_ir_file)) {
    VLOG(0) << "RunBackend() - For module with prefix '" << prefix
            << "', we did not found a LLVM file to load.";
  }

  if (matched_filename != std::end(xla_gpu_llvm_ir_file)) {
    VLOG(0) << "RunBackend() - Will load LLVM from file: " << *matched_filename;
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

NVPTXCompiler::NVPTXCompiler()
    : GpuCompiler(stream_executor::cuda::kCudaPlatformId, nvptx::kTargetTriple,
                  nvptx::kDataLayout) {}

HloDataflowAnalysis::CanShareBuffer NVPTXCompiler::GetCanShareBuffer() {
  return &CanShareBufferHint;
}

GpuVersion NVPTXCompiler::GetGpuVersion(se::StreamExecutor* stream_exec) {
  return stream_exec->GetDeviceDescription().cuda_compute_capability();
}

StatusOr<std::pair<std::string, std::vector<uint8>>>
NVPTXCompiler::CompileTargetBinary(const HloModuleConfig& module_config,
                                   llvm::Module* llvm_module,
                                   GpuVersion gpu_version,
                                   se::StreamExecutor* stream_exec,
                                   bool relocatable,
                                   const HloModule* debug_module) {
  std::string libdevice_dir;
  {
    tensorflow::mutex_lock lock(mutex_);

    // Find the directory containing libdevice.  To avoid searching for it every
    // time, we have a one-element cache, keyed on the module's config's
    // cuda_data_dir.
    if (cached_libdevice_dir_.empty()) {
      cached_libdevice_dir_ = GetLibdeviceDir(module_config);
    }
    libdevice_dir = cached_libdevice_dir_;
  }
  VLOG(2) << "Libdevice dir = " << libdevice_dir << "\n";
  std::unique_ptr<llvm::Module> loaded_module =
      MaybeLoadLLVMFromFile(debug_module, llvm_module);
  llvm::Module* selected_module = nullptr;
  if (loaded_module) {
    selected_module = loaded_module.get();
  } else {
    selected_module = llvm_module;
  }

  string ptx;
  if (!(debug_module &&
        MaybeLoadPtxFromFile(module_config, debug_module, &ptx))) {
    XLA_SCOPED_LOGGING_TIMER(
        "NVPTXCompiler::CompileTargetBinary - CompileToPtx");
    TF_ASSIGN_OR_RETURN(ptx, nvptx::CompileToPtx(selected_module, gpu_version,
                                                 module_config, libdevice_dir));
  }

  std::vector<uint8> cubin = CompileGpuAsmOrGetCachedResult(
      stream_exec, ptx, absl::get<se::CudaComputeCapability>(gpu_version),
      module_config, relocatable);

  return std::pair<std::string, std::vector<uint8>>(std::move(ptx),
                                                    std::move(cubin));
}

std::vector<uint8> NVPTXCompiler::CompileGpuAsmOrGetCachedResult(
    se::StreamExecutor* stream_exec, const string& ptx,
    se::CudaComputeCapability cc, const HloModuleConfig& hlo_module_config,
    bool relocatable) {
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
        std::forward_as_tuple(ptx, cc.major, cc.minor, relocatable),
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
        auto ptxas_config = PtxOptsFromConfig(hlo_module_config);
        if (relocatable) {
          ptxas_config.extra_flags.push_back("-c");
        }
        StatusOr<std::vector<uint8>> maybe_cubin = se::CompileGpuAsm(
            stream_exec->device_ordinal(), cache_ptx->c_str(), ptxas_config);

        if (maybe_cubin.ok()) {
          cache_value->cubin_data = std::move(maybe_cubin).ValueOrDie();
          VLOG(2) << "Compiled PTX size:" << ptx.size()
                  << " CUBIN size: " << cache_value->cubin_data.size();
        } else {
          if (maybe_cubin.status().code() ==
              tensorflow::error::Code::NOT_FOUND) {
            if (!hlo_module_config.debug_options()
                     .xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found()) {
              PrintCantFindCudaMessage(
                  "Can't find ptxas binary in ${CUDA_DIR}/bin.  Custom ptxas "
                  "location can be specified using $PATH.",
                  hlo_module_config);
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
          } else if (maybe_cubin.status().code() !=
                     tensorflow::error::Code::UNIMPLEMENTED) {
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

StatusOr<std::vector<uint8>> NVPTXCompiler::LinkModules(
    se::StreamExecutor* stream_exec, std::vector<std::vector<uint8>> modules) {
  std::vector<stream_executor::CubinOrPTXImage> images;
  images.reserve(modules.size());
  for (auto& module : modules) {
    images.push_back({"", std::move(module)});
  }
  return LinkGpuAsm(static_cast<se::gpu::GpuContext*>(
                        stream_exec->implementation()->GpuContextHack()),
                    images);
}

}  // namespace gpu
}  // namespace xla
