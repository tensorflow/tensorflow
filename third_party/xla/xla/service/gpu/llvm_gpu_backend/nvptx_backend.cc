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

#include "xla/service/gpu/llvm_gpu_backend/nvptx_backend.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/PassRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Scalar.h"
#include "xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "xla/service/gpu/llvm_gpu_backend/load_ir_module.h"
#include "xla/service/gpu/llvm_gpu_backend/nvptx_libdevice_path.h"
#include "xla/service/gpu/metrics.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/stream_executor/cuda/ptx_compiler_helpers.h"
#include "xla/stream_executor/cuda/subprocess_compilation.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::gpu::nvptx {

namespace {

// Default inline threshold value to use in llvm.
const int kDefaultInlineThreshold = 1100;

// Emits the given module to PTX. target_machine is an initialized TargetMachine
// for the NVPTX target.
std::string EmitModuleToPTX(llvm::Module* module,
                            llvm::TargetMachine* target_machine) {
  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("XlaEmitGpuAsm:#module=%s#",
                           module->getName().str());
  });
  std::string ptx;
  llvm::raw_string_ostream stream(ptx);
  llvm::buffer_ostream pstream(stream);
  llvm::legacy::PassManager pm;
  pm.add(new llvm::TargetLibraryInfoWrapperPass(
      llvm::Triple(module->getTargetTriple())));
  target_machine->addPassesToEmitFile(pm, pstream, nullptr,
                                      llvm::CodeGenFileType::AssemblyFile);
  pm.run(*module);
  return ptx;
}

// Links libdevice into the given module if the module needs libdevice.
absl::Status LinkLibdeviceIfNecessary(llvm::Module* module,
                                      const std::string& libdevice_path) {
  if (!CouldNeedDeviceBitcode(*module)) {
    return absl::OkStatus();
  }

  if (!tsl::Env::Default()->FileExists(libdevice_path).ok()) {
    LOG(WARNING)
        << "libdevice is required by this HLO module but was not found at "
        << libdevice_path;
    return xla::Internal("libdevice not found at %s", libdevice_path);
  }

  VLOG(1) << "Linking with libdevice from: " << libdevice_path;
  return LinkWithBitcodeVector(module, {libdevice_path});
}

absl::Status NVPTXTargetModuleLinker(llvm::Module* module,
                                     se::GpuComputeCapability gpu_version,
                                     const DebugOptions& debug_options,
                                     const std::string& device_bitcode_path) {
  // Link the input module with libdevice, to pull in implementations of some
  // builtins.
  TF_RETURN_IF_ERROR(LinkLibdeviceIfNecessary(module, device_bitcode_path));

  // Set the flush-denormals-to-zero flag on the module so the NVVM reflect pass
  // can access it.
  module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz",
                        debug_options.xla_gpu_ftz());

  // If ftz is enabled, set it as an attribute on every function in the module.
  if (debug_options.xla_gpu_ftz()) {
    for (llvm::Function& fn : *module) {
      fn.addFnAttr("denormal-fp-math-f32", "preserve-sign");
    }
  }

  return absl::OkStatus();
}

std::unique_ptr<llvm::TargetMachine> NVPTXGetTargetMachine(
    llvm::Triple target_triple, se::CudaComputeCapability compute_capability,
    const DebugOptions& debug_options) {
  absl::StatusOr<stream_executor::SemanticVersion> runtime_cuda_version =
      stream_executor::GetAsmCompilerVersion(
          debug_options.xla_gpu_cuda_data_dir());

  constexpr stream_executor::SemanticVersion kCompileTimeCudaVersion{
      CUDA_VERSION / 1000, (CUDA_VERSION / 10) % 100, CUDA_VERSION % 10};

  auto highest_supported_cuda_version = [&] {
    if (runtime_cuda_version.ok()) {
      return std::min(runtime_cuda_version.value(), kCompileTimeCudaVersion);
    }

    return kCompileTimeCudaVersion;
  }();

  auto ptx_version = nvptx::DetermineHighestSupportedPtxVersionFromCudaVersion(
      highest_supported_cuda_version);
  int highest_supported_ptx_version =
      ptx_version.major() * 10 + ptx_version.minor();

  VLOG(1) << "Targeting PTX version: " << highest_supported_ptx_version;
  std::string feature_str =
      absl::StrFormat("+ptx%d", highest_supported_ptx_version);

  return GetTargetMachine(target_triple, nvptx::GetSmName(compute_capability),
                          debug_options, feature_str);
}

// One-time module initializer.
// Must be called only once -- DO NOT CALL DIRECTLY.
void NVPTXBackendInit() {
  // Initialize the NVPTX target; it's the only target we link with, so call its
  // specific initialization functions instead of the catch-all InitializeAll*.
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();

  // Initialize the LLVM optimization passes.
  llvm::PassRegistry* registry = llvm::PassRegistry::getPassRegistry();
  InitializePasses(registry);
}

}  // namespace

std::vector<std::string> GetNVPTXBackendOptions(
    const DebugOptions& debug_options) {
  // Feed all customized flags here, so we can override them with llvm_cl_opts
  // without redeploy the compiler for development purpose.
  std::vector<std::string> backend_llvm_opts;

  // This flag tunes a threshold in branch folding. The default threshold, which
  // is one, is not suitable for CUDA programs where branches are more expensive
  // than for CPU programs. Setting the threshold to 2 improves the latency of
  // TwoDPatchDotProductKernel_IND_3_ND_48 by over 5%, and does not affect the
  // latency of other benchmarks so far.
  //
  // I also tried setting this threshold to other values:
  // * 3-6 gives similar results as 2;
  // * >6 start hurting the performance of at least dot product kernels.
  //
  // TODO(jingyue): The current threshold only considers the number of IR
  // instructions which do not accurately reflect the true cost. We need a
  // better cost model.
  backend_llvm_opts.emplace_back("-bonus-inst-threshold=2");

  // Use div.full -- it matters for some float-division heavy benchmarks.
  // Using div.approx produces incorrect result for float32(max)/float32(max).
  backend_llvm_opts.emplace_back("-nvptx-prec-divf32=1");

  // SLPVectorizer is useful (vectorizes f16x2 ops) but slow.  Most of the
  // slowness appears to be in trying to form horizontal reductions, which don't
  // exist in PTX *anyway*.  Disable these.  While we're here, tweak
  // SLPVectorizer so it doesn't try to create large vectors -- f16x2 are the
  // only vectors supported in PTX.
  backend_llvm_opts.emplace_back("-slp-vectorize-hor=false");
  backend_llvm_opts.emplace_back("-slp-max-reg-size=32");

  // Extra backend options must go after regular backend options in order to be
  // able for the later to override the former.
  auto backend_extra_llvm_opts = llvm_ir::ExtractXlaBackendExtraOptions(
      debug_options.xla_backend_extra_options());
  backend_llvm_opts.insert(backend_llvm_opts.end(),
                           backend_extra_llvm_opts.cbegin(),
                           backend_extra_llvm_opts.cend());

  return backend_llvm_opts;
}

std::string GetSmName(se::CudaComputeCapability compute_capability) {
  int compute_capability_version =
      compute_capability.major * 10 + compute_capability.minor;
  int sm_version = 30;
  // If the current compute capability isn't known, fallback to the
  // most recent version before it.
  int supported_versions[] = {121, 120, 103, 101, 100, 90, 89, 87,
                              86,  80,  75,  72,  70,  62, 61, 60,
                              53,  52,  50,  37,  35,  32, 30};
  for (int v : supported_versions) {
    if (v <= compute_capability_version) {
      sm_version = v;
      break;
    }
  }

  // If the current CC isn't supported by LLVM and it is newer then
  // the max supported LLVM version, do not warn about it. The end
  // user can't do anything about this. E.g., PTX compiled for SM75 will
  // run on SM80 too.
  if (sm_version != compute_capability_version &&
      compute_capability_version < supported_versions[0]) {
    LOG(WARNING) << "Unknown compute capability "
                 << compute_capability.ToString()
                 << ". Defaulting to telling LLVM that we're compiling for sm_"
                 << sm_version;
  }
  // On Hopper, default to sm_90a so that all instructions can be used. But
  // only sm_90 is forward compatible, so don't use sm_90a with newer hardware:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ptx-compatibility
  // Similarly for sm_10#a and sm_12#a (Blackwell).
  absl::string_view extension =
      stream_executor::ShouldUsePtxExtension(compute_capability) ? "a" : "";
  return absl::StrCat("sm_", sm_version, extension);
}

absl::StatusOr<std::string> CompileToPtx(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options,
    std::function<void(llvm::TargetMachine*)> configure_target) {
  static absl::once_flag backend_init_flag;
  absl::call_once(backend_init_flag, NVPTXBackendInit);
  auto llvm_opts = GetNVPTXBackendOptions(debug_options);
  llvm_ir::LLVMCommandLineOptionsLock llvm_lock(llvm_opts);

  std::string ptx;
  std::unique_ptr<llvm::TargetMachine> target_machine;
  {
    tsl::profiler::TraceMe activity(
        [&] { return absl::StrCat("Compiling IR:", module->getName().str()); },
        tsl::profiler::TraceMeLevel::kInfo);
    XLA_SCOPED_LOGGING_TIMER("Compile module " + module->getName().str());

    // If the module has no functions or globals, there's nothing to compile.
    // Just return an empty string.
    if (module->empty() && module->global_empty()) {
      VLOG(2) << "Module '" << module->getName().str()
              << "' is empty. Skipping compilation.";
      return std::string();
    }

    auto compute_capability =
        std::get_if<se::CudaComputeCapability>(&gpu_version);
    if (!compute_capability) {
      return xla::Internal("Incompatible compute capability was specified.");
    }

    llvm::Triple default_target_triple("nvptx64-unknown-unknown");
    // Construct LLVM TargetMachine for NVPTX.
    std::unique_ptr<llvm::TargetMachine> target_machine = NVPTXGetTargetMachine(
        default_target_triple, *compute_capability, debug_options);

    // Apply target machine configuration from call-back if available.
    if (configure_target) {
      configure_target(target_machine.get());
    }

    uint64_t start_usecs = tsl::Env::Default()->NowMicros();

    // Link with libdevice, and optimize the LLVM module.
    TF_RETURN_IF_ERROR(LinkAndOptimizeModule(
        module, gpu_version, debug_options,
        LibDevicePath(debug_options.xla_gpu_cuda_data_dir()),
        NVPTXTargetModuleLinker, default_target_triple, target_machine.get(),
        kDefaultInlineThreshold));

    uint64_t end_usecs = tsl::Env::Default()->NowMicros();
    RecordLlvmPassesDuration(end_usecs - start_usecs);

    start_usecs = tsl::Env::Default()->NowMicros();

    // Lower optimized LLVM module to PTX.
    ptx = EmitModuleToPTX(module, target_machine.get());

    end_usecs = tsl::Env::Default()->NowMicros();
    RecordLlvmToPtxDuration(end_usecs - start_usecs);
  }
  return ptx;
}
}  // namespace xla::gpu::nvptx
