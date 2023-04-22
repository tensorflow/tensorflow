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

#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Linker/Linker.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/dump_ir_pass.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/utils.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/env_var.h"

#if !defined(PLATFORM_GOOGLE) && TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif

namespace xla {
namespace gpu {
namespace {

static llvm::codegen::RegisterCodeGenFlags CGF;

// Inline threshold value to use in LLVM AMDGPU backend.
const int kAMDGPUInlineThreshold = 0x100000;

// Default inline threshold value to use in llvm.
const int kDefaultInlineThreshold = 1100;

// Gets the GPU name as it's known to LLVM for a given compute
// capability.  If we see an unrecognized compute capability, we
// return the highest one that is known and below the selected device.
static string GetSmName(se::CudaComputeCapability compute_capability) {
  int compute_capability_version =
      compute_capability.major * 10 + compute_capability.minor;
  int sm_version = 30;
  // If the current compute capability isn't known, fallback to the
  // most recent version before it.
  int supported_versions[] = {75, 72, 70, 62, 61, 60, 53,
                              52, 50, 37, 35, 32, 30};
  for (int v : supported_versions) {
    if (v <= compute_capability_version) {
      sm_version = v;
      break;
    }
  }

  // If the current CC isn't supported by LLVM and it is newer then
  // the max supported LLVM version, do not warn about it. The end
  // user can't do anything about this. PTX compiled for SM75 will
  // run on SM80 too.
  if (sm_version != compute_capability_version &&
      compute_capability_version < supported_versions[0]) {
    LOG(WARNING) << "Unknown compute capability "
                 << compute_capability.ToString()
                 << ". Defaulting to telling LLVM that we're compiling for sm_"
                 << sm_version;
  }
  return absl::StrCat("sm_", sm_version);
}

// Convenience function for producing a name of a temporary compilation product
// from the input filename.
string MakeNameForTempProduct(absl::string_view input_filename,
                              absl::string_view extension) {
  return ReplaceFilenameExtension(tensorflow::io::Basename(input_filename),
                                  extension);
}

// Initializes LLVM passes. Uses the PassRegistry mechanism.
void InitializePasses(llvm::PassRegistry* pass_registry) {
  llvm::initializeCore(*pass_registry);
  llvm::initializeCodeGen(*pass_registry);
  llvm::initializeScalarOpts(*pass_registry);
  llvm::initializeObjCARCOpts(*pass_registry);
  llvm::initializeVectorization(*pass_registry);
  llvm::initializeIPO(*pass_registry);
  llvm::initializeAnalysis(*pass_registry);
  llvm::initializeTransformUtils(*pass_registry);
  llvm::initializeInstCombine(*pass_registry);
  llvm::initializeInstrumentation(*pass_registry);
  llvm::initializeTarget(*pass_registry);
  llvm::initializeCodeGenPreparePass(*pass_registry);
}

// Returns the TargetMachine, given a triple.
std::unique_ptr<llvm::TargetMachine> GetTargetMachine(
    llvm::Triple triple, absl::string_view cpu_name,
    const HloModuleConfig& hlo_module_config, absl::string_view feature_str) {
  std::string error;
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget("", triple, error);
  if (target == nullptr) {
    LOG(FATAL) << "Unable to find Target for triple '" << triple.str() << "'"
               << " -- " << error;
    return nullptr;
  }

  llvm::TargetOptions target_options =
      llvm::codegen::InitTargetOptionsFromCodeGenFlags(llvm::Triple());

  // Set the verbose assembly options.
  target_options.MCOptions.AsmVerbose = false;

  // The selection of codegen optimization level is copied from function
  // GetCodeGenOptLevel in //third_party/llvm/llvm/tools/opt/opt.cpp.
  llvm::CodeGenOpt::Level codegen_opt_level;
  switch (hlo_module_config.debug_options().xla_backend_optimization_level()) {
    case 1:
      codegen_opt_level = llvm::CodeGenOpt::Less;
      break;
    case 2:
      codegen_opt_level = llvm::CodeGenOpt::Default;
      break;
    case 3:
      codegen_opt_level = llvm::CodeGenOpt::Aggressive;
      break;
    default:
      codegen_opt_level = llvm::CodeGenOpt::None;
  }
  return absl::WrapUnique(target->createTargetMachine(
      triple.str(), llvm_ir::AsStringRef(cpu_name),
      llvm_ir::AsStringRef(feature_str), target_options,
      llvm::codegen::getExplicitRelocModel(),
      llvm::codegen::getExplicitCodeModel(), codegen_opt_level));
}

// Adds the standard LLVM optimization passes, based on the speed optimization
// level (opt_level) and size optimization level (size_level). Both module
// and function-level passes are added, so two pass managers are passed in and
// modified by this function.
void AddOptimizationPasses(unsigned opt_level, unsigned size_level,
                           llvm::TargetMachine* target_machine,
                           llvm::legacy::PassManagerBase* module_passes,
                           llvm::legacy::FunctionPassManager* function_passes,
                           int inline_threshold) {
  llvm::PassManagerBuilder builder;
  builder.OptLevel = opt_level;
  builder.SizeLevel = size_level;

  if (opt_level > 1) {
    builder.Inliner = llvm::createFunctionInliningPass(inline_threshold);
  } else {
    // Only inline functions marked with "alwaysinline".
    builder.Inliner = llvm::createAlwaysInlinerLegacyPass();
  }

  builder.DisableUnrollLoops = opt_level == 0;
  builder.LoopVectorize = opt_level > 0;
  builder.SLPVectorize = opt_level > 1 && size_level < 2;

  // NVPTX's early-as-possible passes include NVVM reflect.
  target_machine->adjustPassManager(builder);

  builder.populateFunctionPassManager(*function_passes);
  builder.populateModulePassManager(*module_passes);
}

// Emits the given module to a bit code file.
void EmitBitcodeToFile(const llvm::Module& module, absl::string_view filename) {
  std::error_code error_code;
  llvm::ToolOutputFile outfile(string(filename).c_str(), error_code,
                               llvm::sys::fs::OF_None);
  if (error_code) {
    LOG(FATAL) << "opening bitcode file for writing: " << error_code.message();
  }

  llvm::WriteBitcodeToFile(module, outfile.os());
  outfile.keep();
}

// Emits the given module to PTX. target_machine is an initialized TargetMachine
// for the NVPTX target.
string EmitModuleToPTX(llvm::Module* module,
                       llvm::TargetMachine* target_machine) {
  std::string ptx;
  {
    llvm::raw_string_ostream stream(ptx);
    llvm::buffer_ostream pstream(stream);
    // The extension is stripped by IrDumpingPassManager, so we need to
    // get creative to add a suffix.
    IrDumpingPassManager codegen_passes(
        MakeNameForTempProduct(module->getModuleIdentifier(), "-nvptx.dummy"),
        "", false);
    codegen_passes.add(new llvm::TargetLibraryInfoWrapperPass(
        llvm::Triple(module->getTargetTriple())));

    target_machine->addPassesToEmitFile(codegen_passes, pstream, nullptr,
                                        llvm::CGFT_AssemblyFile);
    codegen_passes.run(*module);
  }

  return ptx;
}

// LLVM has an extensive flags mechanism of its own, which is only accessible
// through the command line. Internal libraries within LLVM register parsers for
// flags, with no other way to configure them except pass these flags.
// To do this programmatically, we invoke ParseCommandLineOptions manually with
// a "fake argv".
// Note: setting flags with this method is stateful, since flags are just
// static globals within LLVM libraries.
void FeedLLVMWithFlags(const std::vector<string>& cl_opts) {
  std::vector<const char*> fake_argv = {""};
  for (const string& cl_opt : cl_opts) {
    fake_argv.push_back(cl_opt.c_str());
  }
  llvm::cl::ParseCommandLineOptions(fake_argv.size(), &fake_argv[0]);
}

// Returns whether the module could use any device bitcode library functions.
bool CouldNeedDeviceBitcode(const llvm::Module& module) {
  for (const llvm::Function& function : module.functions()) {
    // The list of prefixes should be in sync with library functions used in
    // target_util.cc.
    if (!function.isIntrinsic() && function.isDeclaration() &&
        (function.getName().startswith("__nv_") ||
         function.getName().startswith("__ocml_") ||
         function.getName().startswith("__ockl_"))) {
      return true;
    }
  }
  return false;
}

// Links the module with a vector of path to bitcode modules.
// The caller must guarantee that the paths exist.
Status LinkWithBitcodeVector(llvm::Module* module,
                             const std::vector<string>& bitcode_path_vector) {
  llvm::Linker linker(*module);

  for (auto& bitcode_path : bitcode_path_vector) {
    if (!tensorflow::Env::Default()->FileExists(bitcode_path).ok()) {
      LOG(ERROR) << "bitcode module is required by this HLO module but was "
                    "not found at "
                 << bitcode_path;
      return xla::InternalError("bitcode module not found at %s", bitcode_path);
    }

    std::unique_ptr<llvm::Module> bitcode_module =
        LoadIRModule(bitcode_path, &module->getContext());
    // Ignore the data layout of the module we're importing. This avoids a
    // warning from the linker.
    bitcode_module->setDataLayout(module->getDataLayout());
    if (linker.linkInModule(
            std::move(bitcode_module), llvm::Linker::Flags::LinkOnlyNeeded,
            [](llvm::Module& M, const llvm::StringSet<>& GVS) {
              internalizeModule(M, [&GVS](const llvm::GlobalValue& GV) {
                return !GV.hasName() || (GVS.count(GV.getName()) == 0);
              });
            })) {
      return xla::InternalError("Error linking bitcode module from %s",
                                bitcode_path);
    }
  }
  return Status::OK();
}

// Links libdevice into the given module if the module needs libdevice.
Status LinkLibdeviceIfNecessary(llvm::Module* module,
                                const string& libdevice_dir_path) {
  if (!CouldNeedDeviceBitcode(*module)) {
    return Status::OK();
  }

  // CUDA 9+ uses a single libdevice file for all devices, and we don't support
  // older CUDAs.
  string libdevice_path =
      tensorflow::io::JoinPath(libdevice_dir_path, "libdevice.10.bc");
  if (!tensorflow::Env::Default()->FileExists(libdevice_path).ok()) {
    LOG(WARNING)
        << "libdevice is required by this HLO module but was not found at "
        << libdevice_path;
    return xla::InternalError("libdevice not found at %s", libdevice_path);
  }

  VLOG(1) << "Linking with libdevice from: " << libdevice_path;
  return LinkWithBitcodeVector(module, {libdevice_path});
}

Status NVPTXTargetModuleLinker(llvm::Module* module, GpuVersion gpu_version,
                               const HloModuleConfig& hlo_module_config,
                               const string& device_bitcode_dir_path) {
  // Link the input module with libdevice, to pull in implementations of some
  // builtins.
  TF_RETURN_IF_ERROR(LinkLibdeviceIfNecessary(module, device_bitcode_dir_path));

  // Set the flush-denormals-to-zero flag on the module so the NVVM reflect pass
  // can access it.
  module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz",
                        hlo_module_config.debug_options().xla_gpu_ftz());

  // If ftz is enabled, set it as an attribute on every function in the module.
  if (hlo_module_config.debug_options().xla_gpu_ftz()) {
    for (llvm::Function& fn : *module) {
      fn.addFnAttr("denormal-fp-math-f32", "preserve-sign");
    }
  }

  return Status::OK();
}

std::unique_ptr<llvm::TargetMachine> NVPTXGetTargetMachine(
    llvm::Triple target_triple, se::CudaComputeCapability compute_capability,
    const HloModuleConfig& hlo_module_config) {
  // Figure out the exact name of the processor as known to the NVPTX backend
  // from the gpu_architecture flag.
  return GetTargetMachine(target_triple, GetSmName(compute_capability),
                          hlo_module_config, "+ptx60");
}

using TargetModuleLinker = std::function<Status(
    llvm::Module*, GpuVersion, const HloModuleConfig&, const string&)>;

Status LinkAndOptimizeModule(llvm::Module* module, GpuVersion gpu_version,
                             const HloModuleConfig& hlo_module_config,
                             const string& device_bitcode_dir_path,
                             TargetModuleLinker module_linker,
                             llvm::Triple default_target_triple,
                             llvm::TargetMachine* target_machine,
                             int inline_threshold) {
  TF_RETURN_IF_ERROR(module_linker(module, gpu_version, hlo_module_config,
                                   device_bitcode_dir_path));

  bool dump_ir = hlo_module_config.debug_options().xla_gpu_dump_llvmir();
  std::string outputs_dir;
  tensorflow::io::GetTestUndeclaredOutputsDir(&outputs_dir);
  IrDumpingPassManager module_passes(module->getModuleIdentifier(), outputs_dir,
                                     dump_ir);

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  llvm::TargetLibraryInfoWrapperPass* tliwp =
      new llvm::TargetLibraryInfoWrapperPass(
          llvm::Triple(module->getTargetTriple()));
  module_passes.add(tliwp);

  // Try to fetch the target triple from the module. If not present, set a
  // default target triple.
  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());
  if (target_triple.getArch() == llvm::Triple::UnknownArch) {
    LOG(WARNING) << "target triple not found in the module";
    target_triple = default_target_triple;
  }

  module_passes.add(llvm::createTargetTransformInfoWrapperPass(
      target_machine->getTargetIRAnalysis()));

  // The LLVM IR verifier performs sanity checking on the IR. This helps
  // discover problems and report them in a meaningful manner, rather than let
  // later passes report obscure assertions because of unfulfilled invariants.
  module_passes.add(llvm::createVerifierPass());

  // Create the function-level pass manager. It needs data layout information
  // too.
  llvm::legacy::FunctionPassManager function_passes(module);

  int32 opt_level =
      hlo_module_config.debug_options().xla_backend_optimization_level();

  if (opt_level < 2) {
    LOG(ERROR) << std::string(80, '*');
    LOG(ERROR) << "The XLA GPU backend doesn't support unoptimized code "
                  "generation but ";
    LOG(ERROR) << "--xla_backend_optimization_level is set to " << opt_level
               << "!";
    LOG(ERROR) << "(Supported configuration is "
                  "--xla_backend_optimization_level >= 2.)";
    LOG(ERROR) << std::string(80, '*');
  }

  // Add optimization passes, and set inliner threshold.
  AddOptimizationPasses(opt_level,
                        /*size_level=*/0, target_machine, &module_passes,
                        &function_passes, inline_threshold);

  // Loop unrolling exposes more opportunities for SROA. Therefore, we run SROA
  // again after the standard optimization passes [http://b/13329423].
  // TODO(jingyue): SROA may further expose more optimization opportunities such
  // as more precise alias analysis and more function inlining (SROA may change
  // the inlining cost of a function). For now, running SROA already emits good
  // enough code for the evaluated benchmarks. We may want to run more
  // optimizations later.
  if (opt_level > 0) {
    // LLVM's optimizer turns on SROA when the optimization level is greater
    // than 0. We mimic this behavior here.
    module_passes.add(llvm::createSROAPass());
  }

  // Verify that the module is well formed after optimizations ran.
  module_passes.add(llvm::createVerifierPass());

  // Done populating the pass managers. Now run them.

  function_passes.doInitialization();
  for (auto func = module->begin(); func != module->end(); ++func) {
    function_passes.run(*func);
  }
  function_passes.doFinalization();
  module_passes.run(*module);

  return Status::OK();
}

// One-time module initializer.
// Must be called only once -- DO NOT CALL DIRECTLY.
void NVPTXBackendInit(const HloModuleConfig& hlo_module_config) {
  // Feed all customized flags here, so we can override them with llvm_cl_opts
  // without redeploy the compiler for development purpose.

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
  FeedLLVMWithFlags({"-bonus-inst-threshold=2"});
  // Increase limit when scanning memory dependencies.  This helps to reduce
  // more redundant load instructions.
  //
  // The specific value is currently large enough for s3d in shoc benchmark,
  // which contains a lot of load instructions and many arithmetic instructions
  // between those loads.
  FeedLLVMWithFlags({"-memdep-block-scan-limit=500"});

  // Use div.full -- it matters for some float-division heavy benchmarks.
  // Using div.approx produces incorrect result for float32(max)/float32(max).
  FeedLLVMWithFlags({"-nvptx-prec-divf32=1"});

  llvm_ir::InitializeLLVMCommandLineOptions(hlo_module_config);

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

namespace nvptx {

StatusOr<string> CompileToPtx(
    llvm::Module* module, GpuVersion gpu_version,
    const HloModuleConfig& hlo_module_config, const string& libdevice_dir_path,
    std::function<void(llvm::TargetMachine*)> configure_target) {
  static absl::once_flag backend_init_flag;
  absl::call_once(backend_init_flag, NVPTXBackendInit, hlo_module_config);

  string ptx;
  std::unique_ptr<llvm::TargetMachine> target_machine;
  {
    tensorflow::profiler::TraceMe activity(
        [&] { return absl::StrCat("Compiling IR:", module->getName().str()); },
        tensorflow::profiler::TraceMeLevel::kInfo);
    XLA_SCOPED_LOGGING_TIMER("Compile module " + module->getName().str());

    // If the module has no functions or globals, there's nothing to compile.
    // Just return an empty string.
    if (module->empty() && module->global_empty()) {
      VLOG(2) << "Module '" << module->getName().str()
              << "' is empty. Skipping compilation.";
      return string();
    }

    auto compute_capability =
        absl::get_if<se::CudaComputeCapability>(&gpu_version);
    if (!compute_capability) {
      return xla::InternalError(
          "Incompatible compute capability was specified.");
    }

    llvm::Triple default_target_triple("nvptx64-unknown-unknown");
    // Construct LLVM TargetMachine for NVPTX.
    std::unique_ptr<llvm::TargetMachine> target_machine = NVPTXGetTargetMachine(
        default_target_triple, *compute_capability, hlo_module_config);

    // Apply target machine configuration from call-back if available.
    if (configure_target) {
      configure_target(target_machine.get());
    }

    // Link with libdevice, and optimize the LLVM module.
    TF_RETURN_IF_ERROR(LinkAndOptimizeModule(
        module, gpu_version, hlo_module_config, libdevice_dir_path,
        NVPTXTargetModuleLinker, default_target_triple, target_machine.get(),
        kDefaultInlineThreshold));

    // Lower optimized LLVM module to PTX.
    ptx = EmitModuleToPTX(module, target_machine.get());
  }
  return ptx;
}

}  // namespace nvptx

namespace {

// Gets the ROCm-Device-Libs filenames for a particular AMDGPU version.
std::vector<string> GetROCDLPaths(std::string amdgpu_version,
                                  const string& rocdl_dir_path) {
  // AMDGPU version-neutral bitcodes.
#if TF_ROCM_VERSION >= 30900
  static std::vector<string>* rocdl_filenames = new std::vector<string>(
      {"hc.bc", "opencl.bc", "ocml.bc", "ockl.bc", "oclc_finite_only_off.bc",
       "oclc_daz_opt_off.bc", "oclc_correctly_rounded_sqrt_on.bc",
       "oclc_unsafe_math_off.bc", "oclc_wavefrontsize64_on.bc"});
#else
  static std::vector<string>* rocdl_filenames = new std::vector<string>(
      {"hc.amdgcn.bc", "opencl.amdgcn.bc", "ocml.amdgcn.bc", "ockl.amdgcn.bc",
       "oclc_finite_only_off.amdgcn.bc", "oclc_daz_opt_off.amdgcn.bc",
       "oclc_correctly_rounded_sqrt_on.amdgcn.bc",
       "oclc_unsafe_math_off.amdgcn.bc", "oclc_wavefrontsize64_on.amdgcn.bc"});
#endif

  // Construct full path to ROCDL bitcode libraries.
  std::vector<string> result;
  for (auto& filename : *rocdl_filenames) {
    result.push_back(tensorflow::io::JoinPath(rocdl_dir_path, filename));
  }

  // Add AMDGPU version-specific bitcodes.
  std::vector<std::string> tokens = absl::StrSplit(amdgpu_version, ':');
  if (!tokens.empty() && tokens[0].size() >= 3) {
    amdgpu_version = tokens[0].substr(3);
  }
  result.push_back(tensorflow::io::JoinPath(
      rocdl_dir_path,
#if TF_ROCM_VERSION >= 30900
      absl::StrCat("oclc_isa_version_", amdgpu_version, ".bc")));
#else
      absl::StrCat("oclc_isa_version_", amdgpu_version, ".amdgcn.bc")));
#endif
  return result;
}

struct HsacoCacheEntry {
  uint64 hash;
  std::string ir;
  std::string gfx;
  std::vector<uint8> hsaco;
};

struct HsacoCache {
 protected:
  std::vector<HsacoCacheEntry> cache;
  std::mutex m_mutex;
  int request_count = 0;
  int hit_count = 0;

 public:
  static bool Find(const std::string& ir, uint64_t& hash,
                   const std::string& gfx, std::vector<uint8>& hsaco);
  static void Add(const std::string& ir, uint64_t hash, const std::string& gfx,
                  const std::vector<uint8>& hsaco);
};

static HsacoCache g_hsacoCache;

bool HsacoCache::Find(const std::string& ir, uint64_t& hash,
                      const std::string& gfx, std::vector<uint8>& hsaco) {
  std::lock_guard<std::mutex> lg(g_hsacoCache.m_mutex);
  hash = std::hash<std::string>{}(ir);
  bool hit = false;
  for (auto& x : g_hsacoCache.cache) {
    if (x.hash != hash) continue;
    if (x.gfx != gfx) continue;
    if (x.ir != ir) continue;
    hsaco = x.hsaco;
    hit = true;
    break;
  }
  g_hsacoCache.request_count++;
  if (hit) g_hsacoCache.hit_count++;
  if (!(g_hsacoCache.request_count % 50))
    VLOG(1) << "HSACO cache: " << g_hsacoCache.request_count << " requests, "
            << g_hsacoCache.hit_count << " hits";
  return hit;
}

void HsacoCache::Add(const std::string& ir, uint64_t hash,
                     const std::string& gfx, const std::vector<uint8>& hsaco) {
  std::lock_guard<std::mutex> lg(g_hsacoCache.m_mutex);
  g_hsacoCache.cache.resize(g_hsacoCache.cache.size() + 1);
  g_hsacoCache.cache.back().ir = ir;
  g_hsacoCache.cache.back().hash = hash;
  g_hsacoCache.cache.back().gfx = gfx;
  g_hsacoCache.cache.back().hsaco = hsaco;
}

// Emits the given module to HSA Code Object. target_machine is an initialized
// TargetMachine for the AMDGPU target.
StatusOr<std::vector<uint8>> EmitModuleToHsaco(
    llvm::Module* module, llvm::TargetMachine* target_machine) {
  auto* env = tensorflow::Env::Default();
  std::vector<std::string> tempdir_vector;
  env->GetLocalTempDirectories(&tempdir_vector);
  if (tempdir_vector.empty()) {
    return xla::InternalError(
        "Unable to locate a temporary directory for compile-time artifacts.");
  }
  std::string tempdir_name = tempdir_vector.front();
  VLOG(1) << "Compile-time artifacts located at: " << tempdir_name;

  bool keep_tempfiles = false;
  TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("TF_ROCM_KEEP_XLA_TEMPFILES",
                                             /*default_val=*/false,
                                             &keep_tempfiles));
  // Prepare filenames for all stages of compilation:
  // IR, binary ISA, and HSACO.
  std::string random_number = std::to_string(tensorflow::random::New64());
  std::string ir_filename =
      absl::StrCat(module->getModuleIdentifier(), random_number + ".ll");
  std::string ir_path = tensorflow::io::JoinPath(tempdir_name, ir_filename);

  std::string ir_opt_filename =
      absl::StrCat(module->getModuleIdentifier(), random_number + "_opt.ll");
  std::string ir_opt_path =
      tensorflow::io::JoinPath(tempdir_name, ir_opt_filename);

  std::string isabin_filename =
      absl::StrCat(module->getModuleIdentifier(), random_number + ".o");
  std::string isabin_path =
      tensorflow::io::JoinPath(tempdir_name, isabin_filename);

  std::string hsaco_filename =
      absl::StrCat(module->getModuleIdentifier(), random_number + ".hsaco");
  std::string hsaco_path =
      tensorflow::io::JoinPath(tempdir_name, hsaco_filename);

  std::error_code ec;

  // Dump LLVM IR.
  std::unique_ptr<llvm::raw_fd_ostream> ir_fs(
      new llvm::raw_fd_ostream(ir_path, ec, llvm::sys::fs::OF_None));
  module->print(*ir_fs, nullptr);
  ir_fs->flush();

  // Emit GCN ISA binary.
  // The extension is stripped by IrDumpingPassManager, so we need to
  // get creative to add a suffix.
  std::string module_id = module->getModuleIdentifier();
  IrDumpingPassManager codegen_passes(
      ReplaceFilenameExtension(tensorflow::io::Basename(module_id),
                               random_number + "-amdgpu.dummy"),
      "", false);
  codegen_passes.add(new llvm::TargetLibraryInfoWrapperPass(
      llvm::Triple(module->getTargetTriple())));
  llvm::SmallVector<char, 0> stream;
  llvm::raw_svector_ostream pstream(stream);
  std::unique_ptr<llvm::raw_fd_ostream> isabin_fs(
      new llvm::raw_fd_ostream(isabin_path, ec, llvm::sys::fs::OF_Text));
  module->setDataLayout(target_machine->createDataLayout());
  target_machine->addPassesToEmitFile(codegen_passes, *isabin_fs, nullptr,
                                      llvm::CGFT_ObjectFile);
  codegen_passes.run(*module);
  isabin_fs->flush();

  if (keep_tempfiles) {
    std::unique_ptr<llvm::raw_fd_ostream> ir_fs(
        new llvm::raw_fd_ostream(ir_opt_path, ec, llvm::sys::fs::OF_None));
    module->print(*ir_fs, nullptr);
    ir_fs->flush();
  }
  // Locate lld.
  // TODO(whchung@gmail.com): change to tensorflow::ROCmRoot() after
  // ROCm-Device-Libs PR.
  std::string lld_path_1 = tensorflow::io::JoinPath("/opt/rocm", "hcc/bin");
  std::string lld_path_2 = tensorflow::io::JoinPath("/opt/rocm", "llvm/bin");
  auto lld_program =
      llvm::sys::findProgramByName("ld.lld", {lld_path_1, lld_path_2});
  if (!lld_program) {
    return xla::InternalError("unable to find ld.lld in PATH: %s",
                              lld_program.getError().message());
  }
  std::vector<llvm::StringRef> lld_args{
      llvm_ir::AsStringRef("ld.lld"),
      llvm_ir::AsStringRef("-flavor"),
      llvm_ir::AsStringRef("gnu"),
      llvm_ir::AsStringRef("-shared"),
      llvm_ir::AsStringRef(isabin_path),
      llvm_ir::AsStringRef("-o"),
      llvm_ir::AsStringRef(hsaco_path),
  };

  std::string error_message;
  int lld_result =
      llvm::sys::ExecuteAndWait(*lld_program, llvm_ir::AsArrayRef(lld_args),
                                llvm::None, {}, 0, 0, &error_message);
  if (lld_result) {
    return xla::InternalError("ld.lld execute fail: %s, error code %d",
                              error_message, lld_result);
  }

  // Read HSACO.
  std::ifstream hsaco_file(hsaco_path, std::ios::binary | std::ios::ate);
  std::ifstream::pos_type hsaco_file_size = hsaco_file.tellg();

  std::vector<uint8> hsaco(hsaco_file_size);
  hsaco_file.seekg(0, std::ios::beg);
  hsaco_file.read(reinterpret_cast<char*>(&hsaco[0]), hsaco_file_size);
  hsaco_file.close();
  if (!keep_tempfiles) {
    remove(ir_path.c_str());
    remove(isabin_path.c_str());
    remove(hsaco_path.c_str());
  }
  return hsaco;
}

// Links ROCm-Device-Libs into the given module if the module needs it.
Status LinkROCDLIfNecessary(llvm::Module* module, std::string amdgpu_version,
                            const string& rocdl_dir_path) {
  if (!CouldNeedDeviceBitcode(*module)) {
    return Status::OK();
  }

  return LinkWithBitcodeVector(module,
                               GetROCDLPaths(amdgpu_version, rocdl_dir_path));
}

Status AMDGPUTargetModuleLinker(llvm::Module* module, GpuVersion gpu_version,
                                const HloModuleConfig& hlo_module_config,
                                const string& device_bitcode_dir_path) {
  // Link the input module with ROCDL.
  auto amdgpu_version = absl::get_if<std::string>(&gpu_version);
  if (!amdgpu_version) {
    return xla::InternalError(
        "Incompatible AMD GCN ISA version was specified.");
  }
  TF_RETURN_IF_ERROR(
      LinkROCDLIfNecessary(module, *amdgpu_version, device_bitcode_dir_path));

  // If ftz is enabled, set it as an attribute on every function in the module.
  if (hlo_module_config.debug_options().xla_gpu_ftz()) {
    for (llvm::Function& fn : *module) {
      fn.addFnAttr("denormal-fp-math-f32", "preserve-sign");
    }
  }

  return Status::OK();
}

// The following routine maps a feature token extracted from the
// hipDeviceProp_t::gcnArchName string, and maps it to a valid feature_str
// to be used for creating the AMDGPUTarget.
// This mapping is currently in a state of flux because TF XLA uses its
// own copy of LLVM, which is different from the LLVM version used by
// hipcc/runtime in the ROCm install. Ordinarily this is not a problem,
// but right now, the LLVM version used by hipcc/runtime has "targetID"
// related changes which have not yet been upstreamed (to the LLVM repo)
// When that upstreaming happens (and TF LLVM pointer moves past the
// upstream commit), the following mapping will need to change
std::string MapGCNArchNameTokenToFeatureStr(const std::string& token) {
  if (token == "sramecc+") {
    return "+sramecc";
  } else if (token == "sramecc-") {
    return "-sramecc";
  } else if (token == "xnack+") {
    return "+xnack";
  } else if (token == "xnack-") {
    return "-xnack";
  }
  return "";
}

std::pair<std::string, std::string> GetFeatureStrFromGCNArchName(
    const std::string& gcn_arch_name) {
  std::string feature_str;

  std::string gfx = gcn_arch_name;
#if TF_ROCM_VERSION < 30900
  // For ROCm versions older than 3.9, hardcode it to "+code-object-v3"
  // This is simply to preserve how things were...nohing else
  feature_str = "+code-object-v3";
#elif TF_ROCM_VERSION < 40000
  // For ROCM versions 3.9 and 3.10, hardcode it to empty string
  feature_str = "";
#else
  // For ROCm versions 4.0 and greater, we need to specify the correct
  // feature str, based on the underlying GPU HW to get max performance.
  std::vector<std::string> tokens = absl::StrSplit(gcn_arch_name, ':');
  std::vector<std::string> mapped_tokens;
  if (tokens.size() > 0) gfx = tokens[0];
  for (auto it = tokens.begin(); it != tokens.end(); it++) {
    // Skip the first token, that is the gfxNNN str
    // The rest of the tokens are the feature/targetid strings
    if (it != tokens.begin()) {
      std::string token(*it);
      std::string mapped_token = MapGCNArchNameTokenToFeatureStr(token);
      mapped_tokens.push_back(mapped_token);
    }
  }
  feature_str = absl::StrJoin(mapped_tokens, ",");
#endif

  return std::make_pair(gfx, feature_str);
}

std::unique_ptr<llvm::TargetMachine> AMDGPUGetTargetMachine(
    llvm::Triple target_triple, GpuVersion gpu_version,
    const HloModuleConfig& hlo_module_config) {
  auto amdgpu_version = absl::get_if<std::string>(&gpu_version);
  std::string gcn_arch_name = *amdgpu_version;
  auto arch = GetFeatureStrFromGCNArchName(gcn_arch_name);
  return GetTargetMachine(std::move(target_triple), arch.first,
                          hlo_module_config, arch.second);
}

void AMDGPUBackendInit(const HloModuleConfig& hlo_module_config) {
  llvm_ir::InitializeLLVMCommandLineOptions(hlo_module_config);

  // Initialize the AMDGPU target; it's the only target we link with, so call
  // its specific initialization functions instead of the catch-all
  // InitializeAll*.
#if TENSORFLOW_USE_ROCM
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();

#if TF_ROCM_VERSION < 40100
  // Use code-object-v3 for ROCm versions 4.0.1 and lower, since the
  // HIP runtime for those ROCm versions expects the v3 HSACO objects
  // Default is now v4 for newer LLVM versions (starting around 210326)
  FeedLLVMWithFlags({"--amdhsa-code-object-version=3"});
#endif

#endif

  llvm::PassRegistry* registry = llvm::PassRegistry::getPassRegistry();
  InitializePasses(registry);
}

}  // namespace

namespace amdgpu {
StatusOr<std::vector<uint8>> CompileToHsaco(
    llvm::Module* module, GpuVersion gpu_version,
    const HloModuleConfig& hlo_module_config, const string& rocdl_dir_path) {
  static absl::once_flag backend_init_flag;
  absl::call_once(backend_init_flag, AMDGPUBackendInit, hlo_module_config);

  std::vector<uint8> hsaco;
  std::unique_ptr<llvm::TargetMachine> target_machine;
  std::string str;
  llvm::raw_string_ostream stream(str);
  stream << *module;
  // Delete the first two lines, since they usually vary even when the rest of
  // the code is the same (but verify that they are what we expect).
  if (str.size() >= 13 && str.substr(0, 13) == "; ModuleID = ") {
    auto pos = str.find('\n');
    if (pos != std::string::npos) str = str.substr(pos + 1);
  }
  if (str.size() >= 18 && str.substr(0, 18) == "source_filename = ") {
    auto pos = str.find('\n');
    if (pos != std::string::npos) str = str.substr(pos + 1);
  }
  str += hlo_module_config.compilation_cache_key();
  {
    tensorflow::profiler::TraceMe activity(
        [&] { return absl::StrCat("Compiling IR", module->getName().str()); },
        tensorflow::profiler::TraceMeLevel::kInfo);
    XLA_SCOPED_LOGGING_TIMER("Compile module " + module->getName().str());

    auto amdgpu_version = absl::get_if<std::string>(&gpu_version);
    if (!amdgpu_version) {
      return xla::InternalError(
          "Incompatible AMD GCN ISA version was specified.");
    }
    uint64_t hash;
    if (HsacoCache::Find(str, hash, *amdgpu_version, hsaco)) {
      VLOG(1) << "HSACO cache hit";
      return hsaco;
    }
    VLOG(1) << "HSACO cache miss";
    bool dump_lls = false;
    if (dump_lls) {
      static int hsaco_count = 0;
      std::string name = "/tmp/" + std::to_string(hsaco_count) + ".ll";
      hsaco_count++;
      std::ofstream ofs(name);
      ofs << str;
      ofs.close();
    }

    llvm::Triple default_target_triple("amdgcn--amdhsa-amdgiz");
    // Construct LLVM TargetMachine for AMDGPU.
    std::unique_ptr<llvm::TargetMachine> target_machine =
        AMDGPUGetTargetMachine(default_target_triple, gpu_version,
                               hlo_module_config);

    // Link with ROCm-Device-Libs, and optimize the LLVM module.
    TF_RETURN_IF_ERROR(LinkAndOptimizeModule(
        module, gpu_version, hlo_module_config, rocdl_dir_path,
        AMDGPUTargetModuleLinker, default_target_triple, target_machine.get(),
        kAMDGPUInlineThreshold));

    // Lower optimized LLVM module to HSA code object.
    TF_ASSIGN_OR_RETURN(hsaco, EmitModuleToHsaco(module, target_machine.get()));
    HsacoCache::Add(str, hash, *amdgpu_version, hsaco);
  }
  return hsaco;
}

}  // namespace amdgpu

}  // namespace gpu
}  // namespace xla
