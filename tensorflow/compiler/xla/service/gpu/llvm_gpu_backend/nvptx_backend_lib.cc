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

#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/nvptx_backend_lib.h"

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/dump_ir_pass.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/utils.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.inc"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"

namespace xla {
namespace gpu {
namespace {

// Default inline threshold value to use in llvm.
const int kDefaultInlineThreshold = 1100;

// Gets the libdevice filename for a particular compute capability.  When
// presented with a GPU we don't recognize, we just return the libdevice from
// compute_20.
static string GetLibdeviceFilename(const string& libdevice_dir_path,
                                   std::pair<int, int> compute_capability) {
  // Since CUDA 9.0, all GPU versions are included in a single file
  const char* unified_libdevice_filename = "libdevice.10.bc";
  std::vector<string> unified_libdevice_files;
  const Status status = tensorflow::Env::Default()->GetMatchingPaths(
      tensorflow::io::JoinPath(libdevice_dir_path, unified_libdevice_filename),
      &unified_libdevice_files);
  if (status.ok() && unified_libdevice_files.size() == 1) {
    return unified_libdevice_filename;
  }
  // There are only four libdevice files: compute_{20,30,35,50}.  Each GPU
  // version gets mapped to one of these.  Note in particular that sm_60 and
  // sm_61 map to libdevice.compute_30.
  static auto* m = new std::map<std::pair<int, int>, int>({{{2, 0}, 20},
                                                           {{2, 1}, 20},
                                                           {{3, 0}, 30},
                                                           {{3, 2}, 30},
                                                           {{3, 5}, 35},
                                                           {{3, 7}, 35},
                                                           {{5, 0}, 50},
                                                           {{5, 2}, 50},
                                                           {{5, 3}, 50},
                                                           {{6, 0}, 30},
                                                           {{6, 1}, 30},
                                                           {{6, 2}, 30}});
  int libdevice_version = 20;
  auto it = m->find(compute_capability);
  if (it != m->end()) {
    libdevice_version = it->second;
  } else {
    LOG(WARNING) << "Unknown compute capability (" << compute_capability.first
                 << ", " << compute_capability.second << ") ."
                 << "Defaulting to libdevice for compute_" << libdevice_version;
  }
  return tensorflow::strings::StrCat("libdevice.compute_", libdevice_version,
                                     ".10.bc");
}

// Gets the GPU name as it's known to LLVM for a given compute capability.  If
// we see an unrecognized compute capability, we return "sm_30".
static string GetSmName(std::pair<int, int> compute_capability) {
  static auto* m = new std::map<std::pair<int, int>, int>({{{2, 0}, 20},
                                                           {{2, 1}, 21},
                                                           {{3, 0}, 30},
                                                           {{3, 2}, 32},
                                                           {{3, 5}, 35},
                                                           {{3, 7}, 37},
                                                           {{5, 0}, 50},
                                                           {{5, 2}, 52},
                                                           {{5, 3}, 53},
                                                           {{6, 0}, 60},
                                                           {{6, 1}, 61},
                                                           {{6, 2}, 62},
                    // TODO: Change this to 70 once LLVM NVPTX supports it
                                                           {{7, 0}, 60}});
  int sm_version = 30;
  auto it = m->find(compute_capability);
  if (it != m->end()) {
    sm_version = it->second;
  } else {
    LOG(WARNING) << "Unknown compute capability (" << compute_capability.first
                 << ", " << compute_capability.second << ") ."
                 << "Defaulting to telling LLVM that we're compiling for sm_"
                 << sm_version;
  }
  return tensorflow::strings::StrCat("sm_", sm_version);
}

// Convenience function for producing a name of a temporary compilation product
// from the input filename.
string MakeNameForTempProduct(const std::string& input_filename,
                              tensorflow::StringPiece extension) {
  return ReplaceFilenameExtension(
      tensorflow::io::Basename(llvm_ir::AsString(input_filename)), extension);
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
    llvm::Triple triple, tensorflow::StringPiece cpu_name,
    const HloModuleConfig& hlo_module_config) {
  std::string error;
  const llvm::Target* target = TargetRegistry::lookupTarget("", triple, error);
  if (target == nullptr) {
    LOG(FATAL) << "Unable to find Target for triple '" << triple.str() << "'"
               << " -- " << error;
    return nullptr;
  }

  TargetOptions target_options = InitTargetOptionsFromCodeGenFlags();
  llvm_ir::SetTargetOptions(
      /*fast_math_enabled=*/hlo_module_config.debug_options()
          .xla_enable_fast_math(),
      &target_options);

  // Enable FMA synthesis.
  target_options.AllowFPOpFusion = FPOpFusion::Fast;

  // Set the verbose assembly options.
  target_options.MCOptions.AsmVerbose = false;

  // The selection of codegen optimization level is copied from function
  // GetCodeGenOptLevel in //third_party/llvm/llvm/tools/opt/opt.cpp.
  CodeGenOpt::Level codegen_opt_level;
  switch (hlo_module_config.debug_options().xla_backend_optimization_level()) {
    case 1:
      codegen_opt_level = CodeGenOpt::Less;
      break;
    case 2:
      codegen_opt_level = CodeGenOpt::Default;
      break;
    case 3:
      codegen_opt_level = CodeGenOpt::Aggressive;
      break;
    default:
      codegen_opt_level = CodeGenOpt::None;
  }
  return WrapUnique(target->createTargetMachine(
      triple.str(), llvm_ir::AsStringRef(cpu_name), "+ptx42", target_options,
      Optional<Reloc::Model>(RelocModel), Optional<CodeModel::Model>(CMModel),
      codegen_opt_level));
}

// Adds the standard LLVM optimization passes, based on the speed optimization
// level (opt_level) and size optimization level (size_level). Both module
// and function-level passes are added, so two pass managers are passed in and
// modified by this function.
void AddOptimizationPasses(unsigned opt_level, unsigned size_level,
                           llvm::TargetMachine* target_machine,
                           llvm::legacy::PassManagerBase* module_passes,
                           llvm::legacy::FunctionPassManager* function_passes) {
  PassManagerBuilder builder;
  builder.OptLevel = opt_level;
  builder.SizeLevel = size_level;

  if (opt_level > 1) {
    builder.Inliner = llvm::createFunctionInliningPass(kDefaultInlineThreshold);
  } else {
    // Only inline functions marked with "alwaysinline".
    builder.Inliner = llvm::createAlwaysInlinerLegacyPass();
  }

  builder.DisableUnitAtATime = false;
  builder.DisableUnrollLoops = opt_level == 0;
  builder.LoopVectorize = opt_level > 0;
  builder.SLPVectorize = opt_level > 1 && size_level < 2;

  // NVPTX's early-as-possible passes include NVVM reflect.
  target_machine->adjustPassManager(builder);

  builder.populateFunctionPassManager(*function_passes);
  builder.populateModulePassManager(*module_passes);
}

// Emits the given module to a bit code file.
void EmitBitcodeToFile(const Module& module, tensorflow::StringPiece filename) {
  std::error_code error_code;
  llvm::ToolOutputFile outfile(filename.ToString().c_str(), error_code,
                               llvm::sys::fs::F_None);
  if (error_code) {
    LOG(FATAL) << "opening bitcode file for writing: " << error_code.message();
  }

  llvm::WriteBitcodeToFile(module, outfile.os());
  outfile.keep();
}

// Emits the given module to PTX. target_machine is an initialized TargetMachine
// for the NVPTX target.
string EmitModuleToPTX(Module* module, llvm::TargetMachine* target_machine) {
  std::string ptx;  // need a std::string instead of a ::string.
  {
    llvm::raw_string_ostream stream(ptx);
    llvm::buffer_ostream pstream(stream);
    // The extension is stripped by IrDumpingPassManager, so we need to
    // get creative to add a suffix.
    string module_id(llvm_ir::AsString(module->getModuleIdentifier()));
    IrDumpingPassManager codegen_passes(
        ReplaceFilenameExtension(tensorflow::io::Basename(module_id),
                                 "-nvptx.dummy"),
        "", false);
    codegen_passes.add(new llvm::TargetLibraryInfoWrapperPass(
        llvm::Triple(module->getTargetTriple())));

    target_machine->addPassesToEmitFile(codegen_passes, pstream, nullptr,
                                        llvm::TargetMachine::CGFT_AssemblyFile);
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

// Returns whether the module could use any libdevice functions. This function
// may have false positives -- the module might not use libdevice even if this
// function returns true.
bool CouldNeedLibdevice(const llvm::Module& module) {
  for (const llvm::Function& function : module.functions()) {
    // This is a conservative approximation -- not all such functions are in
    // libdevice.
    if (!function.isIntrinsic() && function.isDeclaration()) {
      return true;
    }
  }
  return false;
}

// Links libdevice into the given module if the module needs libdevice.
Status LinkLibdeviceIfNecessary(llvm::Module* module,
                                std::pair<int, int> compute_capability,
                                const string& libdevice_dir_path) {
  if (!CouldNeedLibdevice(*module)) {
    return Status::OK();
  }

  llvm::Linker linker(*module);
  string libdevice_path = tensorflow::io::JoinPath(
      libdevice_dir_path, GetLibdeviceFilename(libdevice_dir_path,
                                               compute_capability));
  TF_RETURN_IF_ERROR(tensorflow::Env::Default()->FileExists(libdevice_path));
  VLOG(1) << "Linking with libdevice from: " << libdevice_path;
  std::unique_ptr<llvm::Module> libdevice_module =
      LoadIRModule(libdevice_path, &module->getContext());
  if (linker.linkInModule(
          std::move(libdevice_module), llvm::Linker::Flags::LinkOnlyNeeded,
          [](Module& M, const StringSet<>& GVS) {
            internalizeModule(M, [&M, &GVS](const GlobalValue& GV) {
              return !GV.hasName() || (GVS.count(GV.getName()) == 0);
            });
          })) {
    return tensorflow::errors::Internal(tensorflow::strings::StrCat(
        "Error linking libdevice from ", libdevice_path));
  }
  return Status::OK();
}

StatusOr<string> CompileModuleToPtx(llvm::Module* module,
                                    std::pair<int, int> compute_capability,
                                    const HloModuleConfig& hlo_module_config,
                                    const string& libdevice_dir_path) {
  // If the module has no functions or globals, there's nothing to compile. Just
  // return an empty string.
  if (module->empty() && module->global_empty()) {
    VLOG(2) << "Module '" << llvm_ir::AsString(module->getName())
            << "' is empty. Skipping compilation.";
    return string();
  }
  // Link the input module with libdevice, to pull in implementations of some
  // builtins.
  TF_RETURN_IF_ERROR(
      LinkLibdeviceIfNecessary(module, compute_capability, libdevice_dir_path));

  // Set the flush-denormals-to-zero flag on the module so the NVVM reflect pass
  // can access it.
  module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz",
                        hlo_module_config.debug_options().xla_gpu_ftz());

  // If ftz is enabled, set it as an attribute on every function in the module.
  if (hlo_module_config.debug_options().xla_gpu_ftz()) {
    for (llvm::Function& fn : *module) {
      fn.addFnAttr("nvptx-f32ftz", "true");
    }
  }

  IrDumpingPassManager module_passes(module->getModuleIdentifier(), "", false);

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
    target_triple = llvm::Triple("nvptx64-unknown-unknown");
  }

  // Figure out the exact name of the processor as known to the NVPTX backend
  // from the gpu_architecture flag.
  std::unique_ptr<llvm::TargetMachine> target_machine = GetTargetMachine(
      target_triple, GetSmName(compute_capability), hlo_module_config);
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

  CHECK_GE(opt_level, 2)
      << "The XLA GPU backend doesn't support unoptimized code generation";

  AddOptimizationPasses(opt_level,
                        /*size_level=*/0, target_machine.get(), &module_passes,
                        &function_passes);

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

  // Finally, produce PTX.
  return EmitModuleToPTX(module, target_machine.get());
}

// One-time module initializer.
// Must be called only once -- DO NOT CALL DIRECTLY.
void GPUBackendInit(const HloModuleConfig& hlo_module_config) {
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
  // TODO(jingyue): The current threshold only considers the numbr of IR
  // instructions which do not accurately reflect the true cost. We need a
  // better cost model.
  FeedLLVMWithFlags({"-bonus-inst-threshold=2"});
  // TODO(b/22073864): Increase limit when scan memory dependency.
  // This helps to reduce more redundant load instructions.
  //
  // The specific value is currently large enough for s3d in shoc benchmark,
  // which contains a lot of load instructions and many arithmetic instructions
  // between those loads.
  FeedLLVMWithFlags({"-memdep-block-scan-limit=500"});

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

StatusOr<string> CompileToPtx(llvm::Module* module,
                              std::pair<int, int> compute_capability,
                              const HloModuleConfig& hlo_module_config,
                              const string& libdevice_dir_path) {
  static std::once_flag backend_init_flag;
  std::call_once(backend_init_flag, GPUBackendInit, hlo_module_config);

  string ptx;
  {
    tensorflow::tracing::ScopedActivity activity(
        "Compiling IR", llvm_ir::AsString(module->getName()),
        /*is_expensive=*/true);
    XLA_SCOPED_LOGGING_TIMER("Compile module " +
                             llvm_ir::AsString(module->getName()));
    TF_ASSIGN_OR_RETURN(
        ptx, CompileModuleToPtx(module, compute_capability, hlo_module_config,
                                libdevice_dir_path));
  }
  return ptx;
}

}  // namespace gpu
}  // namespace xla
