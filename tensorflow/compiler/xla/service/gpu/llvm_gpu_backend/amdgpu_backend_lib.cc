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

#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/amdgpu_backend_lib.h"

#include <array>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <fstream>

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
#include "llvm/LinkAllIR.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Linker/Linker.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace gpu {
namespace {

// Default inline threshold value to use in llvm.
const int kDefaultInlineThreshold = 1048576;

// Gets the ROCm-Device-Libs filenames for a particular AMDGPU version.
static std::vector<string> GetROCDLFilenames(int amdgpu_version) {
  // AMDGPU version-neutral bitcodes
  std::vector<string> result {
    "hc.amdgcn.bc",
    "opencl.amdgcn.bc",
    "ocml.amdgcn.bc",
    "ockl.amdgcn.bc",
    "irif.amdgcn.bc",
    "oclc_finite_only_off.amdgcn.bc",
    "oclc_daz_opt_off.amdgcn.bc",
    "oclc_correctly_rounded_sqrt_on.amdgcn.bc",
    "oclc_unsafe_math_off.amdgcn.bc"
  };

  // AMDGPU version-specific bitcodes
  result.push_back(tensorflow::strings::StrCat("oclc_isa_version_",
                                               amdgpu_version, ".amdgcn.bc"));
  return std::move(result);
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
  // GetCodeGenOptLevel in //external/llvm/tools/opt/opt.cpp.
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
      triple.str(), llvm_ir::AsStringRef(cpu_name), "", target_options,
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

// Emits the given module to HSA Code Object. target_machine is an initialized
// TargetMachine for the AMDGPU target.
std::vector<char> EmitModuleToHsaco(Module* module, llvm::TargetMachine* target_machine) {
  char tempdir_template[] = "/tmp/amdgpu_xla-XXXXXX";
  char* tempdir_name = mkdtemp(tempdir_template);

  VLOG(1) << "Compile-time artifacts located at: " << tempdir_name;

  // prepare filenames for all stages of compilation:
  // IR, ISA, binary ISA, and HSACO
  std::string ir_filename = tensorflow::strings::StrCat(module->getModuleIdentifier(), ".ll");
  std::string ir_path = tensorflow::io::JoinPath(tempdir_name, ir_filename);

  std::string isabin_filename = tensorflow::strings::StrCat(module->getModuleIdentifier(), ".s");
  std::string isabin_path = tensorflow::io::JoinPath(tempdir_name, isabin_filename);

  std::string hsaco_filename = tensorflow::strings::StrCat(module->getModuleIdentifier(), ".hsaco");
  std::string hsaco_path = tensorflow::io::JoinPath(tempdir_name, hsaco_filename);

  std::error_code ec;
  SmallString<128> path;

  // dump LLVM IR
  std::unique_ptr<llvm::raw_fd_ostream> ir_fs(new llvm::raw_fd_ostream(ir_path, ec, llvm::sys::fs::F_None));
  module->print(*ir_fs, nullptr);
  ir_fs->flush();

  // emit GCN ISA binary
  std::string gcnisa_binary;  // need a std::string instead of a ::string.
  {
    llvm::raw_string_ostream stream(gcnisa_binary);
    llvm::buffer_ostream pstream(stream);
    // The extension is stripped by IrDumpingPassManager, so we need to
    // get creative to add a suffix.
    string module_id(llvm_ir::AsString(module->getModuleIdentifier()));
    IrDumpingPassManager codegen_passes(
        ReplaceFilenameExtension(tensorflow::io::Basename(module_id),
                                 "-amdgpu.dummy"),
        "", false);
    codegen_passes.add(new llvm::TargetLibraryInfoWrapperPass(
        llvm::Triple(module->getTargetTriple())));

    target_machine->addPassesToEmitFile(codegen_passes, pstream, nullptr,
                                        llvm::TargetMachine::CGFT_ObjectFile);
    codegen_passes.run(*module);
  }

  // dump GCN ISA binary
  std::unique_ptr<llvm::raw_fd_ostream> isabin_fs(new llvm::raw_fd_ostream(isabin_path, ec, llvm::sys::fs::F_None));
  *isabin_fs << gcnisa_binary;
  isabin_fs->flush();

  // execute ld.lld to convert GCN ISA binary into HSACO
  auto lld_program = llvm::sys::findProgramByName("ld.lld");
  if (!lld_program) {
    LOG(FATAL) << "unable to find ld.lld in PATH: "
               << lld_program.getError().message();
  }
  std::vector<llvm::StringRef> lld_args {
    llvm_ir::AsStringRef("ld.lld"),
    llvm_ir::AsStringRef("-flavor"),
    llvm_ir::AsStringRef("gnu"),
    llvm_ir::AsStringRef("-shared"),
    llvm_ir::AsStringRef("isabin_path"),
    llvm_ir::AsStringRef("-o"),
    llvm_ir::AsStringRef("hsaco_path"),
  };
  lld_args[4] = llvm_ir::AsStringRef(isabin_path.c_str());
  lld_args[6] = llvm_ir::AsStringRef(hsaco_path.c_str());

  std::string error_message;
  int lld_result = llvm::sys::ExecuteAndWait(*lld_program,
                                             llvm_ir::AsArrayRef(lld_args),
                                             llvm::None, {}, 0, 0,
                                             &error_message); 

  if (lld_result) {
    LOG(FATAL) << "ld.lld execute fail: " << error_message;
  }

  // read HSACO
  std::ifstream hsaco_file(hsaco_path, std::ios::binary|std::ios::ate);
  std::ifstream::pos_type hsaco_file_size = hsaco_file.tellg();

  std::vector<char> hsaco(hsaco_file_size);
  hsaco_file.seekg(0, std::ios::beg);
  hsaco_file.read(&hsaco[0], hsaco_file_size);

  return std::move(hsaco);
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

// Returns whether the module could use any ROCm-Device-Libs functions. This
// function may have false positives -- the module might not use rocdl even if
// this function returns true.
bool CouldNeedROCDL(const llvm::Module& module) {
  for (const llvm::Function& function : module.functions()) {
    // This is a conservative approximation -- not all such functions are in
    // ROCm-Device-Libs.
    if (!function.isIntrinsic() && function.isDeclaration()) {
      return true;
    }
  }
  return false;
}

// Links ROCm-Device-Libs into the given module if the module needs it.
tensorflow::Status LinkROCDLIfNecessary(
    llvm::Module* module, int amdgpu_version,
    const string& rocdl_dir_path) {
  if (!CouldNeedROCDL(*module)) {
    return tensorflow::Status::OK();
  }

  llvm::Linker linker(*module);

  // ROCm-Device-Libs is shipped as a collection of bitcodes
  std::vector<string> rocdl_bitcode_vector = GetROCDLFilenames(amdgpu_version);

  for (auto& rocdl_bitcode : rocdl_bitcode_vector) {
    string rocdl_path = tensorflow::io::JoinPath(rocdl_dir_path,
                                                 rocdl_bitcode);
    TF_RETURN_IF_ERROR(tensorflow::Env::Default()->FileExists(rocdl_path));
    VLOG(1) << "Linking with ROCDL bitcode from: " << rocdl_path;
    std::unique_ptr<llvm::Module> rocdl_module =
        LoadIRModule(rocdl_path, &module->getContext());
    if (linker.linkInModule(
            std::move(rocdl_module), llvm::Linker::Flags::LinkOnlyNeeded,
            [](Module& M, const StringSet<>& GVS) {
              internalizeModule(M, [&M, &GVS](const GlobalValue& GV) {
                return !GV.hasName() || (GVS.count(GV.getName()) == 0);
              });
            })) {
      return tensorflow::errors::Internal(tensorflow::strings::StrCat(
          "Error linking ROCm-Device-Libs from ", rocdl_path));
    }
  }
  return tensorflow::Status::OK();
}

StatusOr<std::vector<char>> CompileModuleToHsaco(llvm::Module* module,
                                      int amdgpu_version,
                                      const HloModuleConfig& hlo_module_config,
                                      const string& rocdl_dir_path) {
  // Link the input module with ROCDL, to pull in implementations of some
  // builtins.
  TF_RETURN_IF_ERROR(
      LinkROCDLIfNecessary(module, amdgpu_version, rocdl_dir_path));

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
    target_triple = llvm::Triple("amdgcn--amdhsa-amdgiz");
  }

  // Figure out the exact name of the processor as known to the AMDGPU backend
  // from the gpu_architecture flag.
  std::unique_ptr<llvm::TargetMachine> target_machine = GetTargetMachine(
      target_triple, tensorflow::strings::StrCat("gfx", amdgpu_version),
      hlo_module_config);
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
  // TODO(jingyue): SROA may further expose more optimization opportunities, such
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

  // Finally, produce HSA Code Object.
  return std::move(EmitModuleToHsaco(module, target_machine.get()));
}

// One-time module initializer.
// Must be called only once -- DO NOT CALL DIRECTLY.
void GPUBackendInit() {
  // Feed all customized flags here, so we can override them with llvm_cl_opts
  // without redeploy the compiler for development purpose.

  // Initialize the AMDGPU target; it's the only target we link with, so call its
  // specific initialization functions instead of the catch-all InitializeAll*.
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();

  // Initialize the LLVM optimization passes.
  llvm::PassRegistry* registry = llvm::PassRegistry::getPassRegistry();
  InitializePasses(registry);
}

}  // namespace

StatusOr<std::vector<char>> CompileToHsaco(llvm::Module* module,
                                int amdgpu_version,
                                const HloModuleConfig& hlo_module_config,
                                const string& rocdl_dir_path) {
  static std::once_flag backend_init_flag;
  std::call_once(backend_init_flag, GPUBackendInit);

  std::vector<char> hsaco;
  {
    ScopedLoggingTimer compilation_timer(
        "Compile module " + llvm_ir::AsString(module->getName()),
        /*vlog_level=*/2);
    TF_ASSIGN_OR_RETURN(
        hsaco, CompileModuleToHsaco(module, amdgpu_version, hlo_module_config,
                                  rocdl_dir_path));
  }
  return std::move(hsaco);
}

}  // namespace gpu
}  // namespace xla
