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

#include "xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <system_error>  // NOLINT
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
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
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Scalar.h"
#include "xla/service/gpu/llvm_gpu_backend/load_ir_module.h"
#include "xla/service/gpu/llvm_gpu_backend/utils.h"
#include "xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla {
namespace gpu {

namespace {
static llvm::codegen::RegisterCodeGenFlags CGF;
}

// Initializes LLVM passes. Uses the PassRegistry mechanism.
void InitializePasses(llvm::PassRegistry* pass_registry) {
  llvm::initializeCore(*pass_registry);
  llvm::initializeCodeGen(*pass_registry);
  llvm::initializeScalarOpts(*pass_registry);
  llvm::initializeVectorization(*pass_registry);
  llvm::initializeIPO(*pass_registry);
  llvm::initializeAnalysis(*pass_registry);
  llvm::initializeTransformUtils(*pass_registry);
  llvm::initializeInstCombine(*pass_registry);
  llvm::initializeTarget(*pass_registry);
  llvm::initializeCodeGenPrepareLegacyPassPass(*pass_registry);
}

// Returns the TargetMachine, given a triple.
std::unique_ptr<llvm::TargetMachine> GetTargetMachine(
    llvm::Triple triple, absl::string_view cpu_name,
    const DebugOptions& debug_options, absl::string_view feature_str) {
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
  llvm::CodeGenOptLevel codegen_opt_level;
  switch (debug_options.xla_backend_optimization_level()) {
    case 1:
      codegen_opt_level = llvm::CodeGenOptLevel::Less;
      break;
    case 2:
      codegen_opt_level = llvm::CodeGenOptLevel::Default;
      break;
    case 3:
      codegen_opt_level = llvm::CodeGenOptLevel::Aggressive;
      break;
    default:
      codegen_opt_level = llvm::CodeGenOptLevel::None;
  }
  return absl::WrapUnique(target->createTargetMachine(
      triple.str(), llvm_ir::AsStringRef(cpu_name),
      llvm_ir::AsStringRef(feature_str), target_options,
      llvm::codegen::getExplicitRelocModel(),
      llvm::codegen::getExplicitCodeModel(), codegen_opt_level));
}

// Returns whether the module could use any device bitcode library functions.
bool CouldNeedDeviceBitcode(const llvm::Module& module) {
  for (const llvm::Function& function : module.functions()) {
    // The list of prefixes should be in sync with library functions used in
    // target_util.cc.
    if (!function.isIntrinsic() && function.isDeclaration() &&
        (function.getName().starts_with("__nv_") ||
         function.getName().starts_with("__ocml_") ||
         function.getName().starts_with("__ockl_"))) {
      return true;
    }
  }
  return false;
}

// Links the module with a vector of path to bitcode modules.
// The caller must guarantee that the paths exist.
absl::Status LinkWithBitcodeVector(
    llvm::Module* module, const std::vector<std::string>& bitcode_path_vector) {
  llvm::Linker linker(*module);

  for (auto& bitcode_path : bitcode_path_vector) {
    if (!tsl::Env::Default()->FileExists(bitcode_path).ok()) {
      LOG(ERROR) << "bitcode module is required by this HLO module but was "
                    "not found at "
                 << bitcode_path;
      return xla::Internal("bitcode module not found at %s", bitcode_path);
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
      return xla::Internal("Error linking bitcode module from %s",
                           bitcode_path);
    }
  }
  return absl::OkStatus();
}

namespace {

// NOLINTBEGIN: clang-diagnostic-unused-function
// Convenience function for producing a name of a temporary compilation product
// from the input filename.
std::string MakeNameForTempProduct(absl::string_view input_filename,
                                   absl::string_view extension) {
  return ReplaceFilenameExtension(tsl::io::Basename(input_filename), extension);
}
// NOLINTEND: clang-diagnostic-unused-function

void DumpModule(const std::string output_filename, const llvm::Module* module) {
  std::string content;
  llvm::raw_string_ostream string_stream(content);
  module->print(string_stream, /*AAW=*/nullptr);

  auto status =
      WriteStringToFile(tsl::Env::Default(), output_filename, content);
  if (!status.ok()) {
    LOG(FATAL) << "Unable to write " << output_filename
               << " to dump LLVM IR: " << status.message();
  }
}

const llvm::Module* GetModule(llvm::Any IR) {
  if (const auto** M = llvm::any_cast<const llvm::Module*>(&IR)) return *M;

  if (const auto** F = llvm::any_cast<const llvm::Function*>(&IR)) {
    return (*F)->getParent();
  }

  if (const auto** C = llvm::any_cast<const llvm::LazyCallGraph::SCC*>(&IR)) {
    return (*C)->begin()->getFunction().getParent();
  }

  if (const auto** L = llvm::any_cast<const llvm::Loop*>(&IR)) {
    const llvm::Function* F = (*L)->getHeader()->getParent();
    return F->getParent();
  }

  return nullptr;
}

auto DumpCallbackForModule(std::string module_identifier,
                           std::string outputs_dir) {
  int i = 0;
  return [=](llvm::StringRef pass, llvm::Any ir) mutable {
    const llvm::Module* module = GetModule(ir);
    if (!module) {
      return;
    }

    const std::string basename = ReplaceFilenameExtension(
        absl::string_view(tsl::io::Basename(module_identifier)),
        absl::StrFormat("pass-%03d.before.%s.ll", i++,
                        absl::string_view(pass.str())));
    DumpModule(tsl::io::JoinPath(outputs_dir, basename), module);
  };
}

}  // namespace

absl::Status LinkAndOptimizeModule(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options, const std::string& device_bitcode_path,
    TargetModuleLinker module_linker, llvm::Triple default_target_triple,
    llvm::TargetMachine* target_machine, int inline_threshold) {
  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("XlaOptimizeLlvmIr:#module=%s#",
                           module->getName().str());
  });
  TF_RETURN_IF_ERROR(
      module_linker(module, gpu_version, debug_options, device_bitcode_path));

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  if (target_machine) {
    fam.registerPass([&] { return target_machine->getTargetIRAnalysis(); });
  }

  llvm::PipelineTuningOptions pto;
  pto.SLPVectorization = true;
  pto.InlinerThreshold = inline_threshold;

  llvm::PassInstrumentationCallbacks pic;

  llvm::StandardInstrumentations si(module->getContext(), false);
  si.registerCallbacks(pic, &mam);

  llvm::PassBuilder pb(target_machine, pto, std::nullopt, &pic);
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  if (debug_options.xla_gpu_dump_llvmir()) {
    std::string outputs_dir;
    if (!tsl::io::GetTestUndeclaredOutputsDir(&outputs_dir)) {
      outputs_dir = debug_options.xla_dump_to();
    }
    if (!outputs_dir.empty()) {
      pic.registerBeforeNonSkippedPassCallback(
          DumpCallbackForModule(module->getModuleIdentifier(), outputs_dir));
    } else {
      LOG(ERROR) << "--xla_gpu_dump_llvmir is set, but neither the environment "
                 << "variable TEST_UNDECLARED_OUTPUTS_DIR nor the flag "
                 << "--xla_dump_to is set, so the llvm dumps are disabled.";
    }
  }

  llvm::OptimizationLevel ol;
  switch (debug_options.xla_backend_optimization_level()) {
    case 0:
      ol = llvm::OptimizationLevel::O0;
      break;
    case 1:
      ol = llvm::OptimizationLevel::O1;
      break;
    case 2:
      ol = llvm::OptimizationLevel::O2;
      break;
    case 3:
      ol = llvm::OptimizationLevel::O3;
      break;
  }

  llvm::ModulePassManager mpm;
  mpm.addPass(llvm::VerifierPass());
  if (ol == llvm::OptimizationLevel::O0) {
    mpm.addPass(pb.buildO0DefaultPipeline(ol));
  } else {
    mpm.addPass(pb.buildPerModuleDefaultPipeline(ol));
  }
  mpm.addPass(llvm::VerifierPass());

  mpm.run(*module, mam);

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
