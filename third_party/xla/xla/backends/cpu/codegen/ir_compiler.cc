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

#include "xla/backends/cpu/codegen/ir_compiler.h"

#include <memory>
#include <utility>

#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Instrumentation/DataFlowSanitizer.h"
#include "xla/backends/cpu/codegen/polynomial_approximations.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"

namespace xla::cpu {

static llvm::OptimizationLevel GetOptimizationLevel(
    IrCompiler::Options options) {
  if (options.optimize_for_size) {
    return llvm::OptimizationLevel::Os;
  }

  switch (options.opt_level) {
    case llvm::CodeGenOptLevel::None:
      return llvm::OptimizationLevel::O0;
    case llvm::CodeGenOptLevel::Less:
      return llvm::OptimizationLevel::O1;
    case llvm::CodeGenOptLevel::Default:
      return llvm::OptimizationLevel::O2;
    case llvm::CodeGenOptLevel::Aggressive:
      return llvm::OptimizationLevel::O3;
  }
}

IrCompiler::IrCompiler(TargetMachineBuilder target_machine_builder,
                       Options options, CompilationHooks hooks)
    : IRCompiler(llvm::orc::IRSymbolMapper::ManglingOptions()),
      target_machine_builder_(std::move(target_machine_builder)),
      options_(std::move(options)),
      hooks_(std::move(hooks)) {}

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> IrCompiler::operator()(
    llvm::Module& module) {
  VLOG(2) << "IR before optimizations";
  XLA_VLOG_LINES(2, llvm_ir::DumpToString(&module));

  // Get a target machine for compilation. If compilations run concurrently on
  // multiple threads, `IrCompiler` user (in most cases `SimpleOrcJIT`)
  // must guarantee that target machine builder will return a unique
  // TargetMachine for each compilation, as it is not thread safe.
  absl::StatusOr<std::shared_ptr<llvm::TargetMachine>> target_machine =
      target_machine_builder_();

  if (!target_machine.ok()) {
    return llvm::make_error<llvm::StringError>(
        llvm::errc::invalid_argument,
        absl::StrFormat(
            "Failed to create target machine for IR compilation: %s",
            target_machine.status().message()));
  }

  {  // Synchronize access to user-defined hooks.
    absl::MutexLock lock(&mutex_);
    if (hooks_.pre_optimization) {
      hooks_.pre_optimization(module);
    }
  }

  llvm::PipelineTuningOptions pto;
  pto.LoopVectorization = !options_.optimize_for_size;
  pto.SLPVectorization =
      !options_.optimize_for_size && !options_.disable_slp_vectorizer;
  pto.LoopUnrolling = false;

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  llvm::PassInstrumentationCallbacks pic;
  llvm::StandardInstrumentations si(module.getContext(), false);
  si.registerCallbacks(pic, &mam);

  llvm::PassBuilder pb(target_machine->get(), pto, {}, &pic);

  // Add the appropriate TargetLibraryInfo.
  llvm::Triple target_triple((*target_machine)->getTargetTriple());
  auto target_library_info_impl =
      std::make_unique<llvm::TargetLibraryInfoImpl>(target_triple);
  target_library_info_impl->addVectorizableFunctions(
      PolynomialApproximationsVectorization());

  fam.registerPass(
      [&] { return llvm::TargetLibraryAnalysis(*target_library_info_impl); });

  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::ModulePassManager pm;

  if (options_.dfsan_enabled) {
    pm.addPass(llvm::DataFlowSanitizerPass(options_.dfsan_abi_list_files));
  }

  llvm::OptimizationLevel opt_level = GetOptimizationLevel(options_);
  if (opt_level == llvm::OptimizationLevel::O0) {
    pm.addPass(pb.buildO0DefaultPipeline(opt_level));
  } else {
    pm.addPass(pb.buildPerModuleDefaultPipeline(opt_level));
  }

  CHECK(!llvm::verifyModule(module, &llvm::dbgs()));

  pm.run(module, mam);

  CHECK(!llvm::verifyModule(module, &llvm::dbgs()));

  RewriteToPolynomialApproximations(&module, options_.fast_math_flags);

  // Buffer for holding machine code prior to constructing the ObjectFile.
  llvm::SmallVector<char, 0> mc_stream_buffer;
  llvm::raw_svector_ostream ostream(mc_stream_buffer);

  VLOG(2) << "IR after optimizations";
  XLA_VLOG_LINES(2, llvm_ir::DumpToString(&module));

  {  // Synchronize access to user-defined hooks.
    absl::MutexLock lock(&mutex_);
    if (hooks_.post_optimization) {
      hooks_.post_optimization(module);
    }
  }

  // Generate code.
  llvm::MCContext* mc_context;
  llvm::legacy::PassManager codegen_passes;
  (*target_machine)->addPassesToEmitMC(codegen_passes, mc_context, ostream);
  codegen_passes.run(module);

  std::unique_ptr<llvm::MemoryBuffer> mc_memory_buffer(
      new llvm::SmallVectorMemoryBuffer(std::move(mc_stream_buffer)));

  {  // Synchronize access to user-defined hooks.
    absl::MutexLock lock(&mutex_);
    if (hooks_.post_codegen) {
      llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>> obj_file =
          llvm::object::ObjectFile::createObjectFile(*mc_memory_buffer);
      if (obj_file) {
        hooks_.post_codegen(module, *obj_file.get());
      } else {
        LOG(WARNING) << "Could not convert memory buffer to object file";
      }
    }
  }

  return std::move(mc_memory_buffer);
}

}  // namespace xla::cpu
