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

#include "tensorflow/compiler/xla/service/cpu/compiler_functor.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Instrumentation/DataFlowSanitizer.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/llvm_ir_runtime.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {
namespace cpu {

static std::vector<llvm::VecDesc> VectorFunctionsForTargetLibraryInfoImpl() {
  std::vector<llvm::VecDesc> result = {
      {"tanhf", runtime::kTanhV4F32SymbolName, llvm::ElementCount::getFixed(4)},
      {"llvm.tanh.f32", runtime::kTanhV4F32SymbolName,
       llvm::ElementCount::getFixed(4)},

      {"tanhf", runtime::kTanhV8F32SymbolName, llvm::ElementCount::getFixed(8)},
      {"llvm.tanh.f32", runtime::kTanhV8F32SymbolName,
       llvm::ElementCount::getFixed(8)},

      {"tanhf", runtime::kTanhV16F32SymbolName,
       llvm::ElementCount::getFixed(16)},
      {"llvm.tanh.f32", runtime::kTanhV16F32SymbolName,
       llvm::ElementCount::getFixed(16)},

      {"expf", runtime::kExpV4F32SymbolName, llvm::ElementCount::getFixed(4)},
      {"llvm.exp.f32", runtime::kExpV4F32SymbolName,
       llvm::ElementCount::getFixed(4)},

      {"expf", runtime::kExpV8F32SymbolName, llvm::ElementCount::getFixed(8)},
      {"llvm.exp.f32", runtime::kExpV8F32SymbolName,
       llvm::ElementCount::getFixed(8)},

      {"expf", runtime::kExpV16F32SymbolName, llvm::ElementCount::getFixed(16)},
      {"llvm.exp.f32", runtime::kExpV16F32SymbolName,
       llvm::ElementCount::getFixed(16)},

      {"logf", runtime::kLogV4F32SymbolName, llvm::ElementCount::getFixed(4)},
      {"llvm.log.f32", runtime::kLogV4F32SymbolName,
       llvm::ElementCount::getFixed(4)},

      {"logf", runtime::kLogV8F32SymbolName, llvm::ElementCount::getFixed(8)},
      {"llvm.log.f32", runtime::kLogV8F32SymbolName,
       llvm::ElementCount::getFixed(8)},

      {"logf", runtime::kLogV16F32SymbolName, llvm::ElementCount::getFixed(16)},
      {"llvm.log.f32", runtime::kLogV16F32SymbolName,
       llvm::ElementCount::getFixed(16)},
  };
  return result;
}

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> CompilerFunctor::operator()(
    llvm::Module& module) {
  VLOG(2) << "IR before optimizations";
  XLA_VLOG_LINES(2, llvm_ir::DumpToString(&module));

  if (pre_optimization_hook_) {
    pre_optimization_hook_(module);
  }

  llvm::OptimizationLevel opt_level;
  if (optimize_for_size_) {
    opt_level = llvm::OptimizationLevel::Os;
  } else {
    switch (opt_level_) {
      case 0:
        opt_level = llvm::OptimizationLevel::O0;
        break;
      case 1:
        opt_level = llvm::OptimizationLevel::O1;
        break;
      case 2:
        opt_level = llvm::OptimizationLevel::O2;
        break;
      case 3:
        opt_level = llvm::OptimizationLevel::O3;
        break;
    }
  }

  llvm::PipelineTuningOptions pto;
  pto.LoopVectorization = !optimize_for_size_;
  pto.SLPVectorization = !optimize_for_size_ && !disable_slp_vectorizer_;
  pto.LoopUnrolling = false;

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  llvm::PassInstrumentationCallbacks pic;
  llvm::StandardInstrumentations si(module.getContext(), false);
  si.registerCallbacks(pic, &mam);

  llvm::PassBuilder pb(target_machine_, pto, {}, &pic);

  // Add the appropriate TargetLibraryInfo.
  llvm::Triple target_triple(target_machine_->getTargetTriple());
  auto target_library_info_impl =
      std::make_unique<llvm::TargetLibraryInfoImpl>(target_triple);
  target_library_info_impl->addVectorizableFunctions(
      VectorFunctionsForTargetLibraryInfoImpl());

  fam.registerPass(
      [&] { return llvm::TargetLibraryAnalysis(*target_library_info_impl); });

  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::ModulePassManager pm;

  if (dfsan_enabled_) {
    pm.addPass(llvm::DataFlowSanitizerPass(dfsan_abi_list_files_));
  }

  if (opt_level == llvm::OptimizationLevel::O0) {
    pm.addPass(pb.buildO0DefaultPipeline(opt_level));
  } else {
    pm.addPass(pb.buildPerModuleDefaultPipeline(opt_level));
  }

  CHECK(!llvm::verifyModule(module, &llvm::dbgs()));

  pm.run(module, mam);

  CHECK(!llvm::verifyModule(module, &llvm::dbgs()));

  runtime::RewriteIRRuntimeFunctions(&module, fast_math_flags_);

  // Buffer for holding machine code prior to constructing the ObjectFile.
  llvm::SmallVector<char, 0> stream_buffer;
  llvm::raw_svector_ostream ostream(stream_buffer);

  VLOG(2) << "IR after optimizations";

  if (post_optimization_hook_) {
    post_optimization_hook_(module);
  }

  // Generate code.
  llvm::MCContext* mc_context;
  llvm::legacy::PassManager codegen_passes;
  target_machine_->addPassesToEmitMC(codegen_passes, mc_context, ostream);
  codegen_passes.run(module);

  std::unique_ptr<llvm::MemoryBuffer> memory_buffer(
      new llvm::SmallVectorMemoryBuffer(std::move(stream_buffer)));

  if (post_codegen_hook_) {
    llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>> obj_file =
        llvm::object::ObjectFile::createObjectFile(*memory_buffer);
    if (obj_file) {
      post_codegen_hook_(*obj_file.get());
    } else {
      LOG(WARNING) << "Could convert memory buffer to object file!";
    }
  }

  return std::move(memory_buffer);
}

}  // namespace cpu
}  // namespace xla
