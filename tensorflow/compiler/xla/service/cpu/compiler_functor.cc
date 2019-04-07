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

#include "absl/memory/memory.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/llvm_ir_runtime.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace cpu {

/* Create filtered versions of the LLVM Pass Managers to filter out some
of the expensive passes.
Profiling:
   learning/brain/google/xla/benchmarks:inception_cpu_benchmark
   learning/brain/google/xla/benchmarks:cifarnet
pointed to LICM and IndVarSimplify as the hottest passes.
LICM is known to exhibit O(n^2) time in the number of instructions.
IndVarSimplify is slow due to SCEV. If loops are emitted in canonical form,
this pass is not necessary.
Disabling these as a starting point.
*/
// TODO(b/64227304) Creating a custom pass pipeline will replace this.

namespace {
class FilteredPassManager : public llvm::legacy::PassManager {
 public:
  explicit FilteredPassManager(bool disable_expensive_passes)
      : disable_expensive_passes_(disable_expensive_passes) {}
  void add(llvm::Pass* p) override {
    if (disable_expensive_passes_) {
      llvm::StringRef PassName = p->getPassName();
      if (PassName.contains("Unroll loops")) {
        return;
      }
    }
    llvm::legacy::PassManager::add(p);
  }

 private:
  bool disable_expensive_passes_;
};
}  // anonymous namespace

std::unique_ptr<llvm::MemoryBuffer> CompilerFunctor::operator()(
    llvm::Module& module) const {
  FilteredPassManager module_passes(disable_expensive_passes_);
  llvm::legacy::FunctionPassManager function_passes(&module);

  VLOG(2) << "IR before optimizations";
  XLA_VLOG_LINES(2, llvm_ir::DumpModuleToString(module));

  if (pre_optimization_hook_) {
    pre_optimization_hook_(module);
  }

  // Add the appropriate TargetLibraryInfo and TargetTransformInfo.
  AddTargetInfoPasses(&module_passes);

  // Build up optimization pipeline.
  if (optimize_for_size_) {
    // Optimizing for size turns on -O2 level optimizations.
    //
    // TODO(b/64153864): Although the code generator supports size_level = 2 to
    // turn on more aggressive code size optimizations than size_level = 1, we
    // pass size_level = 1 because in many cases a size_level of 2 does
    // worse. Investigate why.
    AddOptimizationPasses(&module_passes, &function_passes, /*opt_level=*/2,
                          /*size_level=*/1);
  } else {
    AddOptimizationPasses(&module_passes, &function_passes,
                          /*opt_level=*/opt_level_, /*size_level=*/0);
  }

  // Run optimization passes on module.
  function_passes.doInitialization();

  CHECK(!llvm::verifyModule(module, &llvm::dbgs()));

  for (auto func = module.begin(); func != module.end(); ++func) {
    function_passes.run(*func);
  }
  function_passes.doFinalization();
  module_passes.run(module);

  CHECK(!llvm::verifyModule(module, &llvm::dbgs()));

  runtime::RewriteIRRuntimeFunctions(&module, enable_fast_math_);

  // Buffer for holding machine code prior to constructing the ObjectFile.
  llvm::SmallVector<char, 0> stream_buffer;
  llvm::raw_svector_ostream ostream(stream_buffer);

  VLOG(2) << "IR after optimizations";
  XLA_VLOG_LINES(2, llvm_ir::DumpModuleToString(module));

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

  return memory_buffer;
}

static std::vector<llvm::VecDesc> VectorFunctionsForTargetLibraryInfoImpl() {
  std::vector<llvm::VecDesc> result = {
      {"tanhf", runtime::kTanhV4F32SymbolName, 4},
      {"llvm.tanh.f32", runtime::kTanhV4F32SymbolName, 4},

      {"tanhf", runtime::kTanhV8F32SymbolName, 8},
      {"llvm.tanh.f32", runtime::kTanhV8F32SymbolName, 8},

      {"expf", runtime::kExpV4F32SymbolName, 4},
      {"llvm.exp.f32", runtime::kExpV4F32SymbolName, 4},

      {"expf", runtime::kExpV8F32SymbolName, 8},
      {"llvm.exp.f32", runtime::kExpV8F32SymbolName, 8},

      {"logf", runtime::kLogV4F32SymbolName, 4},
      {"llvm.log.f32", runtime::kLogV4F32SymbolName, 4},

      {"logf", runtime::kLogV8F32SymbolName, 8},
      {"llvm.log.f32", runtime::kLogV8F32SymbolName, 8},
  };
  return result;
}

void CompilerFunctor::AddTargetInfoPasses(
    llvm::legacy::PassManagerBase* passes) const {
  llvm::Triple target_triple(target_machine_->getTargetTriple());
  auto target_library_info_impl =
      absl::make_unique<llvm::TargetLibraryInfoImpl>(target_triple);
  target_library_info_impl->addVectorizableFunctions(
      VectorFunctionsForTargetLibraryInfoImpl());
  passes->add(
      new llvm::TargetLibraryInfoWrapperPass(*target_library_info_impl));
  passes->add(createTargetTransformInfoWrapperPass(
      target_machine_->getTargetIRAnalysis()));
}

void CompilerFunctor::AddOptimizationPasses(
    llvm::legacy::PassManagerBase* module_passes,
    llvm::legacy::FunctionPassManager* function_passes, unsigned opt_level,
    unsigned size_level) const {
  llvm::PassManagerBuilder builder;
  builder.OptLevel = opt_level;
  builder.SizeLevel = size_level;

  if (opt_level > 1) {
    builder.Inliner = llvm::createFunctionInliningPass();
  } else {
    // Only inline functions marked with "alwaysinline".
    builder.Inliner = llvm::createAlwaysInlinerLegacyPass();
  }

  builder.DisableUnitAtATime = false;
  builder.DisableUnrollLoops = opt_level == 0;
  builder.LoopVectorize = opt_level > 0 && size_level == 0;
  builder.SLPVectorize = opt_level > 1 && size_level == 0;

  builder.populateFunctionPassManager(*function_passes);
  builder.populateModulePassManager(*module_passes);
}

}  // namespace cpu
}  // namespace xla
