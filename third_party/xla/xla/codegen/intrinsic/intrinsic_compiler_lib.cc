/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/intrinsic/intrinsic_compiler_lib.h"

#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

namespace xla::codegen::intrinsic {

void RunInlineAndOptPasses(llvm::Module& module) {
  llvm::PassBuilder pb;

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::ModulePassManager mpm;
  mpm.addPass(llvm::AlwaysInlinerPass());

  llvm::FunctionPassManager fpm;
  fpm.addPass(llvm::InstCombinePass());
  fpm.addPass(llvm::EarlyCSEPass());
  fpm.addPass(llvm::DCEPass());
  mpm.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(fpm)));
  mpm.addPass(llvm::GlobalDCEPass());

  mpm.run(module, mam);
}

void RemoveFromCompilerUsed(
    llvm::Module& module,
    absl::FunctionRef<bool(absl::string_view)> should_remove) {
  llvm::removeFromUsedLists(module, [&](llvm::Constant* c) {
    if (auto* f = llvm::dyn_cast<llvm::Function>(c)) {
      return should_remove(f->getName());
    }
    return false;
  });
}

}  // namespace xla::codegen::intrinsic
