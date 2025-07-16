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

#include "xla/codegen/math/math_compiler_lib.h"

#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
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

namespace xla::codegen::math {

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
    absl::flat_hash_set<absl::string_view> replaced_functions) {
  if (replaced_functions.empty()) {
    return;
  }

  llvm::GlobalVariable* compiler_used =
      module.getNamedGlobal("llvm.compiler.used");
  if (!compiler_used) {
    return;
  }

  llvm::ConstantArray* old_array =
      llvm::dyn_cast<llvm::ConstantArray>(compiler_used->getInitializer());
  if (!old_array) {
    return;
  }

  // Collect the constants that should be kept.
  std::vector<llvm::Constant*> elements;
  elements.reserve(old_array->getNumOperands());
  for (int i = 0; i < old_array->getNumOperands(); ++i) {
    auto* operand = old_array->getOperand(i);
    llvm::GlobalValue* gv =
        llvm::dyn_cast<llvm::GlobalValue>(operand->stripPointerCasts());

    if (gv && replaced_functions.contains(gv->getName())) {
      continue;
    }
    elements.push_back(operand);
  }

  // If all functions were removed, erase the global entirely.
  if (elements.empty()) {
    compiler_used->eraseFromParent();
    return;
  }

  // If only some functions were removed, modify the existing global in-place.
  if (elements.size() < old_array->getNumOperands()) {
    llvm::ArrayType* new_array_type = llvm::ArrayType::get(
        old_array->getType()->getElementType(), elements.size());
    llvm::Constant* new_array_init =
        llvm::ConstantArray::get(new_array_type, elements);

    // Rename existing llvm.compiler.used so llvm doesn't give our replacement
    // a suffix due to name collission.
    compiler_used->setName("llvm.compiler.used.old");

    // Create a new global llvm.compiler.used with the new contents.
    new llvm::GlobalVariable(module, new_array_type, false,
                             llvm::GlobalValue::AppendingLinkage,
                             new_array_init, "llvm.compiler.used");
    // Replace all uses of the old llvm.compiler.used with the new one.
    compiler_used->replaceAllUsesWith(
        module.getNamedGlobal("llvm.compiler.used"));

    // Remove the old llvm.compiler.used.
    compiler_used->eraseFromParent();
  }
}

}  // namespace xla::codegen::math
