/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace tf_saved_model {

namespace {

#define GEN_PASS_DEF_ADDFUNCTIONSFOREXPORTEDNAMESPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_savedmodel_passes.h.inc"

struct AddFunctionsForExportedNamesPass
    : public impl::AddFunctionsForExportedNamesPassBase<
          AddFunctionsForExportedNamesPass> {
  void runOnOperation() override;
};

// Set the (array of) exported name(s) of a (public) function to just
// contain the given string.
void SetExportedName(func::FuncOp f, StringRef name) {
  OpBuilder b(f);
  f->removeAttr(kTfSavedModelExportedNamesAttr);
  f->setAttr(kTfSavedModelExportedNamesAttr, b.getStrArrayAttr({name}));
}

// Convert a savedmodel public function into a private function.
// This means we need to remove any attributes that are only allowed
// on exported (public) functions.
void Unexport(func::FuncOp f) {
  f.setVisibility(mlir::SymbolTable::Visibility::Private);
  f->removeAttr(kTfSavedModelExportedNamesAttr);
  for (int i = 0; i < f.getNumArguments(); ++i) {
    llvm::ArrayRef<mlir::NamedAttribute> attrs =
        mlir::function_interface_impl::getArgAttrs(f, i);
    for (NamedAttribute a : attrs) {
      if (a.getName().strref().starts_with("tf_saved_model.")) {
        f.removeArgAttr(i, a.getName());
      }
    }
  }
  for (int i = 0; i < f.getNumResults(); ++i) {
    for (NamedAttribute a : f.getResultAttrs(i)) {
      if (a.getName().strref().starts_with("tf_saved_model.")) {
        f.removeResultAttr(i, a.getName());
      }
    }
  }
}

void AddFunctionsForExportedNamesPass::runOnOperation() {
  Block& module_body = getOperation().getRegion().front();
  for (auto f :
       llvm::make_early_inc_range(getOperation().getOps<func::FuncOp>())) {
    auto exported_names = GetExportedNames(f);
    if (exported_names.empty() || !f.isPublic()) continue;

    if (exported_names.size() == 1 && exported_names[0] == f.getName()) {
      // Functions that already export themselves with their MLIR name
      // we can leave alone. This saves one level of indirection.
      return;
    }

    f->removeAttr(kTfSavedModelExportedNamesAttr);  // so we don't clone it

    // Rename to avoid name collisions with itself.
    f.setName(StringAttr::get(f->getContext(), f.getName() + "_internal"));

    for (StringRef name : llvm::reverse(exported_names)) {
      // Create a "trampoline" function with the given name. So given
      //   func bar(...) {exported_names = ["foo"]}
      // we create
      //   func foo(...) {
      //     return bar(...)
      //   }
      func::FuncOp other = f.cloneWithoutRegions();
      other.setName(name);
      SetExportedName(other, name);
      module_body.push_front(other);
      other.addEntryBlock();
      OpBuilder builder(other.getRegion());
      auto call_op = builder.create<mlir::func::CallOp>(
          f.getLoc(), f.getFunctionType().getResults(), f.getSymName(),
          other.getRegion().getArguments());
      builder.create<mlir::func::ReturnOp>(f.getLoc(), call_op.getResults());
    }

    Unexport(f);
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateAddFunctionsForExportedNamesPass() {
  return std::make_unique<AddFunctionsForExportedNamesPass>();
}

}  // namespace tf_saved_model
}  // namespace mlir
