/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

#define DEBUG_TYPE "tf-shape-inference"

namespace mlir {

namespace {

LogicalResult MarkFunctionVisibility(
    ModuleOp module, llvm::function_ref<bool(FuncOp func)> IsExternalVisible) {
  LogicalResult result = success();

  for (auto func : module.getOps<FuncOp>()) {
    FuncOp::Visibility old_visibility = func.getVisibility();

    FuncOp::Visibility visibility = IsExternalVisible(func)
                                        ? FuncOp::Visibility::Public
                                        : FuncOp::Visibility::Private;

    auto get_visibility_name = [](FuncOp::Visibility v) {
      return v == FuncOp::Visibility::Public
                 ? "public"
                 : v == FuncOp::Visibility::Private ? "private" : "nested";
    };

    if (old_visibility != SymbolTable::Visibility::Public &&
        old_visibility != visibility) {
      result = func.emitError()
               << "can't overwrite the visibility of function "
               << func.getName() << " with "
               << get_visibility_name(old_visibility) << " visibility";
    }

    LLVM_DEBUG(llvm::dbgs()
               << "function " << func.getName() << " has "
               << get_visibility_name(visibility) << " visibility \n");

    func.setVisibility(visibility);
  }

  return result;
}

}  // anonymous namespace

namespace TF {

LogicalResult MarkFunctionVisibilityUsingEntryFunctionSpecification(
    ModuleOp module) {
  auto HasEntryFunctionSpecification = [](FuncOp func) -> bool {
    auto attrs = func.getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
    return attrs && !attrs.empty();
  };
  return MarkFunctionVisibility(module, HasEntryFunctionSpecification);
}

namespace {
struct MarkFunctionVisibilityUsingEntryFunctionSpecificationPass
    : public PassWrapper<
          MarkFunctionVisibilityUsingEntryFunctionSpecificationPass,
          OperationPass<ModuleOp>> {
  void runOnOperation() override {
    if (failed(MarkFunctionVisibilityUsingEntryFunctionSpecification(
            getOperation()))) {
      signalPassFailure();
    }
  }
};
}  // namespace

static PassRegistration<
    MarkFunctionVisibilityUsingEntryFunctionSpecificationPass>
    pass("tf-mark-func-visibility",
         "Use tf.entry_function to mark function visibility.");

std::unique_ptr<OperationPass<ModuleOp>>
CreateMarkFunctionVisibilityUsingEntryFunctionSpecificationPass() {
  return std::make_unique<
      MarkFunctionVisibilityUsingEntryFunctionSpecificationPass>();
}

// Marks the main function with public visibility, while other functions are
// marked with private visibility.
LogicalResult MarkOnlyMainFunctionWithPublicVisibility(ModuleOp module) {
  for (auto func : module.getOps<FuncOp>()) {
    if (func.getName() == "main") {
      func.setVisibility(FuncOp::Visibility::Public);
    } else {
      func.setVisibility(FuncOp::Visibility::Private);
    }
  }
  return success();
}

namespace {
struct MarkOnlyMainFunctionWithPublicVisibilityPass
    : public PassWrapper<MarkOnlyMainFunctionWithPublicVisibilityPass,
                         OperationPass<ModuleOp>> {
  void runOnOperation() override {
    if (failed(MarkOnlyMainFunctionWithPublicVisibility(getOperation()))) {
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateMarkOnlyMainFunctionWithPublicVisibilityPass() {
  return std::make_unique<MarkOnlyMainFunctionWithPublicVisibilityPass>();
}

}  // namespace TF

namespace tf_saved_model {

static LogicalResult MarkFunctionVisibilityUsingSavedModelLinkage(
    ModuleOp module) {
  if (!tf_saved_model::HasTfSavedModelSemantics(module)) {
    return success();
  }
  return MarkFunctionVisibility(module, tf_saved_model::IsExported);
}

namespace {
struct MarkFunctionVisibilityUsingSavedModelLinkagePass
    : public PassWrapper<MarkFunctionVisibilityUsingSavedModelLinkagePass,
                         OperationPass<ModuleOp>> {
  void runOnOperation() override {
    if (failed(MarkFunctionVisibilityUsingSavedModelLinkage(getOperation()))) {
      signalPassFailure();
    }
  }
};
}  // namespace

static PassRegistration<MarkFunctionVisibilityUsingSavedModelLinkagePass> pass(
    "tf-saved-model-mark-func-visibility",
    "Use tf_saved_model linkage information to mark function visibility.");

std::unique_ptr<OperationPass<ModuleOp>>
CreateMarkFunctionVisibilityUsingSavedModelLinkagePass() {
  return std::make_unique<MarkFunctionVisibilityUsingSavedModelLinkagePass>();
}

}  // namespace tf_saved_model

}  // namespace mlir
