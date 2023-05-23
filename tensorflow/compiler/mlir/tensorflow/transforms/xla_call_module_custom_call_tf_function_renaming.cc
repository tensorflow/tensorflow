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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/stablehlo_custom_call.h"

namespace mlir {
namespace TF {
namespace {

#define GEN_PASS_DEF_XLACALLMODULECUSTOMCALLTFFUNCTIONRENAMINGPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"  // IWYU pragma: keep

// Name of TF function's string attribute for keeping the function's original
// name after being renamed.
constexpr llvm::StringRef kTfOriginalFuncNameAttrName =
    "tf._original_func_name";

FailureOr<llvm::DenseMap<llvm::StringRef, FlatSymbolRefAttr>>
FindOriginalToCurrentFunctionNameMapping(ModuleOp module) {
  Builder builder(module.getContext());
  llvm::DenseMap<llvm::StringRef, FlatSymbolRefAttr> mapping;

  auto result = module.walk([&](func::FuncOp func) {
    if (auto name =
            func->getAttrOfType<StringAttr>(kTfOriginalFuncNameAttrName)) {
      if (!mapping
               .insert({name, builder.getAttr<FlatSymbolRefAttr>(
                                  func.getSymNameAttr())})
               .second) {
        module.emitOpError() << "contains multiple functions with the same "
                                "tf._original_func_name '"
                             << name << "'";
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    return failure();
  }

  return mapping;
}

LogicalResult RewriteOriginalTfFunctionNameInCustomCalls(
    ModuleOp module,
    const llvm::DenseMap<llvm::StringRef, FlatSymbolRefAttr>& funcs) {
  auto result = module.walk([&](stablehlo::CustomCallOp op) {
    auto f = GetTfHostCallbackName(op);
    if (failed(f)) {
      // Does not call host callback.
      return WalkResult::advance();
    }
    if (*f == nullptr) {
      // Does not have `caller_name`.
      return WalkResult::interrupt();
    }

    auto it = funcs.find(*f);
    if (it == funcs.end()) {
      op.emitOpError() << "refers to unknown tf function '" << *f << "'";
      return WalkResult::interrupt();
    }
    SetTfHostCallbackName(op, it->second);

    return WalkResult::advance();
  });

  return result.wasInterrupted() ? failure() : success();
}

class XlaCallModuleCustomCallTfFunctionRenamingPass
    : public impl::XlaCallModuleCustomCallTfFunctionRenamingPassBase<
          XlaCallModuleCustomCallTfFunctionRenamingPass> {
 public:
  void runOnOperation() override {
    ModuleOp module = getOperation();

    auto funcs = FindOriginalToCurrentFunctionNameMapping(module);
    if (failed(funcs)) {
      return signalPassFailure();
    }

    if (failed(RewriteOriginalTfFunctionNameInCustomCalls(module, *funcs))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateXlaCallModuleCustomCallTfFunctionRenamingPass() {
  return std::make_unique<XlaCallModuleCustomCallTfFunctionRenamingPass>();
}

}  // namespace TF
}  // namespace mlir
