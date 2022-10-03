/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/drop_savedmodel_semantics.h"

#include <vector>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace TFL {
namespace mhlo {

class DropSavedModelSemanticsPass
    : public PassWrapper<DropSavedModelSemanticsPass, OperationPass<ModuleOp>> {
 public:
  StringRef getArgument() const final { return "drop-savedmodel-semantics"; }
  StringRef getDescription() const final {
    return "Drops all tf_saved_model attributes";
  }

  // NOTE: The implementation is mostly copypasted from
  // third_party/tensorflow/compiler/mlir/tfrt/transforms/lower_saved_model.cc
  // with the original code trimmed and adapted as needed.
  void runOnOperation() override {
    auto module = getOperation();
    if (!tf_saved_model::HasTfSavedModelSemantics(module)) return;

    // Clean up functions from tf_saved_model attributes.
    OpBuilder builder(module);
    auto bound_input = builder.getStringAttr("tf_saved_model.bound_input");
    auto exported_names =
        builder.getStringAttr("tf_saved_model.exported_names");
    auto index_path = builder.getStringAttr("tf_saved_model.index_path");
    module.walk([&](func::FuncOp func) {
      func->removeAttr(exported_names);
      for (unsigned i = 0, e = func.getNumArguments(); i != e; ++i) {
        if (func.getArgAttrOfType<FlatSymbolRefAttr>(i, bound_input)) {
          func.removeArgAttr(i, bound_input);
        }
        func.removeArgAttr(i, index_path);
      }
      for (unsigned i = 0, e = func.getNumResults(); i != e; ++i) {
        func.removeResultAttr(i, bound_input);
        func.removeResultAttr(i, index_path);
      }
    });

    // Clean up modules from tf_saved_model attributes.
    module->removeAttr("tf_saved_model.semantics");
  }
};

std::unique_ptr<Pass> CreateDropSavedModelSemanticsPass() {
  return std::make_unique<DropSavedModelSemanticsPass>();
}

static PassRegistration<DropSavedModelSemanticsPass> pass;

}  // namespace mhlo
}  // namespace TFL
}  // namespace mlir
