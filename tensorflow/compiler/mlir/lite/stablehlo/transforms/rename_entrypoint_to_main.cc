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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/rename_entrypoint_to_main.h"

#include <memory>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace odml {

class RenameEntrypointToMainPass
    : public PassWrapper<RenameEntrypointToMainPass, OperationPass<ModuleOp>> {
 public:
  StringRef getArgument() const final { return "rename-entrypoint-to-main"; }
  StringRef getDescription() const final {
    return "Renames the entrypoint in SavedModel to `main`";
  }

  void runOnOperation() override {
    auto fail = [&](Operation* op, std::string message) {
      op->emitError(message);
      signalPassFailure();
    };

    DenseMap<StringRef, func::FuncOp> entrypoints;
    auto module = getOperation();
    module.walk([&](func::FuncOp op) {
      auto visibility = SymbolTable::getSymbolVisibility(op);
      if (visibility != SymbolTable::Visibility::Public) return;
      entrypoints[op.getSymName()] = op;
    });

    if (auto session_initializer =
            tf_saved_model::GetSessionInitializerOp(module)) {
      // clang-format off
      // Skip session initializer functions which are present in saved models.
      // For example:
      //   "tf_saved_model.session_initializer"() {initializers = [@init_all_tables]} : () -> () // NOLINT
      //   func @init_all_tables() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_init_all_tables"]} { // NOLINT
      //     "tf.NoOp"() {device = ""} : () -> ()
      //     return
      //   }
      // clang-format on
      for (auto attr : session_initializer.getInitializers()) {
        auto sym_attr = mlir::dyn_cast<FlatSymbolRefAttr>(attr);
        if (!sym_attr) break;
        entrypoints.erase(sym_attr.getValue());
      }
    }

    if (entrypoints.empty()) {
      return fail(module, "No entrypoints found");
    }
    if (entrypoints.size() == 1) {
      auto entrypoint = entrypoints.begin()->second;
      Builder builder(entrypoint);
      entrypoint.setName(builder.getStringAttr("main"));
      return;
    }

    // In case we have more than 1 entry points, choose the one with
    // 'tf.entry_function' attribute set.
    llvm::SmallVector<func::FuncOp, 4> candidate_funcs;
    for (auto& entrypoint : entrypoints) {
      if (entrypoint.second->hasAttr("tf.entry_function")) {
        candidate_funcs.push_back(entrypoint.second);
      }
    }

    if (candidate_funcs.empty()) {
      return fail(module, "No entrypoints found");
    }
    if (candidate_funcs.size() > 1) {
      return fail(module, "Too many entrypoints found");
    }
    // Found entrypoint
    Builder builder(candidate_funcs[0]);
    candidate_funcs[0].setName(builder.getStringAttr("main"));
  }
};

std::unique_ptr<Pass> CreateRenameEntrypointToMainPass() {
  return std::make_unique<RenameEntrypointToMainPass>();
}

static PassRegistration<RenameEntrypointToMainPass> pass;

}  // namespace odml
}  // namespace mlir
