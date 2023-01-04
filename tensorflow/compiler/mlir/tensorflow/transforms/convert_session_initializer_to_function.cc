/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"

namespace mlir {
namespace tf_saved_model {

namespace {

#define GEN_PASS_DEF_CONVERTSESSIONINITIALIZERTOFUNCTIONPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_savedmodel_passes.h.inc"

struct ConvertSessionInitializerToFunctionPass
    : public impl::ConvertSessionInitializerToFunctionPassBase<
          ConvertSessionInitializerToFunctionPass> {
  void runOnOperation() override;
};

void ConvertSessionInitializerToFunctionPass::runOnOperation() {
  ModuleOp module = getOperation();
  auto session_initializer = tf_saved_model::GetSessionInitializerOp(module);
  if (!session_initializer) return;

  OpBuilder builder(session_initializer);
  const char *name = "session_initializer";

  // In the (unlikely) case of there already being a session initializer
  // function, bail out.
  if (SymbolTable::lookupSymbolIn(module, name)) {
    module->emitWarning("session_initializer function already exists");
    session_initializer.erase();
    return;
  }

  auto init = builder.create<mlir::func::FuncOp>(
      module.getLoc(), name,
      mlir::FunctionType::get(module.getContext(), /*inputs=*/{},
                              /*results=*/{}));

  // Make savedmodel verification happy.
  init->setAttr("tf_saved_model.exported_names",
                builder.getStrArrayAttr({name}));

  builder.setInsertionPointToStart(init.addEntryBlock());

  for (auto attr : session_initializer.getInitializers()) {
    auto sym = attr.dyn_cast<FlatSymbolRefAttr>();
    if (!sym) {
      session_initializer->emitWarning("non-symbol initializer");
      continue;
    }
    Operation *function = SymbolTable::lookupSymbolIn(module, sym);
    func::FuncOp func = llvm::dyn_cast<func::FuncOp>(function);
    if (!func) {
      session_initializer->emitWarning(
          "session initializer doesn't resolve to a function");
      continue;
    }
    if (func.getNumArguments() != 0) {
      session_initializer->emitWarning(
          "encountered session initializers with arguments");
      continue;
    }

    // Since we're now calling this function, savedmodel verification
    // needs it to be private.
    func.setVisibility(mlir::SymbolTable::Visibility::Private);
    func->removeAttr("tf_saved_model.exported_names");

    ArrayRef<Value> args;
    builder.create<mlir::func::CallOp>(session_initializer.getLoc(),
                                       func.getFunctionType().getResults(),
                                       func.getSymName(), args);
  }
  builder.create<mlir::func::ReturnOp>(session_initializer.getLoc());

  session_initializer.erase();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateConvertSessionInitializerToFunctionPass() {
  return std::make_unique<ConvertSessionInitializerToFunctionPass>();
}

}  // namespace tf_saved_model
}  // namespace mlir
