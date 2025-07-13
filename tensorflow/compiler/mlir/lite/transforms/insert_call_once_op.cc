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

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_INSERTCALLONCEOPFROMSESSIONINITIALIZERPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// This pass inserts a TFL::CallOnce op when tf_saved_model's session
// initializer is given.
class InsertCallOnceOpFromSessionInitializerPass
    : public impl::InsertCallOnceOpFromSessionInitializerPassBase<
          InsertCallOnceOpFromSessionInitializerPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      InsertCallOnceOpFromSessionInitializerPass)

 private:
  void runOnOperation() override;
};

void InsertCallOnceOpFromSessionInitializerPass::runOnOperation() {
  ModuleOp module = getOperation();

  for (func::FuncOp init_func_op :
       tf_saved_model::GetInitializerFunctions(module)) {
    for (auto func : module.getOps<func::FuncOp>()) {
      auto dict_attr =
          func->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
      if (!dict_attr) continue;

      OpBuilder builder(func.getContext());
      builder.setInsertionPointToStart(&func.getBlocks().front());
      builder.create<TFL::CallOnceOp>(func.getLoc(), init_func_op.getName());
    }
  }
}

}  // namespace

// Inserts a TFL::CallOnce op when tf_saved_model's session initializer is
// given.
std::unique_ptr<OperationPass<ModuleOp>>
CreateInsertCallOnceOpFromSessionInitializerPass() {
  return std::make_unique<InsertCallOnceOpFromSessionInitializerPass>();
}

}  // namespace TFL
}  // namespace mlir
