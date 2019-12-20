/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This pass will replace a func's bound inputs which are bound to constant
// global tensors with tf.Const ops inside the func's body.
// This can be useful when bringing up backends since it allows running
// stateless models before implementing global tensor support.

#include <map>
#include <set>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace tf_saved_model {
namespace {
struct InlineGlobalTensorsPass : public ModulePass<InlineGlobalTensorsPass> {
  void runOnModule() override;
};

void InlineGlobalTensorsPass::runOnModule() {
  auto module = getModule();
  SymbolTable symbol_table(module);
  for (auto func : module.getOps<FuncOp>()) {
    SmallVector<unsigned, 4> args_to_erase;
    OpBuilder builder(func.getBody());
    for (int i = 0, e = func.getNumArguments(); i < e; i++) {
      auto global_tensor = LookupBoundInput(func, i, symbol_table);
      if (!global_tensor) continue;

      // Don't inline mutable global tensors. They could be holding state across
      // invocations to this function.
      if (global_tensor.is_mutable()) continue;

      // Replace the arg with a tf.Const op in the function body.
      auto const_op = builder.create<TF::ConstOp>(global_tensor.getLoc(),
                                                  global_tensor.value());
      func.getArgument(i)->replaceAllUsesWith(const_op.getResult());
      args_to_erase.push_back(i);
    }
    func.eraseArguments(args_to_erase);
  }
  // We have already inlined all constant tensors, so erase them.
  for (auto global_tensor :
       llvm::make_early_inc_range(module.getOps<GlobalTensorOp>())) {
    if (!global_tensor.is_mutable()) global_tensor.erase();
  }
}

}  // namespace

// For "opt" to pick up this pass.
static PassRegistration<InlineGlobalTensorsPass> pass(
    "tf-saved-model-inline-global-tensors",
    "Inline tf_saved_model.global_tensor's as tf.Const ops in func bodies.");

std::unique_ptr<OpPassBase<ModuleOp>> CreateInlineGlobalTensorsPass() {
  return std::make_unique<InlineGlobalTensorsPass>();
}

}  // namespace tf_saved_model
}  // namespace mlir
