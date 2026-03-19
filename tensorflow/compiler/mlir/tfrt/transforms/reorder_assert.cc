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
#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"

namespace tensorflow {
namespace tfrt_compiler {
namespace {

// Reorder tf.Assert ops or tf.If ops that contains only tf.Assert ops to the
// end of the function, in order to avoid unnecessary control dependencies
// between tf.Assert and other ops.
class ReorderTfAssertPass
    : public mlir::PassWrapper<ReorderTfAssertPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReorderTfAssertPass)

  llvm::StringRef getArgument() const final { return "tfrt-reorder-tf-assert"; }
  llvm::StringRef getDescription() const final {
    return "Move tf.Assert to the end of the function to avoid unnecessary "
           "control dependencies";
  }

  void runOnOperation() override {
    auto module = getOperation();
    for (auto func_op : module.getOps<mlir::func::FuncOp>()) {
      ProcessFunction(func_op);
    }
  }

  void ProcessFunction(mlir::func::FuncOp func_op) {
    auto& block = func_op.front();

    llvm::SmallVector<mlir::Operation*, 2> assert_ops;
    for (mlir::Operation& op : block) {
      if (auto assert_op = llvm::dyn_cast<mlir::TF::AssertOp>(&op)) {
        assert_ops.push_back(assert_op);
      }

      if (auto if_op = llvm::dyn_cast<mlir::TF::IfOp>(&op)) {
        if (IsAssertOnlyIfOp(if_op)) {
          assert_ops.push_back(if_op);
        }
      }
    }

    auto& return_op = block.back();

    for (auto assert_op : assert_ops) {
      assert_op->moveBefore(&return_op);
    }
  }

  bool IsAssertOnlyIfOp(mlir::TF::IfOp op) {
    // If the results of the if op are used by some other ops, we cannot reorder
    // it.
    if (!op->use_empty()) return false;

    // Only reorder if both branches are non-side-effecting or containing only
    // Assert ops.
    if (IsFunctionNonSideEffectingOrAssert(op.then_function()) &&
        IsFunctionNonSideEffectingOrAssert(op.else_function()))
      return true;

    return false;
  }

  bool IsFunctionNonSideEffectingOrAssert(mlir::func::FuncOp func_op) {
    auto& block = func_op.front();
    for (mlir::Operation& op : block) {
      if (!llvm::isa<mlir::TF::AssertOp>(&op) && !mlir::isMemoryEffectFree(&op))
        return false;
    }
    return true;
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateReorderTfAssertPass() {
  return std::make_unique<ReorderTfAssertPass>();
}

static mlir::PassRegistration<ReorderTfAssertPass> register_pass(
    CreateReorderTfAssertPass);

}  // namespace tfrt_compiler
}  // namespace tensorflow
