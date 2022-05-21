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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"

namespace tensorflow {
namespace tfrt_compiler {
namespace {

bool FunctionHasSideEffect(
    mlir::func::FuncOp func_op,
    llvm::DenseMap<mlir::func::FuncOp, bool>& function_side_effect) {
  auto iter = function_side_effect.find(func_op);
  if (iter != function_side_effect.end()) return iter->second;

  auto& block = func_op.front();

  auto op_has_side_effect = [&](mlir::Operation* op) {
    if (auto while_op = llvm::dyn_cast<mlir::TF::WhileOp>(op)) {
      if (while_op.is_stateless()) return false;

      return FunctionHasSideEffect(while_op.cond_function(),
                                   function_side_effect) ||
             FunctionHasSideEffect(while_op.body_function(),
                                   function_side_effect);
    }

    if (auto if_op = llvm::dyn_cast<mlir::TF::IfOp>(op)) {
      if (if_op.is_stateless()) return false;

      return FunctionHasSideEffect(if_op.else_function(),
                                   function_side_effect) ||
             FunctionHasSideEffect(if_op.then_function(), function_side_effect);
    }

    // Though tf.Assert and tf.Timestamp are side-effecting, they do not
    // interfere with any other side-effecting ops. For now, if control flow
    // ops' callee functions contain them, we treat them as non-side-effecting.
    if (llvm::isa<mlir::TF::AssertOp, mlir::TF::TimestampOp>(op)) return false;

    return !mlir::MemoryEffectOpInterface::hasNoEffect(op);
  };

  // Speculatively setting the function to have no side effect to avoid infinite
  // recursion. The correct side effect will be updated later once more
  // operations in the block are checked.
  function_side_effect[func_op] = false;

  for (mlir::Operation& op : block) {
    if (op_has_side_effect(&op)) {
      function_side_effect[func_op] = true;
      return true;
    }
  }

  function_side_effect[func_op] = false;
  return false;
}

// This pass sets `is_stateless` attribute of tf.If and tf.While ops to true if
// their callee functions contains only non-side-effecting ops.
class OptimizeTfControlFlowSideEffectPass
    : public mlir::PassWrapper<OptimizeTfControlFlowSideEffectPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      OptimizeTfControlFlowSideEffectPass)

 private:
  llvm::StringRef getArgument() const final {
    return "tfrt-optimize-tf-control-flow-side-effect";
  }
  llvm::StringRef getDescription() const final {
    return "Set tf control flow ops to stateless if their callee functions "
           "contains only non-side-effecting ops";
  }
  void runOnOperation() override {
    auto module = getOperation();
    llvm::DenseMap<mlir::func::FuncOp, bool> function_side_effect;

    mlir::Builder builder(module.getContext());
    module.walk([&](mlir::Operation* op) {
      if (auto while_op = llvm::dyn_cast<mlir::TF::WhileOp>(op)) {
        if (while_op.is_stateless()) return;

        if (!FunctionHasSideEffect(while_op.cond_function(),
                                   function_side_effect) &&
            !FunctionHasSideEffect(while_op.body_function(),
                                   function_side_effect)) {
          while_op->setAttr("is_stateless", builder.getBoolAttr(true));
        }
      }

      if (auto if_op = llvm::dyn_cast<mlir::TF::IfOp>(op)) {
        if (if_op.is_stateless()) return;

        if (!FunctionHasSideEffect(if_op.else_function(),
                                   function_side_effect) &&
            !FunctionHasSideEffect(if_op.then_function(),
                                   function_side_effect)) {
          if_op->setAttr("is_stateless", builder.getBoolAttr(true));
        }
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateOptimizeTfControlFlowSideEffectPass() {
  return std::make_unique<OptimizeTfControlFlowSideEffectPass>();
}

static mlir::PassRegistration<OptimizeTfControlFlowSideEffectPass>
    register_pass(CreateOptimizeTfControlFlowSideEffectPass);

}  // namespace tfrt_compiler
}  // namespace tensorflow
