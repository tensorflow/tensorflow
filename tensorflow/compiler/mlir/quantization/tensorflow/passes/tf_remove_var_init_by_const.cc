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
#include <utility>

#include "absl/log/log.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace tf_quant {
namespace {

using ::mlir::tf_saved_model::GetInitializerFunction;
using ::mlir::tf_saved_model::kTfSavedModelInitializerRestoreType;

// A pass that removes `tf.AssignVariableOp(tf.VarHandleOp, tf.Const)` patterns
// from the initializer function (type = "restore_op").
//
// Note: initializing values (`tf.Const`s) will be removed and this may result
// in an information loss and uninitialized variable errors. Make sure that this
// effect is desired (e.g. there is a `tf.RestoreV2Op` restoring the variables
// instead).
class RemoveVariableInitializationByConstPass
    : public PassWrapper<RemoveVariableInitializationByConstPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      RemoveVariableInitializationByConstPass)

  StringRef getArgument() const final {
    return "tf-quant-remove-var-init-by-const";
  }

  StringRef getDescription() const final {
    return "Removes `tf.AssignVariableOp(tf.VarHandleOp, tf.Const)` patterns "
           "from the initializer function of type 'restore_op'.";
  }

  void runOnOperation() override;
};

// Finds and removes the `tf.AssignVariableOp(tf.VarHandleOp, tf.Const)`
// pattern. `tf.VarHandleOp` and `tf.Const` are removed unless they are used by
// other ops.
struct RemoveVariableAssignmentByConst
    : public OpRewritePattern<TF::AssignVariableOp> {
  // Inherit the constructors.
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::AssignVariableOp assign_op,
                                PatternRewriter& rewriter) const override {
    Value resource_operand = assign_op.getOperand(0);
    Value assigned_value_operand = assign_op.getOperand(1);

    if (!isa<TF::VarHandleOp>(resource_operand.getDefiningOp()) ||
        !isa<TF::ConstOp>(assigned_value_operand.getDefiningOp())) {
      return failure();
    }

    // `TF::ConstOp` and `TF::VarHandleOp` are not manually erased.
    // `applyPatternsGreedily` performs dead code elimination and unsed
    // ops will be erased during the optimization.
    rewriter.eraseOp(assign_op);
    return success();
  }
};

void RemoveVariableInitializationByConstPass::runOnOperation() {
  MLIRContext& ctx = getContext();

  RewritePatternSet patterns(&ctx);
  patterns.add<RemoveVariableAssignmentByConst>(&ctx);

  ModuleOp module_op = getOperation();
  func::FuncOp init_func_op = GetInitializerFunction(
      module_op, /*initializer_type=*/kTfSavedModelInitializerRestoreType);
  if (init_func_op) {
    if (failed(applyPatternsGreedily(init_func_op, std::move(patterns)))) {
      init_func_op->emitError(
          "Failed to remove variable assignment by const patterns.");
      signalPassFailure();
    }
  } else {
    LOG(INFO) << "Initializer function with type 'restore_op' does not exist. "
                 "'RemoveVariableInitializationByConstPass' is a no-op.";
  }
}

static PassRegistration<RemoveVariableInitializationByConstPass> pass{};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateRemoveVariableInitializationByConstPass() {
  return std::make_unique<RemoveVariableInitializationByConstPass>();
}
}  // namespace tf_quant
}  // namespace mlir
