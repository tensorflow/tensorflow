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

#include "absl/memory/memory.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace {

class ConvertResultsBroadcastableShapeOp : public RewritePattern {
 public:
  ConvertResultsBroadcastableShapeOp(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;

 private:
  template <typename Op>
  LogicalResult RewriteEqOp(Operation* op, PatternRewriter& rewriter) const;

  LogicalResult RewriteOp(
      Operation* op, PatternRewriter& rewriter,
      const std::function<bool(ArrayRef<int64_t>, ArrayRef<int64_t>,
                               SmallVectorImpl<int64_t>&)>&
          get_broadcasted_shape) const;

  LogicalResult RewriteBatchMatMulV2Op(Operation* op,
                                       PatternRewriter& rewriter) const;
};

#define GEN_PASS_DEF_BROADCASTFOLDPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

class BroadcastFoldPass
    : public impl::BroadcastFoldPassBase<BroadcastFoldPass> {
 public:
  void runOnOperation() override;
};

LogicalResult ConvertResultsBroadcastableShapeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  if (op->hasTrait<OpTrait::ResultsBroadcastableShape>())
    return RewriteOp(op, rewriter, OpTrait::util::getBroadcastedShape);

  // tf.Equal and tf.NotEqual ops only satisfy ResultsBroadcastableShape when
  // incompatible_shape_error is `true` (what is also checked by the verifier).
  if (succeeded(RewriteEqOp<TF::EqualOp>(op, rewriter))) return success();
  if (succeeded(RewriteEqOp<TF::NotEqualOp>(op, rewriter))) return success();
  if (succeeded(RewriteBatchMatMulV2Op(op, rewriter))) return success();

  return failure();
}

LogicalResult ConvertResultsBroadcastableShapeOp::RewriteBatchMatMulV2Op(
    Operation* op, PatternRewriter& rewriter) const {
  auto matmul_op = llvm::dyn_cast<TF::BatchMatMulV2Op>(op);
  if (!matmul_op) return failure();

  // Gets the broadcasted output shape for tf.BatchMatMulV2Op. `shape_x` is the
  // shape of op's first/left-hand-side operand and `shape_y` is the shape of
  // op's second/right-hand-side operand.
  const auto get_broadcasted_shape =
      [&](ArrayRef<int64_t> shape_x, ArrayRef<int64_t> shape_y,
          SmallVectorImpl<int64_t>& result_shape) {
        if (shape_x.size() < 2 || shape_y.size() < 2) {
          return false;
        }

        // Checks outer dimensions (i.e., the dimensions higher than 2D) are
        // broadcastable. If true, then get the broadcasted shape for outer
        // dimension.
        if (!OpTrait::util::getBroadcastedShape(
                shape_x.drop_back(2), shape_y.drop_back(2), result_shape)) {
          return false;
        }

        const int x_row =
            matmul_op.getAdjX() ? shape_x.back() : *(shape_x.rbegin() + 1);
        const int x_col =
            !matmul_op.getAdjX() ? shape_x.back() : *(shape_x.rbegin() + 1);

        const int y_row =
            matmul_op.getAdjY() ? shape_y.back() : *(shape_y.rbegin() + 1);
        const int y_col =
            !matmul_op.getAdjY() ? shape_y.back() : *(shape_y.rbegin() + 1);

        // Checks that matrix multiply can perform a valid contraction.
        if (x_col != y_row) {
          result_shape.clear();
          return false;
        }

        result_shape.push_back(x_row);
        result_shape.push_back(y_col);
        return true;
      };

  return RewriteOp(op, rewriter, get_broadcasted_shape);
}

template <typename Op>
LogicalResult ConvertResultsBroadcastableShapeOp::RewriteEqOp(
    Operation* op, PatternRewriter& rewriter) const {
  auto eq_op = llvm::dyn_cast_or_null<Op>(op);
  if (eq_op && eq_op.getIncompatibleShapeError())
    return RewriteOp(op, rewriter, OpTrait::util::getBroadcastedShape);
  return failure();
}

LogicalResult ConvertResultsBroadcastableShapeOp::RewriteOp(
    Operation* op, PatternRewriter& rewriter,
    const std::function<bool(ArrayRef<int64_t>, ArrayRef<int64_t>,
                             SmallVectorImpl<int64_t>&)>& get_broadcasted_shape)
    const {
  if (op->getNumOperands() != 2 || op->getResultTypes().size() != 1)
    return failure();

  // Check that the result shape is fully defined.
  auto result_type =
      op->getResultTypes().front().dyn_cast_or_null<RankedTensorType>();
  if (!result_type || !result_type.hasStaticShape()) return failure();

  bool changed = false;
  for (uint64_t i = 0, e = op->getNumOperands(); i < e; ++i) {
    // Check that the i'th operand is a broadcast.
    auto broadcast = llvm::dyn_cast_or_null<TF::BroadcastToOp>(
        op->getOpOperand(i).get().getDefiningOp());
    if (!broadcast) continue;

    // Check that the operand of the broadcast has fully defined shape.
    auto broadcast_arg_type =
        broadcast.getInput().getType().dyn_cast_or_null<RankedTensorType>();
    if (!broadcast_arg_type || !broadcast_arg_type.hasStaticShape()) continue;

    // Check that the other argument has fully defined shape.
    auto argument_type = op->getOpOperand(1 - i)
                             .get()
                             .getType()
                             .dyn_cast_or_null<RankedTensorType>();
    if (!argument_type || !argument_type.hasStaticShape()) continue;

    // Get the unbroadcasted shapes in the operand order.
    std::array<llvm::ArrayRef<int64_t>, 2> operand_shapes;
    operand_shapes[i] = broadcast_arg_type.getShape();
    operand_shapes[1 - i] = argument_type.getShape();

    // Check that the input of the broadcast and the other operand is broadcast
    // compatible.
    llvm::SmallVector<int64_t, 4> broadcasted_shape;
    if (!get_broadcasted_shape(operand_shapes[0], operand_shapes[1],
                               broadcasted_shape))
      continue;

    // Check that an implicit broadcast between the operand of the broadcast and
    // the other argument would result in the same type as the result type.
    if (broadcasted_shape != result_type.getShape()) continue;

    // Update the operand of the op to be the operand of the broadcast.
    rewriter.modifyOpInPlace(
        op, [&]() { op->getOpOperand(i).set(broadcast.getInput()); });
    changed = true;
  }
  return success(changed);
}

void BroadcastFoldPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();

  patterns.add<ConvertResultsBroadcastableShapeOp>(func.getContext());
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

namespace TF {
std::unique_ptr<OperationPass<func::FuncOp>> CreateBroadcastFoldPass() {
  return std::make_unique<BroadcastFoldPass>();
}
}  // namespace TF

}  // namespace mlir
