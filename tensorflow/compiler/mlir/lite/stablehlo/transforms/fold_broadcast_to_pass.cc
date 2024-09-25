/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep

namespace mlir {
namespace odml {

namespace {

class ConvertResultsBroadcastableShapeOp : public RewritePattern {
 public:
  explicit ConvertResultsBroadcastableShapeOp(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;

 protected:
  LogicalResult RewriteOp(
      Operation* op, PatternRewriter& rewriter,
      const std::function<bool(ArrayRef<int64_t>, ArrayRef<int64_t>,
                               SmallVectorImpl<int64_t>&)>&
          get_broadcasted_shape) const;
};

// Some tfl ops only support implicit broadcasting up to a certain rank.
// Determine op with shapes is valid. TODO: @lukeboyer - Move the
// `TFL_OperandsHaveSameShapesOrBroadcastableShape` runtime verification trait
// into a standard (not runtime verification) trait and change this function to
// use only that interface. Curently there is no way to query derived runtime
// verification traits.
bool IsRankSupported(Operation* op) {
  // These ops have no rank constraints.
  if (llvm::isa<TFL::AddOp, TFL::SubOp, TFL::MulOp>(op)) {
    return true;
  }

  if (auto div_op = llvm::dyn_cast_or_null<TFL::DivOp>(op)) {
    return div_op.getType().getRank() <= 5;
  }

  // Fallback, all implicit broadcast ops in tfl support at least rank 4.
  return llvm::cast<ShapedType>(op->getResultTypes()[0]).getRank() <= 4;
}

LogicalResult ConvertResultsBroadcastableShapeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  if (op->hasTrait<OpTrait::ResultsBroadcastableShape>())
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
      mlir::dyn_cast_or_null<RankedTensorType>(op->getResultTypes().front());
  if (!result_type || !result_type.hasStaticShape()) return failure();

  if (!IsRankSupported(op)) {
    return failure();
  }

  bool changed = false;
  for (uint64_t i = 0, e = op->getNumOperands(); i < e; ++i) {
    // Check that the i'th operand is a broadcast.
    auto broadcast = llvm::dyn_cast_or_null<TFL::BroadcastToOp>(
        op->getOpOperand(i).get().getDefiningOp());
    if (!broadcast) continue;

    // Check that the operand of the broadcast has fully defined shape.
    auto broadcast_arg_type = mlir::dyn_cast_or_null<RankedTensorType>(
        broadcast.getInput().getType());
    if (!broadcast_arg_type || !broadcast_arg_type.hasStaticShape()) continue;

    // Check that the other argument has fully defined shape.
    auto argument_type = mlir::dyn_cast_or_null<RankedTensorType>(
        op->getOpOperand(1 - i).get().getType());
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

class ConvertResultsBroadcastableBatchMatMulShapeOp
    : public ConvertResultsBroadcastableShapeOp {
 public:
  explicit ConvertResultsBroadcastableBatchMatMulShapeOp(MLIRContext* context)
      : ConvertResultsBroadcastableShapeOp(context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;

 private:
  LogicalResult RewriteOp(Operation* op, PatternRewriter& rewriter) const;
};

LogicalResult ConvertResultsBroadcastableBatchMatMulShapeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  if (succeeded(RewriteOp(op, rewriter))) return success();

  return failure();
}

LogicalResult ConvertResultsBroadcastableBatchMatMulShapeOp::RewriteOp(
    Operation* op, PatternRewriter& rewriter) const {
  auto matmul_op = llvm::dyn_cast<TFL::BatchMatMulOp>(op);
  if (!matmul_op) return failure();

  // Gets the broadcasted output shape for tf.BatchMatMulOp. `shape_x` is the
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

  return ConvertResultsBroadcastableShapeOp::RewriteOp(op, rewriter,
                                                       get_broadcasted_shape);
}

}  // namespace

class FoldBroadcastToPass
    : public PassWrapper<FoldBroadcastToPass, OperationPass<func::FuncOp>> {
 public:
  StringRef getArgument() const final { return "fold-broadcast-to-pass"; }
  StringRef getDescription() const final {
    return "Folds tfl.BroadcastTo nodes with subsequent ops";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    auto func = getOperation();

    patterns.add<ConvertResultsBroadcastableShapeOp>(func.getContext());
    patterns.add<ConvertResultsBroadcastableBatchMatMulShapeOp>(
        func.getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

// TODO(weiyiw): Consider having this as canonicalization?
std::unique_ptr<OperationPass<func::FuncOp>> CreateFoldBroadcastToPass() {
  return std::make_unique<FoldBroadcastToPass>();
}

static PassRegistration<FoldBroadcastToPass> pass;

}  // namespace odml
}  // namespace mlir
