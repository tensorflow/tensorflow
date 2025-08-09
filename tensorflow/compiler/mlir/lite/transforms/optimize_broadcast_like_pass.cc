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

#include "tensorflow/compiler/mlir/lite/transforms/optimize_broadcast_like_pass.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/transforms/optimize_broadcast_like_pass_options.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"

namespace mlir {
namespace TFL {

namespace {

using BroadcastedShapeFunction =
    std::function<bool(llvm::ArrayRef<int64_t>, llvm::ArrayRef<int64_t>,
                       SmallVectorImpl<int64_t>&)>;

class ConvertResultsBroadcastableShapeOp : public RewritePattern {
 public:
  explicit ConvertResultsBroadcastableShapeOp(
      MLIRContext* context, const OptimizeBroadcastLikePassOptions& options)
      : RewritePattern(MatchAnyOpTypeTag(), /*PatternBenefit*/ 1, context),
        options_(options) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;

 protected:
  LogicalResult RewriteOp(
      Operation* op, PatternRewriter& rewriter,
      BroadcastedShapeFunction& get_broadcasted_shape) const;

 private:
  const OptimizeBroadcastLikePassOptions& options_;
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
    // TFL_AddOp, TFL_SubOp, and TFL_MulOp support implicit broadcasting up to
    // rank 6.
    return llvm::cast<ShapedType>(op->getResultTypes()[0]).getRank() <= 6;
  }

  if (auto div_op = llvm::dyn_cast_or_null<TFL::DivOp>(op)) {
    return div_op.getType().getRank() <= 5;
  }

  // Fallback, all implicit broadcast ops in tfl support at least rank 4.
  return llvm::cast<ShapedType>(op->getResultTypes()[0]).getRank() <= 4;
}

// Returns true if this value is a splat constant op which can be scalarized.
bool IsScalarizableSplatConstant(Operation* op) {
  if (!llvm::isa<arith::ConstantOp, TFL::ConstOp>(op)) {
    return false;
  }
  auto const_value = op->getResult(0);
  DenseElementsAttr elements_attr;
  if (!matchPattern(const_value, mlir::m_Constant(&elements_attr))) {
    return false;
  }

  auto type = mlir::dyn_cast<ShapedType>(const_value.getType());
  return type && type.hasStaticShape() && (type.getNumElements() > 1) &&
         elements_attr.isSplat();
}

// Returns the input of the broadcast-like op. This is the Value that is
// being broadcasted by the broadcast-like op. This function returns nullptr if
// the op is not a broadcast-like op.
Value PrepareBroadcastLikeOpInput(Operation* op, PatternRewriter& rewriter) {
  if (!op) return nullptr;
  if (auto broadcast_to_op = llvm::dyn_cast<TFL::BroadcastToOp>(op)) {
    return broadcast_to_op.getInput();
  }
  if (auto fill_op = llvm::dyn_cast<TFL::FillOp>(op)) {
    return fill_op.getInput();
  }
  if (IsScalarizableSplatConstant(op)) {
    DenseElementsAttr elements_attr;
    if (!matchPattern(op->getResult(0), mlir::m_Constant(&elements_attr))) {
      return nullptr;
    }
    auto scalar_elements_attr = DenseElementsAttr::get(
        RankedTensorType::get({}, elements_attr.getType().getElementType()),
        elements_attr.getSplatValue<mlir::Attribute>());

    return rewriter.create<arith::ConstantOp>(
        op->getLoc(), scalar_elements_attr.getType(), scalar_elements_attr);
  }
  return nullptr;
}

// Returns true if we can fuse an affine op with consuming binary op.
bool CanFuseAffineOp(Operation* affine_op, Operation* binary_op) {
  if (!isa_and_nonnull<TFL::Conv2DOp, TFL::DepthwiseConv2DOp,
                       TFL::FullyConnectedOp>(affine_op)) {
    return false;
  }

  DenseElementsAttr value;
  // Check that bias are constants if not none.
  Value bias = affine_op->getOperand(2);
  if (!mlir::isa<NoneType>(bias.getType()) &&
      !matchPattern(bias, m_Constant(&value))) {
    return false;
  }
  // If the binary op is mul/div, also check that filter is constant.
  if (isa<TFL::MulOp, TFL::DivOp>(binary_op) &&
      !matchPattern(affine_op->getOperand(1), m_Constant(&value))) {
    return false;
  }

  // We can only fuse F32/BF16.
  auto is_fusable_type = [](Type t) {
    Type element_type = t;
    if (auto shaped_type = mlir::dyn_cast<ShapedType>(t)) {
      element_type = shaped_type.getElementType();
    }
    return element_type.isBF16() || element_type.isF32();
  };
  for (Type t : binary_op->getOperandTypes()) {
    if (!is_fusable_type(t)) {
      return false;
    }
  }

  return true;
}

LogicalResult ConvertResultsBroadcastableShapeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  if (op->hasTrait<OpTrait::ResultsBroadcastableShape>()) {
    BroadcastedShapeFunction broadcasted_shape_function =
        [](ArrayRef<int64_t> a, ArrayRef<int64_t> b,
           SmallVectorImpl<int64_t>& c) {
          return OpTrait::util::getBroadcastedShape(a, b, c);
        };
    return RewriteOp(op, rewriter, broadcasted_shape_function);
  }
  return failure();
}

LogicalResult ConvertResultsBroadcastableShapeOp::RewriteOp(
    Operation* op, PatternRewriter& rewriter,
    BroadcastedShapeFunction& get_broadcasted_shape) const {
  if (op->getNumOperands() != 2 || op->getResultTypes().size() != 1)
    return rewriter.notifyMatchFailure(
        op, "Operand broadcasting not supported for op: " +
                op->getName().getStringRef());

  // Check that the result shape is fully defined.
  auto result_type = llvm::cast<ShapedType>(op->getResultTypes().front());
  if (!result_type || (!options_.unsafe_fuse_dynamic_shaped_broadcast &&
                       !result_type.hasStaticShape()))
    return rewriter.notifyMatchFailure(
        op, "Unsupported result shape for broadcasting on op: " +
                op->getName().getStringRef());

  if (!IsRankSupported(op)) {
    return rewriter.notifyMatchFailure(
        op, "Unsupported rank for broadcasting on op: " +
                op->getName().getStringRef());
  }

  bool changed = false;
  for (size_t i = 0, e = op->getNumOperands(); i < e; ++i) {
    auto broadcast_like_op = op->getOpOperand(i).get().getDefiningOp();

    // TODO(b/381683763) - Add support for fusing broadcast-like opps with
    // PReluOp.
    if (llvm::isa<TFL::PReluOp>(op)) {
      continue;
    }

    // Get the input of the broadcast-like op. This is the Value that is
    // being broadcasted by the broadcast-like op. This function returns
    // nullptr if the op is not a broadcast-like op.
    auto broadcast_like_op_input =
        PrepareBroadcastLikeOpInput(broadcast_like_op, rewriter);
    if (!broadcast_like_op || !broadcast_like_op_input) {
      continue;
    }

    // Check that the operand of the broadcast has fully defined shape.
    auto broadcast_arg_type =
        llvm::cast<ShapedType>(broadcast_like_op_input.getType());
    if (!broadcast_arg_type ||
        (!options_.unsafe_fuse_dynamic_shaped_broadcast &&
         !broadcast_arg_type.hasStaticShape()))
      continue;

    auto other_arg = op->getOpOperand(1 - i).get();
    // If non-splat operand is not fusable affine ops, then no need to apply
    // this transformation. Check valid for only binary ops, to prevent future
    // fusion with other ops from breaking.
    if (llvm::isa<TFL::AddOp, TFL::SubOp, TFL::MulOp, TFL::DivOp>(op) &&
        IsScalarizableSplatConstant(broadcast_like_op) &&
        !CanFuseAffineOp(other_arg.getDefiningOp(), op)) {
      continue;
    }

    // Check that the other argument has fully defined shape.
    auto other_arg_type = llvm::cast<ShapedType>(other_arg.getType());
    if (!other_arg_type || (!options_.unsafe_fuse_dynamic_shaped_broadcast &&
                            !other_arg_type.hasStaticShape()))
      continue;

    // Get the unbroadcasted shapes in the operand order.
    std::array<llvm::ArrayRef<int64_t>, 2> operand_shapes;
    operand_shapes[i] = broadcast_arg_type.getShape();
    operand_shapes[1 - i] = other_arg_type.getShape();

    // Check that the input of the broadcast and the other operand is
    // broadcast compatible.
    llvm::SmallVector<int64_t, 4> broadcasted_shape;
    if (!get_broadcasted_shape(operand_shapes[0], operand_shapes[1],
                               broadcasted_shape))
      continue;

    // Check that an implicit broadcast between the operand of the broadcast
    // and the other argument would result in the same type as the result
    // type.
    if (broadcasted_shape != result_type.getShape()) continue;

    // Update the operand of the op to be the operand of the broadcast.
    rewriter.modifyOpInPlace(
        op, [&]() { op->getOpOperand(i).set(broadcast_like_op_input); });
    changed = true;
  }
  return success(changed);
}

class ConvertResultsBroadcastableBatchMatMulShapeOp
    : public ConvertResultsBroadcastableShapeOp {
 public:
  explicit ConvertResultsBroadcastableBatchMatMulShapeOp(
      MLIRContext* context, const OptimizeBroadcastLikePassOptions& options)
      : ConvertResultsBroadcastableShapeOp(context, options) {}

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
  if (!matmul_op) return rewriter.notifyMatchFailure(op, "Not a BatchMatMulOp");

  // Gets the broadcasted output shape for tf.BatchMatMulOp. `shape_x` is the
  // shape of op's first/left-hand-side operand and `shape_y` is the shape of
  // op's second/right-hand-side operand.
  BroadcastedShapeFunction get_broadcasted_shape =
      [&](llvm::ArrayRef<int64_t> shape_x, llvm::ArrayRef<int64_t> shape_y,
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

class ReorderBroadcastToCast : public RewritePattern {
 public:
  explicit ReorderBroadcastToCast(MLIRContext* context)
      : RewritePattern(TFL::CastOp::getOperationName(), /*PatternBenefit*/ 1,
                       context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult ReorderBroadcastToCast::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto cast_op = llvm::dyn_cast<TFL::CastOp>(op);
  if (!cast_op) return rewriter.notifyMatchFailure(op, "Not a CastOp");

  auto broadcast_to_op = llvm::dyn_cast_or_null<TFL::BroadcastToOp>(
      cast_op.getInput().getDefiningOp());
  if (!broadcast_to_op)
    return rewriter.notifyMatchFailure(op, "Not a BroadcastToOp");

  auto fused_loc = FusedLoc::get(cast_op.getContext(),
                                 {cast_op.getLoc(), broadcast_to_op.getLoc()});

  auto input_value = broadcast_to_op.getInput();
  auto input_type = input_value.getType();
  auto old_cast_op_output_type = cast_op.getOutput().getType();
  auto new_cast_op_output_type =
      old_cast_op_output_type.hasRank()
          ? static_cast<TensorType>(
                RankedTensorType::get(input_type.getShape(),
                                      old_cast_op_output_type.getElementType()))
          : static_cast<TensorType>(UnrankedTensorType::get(
                old_cast_op_output_type.getElementType()));

  auto new_cast_op = rewriter.create<TFL::CastOp>(
      fused_loc, new_cast_op_output_type, input_value);
  auto new_broadcast_to_op = rewriter.create<TFL::BroadcastToOp>(
      fused_loc, old_cast_op_output_type, new_cast_op.getOutput(),
      broadcast_to_op.getShape());

  rewriter.replaceOp(cast_op, new_broadcast_to_op.getOutput());
  return success();
}

#include "tensorflow/compiler/mlir/lite/transforms/generated_optimize_broadcast_like.inc"
}  // namespace

void OptimizeBroadcastLikePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();

  patterns.add<ConvertResultsBroadcastableShapeOp>(func.getContext(),
                                                   GetOptions());
  patterns.add<ConvertResultsBroadcastableBatchMatMulShapeOp>(func.getContext(),
                                                              GetOptions());
  patterns.add<ReorderBroadcastToCast>(func.getContext());
  TFL::populateWithGenerated(patterns);
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}

}  // namespace TFL
}  // namespace mlir
