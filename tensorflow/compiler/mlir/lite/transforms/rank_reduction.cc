/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"

namespace mlir {
namespace TFL {
namespace {

// Helper to squeeze a 1D constant (shape/begin/size) given squeezed indices.
ElementsAttr SqueezeConstantAttr(ElementsAttr attr,
                                 ArrayRef<int> squeezed_indices) {
  if (!attr || squeezed_indices.empty()) return attr;
  auto type = dyn_cast<RankedTensorType>(attr.getType());
  if (!type || type.getRank() != 1) return attr;

  SmallVector<int64_t, 4> new_values;
  int original_size = type.getNumElements();
  for (int i = 0; i < original_size; ++i) {
    bool squeezed = false;
    for (int s : squeezed_indices) {
      if (s == i) {
        squeezed = true;
        break;
      }
    }
    if (!squeezed) {
      if (attr.getElementType().isInteger(32)) {
        new_values.push_back(attr.getValues<int32_t>()[i]);
      } else {
        new_values.push_back(attr.getValues<int64_t>()[i]);
      }
    }
  }

  auto new_type = RankedTensorType::get(
      {static_cast<int64_t>(new_values.size())}, attr.getElementType());
  if (attr.getElementType().isInteger(32)) {
    SmallVector<int32_t, 4> v32;
    for (auto v : new_values) v32.push_back(static_cast<int32_t>(v));
    return DenseIntElementsAttr::get(new_type, v32);
  }
  auto res_type = RankedTensorType::get(
      {static_cast<int64_t>(new_values.size())}, attr.getElementType());
  return DenseIntElementsAttr::get(res_type, new_values);
}

// Helper to squeeze a Nx2 constant (paddings) given squeezed indices.
// Returns nullptr if any squeezed dimension has non-zero padding.
ElementsAttr SqueezePaddingsAttr(ElementsAttr attr,
                                 ArrayRef<int> squeezed_indices) {
  if (!attr || squeezed_indices.empty()) return attr;
  auto type = dyn_cast<RankedTensorType>(attr.getType());
  if (!type || type.getRank() != 2 || type.getDimSize(1) != 2) return attr;

  SmallVector<int64_t, 8> new_values;
  int original_rows = type.getDimSize(0);
  for (int i = 0; i < original_rows; ++i) {
    bool squeezed = false;
    for (int s : squeezed_indices) {
      if (s == i) {
        squeezed = true;
        break;
      }
    }

    int64_t p0, p1;
    if (attr.getElementType().isInteger(32)) {
      p0 = attr.getValues<int32_t>()[{static_cast<uint64_t>(i), 0}];
      p1 = attr.getValues<int32_t>()[{static_cast<uint64_t>(i), 1}];
    } else {
      p0 = attr.getValues<int64_t>()[{static_cast<uint64_t>(i), 0}];
      p1 = attr.getValues<int64_t>()[{static_cast<uint64_t>(i), 1}];
    }

    if (squeezed) {
      if (p0 != 0 || p1 != 0) return nullptr;
    } else {
      new_values.push_back(p0);
      new_values.push_back(p1);
    }
  }

  auto new_type = RankedTensorType::get(
      {static_cast<int64_t>(new_values.size() / 2), 2}, attr.getElementType());
  if (attr.getElementType().isInteger(32)) {
    SmallVector<int32_t, 8> v32;
    for (auto v : new_values) v32.push_back(static_cast<int32_t>(v));
    return DenseIntElementsAttr::get(new_type, v32);
  }
  return DenseIntElementsAttr::get(new_type, new_values);
}

// Helper to get constant value from TFL ConstOp or PseudoConstOp.
DenseIntElementsAttr getIterativeConstantValue(Value v) {
  ElementsAttr attr;
  if (matchPattern(v, m_Constant(&attr))) {
    return dyn_cast<DenseIntElementsAttr>(attr);
  }
  return nullptr;
}

bool canLiftThroughSlice(SliceOp slice_op, ArrayRef<int> squeezed_indices) {
  auto begin_attr = getIterativeConstantValue(slice_op.getBegin());
  auto size_attr = getIterativeConstantValue(slice_op.getSize());
  if (!begin_attr || !size_attr) return false;

  auto begin_values = begin_attr.getValues<int32_t>();
  auto size_values = size_attr.getValues<int32_t>();

  auto slice_input_ty = cast<RankedTensorType>(slice_op.getInput().getType());
  for (int i : squeezed_indices) {
    if (begin_values[i] != 0 || size_values[i] != 1) return false;
    if (slice_input_ty.getDimSize(i) != 1) return false;
  }
  return true;
}

// Reshapes a tensor by squeezing specified indices.
Value SqueezeTensor(Value val, ArrayRef<int> squeezed_indices,
                    PatternRewriter& rewriter, Location loc) {
  auto ty = cast<RankedTensorType>(val.getType());
  SmallVector<int64_t, 4> new_shape;
  for (int i = 0; i < ty.getRank(); ++i) {
    if (!llvm::is_contained(squeezed_indices, i)) {
      new_shape.push_back(ty.getDimSize(i));
    }
  }
  auto new_ty = RankedTensorType::get(new_shape, ty.getElementType());

  SmallVector<int32_t, 4> shape_v32;
  for (auto d : new_shape) shape_v32.push_back(static_cast<int32_t>(d));
  auto shape_const_ty = RankedTensorType::get(
      {static_cast<int64_t>(shape_v32.size())}, rewriter.getI32Type());
  auto shape_const = rewriter.create<ConstOp>(
      loc, shape_const_ty,
      DenseIntElementsAttr::get(shape_const_ty, shape_v32));
  return rewriter.create<ReshapeOp>(loc, new_ty, val, shape_const);
}

#define GEN_PASS_DEF_RANKREDUCTIONPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// Sinks rank-expanding reshapes through unary element-wise ops.
// Pattern: 2D -> Reshape -> 5D -> Unary -> 5D
// Becomes: 2D -> Unary -> 2D -> Reshape -> 5D
template <typename OpType>
struct SinkReshapeThroughUnaryElementwise : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType unary_op,
                                PatternRewriter& rewriter) const override {
    auto reshape_op =
        unary_op->getOperand(0).template getDefiningOp<ReshapeOp>();
    if (!reshape_op) return failure();

    auto input_ty = cast<RankedTensorType>(reshape_op.getInput().getType());
    auto output_ty = cast<RankedTensorType>(reshape_op.getType());
    // Only sink if it's a rank-expanding reshape (adds unit dimensions).
    if (input_ty.getRank() >= output_ty.getRank()) return failure();

    // Create new Unary op on the reshape input.
    auto new_unary_ty = RankedTensorType::get(
        input_ty.getShape(),
        cast<RankedTensorType>(unary_op.getType()).getElementType());
    auto new_unary = rewriter.create<OpType>(unary_op.getLoc(), new_unary_ty,
                                             reshape_op.getInput());
    new_unary->setAttrs(unary_op->getAttrs());

    // Replace old unary op with a reshape of the new unary output.
    rewriter.replaceOpWithNewOp<ReshapeOp>(unary_op, unary_op.getType(),
                                           new_unary.getResult(),
                                           reshape_op.getShape());
    return success();
  }
};

// Lifts rank-collapsing reshapes through unary element-wise ops.
// Pattern: Unary -> Reshape
// Becomes: Reshape -> Unary
template <typename OpType>
struct LiftReshapeThroughUnaryElementwise : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp reshape_op,
                                PatternRewriter& rewriter) const override {
    auto unary_op = reshape_op.getInput().getDefiningOp<OpType>();
    if (!unary_op || unary_op->getNumResults() != 1 ||
        unary_op->getNumOperands() != 1) {
      return failure();
    }

    auto input_ty = cast<RankedTensorType>(reshape_op.getInput().getType());
    auto output_ty = cast<RankedTensorType>(reshape_op.getType());
    // Only lift if it's a rank-collapsing reshape (removes dimensions).
    if (input_ty.getRank() <= output_ty.getRank()) return failure();

    // Create new Reshape op on the unary input.
    auto original_input = unary_op->getOperand(0);
    auto original_input_ty =
        dyn_cast<RankedTensorType>(original_input.getType());
    if (!original_input_ty ||
        original_input_ty.getRank() != input_ty.getRank()) {
      return failure();
    }

    auto new_reshape_ty = RankedTensorType::get(
        output_ty.getShape(), original_input_ty.getElementType());
    auto new_reshape =
        rewriter.create<ReshapeOp>(reshape_op.getLoc(), new_reshape_ty,
                                   original_input, reshape_op.getShape());

    // Create new Unary op on the reshaped input.
    auto new_unary = rewriter.create<OpType>(unary_op.getLoc(), output_ty,
                                             new_reshape.getResult());
    new_unary->setAttrs(unary_op->getAttrs());

    rewriter.replaceOp(reshape_op, new_unary.getResult());
    return success();
  }
};

struct LiftReshapeThroughConcatenation : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp reshape_op,
                                PatternRewriter& rewriter) const override {
    auto concat_op = dyn_cast_or_null<ConcatenationOp>(
        reshape_op.getInput().getDefiningOp());
    if (!concat_op) return failure();

    auto input_ty = cast<RankedTensorType>(reshape_op.getInput().getType());
    auto output_ty = cast<RankedTensorType>(reshape_op.getType());

    int axis = concat_op.getAxis();
    if (axis < 0) axis += input_ty.getRank();

    // Calculate concatenation factor. Assuming all operands have same shape for
    // now.
    auto values = concat_op.getValues();
    if (values.empty()) return failure();
    auto first_op_ty = cast<RankedTensorType>(values[0].getType());
    int64_t operand_axis_size = first_op_ty.getDimSize(axis);
    int64_t output_axis_size = input_ty.getDimSize(axis);

    if (operand_axis_size == 0 || output_axis_size % operand_axis_size != 0)
      return failure();
    int64_t num_operands = values.size();
    // Verify all operands have the same size as the first one for simplicity.
    for (auto val : values) {
      if (cast<RankedTensorType>(val.getType()).getDimSize(axis) !=
          operand_axis_size)
        return failure();
    }

    // Find the new axis in the target shape.
    int64_t vol_before_axis = 1;
    for (int i = 0; i < axis; ++i) vol_before_axis *= input_ty.getDimSize(i);
    int64_t vol_with_axis = vol_before_axis * output_axis_size;

    int new_axis = -1;
    int64_t current_vol = 1;
    for (int i = 0; i < output_ty.getRank(); ++i) {
      int64_t next_vol = current_vol * output_ty.getDimSize(i);
      // The concat boundary must be preserved.
      // This means the volume before 'axis' in input must align with a
      // dimension boundary or be completely contained within a merged
      // dimension. For concatenation to be possible in the target, the target
      // dimension containing the concat boundary must be divisible by the
      // operand count.
      if (current_vol <= vol_before_axis && next_vol >= vol_with_axis) {
        if (output_ty.getDimSize(i) % num_operands == 0) {
          new_axis = i;
          break;
        }
      }
      current_vol = next_vol;
    }

    if (new_axis == -1) return failure();

    SmallVector<Value, 4> new_inputs;
    for (Value operand : values) {
      auto op_ty = cast<RankedTensorType>(operand.getType());
      SmallVector<int64_t, 4> new_op_shape(output_ty.getShape().begin(),
                                           output_ty.getShape().end());
      new_op_shape[new_axis] /= num_operands;

      auto new_op_ty =
          RankedTensorType::get(new_op_shape, op_ty.getElementType());

      SmallVector<int32_t, 4> new_op_shape_v32;
      for (auto d : new_op_shape)
        new_op_shape_v32.push_back(static_cast<int32_t>(d));

      auto new_reshape_const = rewriter.create<ConstOp>(
          reshape_op.getLoc(),
          RankedTensorType::get({static_cast<int64_t>(new_op_shape_v32.size())},
                                rewriter.getI32Type()),
          DenseIntElementsAttr::get(
              RankedTensorType::get(
                  {static_cast<int64_t>(new_op_shape_v32.size())},
                  rewriter.getI32Type()),
              new_op_shape_v32));

      new_inputs.push_back(rewriter.create<ReshapeOp>(
          reshape_op.getLoc(), new_op_ty, operand, new_reshape_const));
    }

    auto new_concat = rewriter.create<ConcatenationOp>(
        concat_op.getLoc(), output_ty, new_inputs, new_axis,
        concat_op.getFusedActivationFunctionAttr());

    rewriter.replaceOp(reshape_op, new_concat.getResult());
    return success();
  }
};

// Reduces BatchMatMul rank by squeezing leading unit dimensions.
// Pattern: BatchMatMul(X, Y) -> Output
// Becomes: Reshape(BatchMatMul(Reshape(X), Reshape(Y))) -> Output
struct ReduceBatchMatMulRank : public OpRewritePattern<BatchMatMulOp> {
  using OpRewritePattern<BatchMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BatchMatMulOp bmm_op,
                                PatternRewriter& rewriter) const override {
    auto lhs_ty = cast<RankedTensorType>(bmm_op.getX().getType());
    auto rhs_ty = cast<RankedTensorType>(bmm_op.getY().getType());
    auto out_ty = cast<RankedTensorType>(bmm_op.getType());

    int rank = out_ty.getRank();
    if (rank <= 3) return failure();

    // Identify unit dimensions we can squeeze.
    int target_rank = 3;
    SmallVector<int, 4> squeezed_indices;
    for (int i = 0; i < rank - 2; ++i) {
      bool can_squeeze = true;
      if (out_ty.getDimSize(i) != 1) can_squeeze = false;

      int lhs_idx = i - (rank - lhs_ty.getRank());
      if (lhs_idx >= 0 && lhs_idx < lhs_ty.getRank() &&
          lhs_ty.getDimSize(lhs_idx) != 1) {
        can_squeeze = false;
      }

      int rhs_idx = i - (rank - rhs_ty.getRank());
      if (rhs_idx >= 0 && rhs_idx < rhs_ty.getRank() &&
          rhs_ty.getDimSize(rhs_idx) != 1) {
        can_squeeze = false;
      }

      if (can_squeeze) {
        squeezed_indices.push_back(i);
        if (rank - squeezed_indices.size() <= target_rank) break;
      }
    }
    if (squeezed_indices.empty()) return failure();

    auto squeeze_input = [&](Value val, RankedTensorType ty) -> Value {
      SmallVector<int, 4> indices;
      for (int i : squeezed_indices) {
        int idx = i - (rank - ty.getRank());
        if (idx >= 0 && idx < ty.getRank()) {
          indices.push_back(idx);
        }
      }
      return SqueezeTensor(val, indices, rewriter, bmm_op.getLoc());
    };

    Value new_lhs = squeeze_input(bmm_op.getX(), lhs_ty);
    Value new_rhs = squeeze_input(bmm_op.getY(), rhs_ty);

    SmallVector<int64_t, 4> new_out_shape;
    for (int i = 0; i < rank; ++i) {
      if (!llvm::is_contained(squeezed_indices, i)) {
        new_out_shape.push_back(out_ty.getDimSize(i));
      }
    }
    auto new_out_ty =
        RankedTensorType::get(new_out_shape, out_ty.getElementType());

    auto new_bmm = rewriter.create<BatchMatMulOp>(
        bmm_op.getLoc(), new_out_ty, new_lhs, new_rhs, bmm_op.getAdjXAttr(),
        bmm_op.getAdjYAttr(), bmm_op.getAsymmetricQuantizeInputsAttr());

    // Reshape back to the original output shape.
    SmallVector<int32_t, 4> out_shape_v32;
    for (auto d : out_ty.getShape())
      out_shape_v32.push_back(static_cast<int32_t>(d));

    auto out_shape_const_ty = RankedTensorType::get(
        {static_cast<int64_t>(out_shape_v32.size())}, rewriter.getI32Type());
    auto out_shape_const = rewriter.create<ConstOp>(
        bmm_op.getLoc(), out_shape_const_ty,
        DenseIntElementsAttr::get(out_shape_const_ty, out_shape_v32));

    rewriter.replaceOpWithNewOp<ReshapeOp>(bmm_op, out_ty, new_bmm,
                                           out_shape_const);
    return success();
  }
};
// Reduces unary operation rank by squeezing leading unit dimensions.
// Pattern: Unary(X) -> Output
// Becomes: Reshape(Unary(Reshape(X))) -> Output
template <typename OpType>
struct ReduceUnaryRank : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter& rewriter) const override {
    auto input_ty = cast<RankedTensorType>(op->getOperand(0).getType());
    auto out_ty = cast<RankedTensorType>(op.getType());

    int rank = out_ty.getRank();
    if (rank < 5) return failure();

    // Identify unit dimensions we can squeeze.
    int target_rank = 4;
    SmallVector<int, 4> squeezed_indices;
    for (int i = 0; i < rank; ++i) {
      if (out_ty.getDimSize(i) == 1 && input_ty.getDimSize(i) == 1) {
        squeezed_indices.push_back(i);
        if (rank - squeezed_indices.size() <= target_rank) break;
      }
    }
    if (squeezed_indices.empty()) return failure();

    SmallVector<int64_t, 4> new_shape;
    for (int i = 0; i < rank; ++i) {
      if (!llvm::is_contained(squeezed_indices, i)) {
        new_shape.push_back(out_ty.getDimSize(i));
      }
    }
    auto new_out_ty = RankedTensorType::get(new_shape, out_ty.getElementType());

    Value new_input = SqueezeTensor(op->getOperand(0), squeezed_indices,
                                    rewriter, op.getLoc());

    auto new_op = rewriter.create<OpType>(op.getLoc(), new_out_ty, new_input);
    new_op->setAttrs(op->getAttrs());

    // Reshape back to the original output shape.
    SmallVector<int32_t, 4> out_shape_v32;
    for (auto d : out_ty.getShape())
      out_shape_v32.push_back(static_cast<int32_t>(d));

    auto final_shape_const_ty = RankedTensorType::get(
        {static_cast<int64_t>(out_shape_v32.size())}, rewriter.getI32Type());
    auto final_shape_const = rewriter.create<ConstOp>(
        op.getLoc(), final_shape_const_ty,
        DenseIntElementsAttr::get(final_shape_const_ty, out_shape_v32));

    rewriter.replaceOpWithNewOp<ReshapeOp>(op, out_ty, new_op.getResult(),
                                           final_shape_const);
    return success();
  }
};

// Reduces binary operation rank by squeezing leading unit dimensions.
// Pattern: Binary(X, Y) -> Output
// Becomes: Reshape(Binary(Reshape(X), Reshape(Y))) -> Output
template <typename OpType>
struct ReduceBinaryRank : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter& rewriter) const override {
    auto lhs_ty = cast<RankedTensorType>(op.getLhs().getType());
    auto rhs_ty = cast<RankedTensorType>(op.getRhs().getType());
    auto out_ty = cast<RankedTensorType>(op.getType());

    int rank = out_ty.getRank();
    if (rank <= 4) return failure();

    // Identify unit dimensions we can squeeze.
    int target_rank = 4;
    SmallVector<int, 4> squeezed_indices;
    for (int i = 0; i < rank; ++i) {
      bool can_squeeze = true;
      if (out_ty.getDimSize(i) != 1) can_squeeze = false;

      int lhs_idx = i - (rank - lhs_ty.getRank());
      if (lhs_idx >= 0 && lhs_idx < lhs_ty.getRank() &&
          lhs_ty.getDimSize(lhs_idx) != 1) {
        can_squeeze = false;
      }

      int rhs_idx = i - (rank - rhs_ty.getRank());
      if (rhs_idx >= 0 && rhs_idx < rhs_ty.getRank() &&
          rhs_ty.getDimSize(rhs_idx) != 1) {
        can_squeeze = false;
      }

      if (can_squeeze) {
        squeezed_indices.push_back(i);
        if (rank - squeezed_indices.size() <= target_rank) break;
      }
    }
    if (squeezed_indices.empty()) return failure();

    auto squeeze_input = [&](Value val, RankedTensorType ty) -> Value {
      SmallVector<int, 4> indices;
      for (int i : squeezed_indices) {
        int idx = i - (rank - ty.getRank());
        if (idx >= 0 && idx < ty.getRank()) {
          indices.push_back(idx);
        }
      }
      return SqueezeTensor(val, indices, rewriter, op.getLoc());
    };

    Value new_lhs = squeeze_input(op.getLhs(), lhs_ty);
    Value new_rhs = squeeze_input(op.getRhs(), rhs_ty);

    SmallVector<int64_t, 4> new_out_shape;
    for (int i = 0; i < rank; ++i) {
      if (!llvm::is_contained(squeezed_indices, i)) {
        new_out_shape.push_back(out_ty.getDimSize(i));
      }
    }
    auto new_out_ty =
        RankedTensorType::get(new_out_shape, out_ty.getElementType());

    auto new_op = [&]() {
      if constexpr (std::is_same_v<OpType, LogicalAndOp> ||
                    std::is_same_v<OpType, LogicalOrOp> ||
                    std::is_same_v<OpType, MaximumOp> ||
                    std::is_same_v<OpType, MinimumOp>) {
        return rewriter.create<OpType>(op.getLoc(), new_out_ty, new_lhs,
                                       new_rhs);
      } else {
        return rewriter.create<OpType>(op.getLoc(), new_out_ty, new_lhs,
                                       new_rhs,
                                       op.getFusedActivationFunctionAttr());
      }
    }();

    // Reshape back to the original output shape.
    SmallVector<int32_t, 4> out_shape_v32;
    for (auto d : out_ty.getShape())
      out_shape_v32.push_back(static_cast<int32_t>(d));

    auto final_shape_const_ty = RankedTensorType::get(
        {static_cast<int64_t>(out_shape_v32.size())}, rewriter.getI32Type());
    auto final_shape_const = rewriter.create<ConstOp>(
        op.getLoc(), final_shape_const_ty,
        DenseIntElementsAttr::get(final_shape_const_ty, out_shape_v32));

    rewriter.replaceOpWithNewOp<ReshapeOp>(op, out_ty, new_op.getResult(),
                                           final_shape_const);
    return success();
  }
};

// Reduces SelectV2 operation rank by squeezing leading unit dimensions.
struct ReduceSelectV2Rank : public OpRewritePattern<SelectV2Op> {
  using OpRewritePattern<SelectV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectV2Op op,
                                PatternRewriter& rewriter) const override {
    auto mask_ty = dyn_cast<RankedTensorType>(op.getCondition().getType());
    auto on_true_ty = dyn_cast<RankedTensorType>(op.getX().getType());
    auto on_false_ty = dyn_cast<RankedTensorType>(op.getY().getType());
    auto out_ty = cast<RankedTensorType>(op.getType());

    int rank = out_ty.getRank();
    if (rank <= 4) return failure();

    // Identify unit dimensions we can squeeze.
    int target_rank = 4;
    SmallVector<int, 4> squeezed_indices;
    for (int i = 0; i < rank; ++i) {
      if (out_ty.getDimSize(i) != 1) continue;

      auto check_ty = [&](RankedTensorType ty) {
        if (!ty) return true;
        int idx = i - (rank - ty.getRank());
        return idx < 0 || idx >= ty.getRank() || ty.getDimSize(idx) == 1;
      };

      if (check_ty(mask_ty) && check_ty(on_true_ty) && check_ty(on_false_ty)) {
        squeezed_indices.push_back(i);
        if (rank - squeezed_indices.size() <= target_rank) break;
      }
    }
    if (squeezed_indices.empty()) return failure();

    auto squeeze_input = [&](Value val, RankedTensorType ty) -> Value {
      if (!ty) return val;
      SmallVector<int, 4> indices;
      for (int i : squeezed_indices) {
        int idx = i - (rank - ty.getRank());
        if (idx >= 0 && idx < ty.getRank()) {
          indices.push_back(idx);
        }
      }
      return SqueezeTensor(val, indices, rewriter, op.getLoc());
    };

    Value new_mask = squeeze_input(op.getCondition(), mask_ty);
    Value new_on_true = squeeze_input(op.getX(), on_true_ty);
    Value new_on_false = squeeze_input(op.getY(), on_false_ty);

    SmallVector<int64_t, 4> new_shape;
    for (int i = 0; i < rank; ++i) {
      if (!llvm::is_contained(squeezed_indices, i)) {
        new_shape.push_back(out_ty.getDimSize(i));
      }
    }
    auto new_out_ty = RankedTensorType::get(new_shape, out_ty.getElementType());

    auto new_op = rewriter.create<SelectV2Op>(op.getLoc(), new_out_ty, new_mask,
                                              new_on_true, new_on_false);

    // Reshape back.
    SmallVector<int32_t, 4> out_shape_v32;
    for (auto d : out_ty.getShape())
      out_shape_v32.push_back(static_cast<int32_t>(d));

    auto final_shape_const_ty = RankedTensorType::get(
        {static_cast<int64_t>(out_shape_v32.size())}, rewriter.getI32Type());
    auto final_shape_const = rewriter.create<ConstOp>(
        op.getLoc(), final_shape_const_ty,
        DenseIntElementsAttr::get(final_shape_const_ty, out_shape_v32));

    rewriter.replaceOpWithNewOp<ReshapeOp>(op, out_ty, new_op.getResult(),
                                           final_shape_const);
    return success();
  }
};

// Reduces pad operation rank by squeezing leading unit dimensions.
// Pattern: Pad(X, paddings) -> Output
// Becomes: Reshape(Pad(Reshape(X), SqueezedPaddings)) -> Output
template <typename PadOpTy>
struct ReducePadRank : public OpRewritePattern<PadOpTy> {
  using OpRewritePattern<PadOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadOpTy pad_op,
                                PatternRewriter& rewriter) const override {
    auto input_ty = cast<RankedTensorType>(pad_op.getInput().getType());
    auto out_ty = cast<RankedTensorType>(pad_op.getType());

    int rank = out_ty.getRank();
    if (rank <= 4) return failure();

    ElementsAttr paddings_attr;
    if (!matchPattern(pad_op.getPadding(), m_Constant(&paddings_attr)))
      return failure();

    // Identify unit dimensions we can squeeze.
    int target_rank = 4;
    SmallVector<int, 4> squeezed_indices;
    for (int i = 0; i < rank; ++i) {
      if (out_ty.getDimSize(i) == 1 && input_ty.getDimSize(i) == 1) {
        squeezed_indices.push_back(i);
        if (rank - squeezed_indices.size() <= target_rank) break;
      }
    }
    if (squeezed_indices.empty()) return failure();

    auto new_paddings_attr =
        SqueezePaddingsAttr(paddings_attr, squeezed_indices);
    if (!new_paddings_attr) return failure();

    auto new_paddings_const = rewriter.create<ConstOp>(
        pad_op.getLoc(), new_paddings_attr.getType(), new_paddings_attr);

    SmallVector<int64_t, 4> new_shape;
    for (int i = 0; i < rank; ++i) {
      if (!llvm::is_contained(squeezed_indices, i)) {
        new_shape.push_back(out_ty.getDimSize(i));
      }
    }
    auto new_out_ty = RankedTensorType::get(new_shape, out_ty.getElementType());

    Value new_input = SqueezeTensor(pad_op.getInput(), squeezed_indices,
                                    rewriter, pad_op.getLoc());

    Value new_pad;
    if constexpr (std::is_same_v<PadOpTy, PadOp>) {
      new_pad = rewriter.create<PadOp>(pad_op.getLoc(), new_out_ty, new_input,
                                       new_paddings_const);
    } else {
      new_pad = rewriter.create<PadV2Op>(pad_op.getLoc(), new_out_ty, new_input,
                                         new_paddings_const,
                                         pad_op.getConstantValues());
    }

    // Reshape back to the original output shape.
    SmallVector<int32_t, 4> out_shape_v32;
    for (auto d : out_ty.getShape())
      out_shape_v32.push_back(static_cast<int32_t>(d));

    auto final_shape_const_ty = RankedTensorType::get(
        {static_cast<int64_t>(out_shape_v32.size())}, rewriter.getI32Type());
    auto final_shape_const = rewriter.create<ConstOp>(
        pad_op.getLoc(), final_shape_const_ty,
        DenseIntElementsAttr::get(final_shape_const_ty, out_shape_v32));

    rewriter.replaceOpWithNewOp<ReshapeOp>(pad_op, out_ty, new_pad,
                                           final_shape_const);
    return success();
  }
};

// Reduces reduction operation rank by squeezing leading unit dimensions.
// Pattern: Reduction(X, axes) -> Output
// Becomes: Reshape(Reduction(Reshape(X), adjusted_axes)) -> Output
template <typename OpType>
struct ReduceReductionRank : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter& rewriter) const override {
    auto input_ty = cast<RankedTensorType>(op.getInput().getType());
    auto out_ty = cast<RankedTensorType>(op.getType());

    int rank = input_ty.getRank();
    if (rank <= 4) return failure();

    // Identify unit dimensions we can squeeze.
    int target_rank = 4;
    SmallVector<int, 4> squeezed_indices;
    for (int i = 0; i < rank; ++i) {
      if (input_ty.getDimSize(i) == 1 && out_ty.getDimSize(i) == 1) {
        squeezed_indices.push_back(i);
        if (rank - squeezed_indices.size() <= target_rank) break;
      }
    }
    if (squeezed_indices.empty()) return failure();

    // Reduction axes must be constant to adjust.
    Value axes_val;
    if constexpr (std::is_same_v<OpType, MeanOp>) {
      axes_val = op.getAxis();
    } else {
      axes_val = op.getAxes();
    }

    DenseIntElementsAttr axes_attr;
    if (!matchPattern(axes_val, m_Constant(&axes_attr))) return failure();

    SmallVector<int32_t, 4> new_axes_vec;
    for (auto axis : axes_attr.getValues<int32_t>()) {
      int32_t a = axis;
      if (a < 0) a += rank;
      // If we reduce a squeezed dimension, it must be keep_dims=true
      // so it remains a unit dimension in the output (which we also squeeze).
      if (llvm::is_contained(squeezed_indices, a)) {
        if (!op.getKeepDims()) return failure();
        continue;
      }
      int32_t new_a = a;
      for (int s : squeezed_indices) {
        if (s < a) new_a--;
      }
      new_axes_vec.push_back(new_a);
    }

    Value new_input =
        SqueezeTensor(op.getInput(), squeezed_indices, rewriter, op.getLoc());

    auto new_axes_const = rewriter.create<ConstOp>(
        op.getLoc(),
        RankedTensorType::get({static_cast<int64_t>(new_axes_vec.size())},
                              rewriter.getI32Type()),
        DenseIntElementsAttr::get(
            RankedTensorType::get({static_cast<int64_t>(new_axes_vec.size())},
                                  rewriter.getI32Type()),
            new_axes_vec));

    SmallVector<int64_t, 4> new_out_shape;
    for (int i = 0; i < rank; ++i) {
      if (!llvm::is_contained(squeezed_indices, i)) {
        new_out_shape.push_back(out_ty.getDimSize(i));
      }
    }
    auto new_out_ty =
        RankedTensorType::get(new_out_shape, out_ty.getElementType());

    auto new_op = rewriter.create<OpType>(op.getLoc(), new_out_ty, new_input,
                                          new_axes_const, op.getKeepDims());

    SmallVector<int32_t, 4> final_shape_v32;
    for (auto d : out_ty.getShape())
      final_shape_v32.push_back(static_cast<int32_t>(d));
    auto final_shape_const_ty = RankedTensorType::get(
        {static_cast<int64_t>(final_shape_v32.size())}, rewriter.getI32Type());
    auto final_shape_const = rewriter.create<ConstOp>(
        op.getLoc(), final_shape_const_ty,
        DenseIntElementsAttr::get(final_shape_const_ty, final_shape_v32));

    rewriter.replaceOpWithNewOp<ReshapeOp>(op, out_ty, new_op.getResult(),
                                           final_shape_const);
    return success();
  }
};

// Reduces cumsum operation rank by squeezing leading unit dimensions.
// Pattern: Cumsum(X, axis) -> Output
// Becomes: Reshape(Cumsum(Reshape(X), adjusted_axis)) -> Output
struct ReduceCumsumRank : public OpRewritePattern<CumsumOp> {
  using OpRewritePattern<CumsumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CumsumOp op,
                                PatternRewriter& rewriter) const override {
    auto input_ty = cast<RankedTensorType>(op.getInput().getType());
    auto out_ty = cast<RankedTensorType>(op.getType());

    int rank = input_ty.getRank();
    if (rank <= 4) return failure();

    // Identify unit dimensions we can squeeze.
    int target_rank = 4;
    SmallVector<int, 4> squeezed_indices;
    for (int i = 0; i < rank; ++i) {
      if (input_ty.getDimSize(i) == 1 && out_ty.getDimSize(i) == 1) {
        squeezed_indices.push_back(i);
        if (rank - squeezed_indices.size() <= target_rank) break;
      }
    }
    if (squeezed_indices.empty()) return failure();

    // Axis must be constant to adjust.
    DenseIntElementsAttr axis_attr;
    if (!matchPattern(op.getAxis(), m_Constant(&axis_attr))) return failure();

    int32_t axis = *axis_attr.getValues<int32_t>().begin();
    if (axis < 0) axis += rank;
    if (llvm::is_contained(squeezed_indices, axis)) return failure();

    int32_t new_axis = axis;
    for (int s : squeezed_indices) {
      if (s < axis) new_axis--;
    }

    Value new_input =
        SqueezeTensor(op.getInput(), squeezed_indices, rewriter, op.getLoc());

    auto new_axis_const = rewriter.create<ConstOp>(
        op.getLoc(), RankedTensorType::get({}, rewriter.getI32Type()),
        DenseIntElementsAttr::get(
            RankedTensorType::get({}, rewriter.getI32Type()),
            {static_cast<int32_t>(new_axis)}));

    SmallVector<int64_t, 4> new_out_shape;
    for (int i = 0; i < rank; ++i) {
      if (!llvm::is_contained(squeezed_indices, i)) {
        new_out_shape.push_back(out_ty.getDimSize(i));
      }
    }
    auto new_out_ty =
        RankedTensorType::get(new_out_shape, out_ty.getElementType());

    auto new_op = rewriter.create<CumsumOp>(
        op.getLoc(), new_out_ty, new_input, new_axis_const.getResult(),
        rewriter.getBoolAttr(op.getExclusive()),
        rewriter.getBoolAttr(op.getReverse()));

    SmallVector<int32_t, 4> final_shape_v32;
    for (auto d : out_ty.getShape())
      final_shape_v32.push_back(static_cast<int32_t>(d));
    auto final_shape_const_ty = RankedTensorType::get(
        {static_cast<int64_t>(final_shape_v32.size())}, rewriter.getI32Type());
    auto final_shape_const = rewriter.create<ConstOp>(
        op.getLoc(), final_shape_const_ty,
        DenseIntElementsAttr::get(final_shape_const_ty, final_shape_v32));

    rewriter.replaceOpWithNewOp<ReshapeOp>(op, out_ty, new_op.getResult(),
                                           final_shape_const);
    return success();
  }
};

struct ReduceSliceRank : public OpRewritePattern<SliceOp> {
  using OpRewritePattern<SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SliceOp op,
                                PatternRewriter& rewriter) const override {
    auto input_ty = cast<RankedTensorType>(op.getInput().getType());
    auto out_ty = cast<RankedTensorType>(op.getType());

    int rank = input_ty.getRank();
    if (rank <= 4) return failure();

    int target_rank = 4;
    // Identify all squeezable unit dimensions.
    SmallVector<int, 4> squeezed_indices;
    for (int i = 0; i < rank; ++i) {
      if (input_ty.getDimSize(i) == 1 && out_ty.getDimSize(i) == 1 &&
          canLiftThroughSlice(op, {i})) {
        squeezed_indices.push_back(i);
        if (rank - squeezed_indices.size() <= target_rank) break;
      }
    }
    if (squeezed_indices.empty()) return failure();

    // Squeeze begin and size constants.
    auto update_const = [&](Value val) -> Value {
      DenseIntElementsAttr attr;
      if (!matchPattern(val, m_Constant(&attr))) return nullptr;

      auto new_attr = SqueezeConstantAttr(attr, squeezed_indices);
      if (!new_attr) return nullptr;

      return rewriter.create<ConstOp>(op.getLoc(), new_attr);
    };

    Value adjusted_begin = update_const(op.getBegin());
    Value adjusted_size = update_const(op.getSize());
    if (!adjusted_begin || !adjusted_size) return failure();

    // Reshape input.
    Value new_input =
        SqueezeTensor(op.getInput(), squeezed_indices, rewriter, op.getLoc());

    // Create new slice.
    SmallVector<int64_t, 4> new_out_shape;
    for (int i = 0; i < rank; ++i) {
      if (!llvm::is_contained(squeezed_indices, i)) {
        new_out_shape.push_back(out_ty.getDimSize(i));
      }
    }
    auto new_out_ty =
        RankedTensorType::get(new_out_shape, out_ty.getElementType());

    auto new_slice = rewriter.create<SliceOp>(
        op.getLoc(), new_out_ty, new_input, adjusted_begin, adjusted_size);

    // Reshape back.
    SmallVector<int32_t, 4> final_shape_v32;
    for (auto d : out_ty.getShape())
      final_shape_v32.push_back(static_cast<int32_t>(d));
    auto final_shape_const_ty = RankedTensorType::get(
        {static_cast<int64_t>(final_shape_v32.size())}, rewriter.getI32Type());
    auto final_shape_const = rewriter.create<ConstOp>(
        op.getLoc(), final_shape_const_ty,
        DenseIntElementsAttr::get(final_shape_const_ty, final_shape_v32));

    rewriter.replaceOpWithNewOp<ReshapeOp>(op, out_ty, new_slice.getResult(),
                                           final_shape_const);
    return success();
  }
};

// Reduces broadcast_to rank by squeezing leading unit dimensions.
// Reduces concatenation operation rank by squeezing leading unit dimensions.
// Pattern: Concatenation(X, Y, ..., axis) -> Output
// Becomes: Reshape(Concatenation(Reshape(X), Reshape(Y), ..., adjusted_axis))
// -> Output
struct ReduceConcatenationRank : public OpRewritePattern<ConcatenationOp> {
  using OpRewritePattern<ConcatenationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatenationOp op,
                                PatternRewriter& rewriter) const override {
    auto out_ty = cast<RankedTensorType>(op.getType());
    int rank = out_ty.getRank();
    if (rank <= 4) return failure();

    int target_rank = 4;
    SmallVector<int, 4> squeezed_indices;
    for (int i = 0; i < rank; ++i) {
      if (out_ty.getDimSize(i) != 1) continue;

      bool can_squeeze = true;
      for (auto val : op.getValues()) {
        auto val_ty = cast<RankedTensorType>(val.getType());
        if (i >= val_ty.getRank() || val_ty.getDimSize(i) != 1) {
          can_squeeze = false;
          break;
        }
      }
      if (can_squeeze) {
        squeezed_indices.push_back(i);
        if (rank - squeezed_indices.size() <= target_rank) break;
      }
    }

    if (squeezed_indices.empty()) return failure();

    int base_axis = op.getAxis();
    if (base_axis < 0) base_axis += rank;
    int new_axis = base_axis;
    for (int s : squeezed_indices) {
      if (s < base_axis) new_axis--;
    }

    SmallVector<Value, 4> new_operands;
    for (auto val : op.getValues()) {
      new_operands.push_back(
          SqueezeTensor(val, squeezed_indices, rewriter, op.getLoc()));
    }

    SmallVector<int64_t, 4> new_out_shape;
    for (int i = 0; i < rank; ++i) {
      if (!llvm::is_contained(squeezed_indices, i)) {
        new_out_shape.push_back(out_ty.getDimSize(i));
      }
    }
    auto new_out_ty =
        RankedTensorType::get(new_out_shape, out_ty.getElementType());

    auto new_concat =
        rewriter.create<ConcatenationOp>(op.getLoc(), new_out_ty, new_operands,
                                         rewriter.getI32IntegerAttr(new_axis),
                                         op.getFusedActivationFunctionAttr());

    SmallVector<int32_t, 4> final_shape_v32;
    for (auto d : out_ty.getShape())
      final_shape_v32.push_back(static_cast<int32_t>(d));
    auto final_shape_const_ty = RankedTensorType::get(
        {static_cast<int64_t>(final_shape_v32.size())}, rewriter.getI32Type());
    auto final_shape_const = rewriter.create<ConstOp>(
        op.getLoc(), final_shape_const_ty,
        DenseIntElementsAttr::get(final_shape_const_ty, final_shape_v32));

    rewriter.replaceOpWithNewOp<ReshapeOp>(op, out_ty, new_concat.getResult(),
                                           final_shape_const);
    return success();
  }
};

struct ReduceBroadcastToRank : public OpRewritePattern<BroadcastToOp> {
  using OpRewritePattern<BroadcastToOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastToOp op,
                                PatternRewriter& rewriter) const override {
    auto input_ty = cast<RankedTensorType>(op.getInput().getType());
    auto out_ty = cast<RankedTensorType>(op.getType());

    int rank = out_ty.getRank();
    if (rank <= 4) return failure();

    // BroadcastTo shape must be constant for this rank reduction.
    DenseIntElementsAttr target_shape_attr;
    if (!matchPattern(op.getShape(), m_Constant(&target_shape_attr)))
      return failure();

    // Identify unit dimensions we can squeeze.
    int target_rank = 4;
    SmallVector<int, 4> squeezed_indices;
    for (int i = 0; i < rank; ++i) {
      if (out_ty.getDimSize(i) != 1) continue;

      int input_idx = i - (rank - input_ty.getRank());
      if (input_idx >= 0 && input_idx < input_ty.getRank() &&
          input_ty.getDimSize(input_idx) != 1) {
        continue;
      }

      squeezed_indices.push_back(i);
      if (rank - squeezed_indices.size() <= target_rank) break;
    }
    if (squeezed_indices.empty()) return failure();

    SmallVector<int, 4> input_indices;
    for (int i : squeezed_indices) {
      int idx = i - (rank - input_ty.getRank());
      if (idx >= 0 && idx < input_ty.getRank()) {
        input_indices.push_back(idx);
      }
    }

    Value new_input =
        SqueezeTensor(op.getInput(), input_indices, rewriter, op.getLoc());

    SmallVector<int64_t, 4> new_target_shape;
    bool is_i64 = target_shape_attr.getElementType().isInteger(64);
    for (int i = 0; i < rank; ++i) {
      if (llvm::is_contained(squeezed_indices, i)) continue;
      if (is_i64) {
        new_target_shape.push_back(
            target_shape_attr.getValues<int64_t>()[static_cast<uint64_t>(i)]);
      } else {
        new_target_shape.push_back(
            target_shape_attr.getValues<int32_t>()[static_cast<uint64_t>(i)]);
      }
    }

    Type shape_element_ty = target_shape_attr.getElementType();
    auto new_target_shape_ty = RankedTensorType::get(
        {static_cast<int64_t>(new_target_shape.size())}, shape_element_ty);

    ElementsAttr new_target_shape_attr;
    if (is_i64) {
      new_target_shape_attr = DenseIntElementsAttr::get(
          new_target_shape_ty,
          llvm::ArrayRef<int64_t>(new_target_shape.data(),
                                  new_target_shape.size()));
    } else {
      SmallVector<int32_t, 4> new_target_shape_v32;
      for (auto d : new_target_shape)
        new_target_shape_v32.push_back(static_cast<int32_t>(d));
      new_target_shape_attr = DenseIntElementsAttr::get(
          new_target_shape_ty,
          llvm::ArrayRef<int32_t>(new_target_shape_v32.data(),
                                  new_target_shape_v32.size()));
    }

    auto new_target_shape_const = rewriter.create<ConstOp>(
        op.getLoc(), new_target_shape_ty, new_target_shape_attr);

    SmallVector<int64_t, 4> new_out_shape;
    for (int i = 0; i < rank; ++i) {
      if (!llvm::is_contained(squeezed_indices, i)) {
        new_out_shape.push_back(out_ty.getDimSize(i));
      }
    }
    auto new_out_ty =
        RankedTensorType::get(new_out_shape, out_ty.getElementType());

    auto new_op = rewriter.create<BroadcastToOp>(
        op.getLoc(), new_out_ty, new_input, new_target_shape_const);

    // Reshape back to the original output shape.
    SmallVector<int32_t, 4> out_shape_v32;
    for (auto d : out_ty.getShape())
      out_shape_v32.push_back(static_cast<int32_t>(d));

    auto out_shape_const_ty = RankedTensorType::get(
        {static_cast<int64_t>(out_shape_v32.size())}, rewriter.getI32Type());
    auto out_shape_const = rewriter.create<ConstOp>(
        op.getLoc(), out_shape_const_ty,
        DenseIntElementsAttr::get(out_shape_const_ty, out_shape_v32));

    rewriter.replaceOpWithNewOp<ReshapeOp>(op, out_ty, new_op, out_shape_const);
    return success();
  }
};

// Fuses consecutive Reshape ops.
// Pattern: Reshape(Reshape(X, shape1), shape2) -> Reshape(X, shape2)
struct FuseConsecutiveReshapes : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp reshape_op,
                                PatternRewriter& rewriter) const override {
    auto src_reshape = reshape_op.getInput().getDefiningOp<ReshapeOp>();
    if (!src_reshape) return failure();

    rewriter.replaceOpWithNewOp<ReshapeOp>(reshape_op, reshape_op.getType(),
                                           src_reshape.getInput(),
                                           reshape_op.getShape());
    return success();
  }
};

// Patterns to remove unit dimensions from Transpose.
// Pattern: Transpose(X, perm) -> Reshape(Transpose(Reshape(X, squeezed),
// new_perm), original_output_shape)
struct ReduceTransposeRank : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern<TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp transpose_op,
                                PatternRewriter& rewriter) const override {
    auto input_ty = cast<RankedTensorType>(transpose_op.getInput().getType());
    if (input_ty.getRank() < 5) return failure();

    SmallVector<int, 4> unit_dims;
    for (int i = 0; i < input_ty.getRank(); ++i) {
      if (input_ty.getDimSize(i) == 1) unit_dims.push_back(i);
    }
    if (unit_dims.empty()) return failure();

    DenseIntElementsAttr perm_attr;
    if (!matchPattern(transpose_op.getPerm(), m_Constant(&perm_attr)))
      return failure();

    SmallVector<int64_t, 4> perm;
    for (auto val : perm_attr.template getValues<int32_t>())
      perm.push_back(val);

    SmallVector<int, 4> input_idx_remap(input_ty.getRank(), -1);
    SmallVector<int64_t, 4> squeezed_shape;
    int next_idx = 0;
    for (int i = 0; i < input_ty.getRank(); ++i) {
      if (input_ty.getDimSize(i) != 1) {
        input_idx_remap[i] = next_idx++;
        squeezed_shape.push_back(input_ty.getDimSize(i));
      }
    }

    // New permutation on non-unit dimensions.
    SmallVector<int32_t, 4> new_perm_v;
    for (int p : perm) {
      if (input_idx_remap[p] != -1) {
        new_perm_v.push_back(input_idx_remap[p]);
      }
    }

    // If all dimensions were unit dimensions, the new transpose would be rank
    // 0. Let's at least keep one dimension if it's rank 1.
    if (new_perm_v.empty()) return failure();

    auto squeezed_ty =
        RankedTensorType::get(squeezed_shape, input_ty.getElementType());
    auto squeezed_shape_const = rewriter.create<ConstOp>(
        transpose_op.getLoc(),
        RankedTensorType::get({static_cast<int64_t>(squeezed_shape.size())},
                              rewriter.getI32Type()),
        rewriter.getI32TensorAttr(SmallVector<int32_t, 4>(
            squeezed_shape.begin(), squeezed_shape.end())));

    auto squeezed_input = rewriter.create<ReshapeOp>(
        transpose_op.getLoc(), squeezed_ty, transpose_op.getInput(),
        squeezed_shape_const);

    auto new_perm_ty = RankedTensorType::get(
        {static_cast<int64_t>(new_perm_v.size())}, rewriter.getI32Type());
    auto new_perm_const =
        rewriter.create<ConstOp>(transpose_op.getLoc(), new_perm_ty,
                                 rewriter.getI32TensorAttr(new_perm_v));

    SmallVector<int64_t, 4> new_transpose_shape;
    for (int p : new_perm_v) new_transpose_shape.push_back(squeezed_shape[p]);
    auto new_transpose_ty =
        RankedTensorType::get(new_transpose_shape, input_ty.getElementType());

    auto new_transpose =
        rewriter.create<TransposeOp>(transpose_op.getLoc(), new_transpose_ty,
                                     squeezed_input, new_perm_const);

    auto output_ty = cast<RankedTensorType>(transpose_op.getType());
    auto final_shape_const = rewriter.create<ConstOp>(
        transpose_op.getLoc(),
        RankedTensorType::get({static_cast<int64_t>(output_ty.getRank())},
                              rewriter.getI32Type()),
        rewriter.getI32TensorAttr(SmallVector<int32_t, 4>(
            output_ty.getShape().begin(), output_ty.getShape().end())));

    rewriter.replaceOpWithNewOp<ReshapeOp>(transpose_op, transpose_op.getType(),
                                           new_transpose, final_shape_const);
    return success();
  }
};

class RankReductionPass
    : public impl::RankReductionPassBase<RankReductionPass> {
 public:
  RankReductionPass() = default;

  void runOnOperation() override {
    func::FuncOp func_op = getOperation();
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LiftReshapeThroughConcatenation>(ctx);
    patterns.add<LiftReshapeThroughUnaryElementwise<CastOp>>(ctx);
    patterns.add<SinkReshapeThroughUnaryElementwise<CastOp>>(ctx);
    // patterns.add<LiftReshapeThroughBroadcastTo>(ctx);
    patterns.add<ReduceTransposeRank>(ctx);
    patterns.add<ReduceBatchMatMulRank>(ctx);
    patterns.add<ReduceBroadcastToRank>(ctx);
    patterns.add<ReduceReductionRank<MeanOp>>(ctx);
    patterns.add<ReduceReductionRank<SumOp>>(ctx);
    patterns.add<ReduceCumsumRank>(ctx);
    patterns.add<ReduceSliceRank>(ctx);
    patterns.add<ReduceConcatenationRank>(ctx);
    patterns.add<ReduceUnaryRank<AbsOp>>(ctx);
    patterns.add<ReduceUnaryRank<CeilOp>>(ctx);
    patterns.add<ReduceUnaryRank<CosOp>>(ctx);
    patterns.add<ReduceUnaryRank<ExpOp>>(ctx);
    patterns.add<ReduceUnaryRank<FloorOp>>(ctx);
    patterns.add<ReduceUnaryRank<LogOp>>(ctx);
    patterns.add<ReduceUnaryRank<LogicalNotOp>>(ctx);
    patterns.add<ReduceUnaryRank<NegOp>>(ctx);
    patterns.add<ReduceUnaryRank<RsqrtOp>>(ctx);
    patterns.add<ReduceUnaryRank<SinOp>>(ctx);
    patterns.add<ReduceUnaryRank<SqrtOp>>(ctx);
    patterns.add<ReduceUnaryRank<SquareOp>>(ctx);
    patterns.add<ReduceUnaryRank<TanhOp>>(ctx);
    patterns.add<ReduceUnaryRank<CastOp>>(ctx);
    patterns.add<ReduceUnaryRank<DequantizeOp>>(ctx);
    patterns.add<ReduceBinaryRank<MulOp>>(ctx);
    patterns.add<ReduceBinaryRank<AddOp>>(ctx);
    patterns.add<ReduceBinaryRank<LogicalAndOp>>(ctx);
    patterns.add<ReduceBinaryRank<DivOp>>(ctx);
    patterns.add<ReduceBinaryRank<LogicalOrOp>>(ctx);
    patterns.add<ReduceBinaryRank<SubOp>>(ctx);
    patterns.add<ReduceBinaryRank<MaximumOp>>(ctx);
    patterns.add<ReduceBinaryRank<MinimumOp>>(ctx);
    patterns.add<ReduceUnaryRank<SoftmaxOp>>(ctx);
    patterns.add<ReduceSelectV2Rank>(ctx);
    patterns.add<ReducePadRank<PadOp>>(ctx);
    patterns.add<ReducePadRank<PadV2Op>>(ctx);
    patterns.add<FuseConsecutiveReshapes>(ctx);
    if (failed(applyPatternsGreedily(func_op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateRankReductionPass() {
  return std::make_unique<RankReductionPass>();
}

}  // namespace TFL
}  // namespace mlir
