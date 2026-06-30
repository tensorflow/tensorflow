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

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
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

#define GEN_PASS_DEF_GPUCOMPATIBILITYPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

struct PromoteStridedSliceBool : public OpRewritePattern<StridedSliceOp> {
  using OpRewritePattern<StridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StridedSliceOp op,
                                PatternRewriter& rewriter) const override {
    auto input_type = llvm::dyn_cast<ShapedType>(op.getInput().getType());
    if (!input_type || !input_type.getElementType().isInteger(1)) {
      return failure();
    }

    Location loc = op.getLoc();

    // 1. Cast i1 to f32
    auto f32_type = input_type.clone(rewriter.getF32Type());
    auto cast_in = rewriter.create<CastOp>(loc, f32_type, op.getInput());

    // 2. New StridedSlice with f32
    auto output_type =
        llvm::cast<ShapedType>(op.getType()).clone(rewriter.getF32Type());
    auto new_op = rewriter.create<StridedSliceOp>(
        loc, output_type, cast_in, op.getBegin(), op.getEnd(), op.getStrides(),
        op.getBeginMask(), op.getEndMask(), op.getEllipsisMask(),
        op.getNewAxisMask(), op.getShrinkAxisMask(), op.getOffset());

    // 3. Cast results back to i1
    rewriter.replaceOpWithNewOp<CastOp>(op, op.getType(), new_op.getResult());
    return success();
  }
};

struct PromoteTileBool : public OpRewritePattern<TileOp> {
  using OpRewritePattern<TileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TileOp op,
                                PatternRewriter& rewriter) const override {
    auto input_type = llvm::dyn_cast<ShapedType>(op.getInput().getType());
    if (!input_type || !input_type.getElementType().isInteger(1)) {
      return failure();
    }

    Location loc = op.getLoc();

    // 1. Cast i1 to i32
    auto i32_type = input_type.clone(rewriter.getI32Type());
    auto cast_in = rewriter.create<CastOp>(loc, i32_type, op.getInput());

    // 2. New Tile with i32
    auto output_type =
        llvm::cast<ShapedType>(op.getType()).clone(rewriter.getI32Type());
    auto new_op =
        rewriter.create<TileOp>(loc, output_type, cast_in, op.getMultiples());

    // 3. Cast results back to i1
    rewriter.replaceOpWithNewOp<CastOp>(op, op.getType(), new_op.getResult());
    return success();
  }
};

struct SwapAddOperands : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter& rewriter) const override {
    auto lhs_type = llvm::dyn_cast<ShapedType>(op.getLhs().getType());
    auto rhs_type = llvm::dyn_cast<ShapedType>(op.getRhs().getType());

    if (!lhs_type || !rhs_type) return failure();

    // Swap if LHS is rank-0 and RHS is higher rank.
    if (lhs_type.getRank() == 0 && rhs_type.getRank() > 0) {
      rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), op.getRhs(),
                                         op.getLhs(),
                                         op.getFusedActivationFunctionAttr());
      return success();
    }
    return failure();
  }
};

template <typename OpType>
struct ExpandScalarOperand : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter& rewriter) const override {
    auto rhs = op.getRhs();
    auto rhs_type = llvm::dyn_cast<ShapedType>(rhs.getType());
    if (!rhs_type || rhs_type.getRank() != 0) return failure();

    if (matchPattern(rhs, m_Constant())) return failure();

    Location loc = op.getLoc();
    auto shape_type = RankedTensorType::get({1}, rewriter.getI32Type());
    auto shape_const = rewriter.create<ConstOp>(
        loc, DenseIntElementsAttr::get(shape_type, {1}));

    auto rank1_type = RankedTensorType::get({1}, rhs_type.getElementType());
    auto reshape =
        rewriter.create<ReshapeOp>(loc, rank1_type, rhs, shape_const);

    rewriter.replaceOpWithNewOp<OpType>(op, op.getType(), op.getLhs(), reshape,
                                        op.getFusedActivationFunctionAttr());
    return success();
  }
};

struct PromoteAndExpandPad : public OpRewritePattern<PadOp> {
  using OpRewritePattern<PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadOp op,
                                PatternRewriter& rewriter) const override {
    auto input_type = llvm::dyn_cast<ShapedType>(op.getInput().getType());
    if (!input_type || input_type.getRank() != 2) return failure();

    DenseIntElementsAttr paddings_attr;
    if (!matchPattern(op.getPadding(), m_Constant(&paddings_attr))) {
      return failure();
    }

    Location loc = op.getLoc();
    bool is_bool = input_type.getElementType().isInteger(1);

    // 1. Reshape 2D -> 3D [1, M, N]
    SmallVector<int64_t, 3> shape_3d = {1, input_type.getShape()[0],
                                        input_type.getShape()[1]};
    auto type_3d = input_type.clone(shape_3d);
    auto shape_const_in = rewriter.create<ConstOp>(
        loc, DenseIntElementsAttr::get(
                 RankedTensorType::get({3}, rewriter.getI32Type()),
                 {(int32_t)1, (int32_t)input_type.getShape()[0],
                  (int32_t)input_type.getShape()[1]}));
    auto reshape_in =
        rewriter.create<ReshapeOp>(loc, type_3d, op.getInput(), shape_const_in);

    // 2. Cast i1 -> f32 if needed
    Value current_input = reshape_in.getResult();
    if (is_bool) {
      auto f32_type = type_3d.clone(rewriter.getF32Type());
      current_input = rewriter.create<CastOp>(loc, f32_type, current_input);
    }

    // 3. Expand paddings from 2x2 to 3x2
    SmallVector<int32_t, 6> new_paddings_data = {0, 0};
    for (auto val : paddings_attr.getValues<int32_t>()) {
      new_paddings_data.push_back(val);
    }
    auto new_paddings_type =
        RankedTensorType::get({3, 2}, rewriter.getI32Type());
    auto new_paddings_const = rewriter.create<ConstOp>(
        loc, DenseIntElementsAttr::get(new_paddings_type, new_paddings_data));

    // 4. New PadOp
    auto output_type_2d = llvm::cast<ShapedType>(op.getType());
    SmallVector<int64_t, 3> output_shape_3d = {1, output_type_2d.getShape()[0],
                                               output_type_2d.getShape()[1]};
    auto base_output_type_3d = output_type_2d.clone(output_shape_3d);
    auto final_pad_output_type_3d =
        is_bool ? base_output_type_3d.clone(rewriter.getF32Type())
                : base_output_type_3d;

    auto new_pad = rewriter.create<PadOp>(loc, final_pad_output_type_3d,
                                          current_input, new_paddings_const);

    // 5. Cast back to i1 if needed
    Value current_output = new_pad.getResult();
    if (is_bool) {
      current_output =
          rewriter.create<CastOp>(loc, base_output_type_3d, current_output);
    }

    // 6. Reshape 3D -> 2D
    auto shape_const_out = rewriter.create<ConstOp>(
        loc, DenseIntElementsAttr::get(
                 RankedTensorType::get({2}, rewriter.getI32Type()),
                 {(int32_t)output_type_2d.getShape()[0],
                  (int32_t)output_type_2d.getShape()[1]}));
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), current_output,
                                           shape_const_out);
    return success();
  }
};

struct SwapBatchMatMul : public OpRewritePattern<BatchMatMulOp> {
  using OpRewritePattern<BatchMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BatchMatMulOp op,
                                PatternRewriter& rewriter) const override {
    // We want to swap if LHS is constant and RHS is not.
    if (!matchPattern(op.getX(), m_Constant()) ||
        matchPattern(op.getY(), m_Constant())) {
      return failure();
    }

    Location loc = op.getLoc();

    // new_adj_x = !old_adj_y
    // new_adj_y = !old_adj_x
    bool new_adj_x = !op.getAdjY();
    bool new_adj_y = !op.getAdjX();

    auto final_type = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!final_type) return failure();
    int rank = final_type.getRank();
    if (rank < 2) return failure();

    // Swapped output shape: [..., N, M] if original is [..., M, N]
    SmallVector<int64_t> bmm_shape(final_type.getShape());
    std::swap(bmm_shape[rank - 1], bmm_shape[rank - 2]);
    auto bmm_type =
        RankedTensorType::get(bmm_shape, final_type.getElementType());

    auto new_bmm = rewriter.create<BatchMatMulOp>(
        loc, bmm_type, op.getY(), op.getX(), new_adj_x, new_adj_y,
        op.getAsymmetricQuantizeInputsAttr());

    // Transpose back.
    SmallVector<int32_t> perm_data;
    for (int i = 0; i < rank - 2; ++i) perm_data.push_back(i);
    perm_data.push_back(rank - 1);
    perm_data.push_back(rank - 2);

    auto perm_type = RankedTensorType::get({rank}, rewriter.getI32Type());
    auto perm_attr = DenseIntElementsAttr::get(perm_type, perm_data);
    auto perm_const = rewriter.create<ConstOp>(loc, perm_attr);

    rewriter.replaceOpWithNewOp<TransposeOp>(op, op.getType(),
                                             new_bmm.getResult(), perm_const);
    return success();
  }
};

struct ConvertBroadcastToToTile : public OpRewritePattern<BroadcastToOp> {
  using OpRewritePattern<BroadcastToOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastToOp op,
                                PatternRewriter& rewriter) const override {
    auto input_type = llvm::dyn_cast<ShapedType>(op.getInput().getType());
    auto output_type = llvm::dyn_cast<ShapedType>(op.getType());

    if (!input_type || !output_type || !input_type.hasStaticShape() ||
        !output_type.hasStaticShape()) {
      return failure();
    }

    auto input_shape = input_type.getShape();
    auto output_shape = output_type.getShape();

    if (input_shape.size() != output_shape.size()) {
      return failure();
    }

    SmallVector<int64_t> multiples;
    for (int i = 0; i < input_shape.size(); ++i) {
      if (input_shape[i] == 0) return failure();
      multiples.push_back(output_shape[i] / input_shape[i]);
    }

    Location loc = op.getLoc();
    auto multiples_type = RankedTensorType::get({(int64_t)multiples.size()},
                                                rewriter.getI64Type());
    auto multiples_attr = DenseIntElementsAttr::get(multiples_type, multiples);
    auto multiples_const = rewriter.create<ConstOp>(loc, multiples_attr);

    rewriter.replaceOpWithNewOp<TileOp>(op, output_type, op.getInput(),
                                        multiples_const);
    return success();
  }
};

struct ConvertBroadcastToSumToMul : public OpRewritePattern<SumOp> {
  using OpRewritePattern<SumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SumOp op,
                                PatternRewriter& rewriter) const override {
    if (!op.getKeepDims()) return failure();

    auto broadcast_op = op.getInput().getDefiningOp<BroadcastToOp>();
    if (!broadcast_op) return failure();

    auto input_type =
        llvm::dyn_cast<ShapedType>(broadcast_op.getInput().getType());
    auto broadcast_type = llvm::dyn_cast<ShapedType>(broadcast_op.getType());
    auto output_type = llvm::dyn_cast<ShapedType>(op.getType());

    if (!input_type || !broadcast_type || !output_type ||
        !input_type.hasStaticShape() || !broadcast_type.hasStaticShape() ||
        !output_type.hasStaticShape()) {
      return failure();
    }

    DenseIntElementsAttr axes_attr;
    if (!matchPattern(op.getAxes(), m_Constant(&axes_attr))) {
      return failure();
    }

    auto input_shape = input_type.getShape();
    auto broadcast_shape = broadcast_type.getShape();

    int64_t scale = 1;
    llvm::SmallDenseSet<int64_t> axes_set;
    for (auto axis : axes_attr.getValues<APInt>()) {
      int64_t val = axis.getSExtValue();
      int64_t norm_axis = val < 0 ? val + broadcast_shape.size() : val;
      axes_set.insert(norm_axis);
    }

    for (int i = 0; i < broadcast_shape.size(); ++i) {
      if (axes_set.contains(i)) {
        if (input_shape[i] != 1) return failure();
        scale *= broadcast_shape[i];
      } else {
        if (input_shape[i] != broadcast_shape[i]) return failure();
      }
    }

    Location loc = op.getLoc();
    auto elem_type = input_type.getElementType();
    TypedAttr attr;
    if (elem_type.isF32()) {
      attr = DenseElementsAttr::get(RankedTensorType::get({}, elem_type),
                                    {(float)scale});
    } else if (elem_type.isInteger(32)) {
      attr = DenseElementsAttr::get(RankedTensorType::get({}, elem_type),
                                    {(int32_t)scale});
    } else if (elem_type.isInteger(64)) {
      attr = DenseElementsAttr::get(RankedTensorType::get({}, elem_type),
                                    {(int64_t)scale});
    } else {
      return failure();
    }

    auto scale_const = rewriter.create<ConstOp>(loc, attr);
    rewriter.replaceOpWithNewOp<MulOp>(op, output_type, broadcast_op.getInput(),
                                       scale_const,
                                       rewriter.getStringAttr("NONE"));
    return success();
  }
};

struct ConstantFoldTile : public OpRewritePattern<TileOp> {
  using OpRewritePattern<TileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TileOp op,
                                PatternRewriter& rewriter) const override {
    DenseElementsAttr input_attr;
    if (!matchPattern(op.getInput(), m_Constant(&input_attr))) return failure();

    DenseIntElementsAttr multiples_attr;
    if (!matchPattern(op.getMultiples(), m_Constant(&multiples_attr))) {
      return failure();
    }

    auto output_type = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!output_type || !output_type.hasStaticShape()) return failure();

    if (input_attr.isSplat()) {
      rewriter.replaceOpWithNewOp<ConstOp>(
          op, DenseElementsAttr::get(output_type,
                                     input_attr.getSplatValue<Attribute>()));
      return success();
    }

    // Handle non-splat tiling.
    auto input_shape = input_attr.getType().getShape();
    auto output_shape = output_type.getShape();
    int rank = output_shape.size();

    SmallVector<Attribute> output_values;
    output_values.reserve(output_type.getNumElements());
    auto input_values = input_attr.getValues<Attribute>();

    for (int64_t i = 0; i < output_type.getNumElements(); ++i) {
      int64_t linear_idx = i;
      SmallVector<int64_t> output_coord(rank);
      for (int r = rank - 1; r >= 0; --r) {
        output_coord[r] = linear_idx % output_shape[r];
        linear_idx /= output_shape[r];
      }

      int64_t input_linear_idx = 0;
      int64_t multiplier = 1;
      for (int r = rank - 1; r >= 0; --r) {
        input_linear_idx += (output_coord[r] % input_shape[r]) * multiplier;
        multiplier *= input_shape[r];
      }
      output_values.push_back(input_values[input_linear_idx]);
    }

    rewriter.replaceOpWithNewOp<ConstOp>(
        op, DenseElementsAttr::get(output_type, output_values));
    return success();
  }
};

struct FuseDequantizeFullyConnected
    : public OpRewritePattern<FullyConnectedOp> {
  using OpRewritePattern<FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FullyConnectedOp op,
                                PatternRewriter& rewriter) const override {
    auto dequant_op = op.getFilter().getDefiningOp<DequantizeOp>();
    if (!dequant_op) return failure();

    rewriter.modifyOpInPlace(
        op, [&]() { op.getFilterMutable().assign(dequant_op.getInput()); });
    return success();
  }
};

class GpuCompatibilityPass
    : public impl::GpuCompatibilityPassBase<GpuCompatibilityPass> {
 public:
  GpuCompatibilityPass() = default;

  void runOnOperation() override {
    auto func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<PromoteStridedSliceBool, PromoteTileBool, PromoteAndExpandPad,
                 SwapBatchMatMul, ConvertBroadcastToToTile,
                 ConvertBroadcastToSumToMul, SwapAddOperands,
                 ExpandScalarOperand<AddOp>, FuseDequantizeFullyConnected,
                 ConstantFoldTile>(&getContext());
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateGpuCompatibilityPass() {
  return std::make_unique<GpuCompatibilityPass>();
}

}  // namespace TFL
}  // namespace mlir
