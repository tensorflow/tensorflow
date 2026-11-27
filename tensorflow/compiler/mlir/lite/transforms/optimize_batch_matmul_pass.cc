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

#include "tensorflow/compiler/mlir/lite/transforms/optimize_batch_matmul_pass.h"

#include <algorithm>
#include <cstdint>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/tflite_passes/optimize_batch_matmul_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"

namespace mlir {
namespace TFL {
namespace {

// Returns `op` as a bf16/f16 -> f32 cast, or null if it is anything else.
//
// Weights kept in a narrow float type reach the matmul behind such a cast,
// since the kernels only run in f32. Flatbuffer export materializes this cast
// when its input is a constant, so the cast does not stop the value from being
// a constant filter.
CastOp AsFloatWideningCast(mlir::Operation* op) {
  auto cast_op = mlir::dyn_cast_or_null<CastOp>(op);
  if (!cast_op) return nullptr;
  mlir::Type in_elem = getElementTypeOrSelf(cast_op.getInput().getType());
  mlir::Type out_elem = getElementTypeOrSelf(cast_op.getType());
  if (!(in_elem.isBF16() || in_elem.isF16()) || !out_elem.isF32()) {
    return nullptr;
  }
  return cast_op;
}

// Checks whether `value` is, or can be folded into, a constant. This covers a
// plain constant, the dequantize wrappers a quantized constant weight hides
// behind, and chains of ops that only rearrange or widen one of those.
bool IsFromFoldableChain(mlir::Value value) {
  mlir::Operation* defining_op = value.getDefiningOp();

  while (defining_op) {
    // `QConstOp` carries its value in an attribute but is neither ConstantLike
    // nor foldable, so it has to be named rather than left to `m_Constant`.
    if (mlir::isa<DequantizeOp, BlockwiseDequantizeOp, QConstOp>(defining_op) ||
        defining_op->hasTrait<mlir::OpTrait::ConstantLike>() ||
        matchPattern(defining_op, mlir::m_Constant())) {
      return true;
    }

    // Look through ops that don't change the constant nature.
    if (auto reshape_op = mlir::dyn_cast<ReshapeOp>(defining_op)) {
      defining_op = reshape_op.getInput().getDefiningOp();
    } else if (auto split_op = mlir::dyn_cast<SplitOp>(defining_op)) {
      defining_op = split_op.getValue().getDefiningOp();
    } else if (auto transpose_op = mlir::dyn_cast<TransposeOp>(defining_op)) {
      defining_op = transpose_op.getInput().getDefiningOp();
    } else if (auto cast_op = AsFloatWideningCast(defining_op)) {
      defining_op = cast_op.getInput().getDefiningOp();
    } else {
      // Stop if the op does not preserve constant nature.
      break;
    }
  }
  return false;
}

// Checks whether a batch_matmul whose rhs is `value` can be rewritten into a
// fully_connected, i.e. whether `value` can become that op's filter.
//
// This is the single gate shared by both directions of the
// batch_matmul/fully_connected rewrite, and the two have to agree on it:
// `ConvertBatchMatMulOp2FullyConnectedOp_Rank2ConstantRhs` converts a matmul
// over such a value into a fully_connected, while the
// `NotFoldableIntoFullyConnected` constraint in `optimize_batch_matmul.td`
// stops `FuseTransposeFCRhsToBatchMatmul` converting it straight back.
bool IsRhsFoldableIntoFullyConnected(mlir::Value value) {
  auto ranked_type = mlir::dyn_cast<RankedTensorType>(value.getType());
  if (!ranked_type || ranked_type.getRank() != 2) {
    return false;
  }
  if (mlir::isa<BlockwiseQuantType>(ranked_type.getElementType())) {
    return false;
  }

  if (IsFromFoldableChain(value)) {
    return true;
  }

  // A signed int4/int8 quantized rhs becomes the filter as-is, so unlike the
  // case above it does not have to be constant.
  auto quantized_type =
      mlir::dyn_cast<quant::QuantizedType>(ranked_type.getElementType());
  return quantized_type != nullptr && quantized_type.isSigned() &&
         (quantized_type.getStorageTypeIntegralWidth() == 4 ||
          quantized_type.getStorageTypeIntegralWidth() == 8);
}

// Converts batch_matmul operation to fully_connected if rhs is a
// constant tensor with rank 2
struct ConvertBatchMatMulOp2FullyConnectedOp_Rank2ConstantRhs
    : public OpRewritePattern<TFL::BatchMatMulOp> {
  using OpRewritePattern<TFL::BatchMatMulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TFL::BatchMatMulOp bmm_op,
                                PatternRewriter& rewriter) const override {
    if (!IsRhsFoldableIntoFullyConnected(bmm_op.getY())) {
      return rewriter.notifyMatchFailure(
          bmm_op,
          "rhs is neither a rank 2 value foldable into a constant nor a rank 2 "
          "signed int4/int8 quantized value.");
    }
    // The filter transpose of a shared blockwise-dequantized rhs cannot be
    // pushed into its constant operands, which would leave a float filter;
    // such a static-range quantized matmul stays a blockwise quantized
    // batch_matmul instead.
    if (mlir::isa_and_nonnull<DequantizeOp>(bmm_op.getX().getDefiningOp()) &&
        mlir::isa_and_nonnull<BlockwiseDequantizeOp>(
            bmm_op.getY().getDefiningOp()) &&
        !bmm_op.getY().hasOneUse()) {
      return rewriter.notifyMatchFailure(
          bmm_op,
          "SRQ batch_matmul with a shared blockwise-dequantized rhs stays "
          "batch_matmul");
    }

    // Create a tfl.transpose op that performs ZX transpose on `input`.
    auto create_z_x_transpose_op = [&](Value input) -> Value {
      RankedTensorType input_type =
          mlir::cast<RankedTensorType>(input.getType());
      const int input_rank = input_type.getRank();

      // Create a 1D I32 tensor for representing the dimension permutation.
      auto permuation_tensor_type =
          RankedTensorType::get({input_rank}, rewriter.getIntegerType(32));
      llvm::SmallVector<Attribute, 4> permute;
      permute.reserve(input_rank);
      // First create an identity permutation tensor.
      for (int i = 0; i < input_rank; i++) {
        permute.push_back(rewriter.getI32IntegerAttr(i));
      }
      // Swaps the last two dimension since the last two dimension will be
      // mapped to X and Z dimension.
      std::iter_swap(permute.begin() + input_rank - 1,
                     permute.begin() + input_rank - 2);
      auto permutation_tensor_op = arith::ConstantOp::create(
          rewriter, bmm_op->getLoc(), permuation_tensor_type,
          DenseElementsAttr::get(permuation_tensor_type, permute));

      auto input_shape = input_type.getShape();
      llvm::SmallVector<int64_t, 4> permuted_shape(input_shape.begin(),
                                                   input_shape.end());
      // Swaps z dimension and x dimension to get permuted shape.
      std::iter_swap(permuted_shape.begin() + input_rank - 1,
                     permuted_shape.begin() + input_rank - 2);

      // If the input is a per-axis quantized type, we should make sure the
      // quantization dimension is correctly set/propagated to the transpose op.
      RankedTensorType output_type;
      if (auto input_qtype = mlir::dyn_cast<quant::UniformQuantizedPerAxisType>(
              getElementTypeOrSelf(input.getType()))) {
        // If the input is a per-axis quantized type, we should make sure the
        // quantization dimension is correctly set/propagated to the transpose
        // op.
        int new_quant_dim = -1;
        // We should be checking if getQuantizedDimension either (rank - 1)-th
        // or (rank - 2)-th dimension because we transpose only the last two
        // dimensions here. If the previous dimension is (rank - 1), we should
        // swap the quantization dimension to (rank - 2) and vice versa.
        // If the quantization dimension is anything else, we should not change
        // it and create the TransposeOp with the original qtype.
        if (input_qtype.getQuantizedDimension() == (input_rank - 1)) {
          new_quant_dim = input_rank - 2;
        } else if (input_qtype.getQuantizedDimension() == (input_rank - 2)) {
          new_quant_dim = input_rank - 1;
        }

        if (new_quant_dim != -1) {
          input_qtype = quant::UniformQuantizedPerAxisType::get(
              input_qtype.getFlags(), input_qtype.getStorageType(),
              input_qtype.getExpressedType(), input_qtype.getScales(),
              input_qtype.getZeroPoints(), new_quant_dim,
              input_qtype.getStorageTypeMin(), input_qtype.getStorageTypeMax());
        }
        output_type = RankedTensorType::getChecked(bmm_op->getLoc(),
                                                   permuted_shape, input_qtype);
      } else {
        output_type =
            RankedTensorType::get(permuted_shape, input_type.getElementType());
      }

      return TFL::TransposeOp::create(rewriter, bmm_op->getLoc(), output_type,
                                      input, permutation_tensor_op.getResult());
    };

    Value input_lhs = bmm_op.getX();
    Value input_rhs = bmm_op.getY();

    Value output_lhs =
        bmm_op.getAdjX() ? create_z_x_transpose_op(input_lhs) : input_lhs;

    // The rhs need to be transposed if adj_y == false AND this matmul will be
    // legalized to tfl.fully_connected
    Value output_rhs = input_rhs;
    if (!bmm_op.getAdjY()) {
      // Transpose below a widening cast rather than above it. Only a
      // transpose directly on a constant gets folded away, and the cast is
      // then left directly on the constant too, which flatbuffer export
      // materializes.
      if (CastOp cast_op = AsFloatWideningCast(input_rhs.getDefiningOp())) {
        Value transposed = create_z_x_transpose_op(cast_op.getInput());
        auto transposed_type =
            mlir::cast<RankedTensorType>(transposed.getType());
        output_rhs = TFL::CastOp::create(
            rewriter, cast_op.getLoc(),
            transposed_type.clone(getElementTypeOrSelf(cast_op.getType())),
            transposed);
      } else {
        output_rhs = create_z_x_transpose_op(input_rhs);
      }
    }

    Type output_type = bmm_op.getResult().getType();
    auto no_input =
        TFL::NoValueOp::create(rewriter, bmm_op->getLoc(),
                               rewriter.getNoneType(), rewriter.getUnitAttr());
    auto fc_op = TFL::FullyConnectedOp::create(
        rewriter, bmm_op->getLoc(), ArrayRef<Type>{output_type},
        /*input=*/output_lhs, /*filter=*/output_rhs, /*bias=*/no_input,
        /*fused_activation_function=*/rewriter.getStringAttr("NONE"),
        /*weights_format=*/rewriter.getStringAttr("DEFAULT"),
        /*keep_num_dims=*/rewriter.getBoolAttr(true),
        /*asymmetric_quantize_inputs=*/mlir::BoolAttr());
    rewriter.replaceOp(bmm_op, {fc_op.getResult(0)});

    return success();
  };
};

// Converts batch_matmul operation with a ones tensor to a reduce_sum.
struct ConvertBatchMatMulOpToReduceSum
    : public OpRewritePattern<TFL::BatchMatMulOp> {
  using OpRewritePattern<TFL::BatchMatMulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TFL::BatchMatMulOp bmm_op,
                                PatternRewriter& rewriter) const override {
    // For simplicity, check if first operand is an identity i.e. `ones_like`.
    // This assumes canonicalization ordered operands this way.
    SplatElementsAttr constant;
    if (!matchPattern(bmm_op.getX(), m_Constant(&constant))) {
      return failure();
    }

    if (!SplatValueEquals(constant, 1.0)) {
      return failure();
    }

    // The input tensors x and y are 2-D or higher with shape:
    //       [..., r_x == 1, c_x] and [..., c_y, r_y].
    // The position of r_* and c_* are determined by the polarity of
    // the adj(X|Y) attribute, respectively.
    // So adjX == True indicates [..., c_x, r_x == 1].
    llvm::ArrayRef<int64_t> lhs_shape =
        mlir::cast<RankedTensorType>(bmm_op.getX().getType()).getShape();
    int rX = lhs_shape.size() - 2;
    int cX = lhs_shape.size() - 1;
    if (bmm_op.getAdjX()) {
      rX = lhs_shape.size() - 1;
      cX = lhs_shape.size() - 2;
    }

    if (lhs_shape[rX] != 1) {
      return failure();
    }

    llvm::ArrayRef<int64_t> rhs_shape =
        mlir::cast<RankedTensorType>(bmm_op.getY().getType()).getShape();
    int rY = rhs_shape.size() - 1;
    int cY = rhs_shape.size() - 2;
    if (bmm_op.getAdjX()) {
      rY = rhs_shape.size() - 2;
      cY = rhs_shape.size() - 1;
    }

    auto reduce_dim_op = TFL::ConstOp::create(
        rewriter, bmm_op->getLoc(),
        DenseIntElementsAttr::get(
            RankedTensorType::get({1}, rewriter.getI32Type()), {cY}));
    auto sum_op =
        TFL::SumOp::create(rewriter, bmm_op->getLoc(), bmm_op.getType(),
                           bmm_op.getY(), reduce_dim_op,
                           /*keep_dims=*/rewriter.getBoolAttr(true));
    rewriter.replaceOp(bmm_op, sum_op);
    return success();
  };

 private:
  bool SplatValueEquals(SplatElementsAttr float_or_int, double rhs) const {
    if (mlir::isa<DenseFPElementsAttr>(float_or_int)) {
      return mlir::cast<DenseFPElementsAttr>(float_or_int)
          .getSplatValue<APFloat>()
          .isExactlyValue(rhs);
    } else if (mlir::cast<DenseIntElementsAttr>(float_or_int)) {
      return float_or_int.getSplatValue<APInt>() == static_cast<int>(rhs);
    }
    return false;
  }
};

// Rewrites a batch_matmul whose LHS is a constant with a single row as a
// broadcast multiply followed by a sum over the contracting dimension:
//
//   BMM(X, Y), X constant [..., 1, K], Y [..., K, N]
//     = SUM(MUL(Y, reshape(X, [..., K, 1])), axis = K, keep_dims)
//
// This generalizes `ConvertBatchMatMulOpToReduceSum` (X all ones) to any
// constant row. The shape comes from a single learned query attended against
// runtime keys, e.g. the attention-pooling probe of SigLIP2 once the query
// projection has been folded: `[12, 1, 64] x [12, 64, 196]`.
//
// GPU delegates treat the LHS of batch_matmul as an activation and reject a
// constant there (ML Drift: "inputs[0] should not be a constant tensor"), and
// swapping the operands is not an option while the constant batched-RHS path
// is unverified (b/581393027). MUL and SUM with a constant operand are plain
// elementwise/reduce ops on every backend. The intermediate product has the
// shape of Y, so this is only applied while Y is small.
struct ConvertConstantRowLhsBatchMatMulToMulSum
    : public OpRewritePattern<TFL::BatchMatMulOp> {
  using OpRewritePattern<TFL::BatchMatMulOp>::OpRewritePattern;

  // Upper bound on the elements of Y (and so of the MUL result) to keep the
  // materialized product cheap relative to the matmul it replaces.
  static constexpr int64_t kMaxRhsElements = 1 << 20;

  LogicalResult matchAndRewrite(TFL::BatchMatMulOp bmm_op,
                                PatternRewriter& rewriter) const override {
    DenseElementsAttr lhs_value;
    if (!matchPattern(bmm_op.getX(), m_Constant(&lhs_value)) ||
        matchPattern(bmm_op.getY(), m_Constant())) {
      return rewriter.notifyMatchFailure(
          bmm_op, "lhs is not the only constant operand");
    }
    // A splat-ones LHS is a plain reduce_sum; leave it to
    // ConvertBatchMatMulOpToReduceSum.
    if (auto splat = mlir::dyn_cast<SplatElementsAttr>(lhs_value);
        splat && mlir::isa<DenseFPElementsAttr>(splat) &&
        splat.getSplatValue<APFloat>().isExactlyValue(1.0)) {
      return rewriter.notifyMatchFailure(bmm_op, "lhs is a splat of ones");
    }

    auto lhs_type = mlir::dyn_cast<RankedTensorType>(bmm_op.getX().getType());
    auto rhs_type = mlir::dyn_cast<RankedTensorType>(bmm_op.getY().getType());
    auto out_type = mlir::dyn_cast<RankedTensorType>(bmm_op.getType());
    if (!lhs_type || !rhs_type || !out_type || !lhs_type.hasStaticShape() ||
        !rhs_type.hasStaticShape() || !out_type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(bmm_op, "shapes are not static");
    }
    if (!lhs_type.getElementType().isF32() ||
        !rhs_type.getElementType().isF32() ||
        !out_type.getElementType().isF32()) {
      return rewriter.notifyMatchFailure(bmm_op, "not a float32 matmul");
    }
    const int64_t rank = out_type.getRank();
    if (rank < 2 || lhs_type.getRank() != rank || rhs_type.getRank() != rank) {
      return rewriter.notifyMatchFailure(bmm_op, "operand ranks differ");
    }
    if (rhs_type.getNumElements() > kMaxRhsElements) {
      return rewriter.notifyMatchFailure(bmm_op, "rhs is too large");
    }

    llvm::ArrayRef<int64_t> lhs_shape = lhs_type.getShape();
    llvm::ArrayRef<int64_t> rhs_shape = rhs_type.getShape();
    const int64_t lhs_rows = lhs_shape[bmm_op.getAdjX() ? rank - 1 : rank - 2];
    const int64_t lhs_cols = lhs_shape[bmm_op.getAdjX() ? rank - 2 : rank - 1];
    // The contracting axis of Y, and the axis of Y the constant row runs along.
    const int64_t rhs_contract_axis = bmm_op.getAdjY() ? rank - 1 : rank - 2;
    if (lhs_rows != 1 || lhs_cols != rhs_shape[rhs_contract_axis]) {
      return rewriter.notifyMatchFailure(bmm_op, "lhs is not a single row");
    }
    // The batch dimensions of X must broadcast onto Y without expanding Y.
    for (int64_t i = 0; i < rank - 2; ++i) {
      if (lhs_shape[i] != 1 && lhs_shape[i] != rhs_shape[i]) {
        return rewriter.notifyMatchFailure(bmm_op,
                                           "lhs batch dims do not broadcast");
      }
    }

    // X holds exactly one row, so laying it along the contracting axis of Y is
    // a reshape of the same elements.
    llvm::SmallVector<int64_t> row_shape(lhs_shape.begin(),
                                         lhs_shape.end() - 2);
    row_shape.push_back(bmm_op.getAdjY() ? 1 : lhs_cols);
    row_shape.push_back(bmm_op.getAdjY() ? lhs_cols : 1);
    auto row_type = RankedTensorType::get(row_shape, rewriter.getF32Type());
    auto row_const = TFL::ConstOp::create(rewriter, bmm_op.getLoc(),
                                          lhs_value.reshape(row_type));

    auto mul_op =
        TFL::MulOp::create(rewriter, bmm_op.getLoc(), rhs_type, bmm_op.getY(),
                           row_const, rewriter.getStringAttr("NONE"));

    llvm::SmallVector<int64_t> sum_shape(rhs_shape.begin(), rhs_shape.end());
    sum_shape[rhs_contract_axis] = 1;
    auto sum_type = RankedTensorType::get(sum_shape, rewriter.getF32Type());
    auto reduce_dim_op = TFL::ConstOp::create(
        rewriter, bmm_op.getLoc(),
        DenseIntElementsAttr::get(
            RankedTensorType::get({1}, rewriter.getI32Type()),
            {static_cast<int32_t>(rhs_contract_axis)}));
    auto sum_op = TFL::SumOp::create(rewriter, bmm_op.getLoc(), sum_type,
                                     mul_op.getResult(), reduce_dim_op,
                                     /*keep_dims=*/rewriter.getBoolAttr(true));

    if (sum_type == out_type) {
      rewriter.replaceOp(bmm_op, sum_op.getResult());
      return success();
    }
    // adj_y: the sum is [..., N, 1] while the matmul result is [..., 1, N].
    llvm::SmallVector<int32_t> out_shape(out_type.getShape().begin(),
                                         out_type.getShape().end());
    auto shape_const = TFL::ConstOp::create(
        rewriter, bmm_op.getLoc(),
        DenseIntElementsAttr::get(
            RankedTensorType::get({rank}, rewriter.getI32Type()), out_shape));
    rewriter.replaceOpWithNewOp<TFL::ReshapeOp>(
        bmm_op, out_type, sum_op.getResult(), shape_const);
    return success();
  }
};

// Pattern to fuse transpose op into RHS of batch_matmul op if the transpose and
// batch_matmul are separated by a reshape op; and the transpose op is used
// exclusively to transpose the contracting dimension and the LHS-Output
// dimension.
// Converts batch_matmul operation to fully_connected if rhs is rank-2
// else converts it to a BatchMatMul op with adj_y = true and transpose fused
// into RHS.
//
// Example:
// % 0 = "tfl.transpose" // Input: [2048, 32, 128] -> [128, 2048, 32]
// % 1 = "tfl.reshape"(%0)  // reshaped [128, 2048, 32] -> [128, 65536]
// % 2 = "tfl.batch_matmul"  // LHS: [4, 128], RHS: [128, 65536] -> [4, 65536]
struct FuseRhsTransposeIntoBatchMatMulOp
    : public OpRewritePattern<TFL::BatchMatMulOp> {
  using OpRewritePattern<TFL::BatchMatMulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TFL::BatchMatMulOp bmm_op,
                                PatternRewriter& rewriter) const override {
    // Exit the pattern if adj_y is true.
    if (bmm_op.getAdjY()) {
      return rewriter.notifyMatchFailure(
          bmm_op, "Pattern does not apply when adj_y is true.");
    }

    // Exit the pattern if the RHS of BatchMatMulOp is not originated from a
    // TFL::TransposeOp->TFL::ReshapeOp.
    auto reshape_op = bmm_op.getY().getDefiningOp<ReshapeOp>();
    if (!reshape_op) {
      return rewriter.notifyMatchFailure(
          bmm_op,
          "RHS is not originated from a transpose->reshape op pattern.");
    }

    auto transpose_op = reshape_op.getInput().getDefiningOp<TransposeOp>();
    if (!transpose_op) {
      return rewriter.notifyMatchFailure(
          bmm_op,
          "RHS is not originated from a transpose->reshape op pattern.");
    }

    // Get the dimensions info of the RHS of BatchMatMulOp.
    auto rhs_dimensions_info = GetBatchMatMulRhsDimensionsInfo(
        mlir::cast<ShapedType>(bmm_op.getY().getType()));

    // Make sure that the reshape op is flattening either the contracting
    // dimension or the output dimension.
    auto reshape_input_shape = GetShape(reshape_op.getInput());
    if (!HasFlattenedContractingDims(reshape_input_shape,
                                     rhs_dimensions_info) &&
        !HasFlattenedOutDims(reshape_input_shape, rhs_dimensions_info)) {
      return rewriter.notifyMatchFailure(
          bmm_op,
          "Reshape op is not flattening the contracting dimension or the "
          "output dimension.");
    }

    // Make sure that the transpose op is only transposing the contracting
    // dimensions and the output dimensions.
    auto transpose_perm_status_or_value =
        GetValueAsIntArray(transpose_op.getPerm());
    auto transpose_input_shape = GetShape(transpose_op.getInput());
    if (transpose_perm_status_or_value.ok() &&
        !HasTransposedContractingAndOutDims(
            transpose_input_shape, transpose_perm_status_or_value.value(),
            rhs_dimensions_info)) {
      return rewriter.notifyMatchFailure(
          bmm_op,
          "Transpose op is not transposing the contracting dimension and the "
          "output dimension.");
    }

    auto rhs_contracting_dimensions =
        rhs_dimensions_info.contracting_dimensions();
    auto rhs_out_dimensions = rhs_dimensions_info.out_dimensions();
    auto rhs_batch_dimensions = rhs_dimensions_info.batch_dimensions();

    // Create a new ReshapeOp, without the TransposeOp, to flatten the
    // contracting dimension and the output dimension, as needed.
    llvm::SmallVector<int32_t> new_reshape_input_shape;
    if (!rhs_dimensions_info.batch_dimensions().AxesArray().empty()) {
      for (auto dim_size : rhs_batch_dimensions.SizesArray()) {
        new_reshape_input_shape.push_back(dim_size);
      }
    }
    new_reshape_input_shape.push_back(rhs_out_dimensions.SizesArray().front());
    new_reshape_input_shape.push_back(
        rhs_contracting_dimensions.SizesArray().front());

    Value new_reshape_shape_value = arith::ConstantOp::create(
        rewriter, bmm_op->getLoc(),
        GetI32ElementsAttr(new_reshape_input_shape, &rewriter));
    auto new_reshape_value = TFL::ReshapeOp::create(rewriter, bmm_op->getLoc(),
                                                    transpose_op.getInput(),
                                                    new_reshape_shape_value);

    // Replace the BatchMatMulOp with a FullyConnectedOp, if the RHS of BMM has
    // no broadcasting dimensions. I.e. RHS of BMM is of Rank 2.
    if (rhs_dimensions_info.batch_dimensions().AxesArray().empty()) {
      auto no_input = TFL::NoValueOp::create(rewriter, bmm_op->getLoc(),
                                             rewriter.getNoneType(),
                                             rewriter.getUnitAttr());
      auto fc_op = TFL::FullyConnectedOp::create(
          rewriter, bmm_op->getLoc(), ArrayRef<Type>{bmm_op.getType()},
          /*input=*/bmm_op.getX(), /*filter=*/new_reshape_value,
          /*bias=*/no_input,
          /*fused_activation_function=*/rewriter.getStringAttr("NONE"),
          /*weights_format=*/rewriter.getStringAttr("DEFAULT"),
          /*keep_num_dims=*/rewriter.getBoolAttr(true),
          /*asymmetric_quantize_inputs=*/mlir::BoolAttr());
      rewriter.replaceOp(bmm_op, {fc_op.getResult(0)});
    } else {
      // Replace the BatchMatMulOp with a BatchMatMulOp with adj_y = true and
      // transpose fused into RHS.
      auto bmm_op_with_adj_y = TFL::BatchMatMulOp::create(
          rewriter, bmm_op->getLoc(), bmm_op.getType(), bmm_op.getX(),
          new_reshape_value, bmm_op.getAdjX(), /*adj_y=*/true,
          mlir::BoolAttr());
      rewriter.replaceOp(bmm_op, {bmm_op_with_adj_y.getResult()});
    }

    return success();
  }
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_optimize_batch_matmul.inc"
}  // namespace

void OptimizeBatchMatmulPass::runOnOperation() {
  auto func = getOperation();
  auto* ctx = &getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<ConvertBatchMatMulOp2FullyConnectedOp_Rank2ConstantRhs,
               ConvertBatchMatMulOpToReduceSum,
               ConvertConstantRowLhsBatchMatMulToMulSum,
               FuseRhsTransposeIntoBatchMatMulOp>(ctx);
  TFL::populateWithGenerated(patterns);
  (void)applyPatternsGreedily(func, std::move(patterns));
}

}  // namespace TFL
}  // namespace mlir
