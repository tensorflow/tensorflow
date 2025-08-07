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
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
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

// Checks whether the producer of `value` is part of a chain that can be folded
// into a constant. This includes DequantizeOp, or chains involving ReshapeOp
// and SplitOp originating from a DequantizeOp.
bool NotFromFoldableChain(mlir::Value value) {
  mlir::Operation* defining_op = value.getDefiningOp();

  while (defining_op) {
    if (mlir::isa<DequantizeOp>(defining_op)) {
      return false;
    }

    // Look through ops that don't change the constant nature.
    if (auto reshape_op = mlir::dyn_cast<ReshapeOp>(defining_op)) {
      defining_op = reshape_op.getInput().getDefiningOp();
    } else if (auto split_op = mlir::dyn_cast<SplitOp>(defining_op)) {
      defining_op = split_op.getValue().getDefiningOp();
    } else {
      // Stop if the op is not Dequantize, Reshape, or Split.
      break;
    }
  }
  return true;
}

// Converts batch_matmul operation to fully_connected if rhs is a
// constant tensor with rank 2
struct ConvertBatchMatMulOp2FullyConnectedOp_Rank2ConstantRhs
    : public OpRewritePattern<TFL::BatchMatMulOp> {
  using OpRewritePattern<TFL::BatchMatMulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TFL::BatchMatMulOp bmm_op,
                                PatternRewriter& rewriter) const override {
    bool is_int_quantized_rank_2_value = false;
    if (auto rhs_type = mlir::dyn_cast<quant::QuantizedType>(
            getElementTypeOrSelf(bmm_op.getY().getType()))) {
      int64_t rhs_type_rank = bmm_op.getY().getType().getRank();
      bool rhs_i4_or_i8 = rhs_type.getStorageTypeIntegralWidth() == 4 ||
                          rhs_type.getStorageTypeIntegralWidth() == 8;
      if (rhs_i4_or_i8 && rhs_type.isSigned() && rhs_type_rank == 2) {
        is_int_quantized_rank_2_value = true;
      }
    }

    ElementsAttr constant = nullptr;
    Value rhs = bmm_op.getY();
    // If there is a reshape, look through it.
    if (auto reshape = rhs.getDefiningOp<ReshapeOp>()) {
      rhs = reshape.getInput();
    }

    DenseElementsAttr dense_constant;
    if (matchPattern(rhs, m_Constant(&dense_constant))) {
      constant = dense_constant;
    } else if (auto dq = rhs.getDefiningOp<DequantizeOp>()) {
      Value q_input = dq.getInput();
      if (auto q = q_input.getDefiningOp<QuantizeOp>()) {
        if (matchPattern(q.getInput(), m_Constant(&dense_constant))) {
          constant = dense_constant;
        }
      } else if (auto pseudo_q = q_input.getDefiningOp<TFL::QConstOp>()) {
        constant = pseudo_q.getValue();
      }
    }

    const bool is_rank_2_constant =
        constant &&
        mlir::cast<ShapedType>(bmm_op.getY().getType()).getRank() == 2;

    if (!is_rank_2_constant && !is_int_quantized_rank_2_value) {
      return rewriter.notifyMatchFailure(
          bmm_op,
          "rhs is neither a constant with rank 2 nor int4 quantized nor a "
          "dequantized value.");
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
      auto permutation_tensor_op = rewriter.create<arith::ConstantOp>(
          bmm_op->getLoc(), permuation_tensor_type,
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

      return rewriter.create<TFL::TransposeOp>(
          bmm_op->getLoc(), output_type, input,
          permutation_tensor_op.getResult());
    };

    Value input_lhs = bmm_op.getX();
    Value input_rhs = bmm_op.getY();

    Value output_lhs =
        bmm_op.getAdjX() ? create_z_x_transpose_op(input_lhs) : input_lhs;

    // The rhs need to be transposed if adj_y == false AND this matmul will be
    // legalized to tfl.fully_connected
    Value output_rhs =
        !bmm_op.getAdjY() ? create_z_x_transpose_op(input_rhs) : input_rhs;

    Type output_type = bmm_op.getResult().getType();
    auto no_input = rewriter.create<TFL::NoValueOp>(
        bmm_op->getLoc(), rewriter.getNoneType(), rewriter.getUnitAttr());
    auto fc_op = rewriter.create<TFL::FullyConnectedOp>(
        bmm_op->getLoc(), ArrayRef<Type>{output_type},
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

    auto reduce_dim_op = rewriter.create<TFL::ConstOp>(
        bmm_op->getLoc(),
        DenseIntElementsAttr::get(
            RankedTensorType::get({1}, rewriter.getI32Type()), {cY}));
    auto sum_op = rewriter.create<TFL::SumOp>(
        bmm_op->getLoc(), bmm_op.getType(), bmm_op.getY(), reduce_dim_op,
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

    Value new_reshape_shape_value = rewriter.create<arith::ConstantOp>(
        bmm_op->getLoc(),
        GetI32ElementsAttr(new_reshape_input_shape, &rewriter));
    auto new_reshape_value = rewriter.create<TFL::ReshapeOp>(
        bmm_op->getLoc(), transpose_op.getInput(), new_reshape_shape_value);

    // Replace the BatchMatMulOp with a FullyConnectedOp, if the RHS of BMM has
    // no broadcasting dimensions. I.e. RHS of BMM is of Rank 2.
    if (rhs_dimensions_info.batch_dimensions().AxesArray().empty()) {
      auto no_input = rewriter.create<TFL::NoValueOp>(
          bmm_op->getLoc(), rewriter.getNoneType(), rewriter.getUnitAttr());
      auto fc_op = rewriter.create<TFL::FullyConnectedOp>(
          bmm_op->getLoc(), ArrayRef<Type>{bmm_op.getType()},
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
      auto bmm_op_with_adj_y = rewriter.create<TFL::BatchMatMulOp>(
          bmm_op->getLoc(), bmm_op.getType(), bmm_op.getX(), new_reshape_value,
          bmm_op.getAdjX(), /*adj_y=*/true, mlir::BoolAttr());
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
  patterns
      .add<ConvertBatchMatMulOp2FullyConnectedOp_Rank2ConstantRhs,
           ConvertBatchMatMulOpToReduceSum, FuseRhsTransposeIntoBatchMatMulOp>(
          ctx);
  TFL::populateWithGenerated(patterns);
  (void)applyPatternsGreedily(func, std::move(patterns));
}

}  // namespace TFL
}  // namespace mlir
