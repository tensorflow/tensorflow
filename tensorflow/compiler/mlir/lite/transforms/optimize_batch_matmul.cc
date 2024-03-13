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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_OPTIMIZEBATCHMATMULPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// Checks whether the producer of `value` is TFL_DequantizeOp. This function
// iteratively finds the defining op if the direct defining op is TFL_SplitOp.
bool NotFromDequant(mlir::Value value) {
  auto dequant_op = value.getDefiningOp<DequantizeOp>();
  if (dequant_op) {
    return false;
  }
  auto split_op = value.getDefiningOp<SplitOp>();
  if (!split_op) {
    return true;
  }
  return !split_op.getValue().getDefiningOp<DequantizeOp>();
}

// Optimize TFLite operations in functions.
class OptimizeBatchMatmulPass
    : public impl::OptimizeBatchMatmulPassBase<OptimizeBatchMatmulPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeBatchMatmulPass)

  OptimizeBatchMatmulPass() = default;
  OptimizeBatchMatmulPass(const OptimizeBatchMatmulPass &) {}

  void runOnOperation() override;
};

// Converts batch_matmul operation to fully_connected if rhs is a
// constant tensor with rank 2
struct ConvertBatchMatMulOp2FullyConnectedOp
    : public OpRewritePattern<TFL::BatchMatMulOp> {
  using OpRewritePattern<TFL::BatchMatMulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TFL::BatchMatMulOp bmm_op,
                                PatternRewriter& rewriter) const override {
    DenseElementsAttr constant;
    if (auto rhs = bmm_op.getY(); !matchPattern(rhs, m_Constant(&constant))) {
      // The constant may be preceded by QDQs in models with QDQ format, so we
      // should set it to the real constant.
      auto dq = dyn_cast_or_null<DequantizeOp>(rhs.getDefiningOp());
      if (!dq) return failure();
      auto q = dyn_cast_or_null<QuantizeOp>(dq.getInput().getDefiningOp());
      if (!q || !matchPattern(q.getInput(), m_Constant(&constant))) {
        return failure();
      }
    }

    // Input rhs must be a constant with rank 2.
    if (constant.getType().getRank() != 2) return failure();

    // Create a tfl.transpose op that performs ZX transpose on `input`.
    auto create_z_x_transpose_op = [&](Value input) -> Value {
      RankedTensorType input_type = input.getType().cast<RankedTensorType>();
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
      return rewriter.create<TFL::TransposeOp>(
          bmm_op->getLoc(),
          RankedTensorType::get(permuted_shape, input_type.getElementType()),
          input, permutation_tensor_op.getResult());
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
        bmm_op.getX().getType().cast<RankedTensorType>().getShape();
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
        bmm_op.getY().getType().cast<RankedTensorType>().getShape();
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
    if (float_or_int.isa<DenseFPElementsAttr>()) {
      return float_or_int.cast<DenseFPElementsAttr>()
          .getSplatValue<APFloat>()
          .isExactlyValue(rhs);
    } else if (float_or_int.cast<DenseIntElementsAttr>()) {
      return float_or_int.getSplatValue<APInt>() == static_cast<int>(rhs);
    }
    return false;
  }
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_optimize_batch_matmul.inc"

void OptimizeBatchMatmulPass::runOnOperation() {
  auto func = getOperation();
  auto* ctx = &getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<ConvertBatchMatMulOp2FullyConnectedOp,
               ConvertBatchMatMulOpToReduceSum>(ctx);
  TFL::populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateOptimizeBatchMatmulPass() {
  return std::make_unique<OptimizeBatchMatmulPass>();
}

static PassRegistration<OptimizeBatchMatmulPass> pass;

}  // namespace TFL
}  // namespace mlir
