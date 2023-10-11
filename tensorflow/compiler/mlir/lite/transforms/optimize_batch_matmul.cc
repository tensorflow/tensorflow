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
    // Input rhs must be a constant with rank 2.
    DenseElementsAttr constant;
    if (!(matchPattern(bmm_op.getY(), m_Constant(&constant)) &&
          constant.getType().getRank() == 2)) {
      return failure();
    }

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

#include "tensorflow/compiler/mlir/lite/transforms/generated_optimize_batch_matmul.inc"

void OptimizeBatchMatmulPass::runOnOperation() {
  auto func = getOperation();
  auto* ctx = &getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<ConvertBatchMatMulOp2FullyConnectedOp>(ctx);
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
