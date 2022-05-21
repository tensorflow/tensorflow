/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

// Maximum rank that is allowed. Other Tensors should be restricted to this
// rank. This can be generalized as a pass parameter depending on the use-cases.
constexpr int64_t kMaxRank = 5;

// Rewrites Reshape -> Transpose -> Reshape sequence of ops originating from
// lowering of ops like SpaceToBatchND.
//
// Input to the first Reshape is Tensor in NHWC format in 4D or 5D.
//
// The first reshape splits spatial dimensions to generated two dimensions for
// each of the spatial dimension. Then, transpose moves the second part of the
// split dimensions to the beginning. The final reshape op combines the first
// dimension with the moved dimensions.
//
// reshape(NxHxWxC) -> (Nx(H/B1)xB1x(W/B2)xB2xC)
// tranpose(Nx(H/B1)xB1x(W/B2)xB2xC) -> (B1xB2xNx(H/B1)x(W/B2)xC)
// reshape(B1xB2xNx(H/B1)x(W/B2)xC) -> ((B1*B2*N)x(H/B1)x(W/B2)xC)
//
// Rank of the intermediate tensors can be reduced by doing one transpose for
// each of the spatial dims in sequence.
struct RewriteReshapeTransposeReshape : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern<TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    Value result = op.getResult();
    TensorType result_ty = result.getType().cast<TensorType>();
    Value operand = op.operand();
    TensorType operand_ty = operand.getType().cast<TensorType>();
    if (!operand_ty.hasStaticShape() || !result_ty.hasStaticShape())
      return rewriter.notifyMatchFailure(op,
                                         "transpose op has non-static types");

    if (result_ty.getRank() <= kMaxRank)
      return rewriter.notifyMatchFailure(op,
                                         "already has right dimensionality");

    if (!operand.hasOneUse() || !result.hasOneUse())
      return rewriter.notifyMatchFailure(
          op, "transpose op operand and result have multiple uses");

    auto def_op = operand.getDefiningOp<ReshapeOp>();
    if (!def_op)
      return rewriter.notifyMatchFailure(
          op, "defining op for operand is not reshape");

    auto user_op = llvm::dyn_cast<ReshapeOp>(result.use_begin().getUser());
    if (!user_op)
      return rewriter.notifyMatchFailure(op,
                                         "user of the result is not reshape");

    Value input = def_op.operand();
    auto input_ty = input.getType().cast<TensorType>();
    auto output_ty = user_op.getType();
    if (!input_ty.hasStaticShape() || !output_ty.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "reshape op input or output type is not static");

    int64_t input_rank = input_ty.getRank();
    int64_t output_rank = output_ty.getRank();
    if (input_rank != output_rank)
      return rewriter.notifyMatchFailure(
          op, "reshape op input and output rank are different");

    int64_t spatial_dims = input_rank - 2;
    if (spatial_dims < 0 || operand_ty.getRank() != 2 + 2 * spatial_dims)
      return rewriter.notifyMatchFailure(
          op, "transpose op operand isn't expanding spatial dims");

    SmallVector<int64_t, 4> block_sizes;
    SmallVector<int64_t, 6> expected_perm(operand_ty.getRank());
    expected_perm[spatial_dims] = 0;

    if (input_ty.getDimSize(0) != operand_ty.getDimSize(0))
      return rewriter.notifyMatchFailure(
          op, "reshape op isn't expanding only spatial dims");
    for (int64_t dim = 0; dim < spatial_dims; dim++) {
      int64_t block_size = operand_ty.getDimSize(1 + dim * 2 + 1);
      if (input_ty.getDimSize(1 + dim) !=
          operand_ty.getDimSize(1 + dim * 2) * block_size)
        return rewriter.notifyMatchFailure(
            op, "reshape op isn't only expanding spatial dims");
      block_sizes.push_back(block_size);

      expected_perm[dim] = 1 + 2 * dim + 1;
      expected_perm[1 + spatial_dims + dim] = 1 + 2 * dim;
    }
    expected_perm[1 + 2 * spatial_dims] = 1 + 2 * spatial_dims;

    SmallVector<int64_t, 6> perm(op.permutation().getValues<int64_t>());
    if (perm != expected_perm)
      return rewriter.notifyMatchFailure(
          op, "reshape op isn't only moving spatial dims");

    SmallVector<int64_t, 4> out_shape;
    out_shape.push_back(result_ty.getDimSize(0));
    for (int64_t dim = 0; dim < spatial_dims; dim++) {
      out_shape[0] *= result_ty.getDimSize(1 + dim);
      out_shape.push_back(result_ty.getDimSize(1 + spatial_dims + dim));
    }
    out_shape.push_back(result_ty.getDimSize(1 + spatial_dims * 2));
    if (llvm::to_vector<4>(output_ty.getShape()) != out_shape)
      return rewriter.notifyMatchFailure(
          op, "reshape op isn't only combining block dims");

    // Now that the input patterns are verified, introduce a sequence of
    // reshape->transpose->reshape for each of the spatial dimensions.  We need
    // to start with the last spatial dimension to preserve the sequence in the
    // first dimension.
    for (int dim = spatial_dims - 1; dim >= 0; dim--) {
      // 1) Reshape to split the particular spatial dimension.
      auto input_ty = input.getType().cast<TensorType>();
      auto intermediate_shape = llvm::to_vector<4>(input_ty.getShape());
      int64_t dim_idx = 1 + dim;
      intermediate_shape[dim_idx] /= block_sizes[dim];
      int64_t block_idx = dim_idx + 1;
      intermediate_shape.insert(intermediate_shape.begin() + block_idx,
                                block_sizes[dim]);
      auto reshaped_ty =
          RankedTensorType::get(intermediate_shape, input_ty.getElementType());

      auto reshape =
          rewriter.create<ReshapeOp>(op.getLoc(), reshaped_ty, input);

      // 2) Transpose to move the block part of the split dimension in the
      // beginning.
      SmallVector<int64_t, 8> perm;
      perm.push_back(block_idx);
      perm.push_back(0);
      for (int i = 1, e = reshaped_ty.getRank(); i != e; i++) {
        if (i != perm[0]) perm.push_back(i);
      }

      auto transpose = rewriter.create<TransposeOp>(
          op.getLoc(), reshape, rewriter.getI64TensorAttr(perm));

      // 3) Reshape to combine the block dimension with the batch dimension.
      intermediate_shape = llvm::to_vector<4>(transpose.getType().getShape());
      intermediate_shape[0] *= intermediate_shape[1];
      intermediate_shape.erase(intermediate_shape.begin() + 1);
      reshaped_ty =
          RankedTensorType::get(intermediate_shape, input_ty.getElementType());

      input = rewriter.create<ReshapeOp>(op.getLoc(), reshaped_ty, transpose);
    }

    rewriter.replaceOp(user_op, input);
    return success();
  }
};

struct RestrictMaxRankPass
    : public RestrictMaxRankPassBase<RestrictMaxRankPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mhlo::MhloDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    // Collect patterns.
    RewritePatternSet patterns(ctx);
    patterns.add<RewriteReshapeTransposeReshape>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            GreedyRewriteConfig()))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createRestrictMaxRankPass() {
  return std::make_unique<RestrictMaxRankPass>();
}

}  // namespace mhlo
}  // namespace mlir
