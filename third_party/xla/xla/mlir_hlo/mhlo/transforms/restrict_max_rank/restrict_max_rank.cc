/* Copyright 2021 The OpenXLA Authors.

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
#include <memory>
#include <utility>

#include "llvm/Support/Casting.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
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

#define GEN_PASS_DEF_RESTRICTMAXRANKPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

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
    TensorType resultTy = result.getType().cast<TensorType>();
    Value operand = op.getOperand();
    TensorType operandTy = operand.getType().cast<TensorType>();
    if (!operandTy.hasStaticShape() || !resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op,
                                         "transpose op has non-static types");

    if (resultTy.getRank() <= kMaxRank)
      return rewriter.notifyMatchFailure(op,
                                         "already has right dimensionality");

    if (!operand.hasOneUse() || !result.hasOneUse())
      return rewriter.notifyMatchFailure(
          op, "transpose op operand and result have multiple uses");

    auto defOp = operand.getDefiningOp<ReshapeOp>();
    if (!defOp)
      return rewriter.notifyMatchFailure(
          op, "defining op for operand is not reshape");

    auto userOp = llvm::dyn_cast<ReshapeOp>(result.use_begin().getUser());
    if (!userOp)
      return rewriter.notifyMatchFailure(op,
                                         "user of the result is not reshape");

    Value input = defOp.getOperand();
    auto inputTy = input.getType().cast<TensorType>();
    auto outputTy = userOp.getType();
    if (!inputTy.hasStaticShape() || !outputTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "reshape op input or output type is not static");

    int64_t inputRank = inputTy.getRank();
    int64_t outputRank = outputTy.getRank();
    if (inputRank != outputRank)
      return rewriter.notifyMatchFailure(
          op, "reshape op input and output rank are different");

    int64_t spatialDims = inputRank - 2;
    if (spatialDims < 0 || operandTy.getRank() != 2 + 2 * spatialDims)
      return rewriter.notifyMatchFailure(
          op, "transpose op operand isn't expanding spatial dims");

    SmallVector<int64_t, 4> blockSizes;
    SmallVector<int64_t, 6> expectedPerm(operandTy.getRank());
    expectedPerm[spatialDims] = 0;

    if (inputTy.getDimSize(0) != operandTy.getDimSize(0))
      return rewriter.notifyMatchFailure(
          op, "reshape op isn't expanding only spatial dims");
    for (int64_t dim = 0; dim < spatialDims; dim++) {
      int64_t blockSize = operandTy.getDimSize(1 + dim * 2 + 1);
      if (inputTy.getDimSize(1 + dim) !=
          operandTy.getDimSize(1 + dim * 2) * blockSize)
        return rewriter.notifyMatchFailure(
            op, "reshape op isn't only expanding spatial dims");
      blockSizes.push_back(blockSize);

      expectedPerm[dim] = 1 + 2 * dim + 1;
      expectedPerm[1 + spatialDims + dim] = 1 + 2 * dim;
    }
    expectedPerm[1 + 2 * spatialDims] = 1 + 2 * spatialDims;

    SmallVector<int64_t, 6> perm(op.getPermutation().getValues<int64_t>());
    if (perm != expectedPerm)
      return rewriter.notifyMatchFailure(
          op, "reshape op isn't only moving spatial dims");

    SmallVector<int64_t, 4> outShape;
    outShape.push_back(resultTy.getDimSize(0));
    for (int64_t dim = 0; dim < spatialDims; dim++) {
      outShape[0] *= resultTy.getDimSize(1 + dim);
      outShape.push_back(resultTy.getDimSize(1 + spatialDims + dim));
    }
    outShape.push_back(resultTy.getDimSize(1 + spatialDims * 2));
    if (llvm::to_vector<4>(outputTy.getShape()) != outShape)
      return rewriter.notifyMatchFailure(
          op, "reshape op isn't only combining block dims");

    // Now that the input patterns are verified, introduce a sequence of
    // reshape->transpose->reshape for each of the spatial dimensions.  We need
    // to start with the last spatial dimension to preserve the sequence in the
    // first dimension.
    for (int dim = spatialDims - 1; dim >= 0; dim--) {
      // 1) Reshape to split the particular spatial dimension.
      auto inputTy = input.getType().cast<TensorType>();
      auto intermediateShape = llvm::to_vector<4>(inputTy.getShape());
      int64_t dimIdx = 1 + dim;
      intermediateShape[dimIdx] /= blockSizes[dim];
      int64_t blockIdx = dimIdx + 1;
      intermediateShape.insert(intermediateShape.begin() + blockIdx,
                               blockSizes[dim]);
      auto reshapedTy =
          RankedTensorType::get(intermediateShape, inputTy.getElementType());

      auto reshape = rewriter.create<ReshapeOp>(op.getLoc(), reshapedTy, input);

      // 2) Transpose to move the block part of the split dimension in the
      // beginning.
      SmallVector<int64_t, 8> perm;
      perm.push_back(blockIdx);
      perm.push_back(0);
      for (int i = 1, e = reshapedTy.getRank(); i != e; i++) {
        if (i != perm[0]) perm.push_back(i);
      }

      auto transpose = rewriter.create<TransposeOp>(
          op.getLoc(), reshape, rewriter.getI64TensorAttr(perm));

      // 3) Reshape to combine the block dimension with the batch dimension.
      intermediateShape = llvm::to_vector<4>(transpose.getType().getShape());
      intermediateShape[0] *= intermediateShape[1];
      intermediateShape.erase(intermediateShape.begin() + 1);
      reshapedTy =
          RankedTensorType::get(intermediateShape, inputTy.getElementType());

      input = rewriter.create<ReshapeOp>(op.getLoc(), reshapedTy, transpose);
    }

    rewriter.replaceOp(userOp, input);
    return success();
  }
};

struct RestrictMaxRankPass
    : public impl::RestrictMaxRankPassBase<RestrictMaxRankPass> {
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
