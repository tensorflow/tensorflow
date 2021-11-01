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

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h.inc"

using mlir::failure;
using mlir::LogicalResult;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::success;
using mlir::Value;
using mlir::linalg::FillOp;
using mlir::linalg::LinalgOp;
using mlir::linalg::LinalgTransformationFilter;
using mlir::linalg::PadTensorOp;
using mlir::linalg::TiledLoopOp;
using mlir::tensor::ExtractSliceOp;

// Replace FillOp(PadTensorOp) -> FillOp(InitTensorOp).
struct FillOfPadTensor : public OpRewritePattern<FillOp> {
  using OpRewritePattern<FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FillOp fill,
                                PatternRewriter &rewriter) const override {
    if (auto pad = fill.output().getDefiningOp<PadTensorOp>()) {
      if (!pad.getResultType().hasStaticShape()) {
        return failure();
      }
      Value init = rewriter.create<mlir::linalg::InitTensorOp>(
          fill.getLoc(), pad.getResultType().getShape(),
          pad.getResultType().getElementType());
      rewriter.replaceOpWithNewOp<FillOp>(fill, fill.value(), init);
      return success();
    }
    return failure();
  }
};

// Replace PadTensorOp(ExtractSliceOp(LinalgOp)) -> LinalgOp if
// `pad_tensor(extract_slice(*))` is a no-op.
struct PadOfExtractOfLinalg : public OpRewritePattern<PadTensorOp> {
  using OpRewritePattern<PadTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadTensorOp pad,
                                PatternRewriter &rewriter) const override {
    if (auto extract = pad.source().getDefiningOp<ExtractSliceOp>()) {
      if (auto linalg = extract.source().getDefiningOp<LinalgOp>()) {
        auto type = extract.source().getType().cast<mlir::RankedTensorType>();
        if (type.hasStaticShape() && type == pad.getType()) {
          rewriter.replaceOp(pad, {extract.source()});
          return success();
        }
      }
    }
    return failure();
  }
};

Value getNeutralOfLinalgOp(OpBuilder &b, mlir::OpOperand &op) {
  auto t = getElementTypeOrSelf(op.get().getType());
  return b.create<mlir::arith::ConstantOp>(op.getOwner()->getLoc(), t,
                                           b.getZeroAttr(t));
}

struct PaddingPattern : public mlir::OpInterfaceRewritePattern<LinalgOp> {
  PaddingPattern(LinalgTransformationFilter filter, mlir::MLIRContext *context,
                 mlir::PatternBenefit benefit = 1)
      : mlir::OpInterfaceRewritePattern<LinalgOp>(context, benefit),
        filter(filter) {}

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    // Check if the op was already processed.
    if (failed(filter.checkAndNotify(rewriter, op))) return failure();

    // Check if all tensor arguments are produced by ExtractSliceOp.
    if (!llvm::all_of(
            op.getInputAndOutputOperands(), [&](mlir::OpOperand *operand) {
              return !operand->get().getType().isa<mlir::RankedTensorType>() ||
                     mlir::isa_and_nonnull<ExtractSliceOp>(
                         operand->get().getDefiningOp());
            }))
      return failure();

    // Attempt to pad the op.
    LinalgOp padded_op;
    if (mlir::failed(mlir::linalg::rewriteAsPaddedOp(
            rewriter, op, getNeutralOfLinalgOp, nullptr, padded_op))) {
      return failure();
    }
    if (padded_op) {
      filter.replaceLinalgTransformationFilter(rewriter, padded_op);
    } else {
      // In case the op did not require padding, mark the op.
      filter.replaceLinalgTransformationFilter(rewriter, op);
    }
    return success();
  }

 private:
  LinalgTransformationFilter filter;
};

struct PadTiledOpsPass : public PadTiledOpsBase<PadTiledOpsPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnFunction() override {
    auto func = getFunction();
    auto filter = LinalgTransformationFilter(
                      llvm::ArrayRef<mlir::Identifier>{},
                      {mlir::Identifier::get("padded", func.getContext())})
                      .addFilter([](mlir::Operation *op) {
                        return success(op->getParentOfType<TiledLoopOp>());
                      });

    auto *ctx = func.getContext();
    mlir::OwningRewritePatternList patterns(ctx);
    patterns.insert<PaddingPattern>(filter, ctx);
    patterns.insert<FillOfPadTensor, PadOfExtractOfLinalg>(ctx);
    mlir::memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
    (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns));

    // Ensure we drop the marker in the end.
    func.walk([](mlir::linalg::LinalgOp op) {
      op->removeAttr(mlir::linalg::LinalgTransforms::kLinalgTransformMarker);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::FunctionPass> CreatePadTiledOpsPass() {
  return std::make_unique<PadTiledOpsPass>();
}

}  // namespace tensorflow
