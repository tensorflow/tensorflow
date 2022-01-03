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

#include <utility>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/passes.h.inc"

using mlir::failure;
using mlir::FailureOr;
using mlir::LogicalResult;
using mlir::Operation;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::success;
using mlir::Value;
using mlir::linalg::FillOp;
using mlir::linalg::InitTensorOp;
using mlir::linalg::PadTensorOp;
using mlir::linalg::TiledLoopOp;
using mlir::linalg::YieldOp;
using mlir::tensor::ExtractSliceOp;

struct UntangleOutputOperandsOfUnusedResults
    : public OpRewritePattern<TiledLoopOp> {
  using OpRewritePattern<TiledLoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TiledLoopOp tiled_loop,
                                PatternRewriter& rewriter) const override {
    bool changed = false;
    auto yield = tiled_loop.getBody()->getTerminator();
    mlir::SmallVector<Value> new_yield_args{yield->getOperands()};

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(tiled_loop.getBody());
    for (auto& zip : llvm::enumerate(llvm::zip(
             tiled_loop.getResults(), tiled_loop.outputs(),
             tiled_loop.getRegionOutputArgs(), yield->getOperands()))) {
      Value result, output_arg, yield_operand, output_bb_arg;
      std::tie(result, output_arg, output_bb_arg, yield_operand) = zip.value();
      if (!result.use_empty()) continue;

      FailureOr<InitTensorOp> init_tensor = FindInitTensorOp(output_arg);
      if (failed(init_tensor)) continue;

      auto cloned_init =
          mlir::cast<InitTensorOp>(rewriter.clone(*init_tensor.getValue()));

      output_bb_arg.replaceUsesWithIf(
          cloned_init.result(), [&](mlir::OpOperand& operand) {
            auto loop = operand.getOwner()->getParentOfType<TiledLoopOp>();
            return loop && loop == tiled_loop;
          });
      changed = true;
      new_yield_args[zip.index()] = output_bb_arg;
    }
    // Update terminator of `tiled_loop`.
    rewriter.setInsertionPointToEnd(tiled_loop.getBody());
    rewriter.replaceOpWithNewOp<YieldOp>(yield, new_yield_args);

    return success(changed);
  }

 private:
  // Traverse the chain of `tiled_loop` operations to find `init_tensor`.
  // There might be several chained loops if the loop peeling was enabled.
  FailureOr<InitTensorOp> FindInitTensorOp(Value output_arg) const {
    while (true) {
      if (auto init_tensor = output_arg.getDefiningOp<InitTensorOp>())
        return init_tensor;

      auto result = output_arg.dyn_cast<mlir::OpResult>();
      if (!result) return failure();

      auto tloop = mlir::dyn_cast<TiledLoopOp>(result.getOwner());
      if (!tloop) return failure();
      output_arg = tloop.outputs()[result.getResultNumber()];
    }
    return failure();
  }
};

// Replace FillOp(PadTensorOp) -> FillOp(InitTensorOp).
struct FillOfPadTensor : public OpRewritePattern<FillOp> {
  using OpRewritePattern<FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FillOp fill,
                                PatternRewriter& rewriter) const override {
    if (auto pad = fill.output().getDefiningOp<PadTensorOp>()) {
      if (!pad.getResultType().hasStaticShape()) {
        return failure();
      }
      Value init = rewriter.create<InitTensorOp>(
          fill.getLoc(), pad.getResultType().getShape(),
          pad.getResultType().getElementType());
      rewriter.replaceOpWithNewOp<FillOp>(fill, fill.value(), init);
      return success();
    }
    return failure();
  }
};

// Rewrite linalg.fill(extract_slice) as linalg.fill(init_tensor).
struct FillOfExtractSlice : public OpRewritePattern<FillOp> {
  using OpRewritePattern<FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FillOp fill,
                                PatternRewriter& rewriter) const override {
    if (!fill.hasTensorSemantics()) return failure();

    auto fill_tensor_type = fill.getOutputTensorTypes().back();

    if (auto extract = fill.output().getDefiningOp<ExtractSliceOp>()) {
      if (!extract.source().getDefiningOp<InitTensorOp>()) return failure();
      llvm::SmallVector<int64_t, 4> static_sizes = llvm::to_vector<4>(
          llvm::map_range(extract.static_sizes().cast<mlir::ArrayAttr>(),
                          [](mlir::Attribute a) -> int64_t {
                            return a.cast<mlir::IntegerAttr>().getInt();
                          }));
      auto init = rewriter.create<InitTensorOp>(
          fill.getLoc(), extract.getDynamicSizes(), static_sizes,
          fill_tensor_type.getElementType());
      rewriter.replaceOpWithNewOp<FillOp>(fill, fill.value(), init);
      return success();
    }
    return failure();
  }
};

// For every unused result of a `tiled_loop`, finds an `init_tensor` producer of
// the corresponding `tiled_loop` output argument and clones `init_tensor` into
// the loop body to replace the uses of output block arg. After that
// TiledLoopOp canonicalization will remove the unused results, output args and
// block args and `FillOfExtractSlice`/`FillOfPadTensor` will reduce the size of
// the cloned `init_tensor`.
struct SinkUnusedOutputsPass
    : public SinkUnusedOutputsBase<SinkUnusedOutputsPass> {
  void runOnFunction() override {
    auto func = getFunction();
    auto ctx = func.getContext();

    mlir::OwningRewritePatternList patterns(ctx);
    TiledLoopOp::getCanonicalizationPatterns(patterns, ctx);
    patterns.insert<FillOfExtractSlice, FillOfPadTensor,
                    UntangleOutputOperandsOfUnusedResults>(ctx);
    (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<mlir::FunctionPass> CreateSinkUnusedOutputs() {
  return std::make_unique<SinkUnusedOutputsPass>();
}

}  // namespace tensorflow
