/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "gml_st/transforms/test_passes.h"

#include <string>
#include <utility>

#include "gml_st/interfaces/bufferizable_op_interface_impl.h"
#include "gml_st/interfaces/tiling_interface_impl.h"
#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/peeling/peeling.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_TESTGMLSTBUFFERIZATION
#define GEN_PASS_DEF_TESTGMLSTLOOPPEELING
#define GEN_PASS_DEF_TESTGMLSTLOOPTILING
#define GEN_PASS_DEF_TESTGMLSTGREEDYFUSION
#include "gml_st/transforms/test_passes.h.inc"

static constexpr char kPeeledLoopsLabel[] = "__peeled_loops__";
static constexpr char kPartialIterationLabel[] = "__partial_iteration__";

/// Peel LoopOps, i.e., split them into two loops: One loop where the
/// `idx`-th loop contains only "full" iterations and a second loop for the
/// remaining partial iteration (if any).
struct TiledLoopPeelingPattern : public OpRewritePattern<LoopOp> {
  TiledLoopPeelingPattern(MLIRContext *ctx, int64_t idx, bool skipPartial)
      : OpRewritePattern<LoopOp>(ctx), idx(idx), skipPartial(skipPartial) {}

  LogicalResult matchAndRewrite(LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> peeledLoops;
    if (loopOp->hasAttr(kPeeledLoopsLabel)) {
      auto attr = loopOp->getAttr(kPeeledLoopsLabel).cast<ArrayAttr>();
      peeledLoops =
          llvm::to_vector<4>(llvm::map_range(attr, [](Attribute attr) {
            return attr.cast<IntegerAttr>().getInt();
          }));
      // Check if the loop was already peeled.
      if (llvm::find(peeledLoops, idx) != peeledLoops.end()) return failure();
    }
    if (skipPartial && loopOp->hasAttr(kPartialIterationLabel))
      // No peeling of loop nests with a partial iteration.
      return failure();

    if (static_cast<int64_t>(loopOp.getIteratorTypes().size()) <= idx)
      return failure();

    // Peel loop and canonicalize.
    auto result = peelAndCanonicalizeGmlStLoop(rewriter, loopOp, idx);
    if (failed(result)) return failure();

    // Apply label, so that the same loop is not rewritten a second time.
    peeledLoops.push_back(idx);
    rewriter.updateRootInPlace(loopOp, [&]() {
      loopOp->setAttr(kPeeledLoopsLabel, rewriter.getI64ArrayAttr(peeledLoops));
    });
    (*result)->setAttr(kPeeledLoopsLabel,
                       rewriter.getI64ArrayAttr(peeledLoops));
    (*result)->setAttr(kPartialIterationLabel, rewriter.getUnitAttr());

    return success();
  }

  /// Index of loop to peel.
  int64_t idx;

  /// If set to true, do not peel LoopOps with a partial iteration.
  bool skipPartial;
};

class TestGmlStLoopPeelingPass
    : public impl::TestGmlStLoopPeelingBase<TestGmlStLoopPeelingPass> {
  void runOnOperation() final {
    auto funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);
    for (unsigned idx : dims)
      patterns.add<TiledLoopPeelingPattern>(ctx, idx, skip_partial);

    (void)(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)));

    // Drop the markers.
    funcOp.walk([](LoopOp op) {
      op->removeAttr(kPeeledLoopsLabel);
      op->removeAttr(kPartialIterationLabel);
    });
  }
};

static constexpr llvm::StringRef kTestTilingAppliedLabel =
    "__test_tiling_applied_label__";

struct LinalgTilingPattern
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  LinalgTilingPattern(MLIRContext *context, linalg::LinalgTilingOptions options,
                      PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
        options(std::move(options)) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(op, kTestTilingAppliedLabel)) return failure();

    FailureOr<linalg::TiledLinalgOp> res =
        gml_st::tileLinalgOp(rewriter, op, options);
    if (failed(res)) return failure();

    setLabel(res->op, kTestTilingAppliedLabel);

    if (res->tensorResults.empty())
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, res->tensorResults);

    return success();
  }

 private:
  linalg::LinalgTilingOptions options;
};

struct TestGmlStLoopTilingPass
    : public impl::TestGmlStLoopTilingBase<TestGmlStLoopTilingPass> {
  TestGmlStLoopTilingPass() = default;
  TestGmlStLoopTilingPass(ArrayRef<int64_t> tileSizes,
                          ArrayRef<StringRef> distributionTypes) {
    this->tile_sizes = tileSizes;
    this->distribution_types = llvm::to_vector<2>(llvm::map_range(
        distributionTypes, [](StringRef ref) { return ref.str(); }));
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    auto distTypes = llvm::to_vector<2>(llvm::map_range(
        distribution_types, [](std::string &str) { return StringRef(str); }));
    auto options = linalg::LinalgTilingOptions()
                       .setTileSizes(tile_sizes)
                       .setDistributionTypes(distTypes);
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<LinalgTilingPattern>(ctx, options);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

    funcOp.walk(
        [](linalg::LinalgOp op) { removeLabel(op, kTestTilingAppliedLabel); });
  }
};

struct TestGmlStBufferizationPass
    : public impl::TestGmlStBufferizationBase<TestGmlStBufferizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<bufferization::BufferizationDialect, memref::MemRefDialect>();
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    gml_st::registerBufferizableOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    bufferization::OneShotBufferizationOptions opts;
    opts.allowReturnAllocs = true;
    opts.bufferizeFunctionBoundaries = true;
    opts.functionBoundaryTypeConversion =
        bufferization::LayoutMapOption::IdentityLayoutMap;

    ModuleOp module = getOperation();
    if (failed(bufferization::runOneShotModuleBufferize(module, opts))) {
      signalPassFailure();
      return;
    }
  }
};

static constexpr llvm::StringRef kTestFusionAppliedLabel =
    "__test_fusion_applied_label__";

struct GreedyFusionPattern : public OpRewritePattern<gml_st::ParallelOp> {
  using OpRewritePattern<gml_st::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gml_st::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(op, kTestFusionAppliedLabel)) return failure();

    rewriter.updateRootInPlace(op, [&]() {
      fuseGreedily(rewriter, op.getRegion().front(), [](Operation *op) {
        return isa<linalg::BroadcastOp, linalg::FillOp, linalg::MapOp>(op);
      });
    });

    setLabel(op, kTestFusionAppliedLabel);
    return success();
  }
};

struct TestGmlStGreedyFusionPass
    : public impl::TestGmlStGreedyFusionBase<TestGmlStGreedyFusionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<GmlStDialect, linalg::LinalgDialect, tensor::TensorDialect>();
    registerGmlStTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<GreedyFusionPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
      return signalPassFailure();

    funcOp.walk([](gml_st::ParallelOp op) {
      removeLabel(op, kTestFusionAppliedLabel);
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createTestGmlStLoopPeelingPass() {
  return std::make_unique<TestGmlStLoopPeelingPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> createTestGmlStLoopTilingPass() {
  return std::make_unique<TestGmlStLoopTilingPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createTestGmlStBufferizationPass() {
  return std::make_unique<TestGmlStBufferizationPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> createTestGmlStGreedyFusionPass() {
  return std::make_unique<TestGmlStGreedyFusionPass>();
}

}  // namespace gml_st
}  // namespace mlir
