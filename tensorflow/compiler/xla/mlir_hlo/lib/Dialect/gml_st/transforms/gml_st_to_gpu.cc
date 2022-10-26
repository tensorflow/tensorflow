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

#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir-hlo/Dialect/gml_st/transforms/vector_utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_GMLSTTOGPUPASS
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h.inc"

using namespace mlir;
using namespace mlir::gml_st;
using mlir::gpu::LaunchOp;
using mlir::memref::SubViewOp;
using mlir::vector::CombiningKind;
using mlir::vector::ExtractOp;
using mlir::vector::MultiDimReductionOp;
using mlir::vector::TransferReadOp;
using mlir::vector::TransferWriteOp;

namespace {
/// Converts a sequence of 3 nested gml_st.parallel ops into a gpu.launch op.
/// Throughout thes pass we will call the first level of nesting "block", the
/// second "warp", and the 3rd "thread" level. The intention is to allude to the
/// fact that these will likely correspond to the CUDA programming concepts of
/// the same name when the IR is lowered to PTX. However, this pass does not
/// make, nor verify all the requirements (e.g., that the warp-level iteration
/// contains exactly 32 steps) for mapping to this level.
///
/// Each gml_st.parallel is expected to only have a single induction variable.
/// The loops representing the block, warp, and thread level are mapped to
/// gridDim.x, blockDim.y, and blockDim.x launch dimensions of gpu.launch,
/// respectively.
///
/// All operations from within the nested gml_st.parallel regions are copied
/// directly into the gpu.launch region, with induction variables replaced by
/// equivalent values computed using the blockIdx.x, threadIdx.y and threadIdx.x
/// indices. Thus, the 3 nested parallel regions are effectively flattened into
/// a single level of nesting within the gpu.launch region.
///
/// At any level of nesting, multiple gml_st.parallel operations are allowed, as
/// long as they have the same iteration space, i.e., the SSA values defining
/// the lower bound, upper bound and the step of all parallels on the same level
/// of nesting are the same values.
struct ParallelOpToGpuPattern : public OpRewritePattern<ParallelOp> {
  using OpRewritePattern<ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ParallelOp root,
                                PatternRewriter& rewriter) const override;
};

struct MultiDimReductionOpToWarpReductionPattern
    : OpRewritePattern<MultiDimReductionOp> {
  using OpRewritePattern<MultiDimReductionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MultiDimReductionOp reductionOp,
                                PatternRewriter& rewriter) const override;
};

struct EliminateMaterializeOfTransferReadPattern
    : OpRewritePattern<MaterializeOp> {
  using OpRewritePattern<MaterializeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MaterializeOp materialize,
                                PatternRewriter& rewriter) const override;
};

struct EliminateDistributeIntoTransferWritePattern
    : OpRewritePattern<TransferWriteOp> {
  using OpRewritePattern<TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferWriteOp transferWrite,
                                PatternRewriter& rewriter) const override;
};

/// Implements the GmlStToGpuPass declared in
/// include/mlir-hlo/Dialect/gml_st/transforms/passes.td.
struct GmlStToGpuPass : public ::impl::GmlStToGpuPassBase<GmlStToGpuPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns
        .add<ParallelOpToGpuPattern, EliminateMaterializeOfTransferReadPattern,
             EliminateDistributeIntoTransferWritePattern,
             MultiDimReductionOpToWarpReductionPattern>(&getContext());
    func::FuncOp func = getOperation();
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
    // Make sure there are no GmlSt ops left.
    WalkResult walk = func.walk([](Operation* op) {
      if (isa<GmlStDialect>(op->getDialect())) {
        op->emitOpError("failed to simtfy");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walk.wasInterrupted()) signalPassFailure();
  }
};
}  // namespace

/// Creates an initial gpu.launch op with launch configuration set to a single
/// thread. The idea is to update those later, as we discover the correct values
/// from the nesting structure.
static LaunchOp createInitialGpuLaunchOp(Location loc, Value defaultSize,
                                         PatternRewriter& rewriter) {
  auto launch =
      rewriter.create<LaunchOp>(loc, defaultSize, defaultSize, defaultSize,
                                defaultSize, defaultSize, defaultSize);
  Block* body = &launch.getBody().front();
  rewriter.setInsertionPointToEnd(body);
  rewriter.create<gpu::TerminatorOp>(loc);
  rewriter.setInsertionPointToStart(body);
  return launch;
}

/// Returns the induction variable index `idx` of gpu.launch that should be used
/// for the given gml_st.parallel's `nestingLevel`.
static LogicalResult getInductionVarIdxForLevel(unsigned nestingLevel,
                                                unsigned& idx) {
  constexpr std::array<unsigned, 3> kNestingToLaunchIdx{
      0,  // block IDs map to blockIdx.x
      4,  // warp IDs map to threadIdx.y
      3,  // thread IDs map to threadIdx.x
  };
  if (nestingLevel >= kNestingToLaunchIdx.size()) return failure();
  idx = kNestingToLaunchIdx[nestingLevel];
  return success();
}

/// Verifies that the loop bounds of `currentBound` (which is a result of
/// affine.apply) are the same ones as the bounds of the `parallel` op.
static LogicalResult verifyLoopBoundsMatch(Value currentBound,
                                           ParallelOp parallel) {
  auto applyOp = currentBound.getDefiningOp<AffineApplyOp>();
  assert(applyOp && "inferred bounds should be expressed as affine.apply");
  OperandRange operands = applyOp.getMapOperands();
  assert(operands.size() == 3 &&
         "affine map expressing the launch bound should have three operands");
  return success(operands[0] == parallel.getUpperBound().front() &&
                 operands[1] == parallel.getLowerBound().front() &&
                 operands[2] == parallel.getStep().front());
}

/// Emits code that infers and returns an iteration-independent version of
/// `upperBound` in cases when the tiling size does not evenly divide the
/// problem size.
///
/// In these cases, `upperBound` depends on other values within the `launch`
/// region, so it cannot be used to infer the launch bounds of `launch`. This
/// function returns an approximation of `upperBound` that does not depend on
/// such values, and emmits code that masks off extra threads (identified by
/// `inductionVar`) that result from using the approximated value.
static Value handleImperfectTile(Location loc, LaunchOp launch,
                                 Value upperBound, Value inductionVar,
                                 PatternRewriter& rewriter) {
  // We are assuming that imperfect tiling is expressed through an affine.min
  // op with an affine map of the form (<something>)[<something>] ->
  // (<something>, tileSize), where <something>s possibly depend on values
  // defined within the regions of nested gml_st.parallel. Since local values
  // are not available outside of the loops, which is needed for launch bounds
  // computation, we only use tileSize to compte the launch bounds. We then mask
  // off the threads that would be computing out-of-bound values.
  // TODO(b/244314345): Replace this pattern matching with a proper way to
  // handle imperfect tiling once we figure this out.
  auto affineMin = upperBound.getDefiningOp<AffineMinOp>();
  if (!affineMin || affineMin.getMap().getNumResults() != 2) return upperBound;
  auto tileSize =
      affineMin.getMap().getResult(1).dyn_cast<AffineConstantExpr>();
  if (!tileSize) return upperBound;

  // Insert a guard in the region to mask off threads that would operate outside
  // the tile bounds.
  auto predicate = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, inductionVar, upperBound);
  auto scfIf =
      rewriter.create<scf::IfOp>(loc, predicate, /*withElseRegion=*/false);
  rewriter.setInsertionPointToStart(&scfIf.getThenRegion().front());

  // Create a constant corresponding to the tile size, and return it as the
  // iteration-independent upper bound.
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(launch);
  return rewriter.create<arith::ConstantIndexOp>(loc, tileSize.getValue());
}

/// Matches the `launchIdx`-th iteration space of `launch` to the iteration
/// space of `parallel`. Returns an SSA value that is a part of the `launch`'s
/// region, and represents the value of `parallel`'s induction variable.
static Value matchLaunchSpaceToLoop(ParallelOp parallel,
                                    const BlockAndValueMapping& bvm,
                                    unsigned launchIdx, LaunchOp launch,
                                    PatternRewriter& rewriter) {
  Location loc = parallel.getLoc();
  Value upperBound = bvm.lookupOrDefault(parallel.getUpperBound().front());
  Value lowerBound = bvm.lookupOrDefault(parallel.getLowerBound().front());
  Value step = bvm.lookupOrDefault(parallel.getStep().front());

  // Compute the value that gml_st.parallel's induction variable would have in
  // each iteration, and make it available to operations within the gpu.launch
  // region.
  AffineMap inductionVarMap = AffineMap::get(
      /*dimCount=*/1, /*symbolCount=*/2,
      rewriter.getAffineDimExpr(0) * rewriter.getAffineSymbolExpr(1) +
          rewriter.getAffineSymbolExpr(0));
  Value inductionVar = rewriter.create<AffineApplyOp>(
      loc, inductionVarMap,
      ValueRange{launch.getBody().getArgument(launchIdx), lowerBound, step});

  // Infer the launch bound from the loop bounds and the step.
  Value iterIndependentUpperBound =
      handleImperfectTile(loc, launch, upperBound, inductionVar, rewriter);
  AffineMap launchBoundMap = AffineMap::get(
      /*dimCount=*/1, /*symbolCount=*/2,
      (rewriter.getAffineDimExpr(0) - rewriter.getAffineSymbolExpr(0))
          .ceilDiv(rewriter.getAffineSymbolExpr(1)));
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(launch);
  launch.setOperand(
      launchIdx, rewriter.create<AffineApplyOp>(
                     loc, launchBoundMap,
                     ValueRange{iterIndependentUpperBound, lowerBound, step}));
  return inductionVar;
}

// Converts the 3 nested gml_st.parallel ops rooted at `root` into a
// gpu.launch op. We do this by creating an empty gpu.launch region and
// copying all the operations in gml_st.parallel into that region,
// recursively copying the bodies of any nested gml_st.parallel regions that
// we encounter.
LogicalResult ParallelOpToGpuPattern::matchAndRewrite(
    ParallelOp root, PatternRewriter& rewriter) const {
  Location loc = root.getLoc();

  if (root->getParentOfType<ParallelOp>())
    return rewriter.notifyMatchFailure(root, "should be the root parallel");

  Value defaultSize = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  LaunchOp launch = createInitialGpuLaunchOp(loc, defaultSize, rewriter);

  BlockAndValueMapping bvm;
  // We need to keep track of which value in the gpu.launch region represents
  // which level of the induction variable in the nested region. This is because
  // we might have multiple gml_st.parallel operations on the same level, and
  // their induction variables should map to the same value in the flattened
  // gpu.launch region.
  SmallVector<Value, 3> nestingLevelToInductionVarMap;
  // This is our stack holding in-flight operations of gml_st.parallel regions
  // that we started to copy over to the gpu.launch region, but are on hold
  // while we are processing a nested gml_st.parallel.
  SmallVector<iterator_range<Block::iterator>, 3> loopIterators;

  // This functor implements the processing of a single parallel op:
  // 1)  update of GPU launch bounds according to the interation space
  // 2)  addition of a nesting level to `loopIterators`, with the iterator
  //     over `parallel`'s body
  auto processParallelOp = [&](ParallelOp parallel) {
    unsigned nestingLevel = loopIterators.size();
    unsigned inductionVarIdx = 0;
    if (failed(getInductionVarIdxForLevel(nestingLevel, inductionVarIdx)))
      return rewriter.notifyMatchFailure(parallel, "is nested too deeply");
    if (parallel.getNumLoops() != 1) {
      return rewriter.notifyMatchFailure(
          parallel, "should only have a single induction variable");
    }

    Value currentBound = launch.getOperand(inductionVarIdx);
    if (nestingLevel < nestingLevelToInductionVarMap.size()) {
      // We already inferred the launch bound for this nesting level.
      if (failed(verifyLoopBoundsMatch(currentBound, parallel))) {
        return rewriter.notifyMatchFailure(
            parallel,
            "should have the same iteration space as other parallel operations "
            "on the same nesting level");
      }
    } else {
      // We are encountering a loop at this level of nesting for the first time.
      assert(currentBound == defaultSize &&
             "launch bound should use the default size");
      nestingLevelToInductionVarMap.push_back(matchLaunchSpaceToLoop(
          parallel, bvm, inductionVarIdx, launch, rewriter));
    }

    bvm.map(parallel.getInductionVars().front(),
            nestingLevelToInductionVarMap[nestingLevel]);
    Block* body = parallel.getBody();
    loopIterators.push_back(llvm::make_range(body->begin(), body->end()));
    return success();
  };

  if (failed(processParallelOp(root))) return failure();

  while (!loopIterators.empty()) {
    auto currentLoop = loopIterators.pop_back_val();
    for (Operation& op : currentLoop) {
      if (auto nestedParallel = dyn_cast<ParallelOp>(&op)) {
        // Push the current state back to loopIterator and start the next level
        // of nesting.
        loopIterators.push_back(
            llvm::make_range(std::next(op.getIterator()), currentLoop.end()));
        if (failed(processParallelOp(nestedParallel))) return failure();
        break;
      }
      if (auto setYield = dyn_cast<SetYieldOp>(&op)) {
        // convert setYield into distribute
        auto parallelOp = setYield->getParentOfType<ParallelOp>();
        assert(parallelOp &&
               "gml_st.set_yield should have a parent gml_st.parallel op");
        for (auto [result, src, set] :
             llvm::zip(parallelOp.getResults(), setYield.getSrcs(),
                       setYield.getSets())) {
          bvm.map(result,
                  rewriter.create<DistributeOp>(op.getLoc(), result.getType(),
                                                bvm.lookupOrDefault(src),
                                                bvm.lookupOrDefault(set)));
        }
        continue;
      }
      // TODO(b/244314146): Figure out what we need to do for operations
      // encountered on upper nesting levels to correctly lower them after the
      // rewrite to gpu.launch.
      Operation* clone = rewriter.clone(op, bvm);
      bvm.map(op.getResults(), clone->getResults());
    }
  }

  rewriter.eraseOp(root);
  return success();
}

static Value createCombineOp(Location loc, Value lhs, Value rhs,
                             CombiningKind kind, PatternRewriter& rewriter) {
  auto helper = [&](auto dummy) {
    return rewriter.create<decltype(dummy)>(loc, lhs, rhs);
  };
  switch (kind) {
    case CombiningKind::ADD:
      return helper(arith::AddFOp());
    case CombiningKind::MUL:
      return helper(arith::MulFOp());
    case CombiningKind::MINUI:
      return helper(arith::MinUIOp());
    case CombiningKind::MINSI:
      return helper(arith::MinSIOp());
    case CombiningKind::MINF:
      return helper(arith::MinFOp());
    case CombiningKind::MAXUI:
      return helper(arith::MaxUIOp());
    case CombiningKind::MAXSI:
      return helper(arith::MaxSIOp());
    case CombiningKind::MAXF:
      return helper(arith::MaxFOp());
    case CombiningKind::AND:
      return helper(arith::AndIOp());
    case CombiningKind::OR:
      return helper(arith::OrIOp());
    case CombiningKind::XOR:
      return helper(arith::XOrIOp());
    default:
      llvm_unreachable("unhandled");
  }
}

LogicalResult MultiDimReductionOpToWarpReductionPattern::matchAndRewrite(
    MultiDimReductionOp reductionOp, PatternRewriter& rewriter) const {
  auto inType = reductionOp.getSourceVectorType();
  auto outType = reductionOp.getDestType().dyn_cast<VectorType>();
  auto isNumElementsEqual = [](auto type, int64_t size) {
    return type && type.getNumElements() == size;
  };
  if (!isNumElementsEqual(inType, 32)) {
    return rewriter.notifyMatchFailure(reductionOp, "expected 32-vector input");
  }
  if (!isNumElementsEqual(outType, 1)) {
    return rewriter.notifyMatchFailure(reductionOp, "expected 1-vector output");
  }
  auto distribute = reductionOp.getSource().getDefiningOp<DistributeOp>();
  if (!distribute) {
    return rewriter.notifyMatchFailure(
        reductionOp, "source not defined by gml_st.distribute");
  }
  // Even if this value was not written into the tile corresponding to the
  // current thread's lane id, this is fine, since it doesn't matter which
  // thread processes which element within a reduction.
  TypedValue<VectorType> lhsVector = distribute.getSource();
  if (!isNumElementsEqual(lhsVector.getType(), 1)) {
    return rewriter.notifyMatchFailure(distribute, "expected 1-vector input");
  }

  // Preamble: extract element from input
  Location loc = reductionOp->getLoc();
  Value lhs = rewriter.create<ExtractOp>(
      loc, lhsVector, SmallVector<int64_t>(lhsVector.getType().getRank(), 0));

  auto createConstant = [&](int32_t value) {
    return rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(value));
  };
  Value width = createConstant(32);
  // Create warp shuffles of increasing offset and interleave with a clone of
  // the accumulate block.
  for (int i = 1; i < 32; i *= 2) {
    auto shuffleOp = rewriter.create<gpu::ShuffleOp>(
        loc, lhs, createConstant(i), width, gpu::ShuffleMode::XOR);
    lhs = createCombineOp(loc, lhs, shuffleOp.getShuffleResult(),
                          reductionOp.getKind(), rewriter);
  }

  // Combine with init element and broadcast result back to vector.
  Value acc = rewriter.create<ExtractOp>(loc, reductionOp.getAcc(), 0);
  lhs = createCombineOp(loc, lhs, acc, reductionOp.getKind(), rewriter);
  rewriter.replaceOpWithNewOp<vector::BroadcastOp>(reductionOp, outType, lhs);

  return success();
}

SubViewOp createSubView(Location loc, Value source, TileOp tile,
                        PatternRewriter& rewriter) {
  auto asIntArray = [](ArrayAttr array) {
    return llvm::to_vector(llvm::map_range(array, [](Attribute attr) {
      return attr.cast<IntegerAttr>().getInt();
    }));
  };
  Type memRefType = SubViewOp::inferResultType(
      source.getType().cast<MemRefType>(), asIntArray(tile.getStaticOffsets()),
      asIntArray(tile.getStaticSizes()), asIntArray(tile.getStaticStrides()));
  return rewriter.create<SubViewOp>(
      loc, memRefType, source, tile.getOffsets(), tile.getSizes(),
      tile.getStrides(), tile.getStaticOffsets(), tile.getStaticSizes(),
      tile.getStaticStrides());
}

LogicalResult EliminateMaterializeOfTransferReadPattern::matchAndRewrite(
    MaterializeOp materialize, PatternRewriter& rewriter) const {
  // Match the following pattern:
  //  gml_st.materialize(
  //  vector.transfer_read Memref:$src[(arith.constant 0)...]
  //  gml_st.tile [$offsets] [$sizes] [$strides])
  auto transferRead = materialize.getSource().getDefiningOp<TransferReadOp>();
  if (!transferRead) {
    return rewriter.notifyMatchFailure(
        materialize, "expected vector.transfer_read as source");
  }
  Value source = transferRead.getSource();
  if (!source.getType().isa<MemRefType>()) {
    return rewriter.notifyMatchFailure(transferRead,
                                       "expected memref as source");
  }
  if (failed(matchSimpleTransferOp(transferRead, rewriter))) return failure();

  auto tile = materialize.getSet().getDefiningOp<TileOp>();
  if (!tile) {
    return rewriter.notifyMatchFailure(materialize,
                                       "expected gml_st.tile as set");
  }

  // Rewrite the pattern as:
  // vector.transfer_read
  //   (memref.subview $src [$offsets] [$sizes] [$strides])
  //   [(arith.constant 0)...]
  // TODO(b/254271932): This might not be correct if there is someone writing
  // to `source` in between `transferRead` and `materialize`. This won't happen
  // for elementwise fusion and softmax, but might become a problem down the
  // line.
  auto subview = createSubView(materialize.getLoc(), source, tile, rewriter);
  rewriter.replaceOpWithNewOp<TransferReadOp>(
      materialize, materialize.getResult().getType().cast<VectorType>(),
      subview, transferRead.getIndices(), transferRead.getPermutationMap(),
      transferRead.getPadding(),
      /*mask=*/nullptr, transferRead.getInBounds().value_or(nullptr));
  return success();
}

LogicalResult EliminateDistributeIntoTransferWritePattern::matchAndRewrite(
    TransferWriteOp transferWrite, PatternRewriter& rewriter) const {
  // Match the following pattern:
  //  vector.transfer_write
  //    (gml_st.distribute $src into
  //      [(gml_st.tile [$offsets] [$sizes] [$strides])])
  //    Memref:$dst[(arith.constant 0)]
  Value destination = transferWrite.getSource();
  if (!destination.getType().isa<MemRefType>()) {
    return rewriter.notifyMatchFailure(transferWrite,
                                       "expected memref as destination");
  }
  if (failed(matchSimpleTransferOp(transferWrite, rewriter))) return failure();

  auto distribute = transferWrite.getVector().getDefiningOp<DistributeOp>();
  if (!distribute) {
    return rewriter.notifyMatchFailure(transferWrite,
                                       "expected distribute as source");
  }
  Value source = distribute.getSource();

  auto tile = distribute.getSet().getDefiningOp<TileOp>();
  if (!tile) {
    return rewriter.notifyMatchFailure(distribute,
                                       "expected gml_st.tile as set");
  }

  // Rewrite the pattern as:
  // vector.transfer_write $src,
  //   (memref.subview $dst [$offsets] [$sizes] [$strides])
  //   [(arith.constant 0)...]
  auto subview =
      createSubView(transferWrite.getLoc(), destination, tile, rewriter);
  rewriter.replaceOpWithNewOp<TransferWriteOp>(
      transferWrite, /*resultType=*/llvm::None, source, subview,
      transferWrite.getIndices(), transferWrite.getPermutationMap(),
      /*mask=*/nullptr, transferWrite.getInBounds().value_or(nullptr));
  return success();
}
