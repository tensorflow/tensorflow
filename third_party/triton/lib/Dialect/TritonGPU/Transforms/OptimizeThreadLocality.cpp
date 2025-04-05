#include <memory>
#include <numeric>

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUOPTIMIZETHREADLOCALITY
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {
// Change the destination layout of reshape ops allowing reorder when used by a
// reduction in order to minimize the amount of cross thread communication for
// the reduction.
struct OptimizeReshapeLayoutPattern : public OpRewritePattern<ReshapeOp> {
  OptimizeReshapeLayoutPattern(MLIRContext *context)
      : OpRewritePattern<ReshapeOp>(context, 1) {}

  LogicalResult matchAndRewrite(ReshapeOp viewOp,
                                PatternRewriter &rewriter) const override {
    if (!viewOp.getAllowReorder())
      return failure();
    std::optional<int> reductionAxis;
    for (Operation *user : viewOp.getResult().getUsers()) {
      if (auto reduceOp = dyn_cast<triton::ReduceOp>(user)) {
        if (reductionAxis) {
          if (reductionAxis != reduceOp.getAxis())
            return failure();
        } else {
          reductionAxis = reduceOp.getAxis();
        }
      }
    }
    if (!reductionAxis)
      return failure();
    RankedTensorType tensorType = viewOp.getType();
    if (auto blocked =
            mlir::dyn_cast<BlockedEncodingAttr>(tensorType.getEncoding())) {
      // If the layout already has all the elements along the reduction
      // dimension in the same thread we can skip.
      if (blocked.getThreadsPerWarp()[*reductionAxis] == 1 &&
          blocked.getWarpsPerCTA()[*reductionAxis] == 1 &&
          blocked.getCTAsPerCGA()[*reductionAxis] == 1)
        return failure();
    }
    ArrayRef<int64_t> shape = tensorType.getShape();
    SmallVector<unsigned> order;
    for (int i : triton::gpu::getOrder(tensorType)) {
      if (i != *reductionAxis)
        order.push_back(i);
    }
    // Make the reduction axis last so that elements won't be distributed
    // amongst threads along this dimension.
    order.push_back(*reductionAxis);
    SmallVector<unsigned> sizePerThread(shape.size(), 1);
    auto mod = viewOp->getParentOfType<ModuleOp>();
    int numWarps = lookupNumWarps(viewOp);
    int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
    int numCTAs = TritonGPUDialect::getNumCTAs(mod);
    auto encoding =
        BlockedEncodingAttr::get(viewOp.getContext(), shape, sizePerThread,
                                 order, numWarps, threadsPerWarp, numCTAs);
    if (encoding == tensorType.getEncoding())
      return failure();
    RankedTensorType newType =
        RankedTensorType::get(shape, tensorType.getElementType(), encoding);
    if (triton::gpu::isExpensiveView(viewOp.getSrc().getType(), newType))
      return failure();
    rewriter.setInsertionPointAfter(viewOp);
    rewriter.modifyOpInPlace(viewOp, [&]() {
      viewOp.getResult().setType(newType);
      viewOp.setEfficientLayout(true);
    });
    auto cvt = rewriter.create<ConvertLayoutOp>(viewOp.getLoc(), tensorType,
                                                viewOp.getResult());
    rewriter.replaceAllUsesExcept(viewOp.getResult(), cvt.getResult(), cvt);
    return success();
  }
};
} // namespace

static RankedTensorType replaceEncoding(RankedTensorType oldType,
                                        Attribute newEncoding) {
  return RankedTensorType::get(oldType.getShape(), oldType.getElementType(),
                               newEncoding);
}

// This function considers a gather op in isolation and attempts to determine
// whether an optimized layout can be applied to the source and index tensors.
static void setOptimizedGatherLayout(GatherOp op, RewriterBase &b) {
  RankedTensorType srcType = op.getSrc().getType();
  RankedTensorType idxType = op.getIndices().getType();

  // Determine a warp-local gather layout that minimizes the number of emitted
  // warp shuffles.
  unsigned numThreadsPerWarp = lookupThreadsPerWarp(b);
  unsigned numWarps = lookupNumWarps(op);

  // If in a gather column, each thread owns `srcSizePerThread[axis]` elements
  // in the source tensor and `idxSizePerThread[axis]` elements in the index
  // tensor (including broadcasting), then the number of index shuffles per
  // column is `srcSizePerThread[axis] * idxSizePerThread[axis]`. This is then
  // replicated over the number of columns in which a thread owns (an equal
  // number of) elements, which is `product(srcSizePerThread[i] for i != axis)`.
  //
  // Thus, the total number of index shuffles is `product(srcSizePerThread) *
  // idxSizePerThread[axis]`. Since we cannot alter the number of threads per
  // warp or the number of warps, `product(srcSizePerThread)` is just a function
  // of the shape.
  //
  // So we want to minimize `idxSizePerThread[axis]`. Note that broadcasting is
  // forbidden in the source tensor but allowed in the index tensor. Choose the
  // smallest value while still ensuring that a warp spans whole columns.
  //
  // In order to prevent broadcasting in the source tensor layout, ensure
  //
  //   sizePerThread(i) * threadsPerWarp(i) * warpsPerCTA(i) = shape(i)
  //
  // For all i != axis in the source tensor. The same relationship must hold for
  // the index tensor. This means we can't just set `idxSizePerThread[axis]` to
  // 1 and compute the rest from that. Find the smallest value where this
  // relationship is still respected.

  // We know that the layouts will be the same between the two tensors except
  // for `sizePerThread[axis]`.
  unsigned axis = op.getAxis();
  unsigned rank = srcType.getRank();
  SmallVector<unsigned> threadsPerWarp(rank);
  SmallVector<unsigned> warpsPerCTA(rank);
  SmallVector<unsigned> order;
  order.push_back(axis);

  // Minimize `sizePerThread[axis]` by putting as many theads along the axis as
  // possible, limited to the actual size of the dimension.
  unsigned maxThreadsInAxis =
      std::min<unsigned>(srcType.getDimSize(axis), numThreadsPerWarp);
  threadsPerWarp[axis] = maxThreadsInAxis;

  // Now spread them along the other dimensions. Do this according to order
  // (arbitrary).
  unsigned threadsToAlloc = numThreadsPerWarp / maxThreadsInAxis;
  for (unsigned dim : getThreadOrder(srcType)) {
    if (dim == axis)
      continue;
    // The gather axis is now the fastest-changing dimension.
    order.push_back(dim);
    unsigned nextThreadAlloc =
        std::min<unsigned>(srcType.getDimSize(dim), threadsToAlloc);
    threadsPerWarp[dim] = nextThreadAlloc;
    threadsToAlloc /= nextThreadAlloc;
  }
  assert(llvm::none_of(threadsPerWarp, [](unsigned c) { return c == 0; }));

  // There must be one warp along the gather axis.
  warpsPerCTA[axis] = 1;
  // Allocate the remaining warps in the same manner.
  unsigned warpsToAlloc = numWarps;
  for (unsigned dim : getWarpOrder(srcType)) {
    if (dim == axis)
      continue;
    unsigned warpsCanFit = srcType.getDimSize(dim) / threadsPerWarp[dim];
    assert(warpsCanFit != 0);
    unsigned nextWarpAlloc = std::min<unsigned>(warpsCanFit, warpsToAlloc);
    warpsPerCTA[dim] = nextWarpAlloc;
    warpsToAlloc /= nextWarpAlloc;
  }
  assert(llvm::none_of(warpsPerCTA, [](unsigned c) { return c == 0; }));

  // Just set `sizePerThread` to 1 along other dimensions and let broadcasting
  // handling it. This also means we can use the same layout between the source
  // and index tensors for simplicity.
  SmallVector<unsigned> sizePerThread(rank, 1);
  sizePerThread[axis] = srcType.getDimSize(axis) / threadsPerWarp[axis];

  // Overflow by broadcasting along the gather axis since this is the most
  // predictable.
  threadsPerWarp[axis] *= threadsToAlloc;
  warpsPerCTA[axis] *= warpsToAlloc;

  assert(product(threadsPerWarp) == numThreadsPerWarp);
  assert(product(warpsPerCTA) == numWarps);

  // Construct the new layout.
  MLIRContext *ctx = srcType.getContext();
  auto baseLayout = cast<LayoutEncodingTrait>(srcType.getEncoding());
  auto ctaLayout =
      CTALayoutAttr::get(ctx, baseLayout.getCTAsPerCGA(),
                         baseLayout.getCTASplitNum(), baseLayout.getCTAOrder());
  auto newLayout = BlockedEncodingAttr::get(ctx, sizePerThread, threadsPerWarp,
                                            warpsPerCTA, order, ctaLayout);

  // Update the layout on the gather op and insert conversions.
  auto cvtSrc = b.create<ConvertLayoutOp>(
      op.getLoc(), replaceEncoding(srcType, newLayout), op.getSrc());
  auto cvtIdx = b.create<ConvertLayoutOp>(
      op.getLoc(), replaceEncoding(idxType, newLayout), op.getIndices());

  b.setInsertionPointAfter(op);
  auto cvtOut =
      b.create<ConvertLayoutOp>(op.getLoc(), op.getType(), op.getResult());
  b.replaceAllUsesExcept(op.getResult(), cvtOut, cvtOut);

  b.modifyOpInPlace(op, [&] {
    op.getSrcMutable().set(cvtSrc);
    op.getIndicesMutable().set(cvtIdx);
    op.getResult().setType(replaceEncoding(op.getType(), newLayout));

    // Mark the layout as optimized on the op to prevent it from being changed.
    op.setEfficientLayout(true);
  });

  // Make sure we did this right.
  assert(GatherLoweringHelper(op).isWarpLocal());
}

namespace {
struct OptimizeGatherLayoutPattern : public mlir::OpRewritePattern<GatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getEfficientLayout())
      return failure();
    setOptimizedGatherLayout(op, rewriter);
    return success();
  }
};
} // namespace

namespace {
class TritonGPUOptimizeThreadLocalityPass
    : public impl::TritonGPUOptimizeThreadLocalityBase<
          TritonGPUOptimizeThreadLocalityPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // First try to optimize the layout of views and gathers.
    mlir::RewritePatternSet layoutPatterns(&getContext());
    layoutPatterns.add<OptimizeReshapeLayoutPattern>(&getContext());
    layoutPatterns.add<OptimizeGatherLayoutPattern>(&getContext());
    if (mlir::applyPatternsGreedily(mod, std::move(layoutPatterns)).failed()) {
      signalPassFailure();
    }

    DenseSet<triton::ReduceOp> reduceOps;
    mod.walk([&](triton::ReduceOp reduce) -> void {
      auto srcType = cast<RankedTensorType>(reduce.getOperands()[0].getType());
      auto rank = srcType.getShape().size();
      auto srcEncoding = srcType.getEncoding();
      auto reductionOp = getReductionOp(reduce);
      if (!reductionOp ||
          !isa<arith::AddFOp, arith::MulFOp, arith::MaximumFOp,
               arith::MaxNumFOp, arith::MinimumFOp, arith::MinNumFOp>(
              reductionOp.value()))
        return;
      // TODO: relax this restriction
      if (!(isa<triton::gpu::BlockedEncodingAttr>(srcEncoding) && rank > 1))
        return;
      // The code currently assumes that the reduction is happening on the most
      // inner dim.
      if (reduce.getAxis() != rank - 1)
        return;
      for (auto operand : reduce->getOperands()) {
        if (!operand.getDefiningOp<triton::LoadOp>())
          return;
      }
      auto elemsPerThread =
          triton::gpu::getElemsPerThread(srcType)[reduce.getAxis()];
      // Not worth applying this optimization if there is only one element per
      // thread on the reduction axis
      if (elemsPerThread == 1)
        return;
      if (!reduce->hasOneUse())
        return;
      Operation *user = *(reduce->getUsers().begin());
      if (!user->hasOneUse())
        return;
      OpOperand &yieldOpOperand = *(user->getUses().begin());
      auto yieldOp = dyn_cast<scf::YieldOp>(yieldOpOperand.getOwner());
      if (!yieldOp)
        return;
      auto operandNumber = yieldOpOperand.getOperandNumber();
      Block *block = reduce->getBlock();
      Operation *parentOp = block->getParentOp();
      auto forOp = dyn_cast<scf::ForOp>(parentOp);
      if (!forOp)
        return;
      auto argNum = yieldOpOperand.getOperandNumber();
      auto oldAccum = forOp.getInitArgs()[argNum];
      auto cstOp = oldAccum.getDefiningOp<arith::ConstantOp>();
      if (!cstOp)
        return;
      reduceOps.insert(reduce);
    });

    IRRewriter builder(&getContext());
    for (auto reduce : reduceOps) {
      builder.setInsertionPoint(reduce);
      auto srcType = cast<RankedTensorType>(reduce.getOperands()[0].getType());
      auto srcShape = srcType.getShape();
      auto srcEncoding = srcType.getEncoding();
      assert(isa<triton::gpu::BlockedEncodingAttr>(srcEncoding) &&
             "Thread locality optimization only supports blocked encoding");
      auto blocked = dyn_cast<triton::gpu::BlockedEncodingAttr>(srcEncoding);
      auto elemsPerThread =
          triton::gpu::getElemsPerThread(srcType)[reduce.getAxis()];
      auto rank = srcShape.size();
      // create new layouts
      auto blocked3d = getThreadLocalityOptimizedEncoding(reduce);
      auto viewOpTensorShape = getThreadLocalityOptimizedShape(reduce);
      auto viewOpTensorType = RankedTensorType::get(
          viewOpTensorShape, srcType.getElementType(), blocked3d);
      auto slice2d = triton::gpu::SliceEncodingAttr::get(mod.getContext(), rank,
                                                         blocked3d);
      // Get forOp
      assert(reduce->hasOneUse());
      OpOperand &use = *(reduce->getUses().begin());
      auto operandNumber = use.getOperandNumber();
      auto oldUpdate = use.getOwner();
      assert(oldUpdate->getNumOperands() == 2);
      auto accumOperandNumber = (operandNumber == 0) ? 1 : 0;
      auto accumOperand = oldUpdate->getOperand(accumOperandNumber);
      assert(isa<BlockArgument>(accumOperand));
      auto blockArg = dyn_cast<BlockArgument>(accumOperand);
      auto blockArgNum = blockArg.getArgNumber();
      auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
      // get oldAccum
      auto oldAccum =
          forOp.getInitArgs()[blockArgNum - forOp.getNumInductionVars()];
      // get old loop user
      Value loopResult =
          forOp.getResult(blockArgNum - forOp.getNumInductionVars());
      assert(loopResult.hasOneUse());
      OpOperand &loopUse = *(loopResult.getUses().begin());
      Operation *loopUser = loopUse.getOwner();
      // get old loop yield
      auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      // create newAccum initialization
      auto newAccum =
          createAccum(builder, reduce, oldAccum, viewOpTensorShape, slice2d);
      // create new loop by copying the old for op signature and appending
      // newAccum to the block arguments
      auto newLoop = replaceForOpWithNewSignature(
          builder, forOp, ValueRange{newAccum->getResult(0)});
      // create thread local reduction (also adds viewOps)
      auto newReduce = createReduce(builder, reduce, viewOpTensorType);

      // create new accum update
      auto newUpdate = createUpdate(builder, newLoop, newReduce, oldUpdate);
      // create new yield
      auto newYield = createYield(builder, newLoop, oldYield,
                                  newUpdate->getResult(0), blockArgNum);
      // create post loop reduction on the original reduce axis
      auto newReduce2 = createPostLoopReduce(builder, newLoop, reduce);
      // add convert_layout to get back to original layout, the result layout
      // should now match the layout of the old accumulator (%cst)
      Type destType = loopResult.getType();
      auto cvtLayout = createConvertLayout(builder, destType, newReduce2);
      // incorporate the original accumulator value into the final result
      auto finalOp = incorporateOriginalAccumulatorValue(builder, oldUpdate,
                                                         cvtLayout, oldAccum);
      // Replace the old loop user with the final result
      loopUser->setOperand(loopUse.getOperandNumber(), finalOp->getResult(0));

      // cleanup
      oldYield.erase();
      forOp.erase();
    }
  };

private:
  std::optional<Operation *> getReductionOp(triton::ReduceOp reduce) const {
    auto numRegions = reduce->getNumRegions();
    if (numRegions != 1)
      return std::nullopt;
    Region &region = reduce->getRegion(0);
    auto numBlocks = region.getBlocks().size();
    if (numBlocks != 1)
      return std::nullopt;
    Block &block = region.front();
    auto blockWithoutTerminator = block.without_terminator();
    auto blockSizeWithoutTerminator = std::distance(
        blockWithoutTerminator.begin(), blockWithoutTerminator.end());
    if (blockSizeWithoutTerminator != 1)
      return std::nullopt;
    Operation *op = &block.front();
    return std::optional<Operation *>(op);
  }
  Operation *incorporateOriginalAccumulatorValue(OpBuilder &builder,
                                                 Operation *oldUpdate,
                                                 Operation *cvtLayout,
                                                 Value oldAccum) const {
    builder.setInsertionPointAfter(cvtLayout);
    IRMapping mapping;
    mapping.map(oldUpdate->getOperand(0), oldAccum);
    mapping.map(oldUpdate->getOperand(1), cvtLayout->getResult(0));
    auto finalOp = cloneWithInferType(builder, &(*oldUpdate), mapping);
    return finalOp;
  }
  Operation *createConvertLayout(OpBuilder &builder, Type destType,
                                 Operation *newReduce) const {
    builder.setInsertionPointAfter(newReduce);
    auto newCvt = builder.create<triton::gpu::ConvertLayoutOp>(
        newReduce->getLoc(), destType, newReduce->getResult(0));
    return newCvt;
  }

  Operation *createPostLoopReduce(OpBuilder &builder, scf::ForOp &loop,
                                  triton::ReduceOp &reduce) const {
    auto resultIndex =
        loop.getBody()->getNumArguments() - 1 - loop.getNumInductionVars();
    auto newLoopResult = loop.getResult(resultIndex);
    builder.setInsertionPointAfter(loop);
    IRMapping mapping;
    mapping.map(*(reduce.getOperands().begin()), newLoopResult);
    auto newReduce2 = cloneWithInferType(builder, &(*reduce), mapping);
    return newReduce2;
  }

  Operation *createYield(OpBuilder &builder, scf::ForOp &loop,
                         scf::YieldOp &oldYield, Value newUpdate,
                         int oldAccumBlockArgNum) const {
    builder.setInsertionPoint(oldYield);
    SmallVector<Value> yieldValues = llvm::to_vector(oldYield.getOperands());
    yieldValues[oldAccumBlockArgNum - 1] =
        loop.getBody()->getArgument(oldAccumBlockArgNum);
    yieldValues.push_back(newUpdate);
    auto newYield =
        builder.create<scf::YieldOp>(oldYield.getLoc(), yieldValues);
    return newYield;
  }

  Operation *createUpdate(OpBuilder &builder, scf::ForOp &loop,
                          Operation *newReduce, Operation *oldUpdate) const {
    auto blockArgNum = loop.getBody()->getNumArguments() - 1;
    auto newArg = loop.getBody()->getArgument(blockArgNum);
    builder.setInsertionPointAfter(newReduce);
    IRMapping mapping;
    mapping.map(oldUpdate->getOperand(0), newArg);
    mapping.map(oldUpdate->getOperand(1), newReduce->getResult(0));
    auto newUpdate = cloneWithInferType(builder, oldUpdate, mapping);
    return newUpdate;
  }

  Operation *createReduce(OpBuilder &builder, triton::ReduceOp reduce,
                          Type viewOpTensorType) const {
    auto srcType = cast<RankedTensorType>(reduce.getOperands()[0].getType());
    auto rank = srcType.getShape().size();
    builder.setInsertionPointAfter(reduce);
    IRMapping mapping;
    for (auto operand : reduce.getOperands()) {
      auto viewOp = builder.create<triton::ReshapeOp>(
          reduce.getLoc(), viewOpTensorType, operand,
          /*allowReorder=*/true, /*efficientLayout=*/true);
      mapping.map(operand, viewOp);
    }

    auto newReduce = cloneWithInferType(builder, &(*reduce), mapping);
    newReduce->setAttr("axis", builder.getI32IntegerAttr(rank));
    auto typeInfer = dyn_cast<InferTypeOpInterface>(newReduce);
    if (typeInfer) {
      SmallVector<Type, 1> newTypes;
      auto success = typeInfer.inferReturnTypes(
          newReduce->getContext(), newReduce->getLoc(),
          newReduce->getOperands(), newReduce->getAttrDictionary(),
          newReduce->getPropertiesStorage(), newReduce->getRegions(), newTypes);
      if (succeeded(success)) {
        for (size_t i = 0; i < newTypes.size(); i++)
          newReduce->getResult(i).setType(newTypes[i]);
      }
    }
    return newReduce;
  }

  // Work around the lack of support for MaxNumFOp and MinNumFOp in
  // arith::getNeutralElement.
  std::optional<TypedAttr> getNeutralElement(Operation *op) const {
    if (isa<arith::MaxNumFOp, arith::MinNumFOp>(op)) {
      OpBuilder builder(op->getContext());

      Type resultType = op->getResult(0).getType();
      const llvm::fltSemantics &semantic =
          llvm::cast<FloatType>(resultType).getFloatSemantics();
      if (isa<arith::MaxNumFOp>(op)) {
        return builder.getFloatAttr(
            resultType, APFloat::getInf(semantic, /*Negative=*/true));
      }
      if (isa<arith::MinNumFOp>(op)) {
        return builder.getFloatAttr(
            resultType, APFloat::getInf(semantic, /*Negative=*/false));
      }
    } else {
      return mlir::arith::getNeutralElement(op);
    }
    llvm_unreachable("Unhandled reduction op");
    return std::nullopt;
  }

  Operation *createAccum(OpBuilder &builder, triton::ReduceOp reduce,
                         Value &oldAccum, SmallVector<int64_t> &shape,
                         Attribute &slice2d) const {
    // Drop the last dimension (thread locality dimension)
    SmallVector<int64_t> accumShape(shape.begin(), shape.end() - 1);
    auto elemType = cast<RankedTensorType>(oldAccum.getType()).getElementType();
    // Create tensor type for the new accumulator
    auto accumType = RankedTensorType::get(accumShape, elemType, slice2d);
    // Create new accumulator
    builder.setInsertionPointAfter(oldAccum.getDefiningOp());
    auto reductionOp = getReductionOp(reduce);
    assert(reductionOp && "Processing a reduce that is not supported!");
    auto neutralVal = getNeutralElement(reductionOp.value());
    assert(neutralVal && "Could not find neutral value for reduction op!");
    auto denseAttr = DenseElementsAttr::get(accumType, neutralVal.value());
    auto newAccum = builder.create<arith::ConstantOp>(oldAccum.getLoc(),
                                                      accumType, denseAttr);
    return newAccum;
  }

  SmallVector<int64_t>
  getThreadLocalityOptimizedShape(triton::ReduceOp reduce) const {
    auto srcType = cast<RankedTensorType>(reduce.getOperands()[0].getType());
    auto srcShape = srcType.getShape();
    auto rank = srcShape.size();
    auto elemsPerThread =
        triton::gpu::getElemsPerThread(srcType)[reduce.getAxis()];
    auto viewOpTensorShape = insertValue(srcShape, rank, 1);
    viewOpTensorShape[reduce.getAxis()] /= elemsPerThread;
    viewOpTensorShape[rank] = elemsPerThread;
    return viewOpTensorShape;
  }

  BlockedEncodingAttr
  getThreadLocalityOptimizedEncoding(triton::ReduceOp reduce) const {
    auto srcType = cast<RankedTensorType>(reduce.getOperands()[0].getType());
    auto rank = srcType.getShape().size();
    auto srcEncoding = srcType.getEncoding();
    auto blocked = dyn_cast<triton::gpu::BlockedEncodingAttr>(srcEncoding);
    auto sizePerThread3d =
        insertValue(blocked.getSizePerThread(), rank,
                    blocked.getSizePerThread()[reduce.getAxis()]);
    sizePerThread3d[reduce.getAxis()] = 1;
    auto threadsPerWarp3d = insertValue(blocked.getThreadsPerWarp(), rank, 1);
    auto warsPerCTA3d = insertValue(blocked.getWarpsPerCTA(), rank, 1);
    auto order3d = insertValue(blocked.getOrder(), 0, rank);
    auto ctasPerCGA3d =
        insertValue(blocked.getCTALayout().getCTAsPerCGA(), rank, 1);
    auto ctasSplitNum3d =
        insertValue(blocked.getCTALayout().getCTASplitNum(), rank, 1);
    auto ctaOrder3d =
        insertValue(blocked.getCTALayout().getCTAOrder(), rank, rank);
    auto ctaLayout3d = triton::gpu::CTALayoutAttr::get(
        reduce.getContext(), ctasPerCGA3d, ctasSplitNum3d, ctaOrder3d);
    auto blocked3d = triton::gpu::BlockedEncodingAttr::get(
        reduce.getContext(), sizePerThread3d, threadsPerWarp3d, warsPerCTA3d,
        order3d, ctaLayout3d);
    return blocked3d;
  }

  template <typename T>
  SmallVector<T> insertValue(ArrayRef<T> vec, unsigned index, int value) const {
    SmallVector<T> res(vec.begin(), vec.end());
    res.insert(res.begin() + index, static_cast<T>(value));
    return res;
  }
  template <typename T>
  SmallVector<T> insertValue(const SmallVector<T> &vec, unsigned index,
                             int value) const {
    SmallVector<T> res(vec.begin(), vec.end());
    res.insert(res.begin() + index, static_cast<T>(value));
    return res;
  }
};
} // namespace

} // namespace gpu
} // namespace triton
} // namespace mlir
