//===----------------------------------------------------------------------===//
//
// This pass tries to prefetch operands (a and b) of tt.dot.
// Those ConvertLayoutOps will be lowered to shared memory loads.
//
// For example:
// %a: tensor<128x32xf16, #enc>
// scf.for %iv = ... iter_args(%a_arg = %a, ...) {
//   %d = tt.dot %a_arg, %b, %c
//   ...
//   scf.yield %a_next, ...
// }
//
// will be translated to
//
// %a: tensor<128x32xf16, #enc>
// %a_tmp = tensor.subview %a[0, 0] [128, 16]
// %a_prefetch = ttg.local_load %a_tmp
// scf.for %iv = ... iter_args(%a_buf = %a, ..., %a_prefetch_arg = %a_prefetch)
// {
//   %x = tt.dot %a_prefetch_arg, %b, %c
//   %a_tmp_rem = tensor.subview %a_buf[0, 16] [128, 16]
//   %a_prefetch_next = ttg.local_load %a_tmp_rem
//   ...
//   scf.yield %next_a, ..., %a_prefetch_next
// }
//===----------------------------------------------------------------------===//

#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritongpu-prefetch"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPREFETCH
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

class Prefetcher {
  /// cache the ForOp we are working on
  scf::ForOp forOp;
  /// cache the YieldOp of this ForOp
  scf::YieldOp yieldOp;
  ///
  // TODO: add a hook to infer prefetchWidth
  unsigned prefetchWidth = 32;

  /// dots to be prefetched
  SetVector<triton::DotOp> dots;
  /// dot => dot operand
  DenseMap<Value, Value> dot2aLoopArg;
  DenseMap<Value, Value> dot2aHeaderDef;
  DenseMap<Value, Value> dot2bLoopArg;
  DenseMap<Value, Value> dot2bHeaderDef;
  DenseMap<Value, Value> dot2aYield;
  DenseMap<Value, Value> dot2bYield;
  DenseMap<Value, SmallVector<Value>> dot2aVals;
  DenseMap<Value, SmallVector<Value>> dot2bVals;
  /// operand => defining
  DenseMap<Value, Value> operand2headPrefetch;

  LogicalResult isForOpOperand(Value v);

  Value generatePrefetch(Value v, unsigned opIdx, bool isPrologue,
                         Attribute dotEncoding, OpBuilder &builder,
                         std::optional<int64_t> offsetK = std::nullopt,
                         std::optional<int64_t> shapeK = std::nullopt);

  void cloneElementwiseOps(Value &bRem, const SmallVector<Value> &vals,
                           OpBuilder &builder);

public:
  Prefetcher() = delete;

  Prefetcher(scf::ForOp forOp) : forOp(forOp) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  LogicalResult initialize();

  void emitPrologue();

  scf::ForOp createNewForOp();
};

void Prefetcher::cloneElementwiseOps(Value &ret, const SmallVector<Value> &vals,
                                     OpBuilder &builder) {
  IRMapping mapping;
  mapping.map(vals[1], ret);
  for (int i = 2; i < vals.size(); i++) {
    Value v = vals[i];
    Value curr = builder.clone(*v.getDefiningOp(), mapping)->getResult(0);
    if (isa<RankedTensorType>(curr.getType())) {
      auto retType = RankedTensorType::get(
          cast<RankedTensorType>(ret.getType()).getShape(),
          cast<RankedTensorType>(curr.getType()).getElementType(),
          cast<RankedTensorType>(curr.getDefiningOp()->getOperand(0).getType())
              .getEncoding());
      curr.setType(retType);
    }
    mapping.map(v, curr);
  }
  if (vals.size() > 1)
    ret = mapping.lookup(vals.back());
}

Value Prefetcher::generatePrefetch(Value v, unsigned opIdx, bool isPrologue,
                                   Attribute dotEncoding, OpBuilder &builder,
                                   std::optional<int64_t> offsetK,
                                   std::optional<int64_t> shapeK) {
  // opIdx: 0 => a, 1 => b
  auto type = cast<triton::gpu::MemDescType>(v.getType());
  SmallVector<int64_t> shape{type.getShape().begin(), type.getShape().end()};
  auto rank = shape.size();
  SmallVector<int64_t> offset(rank, 0);
  Type elementType = type.getElementType();

  // k => (prefetchWidth, k - prefetchWidth)
  int64_t kIdx = opIdx == 0 ? rank - 1 : rank - 2;

  offset[kIdx] = isPrologue ? 0 : prefetchWidth;
  shape[kIdx] = isPrologue ? prefetchWidth : (shape[kIdx] - prefetchWidth);

  if (shapeK)
    shape[kIdx] = *shapeK;
  if (offsetK)
    offset[kIdx] = *offsetK;

  SmallVector<Value> offsetsVal;
  for (int64_t off : offset)
    offsetsVal.push_back(
        builder.create<arith::ConstantIntOp>(v.getLoc(), off, 32));
  Value newSmem = builder.create<triton::gpu::MemDescSubviewOp>(
      v.getLoc(),
      triton::gpu::MemDescType::get(
          shape, elementType, type.getEncoding(), type.getMemorySpace(),
          type.getMutableMemory(), type.getAllocShape()),
      v, offsetsVal);

  // We need to assign kwidth to zero in the case where the parent layout is
  // Blocked, otherwise the verifier emits a failure. The parent layout is
  // Blocked only when Tensor Cores are disabled.
  int kwidth = dyn_cast<triton::gpu::BlockedEncodingAttr>(dotEncoding)
                   ? 0
                   : prefetchWidth / 8;
  auto dotOperandEnc = triton::gpu::DotOperandEncodingAttr::get(
      builder.getContext(), opIdx, dotEncoding, kwidth);
  Value prefetchSlice = builder.create<triton::gpu::LocalLoadOp>(
      v.getLoc(), RankedTensorType::get(shape, elementType, dotOperandEnc),
      newSmem);

  return prefetchSlice;
}

LogicalResult Prefetcher::initialize() {
  Block *loop = forOp.getBody();

  auto getEncoding = [](Value v) {
    return cast<TensorOrMemDesc>(v.getType()).getEncoding();
  };

  SmallVector<triton::DotOp> dotsInFor;
  for (Operation &op : *loop)
    if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
      // Only accepts dotOps encoded as Nvidia MMA v2 or AMD MFMA
      auto dstMmaEnc =
          dyn_cast<NvidiaMmaEncodingAttr>(getEncoding(dotOp.getResult()));
      auto dstMfmaEnc =
          dyn_cast<AMDMfmaEncodingAttr>(getEncoding(dotOp.getResult()));
      if (!dstMfmaEnc && (!dstMmaEnc || dstMmaEnc.getVersionMajor() != 2))
        // Don't rewrite if any other type is found.
        return failure();
      dotsInFor.push_back(dotOp);
    }

  if (dotsInFor.empty())
    return failure();

  // TODO: segfault (original for still has uses)
  // when used in flash attention that has 2 dots in the loop
  if (dotsInFor.size() > 1)
    return failure();

  // returns source of cvt
  auto getPrefetchSrc = [](Value v) -> SmallVector<Value> {
    // walk back to conversion
    Operation *op = v.getDefiningOp();
    bool foundConvertFromShared = false;
    SmallVector<Value> rets;
    rets.push_back(op->getResult(0));
    LDBG("Prefetch src: " << *op);
    while (op) {
      if (op->getNumOperands() != 1)
        break;
      if (!op->getResult(0).hasOneUse())
        break;
      // Similar to issues faced in HoistLayoutConversion pattern in
      // OptimizeDotOperands.cpp, we can't propagate through type casts from
      // predicates as they aren't supported in Triton when encoded with dot_op
      // layout.
      if (isa<arith::UIToFPOp>(op)) {
        Type srcType = getElementTypeOrSelf(op->getOperand(0));
        if (srcType.isInteger(1))
          break;
      }
      // Propagation through ExpandDims is currently not supported. This blindly
      // replaces the encoding with dot encoding & but ExpandDims requires a
      // SliceEncoding. This could be rewritten to support it somehow, but I
      // don't think it's trivial & it's currently crashing.
      if (isa<ExpandDimsOp>(op)) {
        break;
      }
      rets.push_back(op->getOperand(0));
      if (auto cvt = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
        // NYI for other encodings, for example if we have transpose
        // in the chain
        if (isa<DotOperandEncodingAttr>(cvt.getType().getEncoding()))
          foundConvertFromShared = true;
        break;
      }
      op = op->getOperand(0).getDefiningOp();
      if (op)
        LDBG("op: " << *op);
    }
    std::reverse(rets.begin(), rets.end());

    if (foundConvertFromShared)
      return rets;
    return {};
  };

  auto getIncomingOp = [this](Value v) -> Value {
    if (auto arg = mlir::dyn_cast<BlockArgument>(v))
      if (arg.getOwner()->getParentOp() == forOp.getOperation())
        return forOp.getTiedLoopInit(arg)->get();
    return Value();
  };

  auto getYieldOperand = [this](Value v) -> Value {
    auto arg = mlir::cast<BlockArgument>(v);
    unsigned yieldIdx = arg.getArgNumber() - forOp.getNumInductionVars();
    return yieldOp.getOperand(yieldIdx);
  };

  for (triton::DotOp dot : dotsInFor) {
    auto aType = dot.getA().getType();
    auto bType = dot.getB().getType();
    auto aEnc =
        mlir::cast<triton::gpu::DotOperandEncodingAttr>(aType.getEncoding());
    auto bEnc =
        mlir::cast<triton::gpu::DotOperandEncodingAttr>(bType.getEncoding());
    int aKWidth = aEnc.getKWidth();
    int bKWidth = bEnc.getKWidth();
    assert(aKWidth == bKWidth);

    auto kSize = aType.getShape().back();

    // works better with nvidia tensor cores
    unsigned elementWidth = aType.getElementTypeBitWidth();
    if (aKWidth == 0)
      prefetchWidth = 256 / elementWidth;
    else
      prefetchWidth = 8 * aKWidth;

    // Skip prefetching if kSize is less than prefetchWidth
    if (kSize < prefetchWidth)
      continue;
    auto aVals = getPrefetchSrc(dot.getA());
    auto bVals = getPrefetchSrc(dot.getB());

    if (aVals.size() && bVals.size()) {
      Value aSmem = aVals.front();
      Value bSmem = bVals.front();
      Value aHeaderDef = getIncomingOp(aSmem);
      Value bHeaderDef = getIncomingOp(bSmem);
      // Only prefetch loop arg
      if (aHeaderDef && bHeaderDef) {
        dots.insert(dot);
        dot2aVals[dot] = aVals;
        dot2bVals[dot] = bVals;
        dot2aHeaderDef[dot] = aHeaderDef;
        dot2bHeaderDef[dot] = bHeaderDef;
        dot2aLoopArg[dot] = aSmem;
        dot2bLoopArg[dot] = bSmem;
        dot2aYield[dot] = getYieldOperand(aSmem);
        dot2bYield[dot] = getYieldOperand(bSmem);
      }
    }
  }

  return success();
}

void Prefetcher::emitPrologue() {
  OpBuilder builder(forOp);

  for (triton::DotOp dot : dots) {
    Attribute dotEncoding = dot.getType().getEncoding();
    Value aPrefetched =
        generatePrefetch(dot2aHeaderDef[dot], 0, true, dotEncoding, builder);
    cloneElementwiseOps(aPrefetched, dot2aVals[dot], builder);
    Value bPrefetched =
        generatePrefetch(dot2bHeaderDef[dot], 1, true, dotEncoding, builder);
    cloneElementwiseOps(bPrefetched, dot2bVals[dot], builder);

    operand2headPrefetch[dot.getA()] = aPrefetched;
    operand2headPrefetch[dot.getB()] = bPrefetched;
  }
}

scf::ForOp Prefetcher::createNewForOp() {
  OpBuilder builder(forOp);

  SmallVector<Value> loopArgs;
  for (auto v : forOp.getInitArgs())
    loopArgs.push_back(v);
  for (triton::DotOp dot : dots) {
    loopArgs.push_back(operand2headPrefetch[dot.getA()]);
    loopArgs.push_back(operand2headPrefetch[dot.getB()]);
  }

  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), loopArgs);

  builder.setInsertionPointToStart(newForOp.getBody());
  IRMapping mapping;
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  // The insertion point should be placed before the yield op
  auto setInsertionPointBeforeYield = [](OpBuilder &builder,
                                         scf::ForOp newForOp) {
    if (newForOp.getBody()->mightHaveTerminator()) {
      builder.setInsertionPoint(newForOp.getBody()->getTerminator());
    } else {
      builder.setInsertionPointToEnd(newForOp.getBody());
    }
  };

  for (Operation &op : forOp.getBody()->without_terminator()) {
    // If we're currently trying to sink a prefetched dot, we need to stop
    // sinking it (by resetting the insertion point to the end) if we find
    // control flow, or anything that depends on the dot op.
    if (op.getNumRegions() > 0) {
      setInsertionPointBeforeYield(builder, newForOp);
    }
    for (auto operand : op.getOperands()) {
      if (auto def = operand.getDefiningOp()) {
        auto dot = dyn_cast<triton::DotOp>(def);
        if (dot && dots.contains(dot)) {
          setInsertionPointBeforeYield(builder, newForOp);
        }
      }
    }
    Operation *newOp = builder.clone(op, mapping);
    auto dot = dyn_cast<triton::DotOp>(&op);
    if (dot && dots.contains(dot)) {
      Attribute dotEncoding = dot.getType().getEncoding();
      // prefetched dot
      Operation *firstDot = builder.clone(*dot, mapping);
      if (Value a = operand2headPrefetch.lookup(dot.getA()))
        firstDot->setOperand(
            0, newForOp.getTiedLoopRegionIterArg(&*a.use_begin()));
      if (Value b = operand2headPrefetch.lookup(dot.getB()))
        firstDot->setOperand(
            1, newForOp.getTiedLoopRegionIterArg(&*b.use_begin()));

      // remaining part
      int64_t kOff = prefetchWidth;
      int64_t kRem = dot.getA().getType().getShape().back() - prefetchWidth;
      Operation *prevDot = firstDot;
      if (kRem == 0) {
        // There is only one dot while prefetchWidth == kSize so delay issuing
        // it. Meanwhile, newOp should be set to firstDot to make sure the dot
        // result is updated to yield.
        builder.setInsertionPoint(prevDot);
        newOp = firstDot;
      }

      while (kRem != 0) {
        // int64_t kShape = largestPow2(kRem);
        int64_t kShape = prefetchWidth;
        auto insertionPoint = builder.saveInsertionPoint();
        builder.setInsertionPoint(prevDot);
        Value aRem =
            generatePrefetch(mapping.lookup(dot2aLoopArg[dot]), 0, false,
                             dotEncoding, builder, kOff, kShape);
        cloneElementwiseOps(aRem, dot2aVals[dot], builder);
        Value bRem =
            generatePrefetch(mapping.lookup(dot2bLoopArg[dot]), 1, false,
                             dotEncoding, builder, kOff, kShape);
        cloneElementwiseOps(bRem, dot2bVals[dot], builder);
        builder.restoreInsertionPoint(insertionPoint);
        newOp = builder.clone(*dot, mapping);
        newOp->setOperand(0, aRem);
        newOp->setOperand(1, bRem);
        newOp->setOperand(2, prevDot->getResult(0));
        prevDot = newOp;
        kOff += kShape;
        kRem -= kShape;
        if (kRem == 0) {
          // We want to delay issuing the last dot as long as possible, ideally
          // until after the prefetch.  To accomplish this, set the insertion
          // point above the dot.  If we find anything dependent on the dot (at
          // the top of this loop), we resume inserting after it.
          builder.setInsertionPoint(prevDot);
        }
      }
    }
    // update mapping of results
    for (unsigned dstIdx : llvm::seq(unsigned(0), op.getNumResults()))
      mapping.map(op.getResult(dstIdx), newOp->getResult(dstIdx));
  }

  // prefetch next iteration
  SmallVector<Value> yieldValues;
  for (Value v : forOp.getBody()->getTerminator()->getOperands())
    yieldValues.push_back(mapping.lookupOrDefault(v));
  for (triton::DotOp dot : dots) {
    Attribute dotEncoding = dot.getType().getEncoding();
    Value aToYield = generatePrefetch(mapping.lookup(dot2aYield[dot]), 0, true,
                                      dotEncoding, builder);
    cloneElementwiseOps(aToYield, dot2aVals[dot], builder);
    yieldValues.push_back(aToYield);
    // bToYield
    Value bToYield = generatePrefetch(mapping.lookup(dot2bYield[dot]), 1, true,
                                      dotEncoding, builder);
    cloneElementwiseOps(bToYield, dot2bVals[dot], builder);
    yieldValues.push_back(bToYield);
  }
  // Update ops of yield
  builder.setInsertionPointToEnd(newForOp.getBody());
  if (!yieldValues.empty())
    builder.create<scf::YieldOp>(yieldOp.getLoc(), yieldValues);
  return newForOp;
}

} // anonymous namespace

struct PrefetchPass : public impl::TritonGPUPrefetchBase<PrefetchPass> {
  void runOnOperation() override {

    // Canonicalize convert ops to make the pattern matching easier.
    RewritePatternSet cleanUpPatterns(&getContext());
    triton::gpu::ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns,
                                                              &getContext());
    if (mlir::applyPatternsGreedily(getOperation(), std::move(cleanUpPatterns))
            .failed()) {
      signalPassFailure();
    }
    getOperation()->walk([&](scf::ForOp forOp) {
      Prefetcher prefetcher(forOp);

      if (prefetcher.initialize().failed())
        return;

      prefetcher.emitPrologue();

      scf::ForOp newForOp = prefetcher.createNewForOp();

      // replace the original loop
      for (unsigned i = 0; i < forOp->getNumResults(); ++i)
        forOp->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
      forOp->erase();
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
