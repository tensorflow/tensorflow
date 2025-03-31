#include "mlir/Transforms/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUOPTIMIZEACCUMULATORINIT
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {
class TMEMAllocWithUnusedInit
    : public OpRewritePattern<triton::nvidia_gpu::TMEMAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::nvidia_gpu::TMEMAllocOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
    if (op.getSrc() == nullptr)
      return failure();
    SmallVector<Operation *> users(op.getResult().getUsers().begin(),
                                   op.getResult().getUsers().end());
    if (users.size() > 2)
      return failure();
    triton::nvidia_gpu::MMAv5OpInterface mmaOp = nullptr;
    triton::nvidia_gpu::TMEMLoadOp tmemLoad = nullptr;
    for (auto user : users) {
      if (auto load = dyn_cast<triton::nvidia_gpu::TMEMLoadOp>(user)) {
        tmemLoad = load;
      } else if (auto mma =
                     dyn_cast<triton::nvidia_gpu::MMAv5OpInterface>(user)) {
        mmaOp = mma;
      }
    }
    if (!mmaOp)
      return failure();
    if (tmemLoad && !mmaOp->isBeforeInBlock(tmemLoad))
      return failure();
    Value useAccFlag = mmaOp.useAccumulator();
    if (!useAccFlag)
      return failure();
    auto flagConstOp = useAccFlag.getDefiningOp<arith::ConstantOp>();
    if (!flagConstOp)
      return failure();
    if (cast<IntegerAttr>(flagConstOp.getValue()).getInt() != 0)
      return failure();
    op.getSrcMutable().clear();
    return success();
  }
};

bool dotSupportsAccInitFlag(Operation *op) {
  assert(isa<DotOpInterface>(op) &&
         "Expected an op which implements a DotOpInterface");

  if (auto wgDotOp = dyn_cast<triton::nvidia_gpu::WarpGroupDotOp>(op)) {
    // Partial accumulation would require a select op to handle the
    // initialization that would degrade the performance.
    return !wgDotOp.needsPartialAccumulator();
  }
  if (isa<triton::nvidia_gpu::MMAv5OpInterface>(op)) {
    return true;
  }
  return false;
}

std::pair<Value, Operation *> getAccumulatorUseAndDef(Operation *op) {
  assert(isa<DotOpInterface>(op) &&
         "Expected an op which implements a DotOpInterface");

  if (auto wgDotOp = dyn_cast<triton::nvidia_gpu::WarpGroupDotOp>(op)) {
    return std::make_pair(wgDotOp.getC(), wgDotOp);
  }
  if (auto tc05MmaOp = dyn_cast<triton::nvidia_gpu::MMAv5OpInterface>(op)) {
    auto accVal = tc05MmaOp.getAccumulator();
    auto tmemAlloc = accVal.getDefiningOp<triton::nvidia_gpu::TMEMAllocOp>();
    if (!tmemAlloc ||
        tmemAlloc->getParentRegion() != tc05MmaOp->getParentRegion())
      return std::make_pair(nullptr, nullptr);
    triton::nvidia_gpu::TMEMLoadOp tmemLoad = nullptr;
    for (auto user : tmemAlloc.getResult().getUsers()) {
      if (auto load = dyn_cast<triton::nvidia_gpu::TMEMLoadOp>(user)) {
        tmemLoad = load;
        break;
      }
    }
    if (!tmemLoad ||
        tmemLoad->getParentRegion() != tc05MmaOp->getParentRegion())
      return std::make_pair(nullptr, nullptr);
    return std::make_pair(tmemAlloc.getSrc(), tmemLoad);
  }
  assert(false && "Unexpected op which implements a DotOpInterface");
  return std::make_pair(nullptr, nullptr);
}

void setUseAccFlag(Operation *op, Value useAcc) {
  assert(isa<DotOpInterface>(op) &&
         "Expected an op which implements a DotOpInterface");

  if (auto wgDotOp = dyn_cast<triton::nvidia_gpu::WarpGroupDotOp>(op)) {
    wgDotOp.getUseCMutable().assign(useAcc);
  } else if (auto tc05MmaOp =
                 dyn_cast<triton::nvidia_gpu::MMAv5OpInterface>(op)) {
    tc05MmaOp.setUseAccumulator(useAcc);
  } else {
    assert(false && "Unexpected op which implements a DotOpInterface");
  }
}

bool isConstantZeroTensor(Value v) {
  return (matchPattern(v, m_Zero()) || matchPattern(v, m_AnyZeroFloat()));
}

std::optional<std::pair<Operation *, int>>
findZeroInitOp(Value accUse, scf::ForOp forOp, bool &loopArgIsZero) {
  Value v = accUse;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    assert(arg.getOwner() == forOp.getBody());
    if (isConstantZeroTensor(forOp.getInitArgs()[arg.getArgNumber() - 1])) {
      loopArgIsZero = true;
    }
    v = forOp.getBody()->getTerminator()->getOperand(arg.getArgNumber() - 1);
  }

  auto defOp = v.getDefiningOp();
  if (!defOp) {
    return std::nullopt;
  }
  if (auto selOp = dyn_cast<arith::SelectOp>(defOp)) {
    if (!selOp.getCondition().getType().isInteger(1))
      return std::nullopt;
    if (isConstantZeroTensor(selOp.getTrueValue()) ||
        isConstantZeroTensor(selOp.getFalseValue())) {
      return std::make_pair(selOp, 0);
    }
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
    unsigned resultIndex = cast<OpResult>(v).getResultNumber();
    Value thenVal = ifOp.thenYield()->getOperand(resultIndex);
    Value elseVal = ifOp.elseYield()->getOperand(resultIndex);
    if (isConstantZeroTensor(thenVal) || isConstantZeroTensor(elseVal)) {
      // Make sure that the other value is not defined in the if itself, but
      // passed from outside
      if (thenVal.getParentBlock()->getParentOp() == ifOp ||
          elseVal.getParentBlock()->getParentOp() == ifOp) {
        return std::nullopt;
      }
      return std::make_pair(ifOp, resultIndex);
    }
  }
  return std::nullopt;
}

} // namespace

class OptimizeAccumulatorInitPass
    : public impl::TritonGPUOptimizeAccumulatorInitBase<
          OptimizeAccumulatorInitPass> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();
    SmallVector<Operation *> mmaOps;
    m.walk([&](Operation *op) {
      if (isa<DotOpInterface>(op) && dotSupportsAccInitFlag(op))
        mmaOps.push_back(op);
    });

    // for each mma op, find where the accumulator is initialized with zero
    // It can be:
    // 1. A constant zero
    // 2. Initialized with zero as the loop argument
    // 3. Initialized with zero in the if op or with a select op in current
    //   or any of the previous loop iterations
    for (Operation *mmaOp : mmaOps) {
      Location loc = mmaOp->getLoc();

      scf::ForOp forOp = dyn_cast<scf::ForOp>(mmaOp->getParentOp());
      if (!forOp) {
        continue;
      }

      IRRewriter rewriter(forOp);
      rewriter.setInsertionPoint(forOp);

      Value vTrue =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
      Value vFalse =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false));

      // Find the accumulator
      auto [accUse, accDef] = getAccumulatorUseAndDef(mmaOp);
      if (!accUse || !accDef) {
        continue;
      }
      if (isConstantZeroTensor(accUse)) {
        setUseAccFlag(mmaOp, vFalse);
        continue;
      }

      bool loopArgIsZero = false;
      std::optional<std::pair<Operation *, int>> zeroInitOp =
          findZeroInitOp(accUse, forOp, loopArgIsZero);
      if (!zeroInitOp) {
        continue;
      }

      Value loopArgFlagValue = loopArgIsZero ? vFalse : vTrue;
      scf::ForOp newForOp =
          replaceForOpWithNewSignature(rewriter, forOp, {loopArgFlagValue});
      forOp.erase();
      forOp = newForOp;
      loopArgFlagValue =
          forOp.getRegionIterArg(forOp.getNumRegionIterArgs() - 1);

      Value condition = nullptr;
      Value oldValue = nullptr;
      Value zeroValue = nullptr;
      bool thenInitsToZero = false;
      if (auto selOp = dyn_cast<arith::SelectOp>(zeroInitOp->first)) {
        condition = selOp.getCondition();
        oldValue = isConstantZeroTensor(selOp.getTrueValue())
                       ? selOp.getFalseValue()
                       : selOp.getTrueValue();
        zeroValue = isConstantZeroTensor(selOp.getTrueValue())
                        ? selOp.getTrueValue()
                        : selOp.getFalseValue();
        thenInitsToZero = isConstantZeroTensor(selOp.getTrueValue());
      } else {
        assert(isa<scf::IfOp>(*zeroInitOp->first) && "Expected an if op");
        auto ifOp = cast<scf::IfOp>(zeroInitOp->first);
        unsigned resultIndex = zeroInitOp->second;
        condition = ifOp.getCondition();
        Value thenVal = ifOp.thenYield()->getOperand(resultIndex);
        Value elseVal = ifOp.elseYield()->getOperand(resultIndex);
        oldValue = isConstantZeroTensor(thenVal) ? elseVal : thenVal;
        zeroValue = isConstantZeroTensor(thenVal) ? thenVal : elseVal;
        thenInitsToZero = isConstantZeroTensor(thenVal);
      }

      // Create a select op that updates the flag
      rewriter.setInsertionPoint(zeroInitOp->first);
      bool zeroingBeforeMMA = zeroInitOp->first->isBeforeInBlock(mmaOp);
      Value prevFlagValue = zeroingBeforeMMA ? loopArgFlagValue : vTrue;
      auto selectFlagOp = rewriter.create<arith::SelectOp>(
          loc, condition, thenInitsToZero ? vFalse : prevFlagValue,
          thenInitsToZero ? prevFlagValue : vFalse);
      setUseAccFlag(mmaOp, zeroingBeforeMMA ? selectFlagOp : loopArgFlagValue);
      auto forYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      forYield->insertOperands(forYield->getNumOperands(),
                               {zeroingBeforeMMA ? vTrue : selectFlagOp});

      // Stop clearing out the accumulator with zero
      if (auto selOp = dyn_cast<arith::SelectOp>(zeroInitOp->first)) {
        rewriter.setInsertionPoint(selOp);
        rewriter.replaceOp(selOp, oldValue);
      } else {
        auto ifOp = cast<scf::IfOp>(zeroInitOp->first);
        int resultIndex = zeroInitOp->second;
        auto zeroingYield =
            thenInitsToZero ? ifOp.thenYield() : ifOp.elseYield();
        zeroingYield.setOperand(resultIndex, oldValue);
      }
    }

    // Cleanup unused init values in tmem allocs
    mlir::RewritePatternSet patterns(m.getContext());
    patterns.add<TMEMAllocWithUnusedInit>(m.getContext());
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
