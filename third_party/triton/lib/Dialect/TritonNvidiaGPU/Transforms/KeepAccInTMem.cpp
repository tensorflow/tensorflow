#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <memory>

namespace {

using namespace mlir;

namespace ttng = triton::nvidia_gpu;
namespace ttg = triton::gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

static bool bwdFilter(Operation *op) {
  return isa<ttg::ConvertLayoutOp, ttng::TMEMLoadOp>(op) ||
         op->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
         op->hasTrait<OpTrait::Elementwise>();
}

static Type getNewType(Type type, Attribute encoding) {
  RankedTensorType tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType)
    return type;
  return RankedTensorType::get(tensorType.getShape(),
                               tensorType.getElementType(), encoding);
}

class TMEMToGlobal : public OpRewritePattern<triton::StoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::StoreOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
    Value data = op.getValue();
    auto tensorType = dyn_cast<RankedTensorType>(data.getType());
    if (!tensorType)
      return failure();
    llvm::SetVector<Operation *> slice;
    mlir::BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    opt.filter = bwdFilter;
    getBackwardSlice(data, &slice, opt);
    Attribute encoding;
    for (auto op : slice) {
      if (auto tmemLoad = dyn_cast<ttng::TMEMLoadOp>(op)) {
        if (!encoding)
          encoding = tmemLoad.getType().getEncoding();
        if (tmemLoad.getType().getEncoding() != encoding)
          return failure();
      }
    }
    if (!encoding || tensorType.getEncoding() == encoding)
      return failure();
    // Use tmem load encoding to avoid going through shared memory.
    Value newData = rewriter.create<ttg::ConvertLayoutOp>(
        loc, getNewType(data.getType(), encoding), data);
    Value newPointer = rewriter.create<ttg::ConvertLayoutOp>(
        loc, getNewType(op.getPtr().getType(), encoding), op.getPtr());
    Value newMask;
    if (op.getMask())
      newMask = rewriter.create<ttg::ConvertLayoutOp>(
          loc, getNewType(op.getMask().getType(), encoding), op.getMask());
    rewriter.create<triton::StoreOp>(loc, newPointer, newData, newMask,
                                     op.getBoundaryCheck(), op.getCache(),
                                     op.getEvict());
    rewriter.eraseOp(op);
    return success();
  }
};

static void addTMEMLoad(IRRewriter &rewriter, ttng::TMEMAllocOp localAlloc,
                        Operation *user, int argNo) {
  rewriter.setInsertionPoint(user);
  auto load = rewriter.create<ttng::TMEMLoadOp>(
      user->getLoc(), user->getOperand(argNo).getType(),
      localAlloc->getResult(0));
  user->setOperand(argNo, load);
}

static bool canKeepAccInTmem(scf::ForOp forOp, Operation *mmaOp,
                             ttng::TMEMAllocOp &localAlloc,
                             ttng::TMEMLoadOp &localLoad,
                             SmallVector<std::pair<Operation *, int>> &accUsers,
                             unsigned &yieldArgNo) {
  // The expected sequence of instructions:
  // %acc_tm = ttg.local_alloc %acc
  // ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm
  // %acc_res = ttg.local_load %acc_tm
  localAlloc = mmaOp->getOperand(2).getDefiningOp<ttng::TMEMAllocOp>();
  if (!localAlloc) {
    return false;
  }
  for (auto user : localAlloc->getUsers()) {
    if (isa<ttng::TMEMLoadOp>(user)) {
      localLoad = cast<ttng::TMEMLoadOp>(user);
    } else if (user != mmaOp) {
      // The accumulator is used by another operation, not something we
      // expect.
      localLoad = nullptr;
      return false;
    }
  }

  SmallVector<Value> queue;
  queue.push_back(localLoad->getResult(0));
  bool foundDotCycle = false;
  while (!queue.empty()) {
    Value value = queue.pop_back_val();
    for (auto &use : value.getUses()) {
      if (use.getOwner() == localAlloc) {
        foundDotCycle = true;
        continue;
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(use.getOwner())) {
        if (yieldOp->getParentOp() == forOp) {
          yieldArgNo = use.getOperandNumber();
          queue.push_back(forOp.getRegionIterArg(yieldArgNo));
          continue;
        }
        if (auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
          // TODO: Accumulator being used in the yield of ifOp means that
          // it is being modified in the other branch of the ifOp. This is not
          // something we can handle yet.
          return false;
        }
        // Not sure what are we doing here. Back out.
        return false;
      }
      accUsers.emplace_back(use.getOwner(), use.getOperandNumber());
    }
  }
  return foundDotCycle;
}

static void hoistReadModifyWrite(Operation *mmaOp, scf::ForOp forOp) {
  // For the transformation to make sense, the accumulator must be
  // reused by the same MMA operation in subsequent iterations.
  SmallVector<std::pair<Operation *, int>> accUsers;
  ttng::TMEMAllocOp localAlloc = nullptr;
  ttng::TMEMLoadOp localLoad = nullptr;
  unsigned yieldArgNo;
  if (!canKeepAccInTmem(forOp, mmaOp, localAlloc, localLoad, accUsers,
                        yieldArgNo)) {
    return;
  }

  assert(localLoad != nullptr);
  assert(localAlloc != nullptr);
  Type loadType = localLoad->getResult(0).getType();
  IRRewriter rewriter(forOp);
  localAlloc->moveBefore(forOp);
  localAlloc->setOperand(0, forOp.getInitArgs()[yieldArgNo]);
  mmaOp->setOperand(2, localAlloc->getResult(0));
  // Unlink the local_load from the yield. Short circuit the unused yield
  // value with the corresponding iter arg.
  forOp.getBody()->getTerminator()->setOperand(
      yieldArgNo, forOp.getRegionIterArg(yieldArgNo));

  // Add TMEM loads before all the uses
  // TODO: We could be more efficient here, reusing loads instead of
  // creating new ones for each use.
  for (auto [user, argNo] : accUsers) {
    addTMEMLoad(rewriter, localAlloc, user, argNo);
  }

  rewriter.setInsertionPointAfter(forOp);
  auto afterLoopLoad = rewriter.create<ttng::TMEMLoadOp>(
      forOp.getLoc(), loadType, localAlloc->getResult(0));
  forOp->getResult(yieldArgNo).replaceAllUsesWith(afterLoopLoad->getResult(0));

  localLoad->erase();
}

// Hoist invariant tmem_alloc. This could technically be done as general LICM
// but controlling tmem liveranga more precisley is likely to be important.
static void hoistInvariantInputs(Operation *mmaOp, scf::ForOp forOp) {
  for (auto operand : mmaOp->getOperands()) {
    if (forOp.isDefinedOutsideOfLoop(operand))
      continue;
    auto tmemAllocOp = operand.getDefiningOp<ttng::TMEMAllocOp>();
    if (!tmemAllocOp || tmemAllocOp.getType().getMutableMemory())
      continue;
    assert(tmemAllocOp.getSrc());
    Value src = tmemAllocOp.getSrc();
    SmallVector<Operation *> opToHoist = {tmemAllocOp.getOperation()};
    // Also hoist simple unary elementwise that may have sinked into the loop.
    while (Operation *defOp = src.getDefiningOp()) {
      if (forOp.isDefinedOutsideOfLoop(src))
        break;
      if (!(isMemoryEffectFree(defOp) && isSpeculatable(defOp) &&
            defOp->getNumOperands() == 1))
        break;
      opToHoist.push_back(defOp);
      src = defOp->getOperand(0);
    }
    if (!forOp.isDefinedOutsideOfLoop(src))
      continue;
    for (auto op : llvm::reverse(opToHoist)) {
      forOp.moveOutOfLoop(op);
    }
  }
}
class TritonNvidiaGPUKeepAccInTMemPass
    : public TritonNvidiaGPUKeepAccInTMemPassBase<
          TritonNvidiaGPUKeepAccInTMemPass> {
public:
  using TritonNvidiaGPUKeepAccInTMemPassBase<
      TritonNvidiaGPUKeepAccInTMemPass>::TritonNvidiaGPUKeepAccInTMemPassBase;

  void runOnOperation() override {
    auto module = getOperation();

    module.walk([&](scf::ForOp forOp) { runOnForOp(forOp); });

    if (triton::tools::getBoolEnv("STORE_TMEM_TO_GLOBAL_BYPASS_SMEM")) {
      mlir::RewritePatternSet patterns(module.getContext());
      patterns.add<TMEMToGlobal>(module.getContext());
      if (applyPatternsGreedily(module, std::move(patterns)).failed())
        signalPassFailure();
    }
  }

  void runOnForOp(scf::ForOp forOp) {
    SmallVector<Operation *> mmaOps;
    forOp.walk([&](Operation *mmaOp) {
      // Skip MMA nested in another forOp
      if (isa<ttng::MMAv5OpInterface>(mmaOp) &&
          mmaOp->getParentOfType<scf::ForOp>() == forOp) {
        mmaOps.push_back(mmaOp);
      }
    });
    if (mmaOps.empty()) {
      return;
    }

    for (auto mmaOp : mmaOps) {
      hoistReadModifyWrite(mmaOp, forOp);
      hoistInvariantInputs(mmaOp, forOp);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTritonNvidiaGPUKeepAccInTMemPass() {
  return std::make_unique<TritonNvidiaGPUKeepAccInTMemPass>();
}
