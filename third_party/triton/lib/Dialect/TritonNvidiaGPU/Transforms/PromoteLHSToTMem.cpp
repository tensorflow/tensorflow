#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
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

namespace {
template <class MMAOpTy>
Attribute getLHSTMemLayout(MMAOpTy tcGen5MMAOp,
                           ttg::BlockedEncodingAttr srcLayout) {
  auto CTALayout = getCTALayout(srcLayout);
  int numWarps = ttg::lookupNumWarps(tcGen5MMAOp);
  auto accTmemEncoding = dyn_cast<ttng::TensorMemoryEncodingAttr>(
      tcGen5MMAOp.getD().getType().getEncoding());
  auto lhs = tcGen5MMAOp.getA();
  auto lhsShape = lhs.getType().getShape();
  // M has to follow the MMA size, as it is related to the message we are using.
  // N has to follow the number of columns in the LHS.
  int M = accTmemEncoding.getBlockM();
  int N = lhsShape[1];
  Attribute resLayout =
      ttng::getTmemCompatibleLayout(M, N, lhsShape, numWarps, CTALayout);
  return resLayout;
}

template <class MMAOpTy> class LHSToTMem : public OpRewritePattern<MMAOpTy> {
public:
  using OpRewritePattern<MMAOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(MMAOpTy tcGen5MMAOp,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = tcGen5MMAOp->getContext();
    Location loc = tcGen5MMAOp.getLoc();
    auto lhs = tcGen5MMAOp.getA();
    auto localAllocOp = lhs.template getDefiningOp<ttg::LocalAllocOp>();
    if (!localAllocOp)
      return failure();
    // Limit the liverange of the TMem allocations to single block.
    if (localAllocOp->getParentRegion() != tcGen5MMAOp->getParentRegion())
      return failure();
    Value src = localAllocOp.getSrc();
    auto srcType = cast<RankedTensorType>(src.getType());
    auto srcLayout = cast<ttg::BlockedEncodingAttr>(srcType.getEncoding());
    bool layoutTmemCompatible = ttng::isDistributedLayoutTMemCompatible(
        tcGen5MMAOp, srcType, tcGen5MMAOp.getD().getType());
    Attribute newLayout = srcLayout;
    if (!layoutTmemCompatible) {
      if (triton::tools::getBoolEnv("ALLOW_LHS_TMEM_LAYOUT_CONVERSION")) {
        newLayout = getLHSTMemLayout(tcGen5MMAOp, srcLayout);
      } else {
        return failure();
      }
    }
    rewriter.setInsertionPointAfter(localAllocOp);
    if (newLayout != srcLayout) {
      auto ty = cast<RankedTensorType>(src.getType());
      auto newTy =
          RankedTensorType::get(ty.getShape(), ty.getElementType(), newLayout);
      src = rewriter.create<ttg::ConvertLayoutOp>(loc, newTy, src);
    }
    auto accTMemEncoding = dyn_cast<ttng::TensorMemoryEncodingAttr>(
        tcGen5MMAOp.getD().getType().getEncoding());
    ArrayRef<unsigned> CTASplitNum = srcLayout.getCTALayout().getCTASplitNum();
    // TMem encoding for A operand is the same as for D (Acc), but unpacked.
    auto aTMemEncoding = ttng::TensorMemoryEncodingAttr::get(
        context, accTMemEncoding.getBlockM(), lhs.getType().getShape()[1],
        /*unpacked=*/false, CTASplitNum[0], CTASplitNum[1]);
    Attribute tensorMemorySpace =
        triton::nvidia_gpu::TensorMemorySpaceAttr::get(context);
    Type lhsMemDescType = triton::gpu::MemDescType::get(
        lhs.getType().getShape(), lhs.getType().getElementType(), aTMemEncoding,
        tensorMemorySpace,
        /*mutableMemory=*/false);
    Value tMemAlloc =
        rewriter.create<ttng::TMEMAllocOp>(loc, lhsMemDescType, src);
    tcGen5MMAOp.getAMutable().assign(tMemAlloc);
    return success();
  }
};
} // namespace

class TritonNvidiaGPUPromoteLHSToTMemPass
    : public TritonNvidiaGPUPromoteLHSToTMemPassBase<
          TritonNvidiaGPUPromoteLHSToTMemPass> {
public:
  using TritonNvidiaGPUPromoteLHSToTMemPassBase<
      TritonNvidiaGPUPromoteLHSToTMemPass>::
      TritonNvidiaGPUPromoteLHSToTMemPassBase;

  void runOnOperation() override {
    if (!triton::tools::getBoolEnv("ENABLE_LHS_TO_TMEM"))
      return;
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<LHSToTMem<ttng::TCGen5MMAOp>>(context);
    patterns.add<LHSToTMem<ttng::TCGen5MMAScaledOp>>(context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTritonNvidiaGPUPromoteLHSToTMemPass() {
  return std::make_unique<TritonNvidiaGPUPromoteLHSToTMemPass>();
}
