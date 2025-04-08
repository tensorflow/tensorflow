#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-coalesce-async-copy"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace ttg = triton::gpu;

// On gfx9 global and buffer loads directly to shared memory need to write
// coalesced. This pattern converts the layout of the src, mask and other to
// ensure the owned data per thread is contigious and does no exceed the
// supported load vector size to ensure coalesed writes
struct CoalesceAsyncCopyWrites
    : public OpRewritePattern<ttg::AsyncCopyGlobalToLocalOp> {
  CoalesceAsyncCopyWrites(const triton::AMD::TargetInfo &targetInfo,
                          const DenseMap<ttg::AsyncCopyGlobalToLocalOp,
                                         unsigned> &asyncCopyContiguity,
                          MLIRContext *ctx)
      : OpRewritePattern(ctx), targetInfo{targetInfo},
        asyncCopyContiguity{std::move(asyncCopyContiguity)} {}

  LogicalResult matchAndRewrite(ttg::AsyncCopyGlobalToLocalOp copyOp,
                                PatternRewriter &rewriter) const override {
    auto src = copyOp.getSrc();
    auto dst = copyOp.getResult();
    Value mask = copyOp.getMask();
    Value other = copyOp.getOther();

    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<ttg::MemDescType>(dst.getType());

    auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
    if (!blockedEnc)
      return rewriter.notifyMatchFailure(copyOp,
                                         "src encoding must be #blocked");

    auto sharedEnc =
        dyn_cast<ttg::SwizzledSharedEncodingAttr>(dstTy.getEncoding());
    if (!sharedEnc)
      return rewriter.notifyMatchFailure(
          copyOp, "destination encoding must be #SwizzledShared");
    if (sharedEnc.getMaxPhase() > 1)
      return rewriter.notifyMatchFailure(
          copyOp, "swizzled shared encoding not supported");

    // We start from the precomputed contiguity we got from AxisAnalysis.
    unsigned loadContig = 0;
    if (auto it = asyncCopyContiguity.find(copyOp);
        it != asyncCopyContiguity.end())
      loadContig = it->second;
    else
      return copyOp->emitError()
             << "No contiguity information about the copy op";
    assert(loadContig > 0);

    // Further restrict the contiguity based on the contiguity of the src to dst
    // layout e.g. if the order of the blocked and shared encoding is different
    // we can only load one element at a time or if the shared encoding is
    // swizzled we cannot exceed the vector size of the swizzling pattern
    LinearLayout regLayout =
        triton::gpu::toLinearLayout(srcTy.getShape(), blockedEnc);
    LinearLayout sharedLayout =
        triton::gpu::toLinearLayout(srcTy.getShape(), sharedEnc);
    auto regToSharedLayout = regLayout.invertAndCompose(sharedLayout);
    loadContig = std::min<unsigned>(loadContig,
                                    regToSharedLayout.getNumConsecutiveInOut());

    // Select the largest supported load width equal or smaller than loadContig
    auto elemBitWidth = dstTy.getElementTypeBitWidth();
    while (loadContig > 0 && !targetInfo.supportsDirectToLdsLoadBitWidth(
                                 loadContig * elemBitWidth)) {
      loadContig /= 2;
    }

    if (loadContig == 0) {
      return rewriter.notifyMatchFailure(
          copyOp, "could not find layout config to create coalesced writes");
    }

    // Do not rewrite if we already use the correct contiguity (could be from a
    // previous rewrite)
    auto contigPerThread = ttg::getContigPerThread(srcTy);
    auto blockedContig = contigPerThread[blockedEnc.getOrder()[0]];
    if (blockedContig == loadContig) {
      return rewriter.notifyMatchFailure(copyOp,
                                         "already using the correct layout");
    }

    // Get new blocked encoding with loadContig as sizePerThread in the fastest
    // dim
    assert(blockedContig >= loadContig);
    contigPerThread[blockedEnc.getOrder()[0]] = loadContig;
    int numWarps = triton::gpu::lookupNumWarps(copyOp);
    auto mod = copyOp->getParentOfType<ModuleOp>();
    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    auto newBlockEnc = BlockedEncodingAttr::get(
        copyOp.getContext(), srcTy.getShape(), contigPerThread,
        blockedEnc.getOrder(), numWarps, threadsPerWarp,
        blockedEnc.getCTALayout());

    // Convert layout of src, mask and other to new encoding
    auto convertLayout = [&rewriter](auto loc, Value old, auto newEnc) {
      auto oldTy = cast<RankedTensorType>(old.getType());
      RankedTensorType newSrcTy = RankedTensorType::get(
          oldTy.getShape(), oldTy.getElementType(), newEnc);
      return rewriter.create<ttg::ConvertLayoutOp>(loc, newSrcTy, old);
    };

    auto loc = copyOp->getLoc();
    Value cvtSrc = convertLayout(loc, src, newBlockEnc);

    if (mask)
      mask = convertLayout(loc, mask, newBlockEnc);
    if (other)
      other = convertLayout(loc, other, newBlockEnc);

    rewriter.modifyOpInPlace(copyOp, [&]() {
      copyOp.getSrcMutable().assign(cvtSrc);
      if (mask)
        copyOp.getMaskMutable().assign(mask);
      if (other)
        copyOp.getOtherMutable().assign(other);
    });
    return success();
  }

private:
  const triton::AMD::TargetInfo &targetInfo;
  const DenseMap<ttg::AsyncCopyGlobalToLocalOp, unsigned> &asyncCopyContiguity;
};

class TritonAMDGPUCoalesceAsyncCopyPass
    : public TritonAMDGPUCoalesceAsyncCopyBase<
          TritonAMDGPUCoalesceAsyncCopyPass> {
public:
  TritonAMDGPUCoalesceAsyncCopyPass(StringRef archGenName) {
    this->archGenerationName = archGenName.str();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *context = &getContext();

    triton::AMD::TargetInfo targetInfo(archGenerationName);

    mlir::RewritePatternSet patterns(context);

    switch (targetInfo.getISAFamily()) {
    case triton::AMD::ISAFamily::CDNA1:
    case triton::AMD::ISAFamily::CDNA2:
    case triton::AMD::ISAFamily::CDNA3:
    case triton::AMD::ISAFamily::CDNA4: {
      break;
    }
    default:
      return;
    }

    // Precompute the contiguity of all AsyncCopy ops based on the src and
    // mask contiguity/alignment to avoid rebuilding ModuleAxisInfoAnalysis
    // after every IR change.
    triton::ModuleAxisInfoAnalysis axisAnalysis(m);
    DenseMap<ttg::AsyncCopyGlobalToLocalOp, unsigned> asyncCopyContiguity;
    m->walk([&](ttg::AsyncCopyGlobalToLocalOp copyOp) {
      unsigned contiguity =
          mlir::LLVM::AMD::getContiguity(copyOp.getSrc(), axisAnalysis);
      if (auto mask = copyOp.getMask()) {
        contiguity =
            std::min<unsigned>(contiguity, axisAnalysis.getMaskAlignment(mask));
      }
      asyncCopyContiguity.insert({copyOp, contiguity});
    });
    patterns.add<CoalesceAsyncCopyWrites>(targetInfo, asyncCopyContiguity,
                                          context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass>
mlir::createTritonAMDGPUCoalesceAsyncCopyPass(std::string archGenName) {
  return std::make_unique<TritonAMDGPUCoalesceAsyncCopyPass>(
      std::move(archGenName));
}
