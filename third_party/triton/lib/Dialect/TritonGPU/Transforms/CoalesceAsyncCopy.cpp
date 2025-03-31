#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUCOALESCEASYNCCOPY
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

// This pass currently only applies if the following are all true...
//   1) Operand A for WGMMA is to be loaded in registers
//   2) We upcast operand A in registers before the WGMMA
//      (downcasting is not yet supported)
//   3) Pipelining is enabled for loading A
//
// ...then for the AsyncCopyGlobalToLocal op, the SharedEncoding
// vec will be less than BlockedEncoding's sizePerThread for k-dim. E.g. if
// we're upcasting from int8 to bf16, then shared vec is 8 and sizePerThread
// for k is 16. In this case, AsyncCopyGlobalToLocal will generate two
// 8-byte-cp.async's for each contiguous 16B global data owned by each
// thread. This breaks coalescing (i.e. results 2x the minimum required
// transactions).
//
// This issue occurs for cp.async because it combines load and store into one
// instruction. The fix is to clip each dim of sizePerThread by shared vec, so
// that the vectorization of load and store are equal along the contiguous
// dimension. In the above example, each thread will then only own 8B contiguous
// global data.
struct ClipAsyncCopySizePerThread
    : public OpRewritePattern<AsyncCopyGlobalToLocalOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AsyncCopyGlobalToLocalOp copyOp,
                                PatternRewriter &rewriter) const override {
    Value src = copyOp.getSrc();
    Value mask = copyOp.getMask();
    Value other = copyOp.getOther();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<MemDescType>(copyOp.getResult().getType());
    auto blockedEnc = dyn_cast<BlockedEncodingAttr>(srcTy.getEncoding());
    if (!blockedEnc)
      return rewriter.notifyMatchFailure(copyOp,
                                         "src must be of blocked encoding");
    auto sharedEnc = dyn_cast<SwizzledSharedEncodingAttr>(dstTy.getEncoding());
    if (!sharedEnc)
      return failure();
    auto sharedVec = sharedEnc.getVec();

    // obtain max contiguous copy size
    // Note this can be further optimized, as copyContigSize can be even
    // smaller when lowering, depending on contiguity and mask alignment
    // (see AsyncCopyGlobalToLocalOpConversion)
    LinearLayout regLayout =
        triton::gpu::toLinearLayout(srcTy.getShape(), blockedEnc);
    LinearLayout sharedLayout =
        triton::gpu::toLinearLayout(srcTy.getShape(), sharedEnc);
    auto copyContigSize =
        regLayout.invertAndCompose(sharedLayout).getNumConsecutiveInOut();

    // obtain block sizePerThread along contig dim
    auto contigPerThread = getContigPerThread(srcTy);
    auto blockContigSize = contigPerThread[blockedEnc.getOrder()[0]];

    if (blockContigSize <= copyContigSize)
      return rewriter.notifyMatchFailure(
          copyOp,
          "blocked sizePerThread along contiguous dim must be greater than the "
          "max contiguous copy size ");

    contigPerThread[blockedEnc.getOrder()[0]] = copyContigSize;

    // obtain new blockedEnc based on clipped sizePerThread
    auto mod = copyOp->getParentOfType<ModuleOp>();
    int numWarps = lookupNumWarps(copyOp);
    int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
    auto newBlockEnc = BlockedEncodingAttr::get(
        copyOp.getContext(), srcTy.getShape(), contigPerThread,
        blockedEnc.getOrder(), numWarps, threadsPerWarp,
        blockedEnc.getCTALayout());

    // insert cvt's after src, mask, and other
    auto convertBlockLayout = [&](Value src, BlockedEncodingAttr enc) {
      auto ty = cast<TensorType>(src.getType());
      auto newTy =
          RankedTensorType::get(ty.getShape(), ty.getElementType(), enc);
      auto cvt = rewriter.create<ConvertLayoutOp>(copyOp->getLoc(), newTy, src);
      return cvt.getResult();
    };
    src = convertBlockLayout(src, newBlockEnc);
    if (mask)
      mask = convertBlockLayout(mask, newBlockEnc);
    if (other)
      other = convertBlockLayout(other, newBlockEnc);

    rewriter.modifyOpInPlace(copyOp, [&]() {
      copyOp.getSrcMutable().assign(src);
      if (mask)
        copyOp.getMaskMutable().assign(mask);
      if (other)
        copyOp.getOtherMutable().assign(other);
    });

    return success();
  }
};

class CoalesceAsyncCopyPass
    : public impl::TritonGPUCoalesceAsyncCopyBase<CoalesceAsyncCopyPass> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *context = &getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<ClipAsyncCopySizePerThread>(context);

    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
