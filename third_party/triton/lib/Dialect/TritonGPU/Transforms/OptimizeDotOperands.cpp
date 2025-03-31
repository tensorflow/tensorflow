#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <memory>

namespace mlir::triton::gpu {

namespace {
// Given
//   dot(convert(trans(src)) #dot_operand) ->
//   dot(convert(local_load(trans(alloc(src)))))
// change the encoding of the inner convert to a special, swizzled shared
// encoding.
class SwizzleShmemConvert : public OpRewritePattern<ConvertLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp cvtOp,
                                PatternRewriter &rewriter) const override {
    if (!cvtOp->hasOneUse() ||
        !isa<triton::DotOp>(cvtOp->use_begin()->getOwner()))
      return failure();
    // Match outerCvt(trans(innerCvt(x))).
    auto trans = cvtOp.getSrc().getDefiningOp<TransOp>();
    if (!trans || trans.getOrder() != ArrayRef<int32_t>{1, 0})
      return failure();

    RankedTensorType srcTy = trans.getSrc().getType();

    if (auto srcCvt = trans.getSrc().getDefiningOp<ConvertLayoutOp>()) {
      srcTy = srcCvt.getSrc().getType();
    }
    RankedTensorType sharedLoadTy = cvtOp.getType();
    auto cvtEncoding =
        dyn_cast<DotOperandEncodingAttr>(sharedLoadTy.getEncoding());
    if (!cvtEncoding)
      return failure();

    // TODO(Qingyi): need to check whether the CTALayout of innerCvtEnc should
    // be used here. For tests where numCTAs = 1, this is not a problem since
    // all CTALayouts are the same.
    //
    // Set needTrans to true here. newInnerCvtEnc is computed based on
    // argEncoding which is before the transpose. Without needTrans we will
    // compute vec and maxPhase based on incorrect m, n and k size of mma. The
    // type inference of MemDescTransOp simply swap the order but doesn't fix
    // the vec and maxPhase for the YType, hence it would causing incorrect
    // swizzling code.
    auto newInnerCvtEnc = SwizzledSharedEncodingAttr::get(
        getContext(), cvtEncoding, srcTy.getShape(),
        /*order=*/getOrderForMemory(srcTy),
        triton::gpu::getCTALayout(srcTy.getEncoding()), srcTy.getElementType(),
        /*needTrans=*/true);
    if (newInnerCvtEnc == cvtEncoding)
      return failure();
    rewriter.setInsertionPoint(trans);
    auto sharedMemorySpace = SharedMemorySpaceAttr::get(getContext());
    auto alloc = rewriter.create<LocalAllocOp>(
        trans.getLoc(),
        MemDescType::get(srcTy.getShape(), srcTy.getElementType(),
                         newInnerCvtEnc, sharedMemorySpace),
        trans.getSrc());
    auto newTrans = rewriter.create<MemDescTransOp>(trans.getLoc(), alloc,
                                                    ArrayRef<int32_t>({1, 0}));
    rewriter.replaceOpWithNewOp<LocalLoadOp>(trans, sharedLoadTy, newTrans);
    return success();
  }
};

// Rewrite
//
//   dot(alloc(trans() #shared1) ->
//   dot(trans(alloc() #shared2))
//
// if dot is an MMAv3/v5 (because MMAv3/v5 allows us to fold transposes).
class FuseTransMMAV3Plus : public OpRewritePattern<LocalAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LocalAllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    if (!allocOp.getSrc() || !allocOp->hasOneUse() ||
        !isa<triton::nvidia_gpu::WarpGroupDotOp,
             triton::nvidia_gpu::MMAv5OpInterface>(
            *allocOp->getUsers().begin()))
      return failure();

    auto dot = *allocOp->getUsers().begin();
    // Match outerCvt(trans(innerCvt(x))).
    auto trans = allocOp.getSrc().getDefiningOp<TransOp>();
    if (!trans || trans.getOrder() != ArrayRef<int32_t>({1, 0}))
      return failure();

    MemDescType allocType = allocOp.getType();
    auto allocEncoding = cast<NVMMASharedEncodingAttr>(allocType.getEncoding());
    RankedTensorType srcTy = trans.getSrc().getType();

    // MMAv3 with transpose only supports f16 and bf16.  Fall back to MMAv3
    // without transpose for other data types.)
    auto newInnerCvtOrder = getOrderForMemory(srcTy);
    if (auto cvt = trans.getSrc().getDefiningOp<ConvertLayoutOp>()) {
      newInnerCvtOrder = getOrderForMemory(cvt.getSrc().getType());
    }
    auto srcElemTy = allocType.getElementType();
    if (!srcElemTy.isF16() && !srcElemTy.isBF16()) {
      if (allocOp.getResult() == dot->getOperand(0)) {
        newInnerCvtOrder = {0, 1};
      } else if (allocOp.getResult() == dot->getOperand(1)) {
        newInnerCvtOrder = {1, 0};
      }
    }

    // TODO(Qingyi): need to check whether the CTALayout of innerCvtEnc should
    // be used here. For tests where numCTAs = 1, this is not a problem since
    // all CTALayouts are the same.
    auto newInnerEnc = NVMMASharedEncodingAttr::get(
        getContext(), srcTy.getShape(), newInnerCvtOrder,
        allocEncoding.getCTALayout(), srcTy.getElementType(),
        allocEncoding.getFp4Padded());

    MemDescType innerTy =
        MemDescType::get(srcTy.getShape(), srcTy.getElementType(), newInnerEnc,
                         allocType.getMemorySpace());
    auto newAlloc = rewriter.create<LocalAllocOp>(allocOp.getLoc(), innerTy,
                                                  trans.getSrc());
    rewriter.replaceOpWithNewOp<MemDescTransOp>(allocOp, newAlloc,
                                                ArrayRef<int32_t>({1, 0}));
    return success();
  }
};

// Inject TMEM copy instructions into IR to efficiently load blocked scales for
// scaled dot
class UseShmemForScales
    : public OpRewritePattern<triton::nvidia_gpu::TCGen5MMAScaledOp> {
public:
  using OpRewritePattern<
      triton::nvidia_gpu::TCGen5MMAScaledOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::nvidia_gpu::TCGen5MMAScaledOp mmaOp,
                                PatternRewriter &rewriter) const override {
    auto aScale = mmaOp.getAScale();
    auto bScale = mmaOp.getBScale();
    LogicalResult ret = failure();
    if (aScale && isa<triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(
                      aScale.getType().getEncoding())) {
      if (rewriteOperand(mmaOp.getAScaleMutable(), rewriter).succeeded())
        ret = success();
    }
    if (bScale && isa<triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(
                      bScale.getType().getEncoding())) {
      if (rewriteOperand(mmaOp.getBScaleMutable(), rewriter).succeeded())
        ret = success();
    }
    return ret;
  }

private:
  LogicalResult rewriteOperand(OpOperand &opOperand,
                               PatternRewriter &rewriter) const {
    auto src = cast<TypedValue<MemDescType>>(opOperand.get());
    auto tmemAlloc = src.getDefiningOp<triton::nvidia_gpu::TMEMAllocOp>();
    if (!tmemAlloc) {
      return failure();
    }
    auto dstType = tmemAlloc.getResult().getType();

    if (!tmemAlloc.getSrc()) {
      return failure();
    }

    // Look for a sequence
    //    local_load
    // -> reshape(..., (BLOCK_MN / 128, BLOCK_K / scale_vec_size / 4, 32, 4,
    // 4)
    // -> transpose(..., (0, 3, 2, 1, 4))
    // -> reshape(..., (BLOCK_MN, BLOCK_K / scale_vec_size)
    // -> tmem_alloc
    // -> tc_gen_mma_scaled
    // and replace it with local_alloc -> tc_gen_mma_scaled
    auto scale2DShape = dstType.getShape();
    auto blockMN = scale2DShape[0];
    auto numScales = scale2DShape[1];
    const SmallVector<int> transposeOrder{0, 3, 2, 1, 4};
    const SmallVector<int64_t> reshape5DShape{blockMN / 128, numScales / 4, 32,
                                              4, 4};

    auto reshapeOp2D = getNextOp<triton::ReshapeOp>(tmemAlloc.getSrc());
    if (!reshapeOp2D ||
        reshapeOp2D.getResult().getType().getShape() != scale2DShape) {
      return failure();
    }

    auto transOp = getNextOp<triton::TransOp>(reshapeOp2D.getSrc());
    if (!transOp || transOp.getOrder() != ArrayRef<int>(transposeOrder)) {
      return failure();
    }

    auto reshapeOp5D = getNextOp<triton::ReshapeOp>(transOp.getSrc());
    if (!reshapeOp5D || reshapeOp5D.getResult().getType().getShape() !=
                            ArrayRef<int64_t>(reshape5DShape)) {
      return failure();
    }

    auto localLoad = getNextOp<triton::gpu::LocalLoadOp>(reshapeOp5D.getSrc());
    if (!localLoad || !isTmemCopyCompatible(localLoad.getSrc().getType())) {
      return failure();
    }
    opOperand.assign(localLoad.getSrc());
    return success();
  }

  template <typename Op> Op getNextOp(Value op) const {
    while (auto cvtOp = op.getDefiningOp<ConvertLayoutOp>()) {
      op = cvtOp.getSrc();
    }
    return op.getDefiningOp<Op>();
  }

  bool isDescendingOrder(triton::gpu::MemDescType scale) const {
    auto order = triton::gpu::getOrder(scale);
    auto rank = scale.getRank();
    for (int i = 0; i < rank; ++i) {
      if (order[i] != rank - 1 - i)
        return false;
    }
    return true;
  }

  bool isTmemCopyCompatible(triton::gpu::MemDescType scaleType) const {
    // TMEM copy expects that blocked scale "chunks" in SMEM are stored in
    // innermost axes contiguously.
    if (!isDescendingOrder(scaleType))
      return false;

    auto sharedEnc =
        cast<triton::gpu::SwizzledSharedEncodingAttr>(scaleType.getEncoding());
    if (sharedEnc.getMaxPhase() != 1 || sharedEnc.getPerPhase() != 1 ||
        sharedEnc.getVec() != 1) {
      // For now, we do not expect swizzling to be applied to the scale SMEM.
      // This is currently true for non-matmul operand SMEM allocated during
      // pipelining.
      return false;
    }

    if (scaleType.getRank() != 2) {
      // TODO: Add support for higher rank when 5D coalesced load is fixed
      // or 4D TMA is supported.
      return false;
    }

    auto elemBits = scaleType.getElementType().getIntOrFloatBitWidth();

    // We assume that 32x128b chunks are flattened into the inner most axis.
    auto innerMostBits =
        scaleType.getDimSize(scaleType.getRank() - 1) * elemBits;
    return innerMostBits % (32 * 128) == 0;
  }
};

} // namespace

#define GEN_PASS_DEF_TRITONGPUOPTIMIZEDOTOPERANDS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUOptimizeDotOperandsPass
    : public impl::TritonGPUOptimizeDotOperandsBase<
          TritonGPUOptimizeDotOperandsPass> {
public:
  using impl::TritonGPUOptimizeDotOperandsBase<
      TritonGPUOptimizeDotOperandsPass>::TritonGPUOptimizeDotOperandsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    OpPassManager pm;
    pm.addPass(mlir::createCanonicalizerPass());
    if (failed(runPipeline(pm, m)))
      return signalPassFailure();

    mlir::RewritePatternSet patterns(context);
    patterns.add<SwizzleShmemConvert>(context);
    patterns.add<FuseTransMMAV3Plus>(context);
    patterns.add<UseShmemForScales>(context);
    ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::triton::gpu
