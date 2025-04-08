#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include <memory>

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;

template <typename TCGen5MMAOpTy>
class SyncMMALowering : public OpRewritePattern<TCGen5MMAOpTy> {
public:
  using OpRewritePattern<TCGen5MMAOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(TCGen5MMAOpTy op,
                                PatternRewriter &rewriter) const override {
    // If the op doesn't have synchronous semantic skip the pattern.
    if (op.getBarrier())
      return failure();
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
    Attribute sharedMemorySpace = SharedMemorySpaceAttr::get(ctx);
    auto barrierCTALayout = CTALayoutAttr::get(
        /*context=*/ctx, /*CTAsPerCGA=*/{1},
        /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
    auto barrierEncoding =
        SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, barrierCTALayout);
    MemDescType barrierMemDescType =
        MemDescType::get({1}, rewriter.getI64Type(), barrierEncoding,
                         sharedMemorySpace, /*mutableMemory=*/true);
    Value barrierAlloc =
        rewriter.create<LocalAllocOp>(loc, barrierMemDescType, Value());
    rewriter.create<InitBarrierOp>(loc, barrierAlloc, 1);
    op.getBarrierMutable().assign(barrierAlloc);

    rewriter.setInsertionPointAfter(op);
    Value phase = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    rewriter.create<WaitBarrierOp>(loc, barrierAlloc, phase);
    rewriter.create<InvalBarrierOp>(loc, barrierAlloc);
    return success();
  }
};

struct TCGen5MMAScaleSharedToTmemConversion
    : public OpRewritePattern<TCGen5MMAScaledOp> {
  using OpRewritePattern<TCGen5MMAScaledOp>::OpRewritePattern;

  // Create a tmem_copy of scales from shared memory to tmem. `rows` is the M or
  // N of the MMA operation (for LHS or RHS respectively).
  bool lowerScaleToTmem(OpOperand &operand, PatternRewriter &rewriter,
                        int rows) const {
    Location loc = operand.getOwner()->getLoc();
    MLIRContext *context = operand.getOwner()->getContext();
    Attribute tensorMemorySpace = TensorMemorySpaceAttr::get(context);
    auto oldType = cast<MemDescType>(operand.get().getType());
    auto numElems = product(oldType.getShape());
    Type elType = oldType.getElementType();
    SwizzledSharedEncodingAttr oldEncoding =
        cast<SwizzledSharedEncodingAttr>(oldType.getEncoding());
    CTALayoutAttr CTALayout = getCTALayout(oldEncoding);
    ArrayRef<unsigned> CTASplitNum = CTALayout.getCTASplitNum();
    // Distribute the scales across the rows of the MMA operation.
    SmallVector<int64_t> shape = {rows, numElems / rows};
    Attribute scaleEncoding = TensorMemoryScalesEncodingAttr::get(
        context, CTASplitNum[0], CTASplitNum[1]);
    Type scaleAType =
        MemDescType::get(shape, elType, scaleEncoding, tensorMemorySpace,
                         /*mutableMemory=*/true);
    auto tmemAlloc = rewriter.create<TMEMAllocOp>(loc, scaleAType, Value());
    rewriter.create<TMEMCopyOp>(loc, operand.get(), tmemAlloc,
                                /*barrier*/ Value());
    operand.set(tmemAlloc);
    return true;
  }

  LogicalResult matchAndRewrite(TCGen5MMAScaledOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = op->getContext();
    auto aScaleType = op.getAScale().getType();
    auto bScaleType = op.getBScale().getType();
    int blockM = op.getA()
                     .getType()
                     .getShape()[op.getA().getType().getShape().size() - 2];
    int blockN = op.getB()
                     .getType()
                     .getShape()[op.getB().getType().getShape().size() - 1];
    int blockK = op.getA()
                     .getType()
                     .getShape()[op.getA().getType().getShape().size() - 1];
    bool anyChanged = false;
    if (isa<SwizzledSharedEncodingAttr>(aScaleType.getEncoding())) {
      anyChanged = lowerScaleToTmem(op.getAScaleMutable(), rewriter, blockM);
    }
    if (isa<SwizzledSharedEncodingAttr>(bScaleType.getEncoding())) {
      anyChanged = lowerScaleToTmem(op.getBScaleMutable(), rewriter, blockN);
    }
    return LogicalResult::success(anyChanged);
  }
};

class TritonNvidiaGPUMMALoweringPass
    : public TritonNvidiaGPUMMALoweringPassBase<
          TritonNvidiaGPUMMALoweringPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns
        .add<SyncMMALowering<TCGen5MMAOp>, SyncMMALowering<TCGen5MMAScaledOp>,
             TCGen5MMAScaleSharedToTmemConversion>(context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTritonNvidiaGPUMMALoweringPass() {
  return std::make_unique<TritonNvidiaGPUMMALoweringPass>();
}
