#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace {

using namespace mlir;

struct AssertOpConversion : public ConvertOpToLLVMPattern<triton::AssertOp> {
  explicit AssertOpConversion(LLVMTypeConverter &typeConverter,
                              const TargetInfoBase &targetInfo,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AssertOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();
    auto elems = unpackLLElements(loc, adaptor.getCondition(), rewriter);
    auto elemTy = elems[0].getType();
    Value condition = b.int_val(elemTy.getIntOrFloatBitWidth(), 0);
    for (auto elem : elems) {
      if (elemTy.isSignedInteger() || elemTy.isSignlessInteger()) {
        condition = b.or_(
            condition,
            b.icmp_eq(elem, rewriter.create<LLVM::ConstantOp>(
                                loc, elemTy, rewriter.getZeroAttr(elemTy))));
      } else {
        assert(false && "Unsupported type for assert");
        return failure();
      }
    }
    llAssert(op, condition, adaptor.getMessage(), rewriter);
    if (isa<RankedTensorType>(op.getCondition().getType())) {
      // Add a barrier to avoid a race condition in case an assert is followed
      // by an op that may trap if the assert condition is true. Since the
      // tensor in those two operations may have different layout we need to
      // make sure all the threads are done executing the assert before going to
      // the next op.
      b.barrier();
    }
    rewriter.eraseOp(op);
    return success();
  }
  // op: the op at which the assert is inserted. Unlike printf, we need to
  // know about the op to split the block.
  void llAssert(Operation *op, Value condition, StringRef message,
                ConversionPatternRewriter &rewriter) const {

    auto ctx = rewriter.getContext();
    auto loc = op->getLoc();

    StringRef file = "unknown";
    StringRef func = "unknown";
    int line = 0;
    int col = 0;

    while (auto callLoc = dyn_cast<CallSiteLoc>(loc))
      loc = callLoc.getCallee();

    if (auto fileLineColLoc = dyn_cast<FileLineColLoc>(loc)) {
      file = fileLineColLoc.getFilename();
      line = fileLineColLoc.getLine();
      col = fileLineColLoc.getColumn();
    }

    // #block1
    // if (condition) {
    //   #block2
    //   __assertfail(message);
    // }
    // #block3
    Block *prevBlock = op->getBlock();

    Block *ifBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToStart(ifBlock);
    targetInfo.assertFail(rewriter, loc, message, file, func, line);

    // Split a block after the call.
    Block *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(ifBlock);
    rewriter.create<cf::BranchOp>(loc, thenBlock);
    rewriter.setInsertionPointToEnd(prevBlock);
    rewriter.create<cf::CondBranchOp>(loc, condition, ifBlock, thenBlock);
    rewriter.setInsertionPointToStart(thenBlock);
  }

protected:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populateAssertOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<AssertOpConversion>(typeConverter, targetInfo, benefit);
}
