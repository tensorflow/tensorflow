#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "third_party/proton/dialect/include/TritonProtonToLLVM/PatternTritonProtonOpToLLVM.h"

namespace {

struct RecordOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::RecordOp> {
  explicit RecordOpConversion(LLVMTypeConverter &typeConverter,
                              const TargetInfoBase &targetInfo,
                              PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::RecordOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::RecordOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.eraseOp(op);
    return success();
  }

protected:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::proton::populateRecordOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<RecordOpConversion>(typeConverter, targetInfo, benefit);
}
