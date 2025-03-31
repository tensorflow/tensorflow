#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

using namespace mlir;

using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;

namespace mlir::triton::AMD {
LogicalResult convertAMDFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                               const LLVMTypeConverter *typeConverter,
                               ConversionPatternRewriter &rewriter);

LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter);

LogicalResult convertScaledMFMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter);

LogicalResult convertWMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter);
} // namespace mlir::triton::AMD

namespace {
struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // D = A * B + C
    Value D = op.getResult();

    auto dEncoding = cast<RankedTensorType>(D.getType()).getEncoding();
    if (isa<AMDMfmaEncodingAttr>(dEncoding)) {
      return AMD::convertMFMA(op, adaptor, getTypeConverter(), rewriter);
    }
    if (isa<AMDWmmaEncodingAttr>(dEncoding)) {
      return AMD::convertWMMA(op, adaptor, getTypeConverter(), rewriter);
    }

    if (isa<BlockedEncodingAttr>(
            cast<RankedTensorType>(D.getType()).getEncoding()))
      return AMD::convertAMDFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }
};

struct ScaledDotOpConversion
    : public ConvertOpToLLVMPattern<triton::DotScaledOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  int mfmaVersion;
  int nonKDim;
  int kPack;

  ScaledDotOpConversion(LLVMTypeConverter &typeConverter, int mfmaVersion,
                        int nonKDim, int kPack, PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        mfmaVersion(mfmaVersion), nonKDim(nonKDim), kPack(kPack) {}

  LogicalResult
  matchAndRewrite(triton::DotScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return AMD::convertScaledMFMA(op, adaptor, getTypeConverter(), rewriter);
  }
};
} // namespace

namespace mlir::triton::AMD {
void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, benefit);
  patterns.add<ScaledDotOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::AMD
