#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Conversion/MLIRTypes.h"

using namespace mlir;
using namespace mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

struct InThreadTransposeOpConversion
    : public OpConversionPattern<triton::amdgpu::InThreadTransposeOp> {
public:
  explicit InThreadTransposeOpConversion(MLIRContext *ctx,
                                         PatternBenefit benefit)
      : OpConversionPattern(ctx, benefit) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::InThreadTransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(op, op.getType(),
                                                      op.getSrc());
    return success();
  }
};

} // namespace

namespace mlir::triton::AMD {

void populateInThreadTransposeOpToTTGPatterns(RewritePatternSet &patterns,
                                              PatternBenefit benefit) {
  patterns.add<InThreadTransposeOpConversion>(patterns.getContext(), benefit);
}

} // namespace mlir::triton::AMD
