#include "triton/Conversion/TritonGPUToLLVM/FMADotUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace ::mlir::triton::gpu;

namespace {
class GenericFMAVectorMultiplier : public FMAVectorMultiplier {
  OpBuilder &builder;
  Location loc;

public:
  GenericFMAVectorMultiplier(OpBuilder &builder, Location loc)
      : builder(builder), loc(loc) {}

  Value multiplyVectors(ArrayRef<Value> a, ArrayRef<Value> b,
                        Value c) override {
    auto K = a.size();
    assert(b.size() == K);
    Value accum = c;
    for (auto [aElem, bElem] : llvm::zip(a, b))
      accum = builder.create<LLVM::FMulAddOp>(loc, aElem, bElem, accum);
    return accum;
  }
};

} // namespace

LogicalResult convertFMADot(DotOp op, DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();
  GenericFMAVectorMultiplier multiplier(rewriter, loc);
  return parametricConvertFMADot(op, adaptor, typeConverter, rewriter,
                                 multiplier);
}
