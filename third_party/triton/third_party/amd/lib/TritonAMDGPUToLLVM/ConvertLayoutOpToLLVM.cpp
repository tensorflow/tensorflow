#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MemDescType;

namespace SharedToDotOperandMFMA {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor,
                    DotOperandEncodingAttr bEncoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread);
} // namespace SharedToDotOperandMFMA

namespace SharedToDotOperandWMMA {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor,
                    DotOperandEncodingAttr bEncoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread);
} // namespace SharedToDotOperandWMMA

namespace {

struct ConvertLayoutOpMFMAToDotOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  explicit ConvertLayoutOpMFMAToDotOpConversion(
      LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
      PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp>(typeConverter,
                                                             benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = cast<RankedTensorType>(op.getSrc().getType());
    auto dstType = cast<RankedTensorType>(op.getType());

    if (!matchMFMAAndDotOperandShuffleCase(srcType, dstType))
      return failure();

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    SmallVector<Value> inVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    if (inVals.empty() || inVals.size() % 8 != 0)
      return failure();

    auto mfmaLayout = dyn_cast<AMDMfmaEncodingAttr>(srcType.getEncoding());
    assert((mfmaLayout.getMDim() == 16 || mfmaLayout.getMDim() == 32) &&
           "Expected MFMA size 16 or 32");
    assert(triton::gpu::lookupThreadsPerWarp(rewriter) == 64 &&
           "Expected warp size 64 for MFMA");

    auto elemTy = int_ty(8);
    auto vecTy = vec_ty(elemTy, 4);

    Value c16 = b.i32_val(16);
    Value c32 = b.i32_val(32);
    Value c48 = b.i32_val(48);
    Value c64 = b.i32_val(64);

    Value threadId = getThreadId(rewriter, loc);
    Value laneId = b.urem(threadId, c64);

    Value mask0 = b.icmp_slt(laneId, c32);
    Value mask1 = b.icmp_slt(b.urem(laneId, c32), c16);

    Value addrShift16 = b.urem(b.add(laneId, c16), c64);
    Value addrShift32 = b.urem(b.add(laneId, c32), c64);
    Value addrShift48 = b.urem(b.add(laneId, c48), c64);

    SmallVector<Value> outVals;
    for (size_t startIdx = 0; startIdx < inVals.size(); startIdx += 8) {
      Value vec0 = b.undef(vecTy);
      for (size_t vIdx = 0; vIdx < 4; ++vIdx) {
        vec0 = b.insert_element(vecTy, vec0, inVals[startIdx + vIdx],
                                b.i32_val(vIdx));
      }
      Value vec1 = b.undef(vecTy);
      for (size_t vIdx = 0; vIdx < 4; ++vIdx) {
        vec1 = b.insert_element(vecTy, vec1, inVals[startIdx + vIdx + 4],
                                b.i32_val(vIdx));
      }

      Value resVec0, resVec1;
      if (mfmaLayout.getMDim() == 32) {
        /*
        Using wave shuffle to convert layouts (32x32x16 case):
        1) Input MMA layout (32x32, fp8, 16 values):
         _____________________________________________________________
        |(t0  v0 v1 v2 v3) (t32 v0 v1 v2 v3) ... (t32 v12 v13 v14 v15)|
        | ...                                ...                      |
        |(t31 v0 v1 v2 v3) (t63 v0 v1 v2 v3) ... (t63 v12 v13 v14 v15)|
        |_____________________________________________________________|

        2) Output Dot operand layout (two 32x16 tiles, fp8, 8 values each):
         ____________________________________________________________  ___
        |(t0  v0 v1 v2 v3 v4 v5 v6 v7) (t32 v0 v1 v2 v3 v4 v5 v6 v7) ||
        | ...                           ...                          ||...
        |(t31 v0 v1 v2 v3 v4 v5 v6 v7) (t63 v0 v1 v2 v3 v4 v5 v6 v7) ||
        |____________________________________________________________||___
        */

        Value shflVec0 = b.bitcast(
            targetInfo.shuffleIdx(rewriter, loc, b.bitcast(vec0, int_ty(32)),
                                  addrShift32),
            vecTy);
        Value shflVec1 = b.bitcast(
            targetInfo.shuffleIdx(rewriter, loc, b.bitcast(vec1, int_ty(32)),
                                  addrShift32),
            vecTy);

        resVec0 = b.select(mask0, vec0, shflVec1);
        resVec1 = b.select(mask0, shflVec0, vec1);
      } else if (mfmaLayout.getMDim() == 16) {
        /*
        16x16x32 case:
        1) Input MMA layout (two 16x16, fp8, 4 values each):
         _________________________________________________________  ___________
        |(t0  v0 v1 v2 v3) (t16 v0 v1 v2 v3) ... (t48 v0 v1 v2 v3)||(t0  v4 ...
        | ...                                ...                  || ...
        |(t15 v0 v1 v2 v3) (t31 v0 v1 v2 v3) ... (t63 v0 v1 v2 v3)||(t15 v4 ...
        |_________________________________________________________||___________

        2) Output Dot operand layout (16x32 tile, fp8, 8 values):
         ________________________________________________________________
        |(t0  v0 v1 v2 v3 v4 v5 v6 v7) ... (t48 v0 v1 v2 v3 v4 v5 v6 v7) |
        | ...                          ...                               |
        |(t15 v0 v1 v2 v3 v4 v5 v6 v7) ... (t63 v0 v1 v2 v3 v4 v5 v6 v7) |
        |________________________________________________________________|
        */

        Value shflVec0_16 = b.bitcast(
            targetInfo.shuffleIdx(rewriter, loc, b.bitcast(vec0, int_ty(32)),
                                  addrShift16),
            vecTy);
        Value shflVec0_32 = b.bitcast(
            targetInfo.shuffleIdx(rewriter, loc, b.bitcast(vec0, int_ty(32)),
                                  addrShift32),
            vecTy);
        Value shflVec1_32 = b.bitcast(
            targetInfo.shuffleIdx(rewriter, loc, b.bitcast(vec1, int_ty(32)),
                                  addrShift32),
            vecTy);
        Value shflVec1_48 = b.bitcast(
            targetInfo.shuffleIdx(rewriter, loc, b.bitcast(vec1, int_ty(32)),
                                  addrShift48),
            vecTy);

        resVec0 = b.select(mask0, b.select(mask1, vec0, shflVec0_16),
                           b.select(mask1, shflVec1_32, shflVec1_48));
        resVec1 = b.select(mask0, b.select(mask1, shflVec0_16, shflVec0_32),
                           b.select(mask1, shflVec1_48, vec1));
      }

      for (size_t vIdx = 0; vIdx < 4; ++vIdx) {
        outVals.push_back(b.extract_element(elemTy, resVec0, b.i32_val(vIdx)));
      }
      for (size_t vIdx = 0; vIdx < 4; ++vIdx) {
        outVals.push_back(b.extract_element(elemTy, resVec1, b.i32_val(vIdx)));
      }
    }

    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

protected:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::AMD::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpMFMAToDotOpConversion>(typeConverter, targetInfo,
                                                     benefit);
}
