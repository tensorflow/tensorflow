#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

struct ConvertLayoutOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  ConvertLayoutOpConversion(const LLVMTypeConverter &typeConverter,
                            const NVIDIA::TargetInfo &targetInfo,
                            PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isa<MmaEncodingTrait, BlockedEncodingAttr, SliceEncodingAttr>(
            srcLayout) &&
        isa<MmaEncodingTrait, BlockedEncodingAttr, SliceEncodingAttr>(
            dstLayout)) {
      if (shouldUseDistSmem(srcLayout, dstLayout))
        return lowerDistToDistWithDistSmem(op, adaptor, rewriter, targetInfo);
    }
    if (isa<NvidiaMmaEncodingAttr>(srcLayout) &&
        isa<DotOperandEncodingAttr>(dstLayout)) {
      return lowerMmaToDotOperand(op, adaptor, rewriter);
    }

    return failure();
  }

private:
  LogicalResult
  lowerDistToDistWithDistSmem(triton::gpu::ConvertLayoutOp op,
                              OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter,
                              const TargetInfoBase &targetInfo) const {
    MLIRContext *ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto typeConverter = getTypeConverter();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    auto srcLayout = srcTy.getEncoding();
    auto dstLayout = dstTy.getEncoding();
    auto srcShapePerCTA = getShapePerCTA(srcTy);
    auto srcCTAsPerCGA = triton::gpu::getCTAsPerCGA(srcLayout);
    auto srcCTAOrder = triton::gpu::getCTAOrder(srcLayout);
    unsigned rank = srcShapePerCTA.size();

    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);

    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    smemBase = b.bitcast(smemBase, elemPtrTy);
    auto smemShape = convertType<unsigned, int64_t>(srcShapePerCTA);

    // Store to local shared memory
    {
      auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      auto inIndices = emitIndices(loc, rewriter, targetInfo, srcLayout, srcTy,
                                   /*withCTAOffset*/ false);

      assert(inIndices.size() == inVals.size() &&
             "Unexpected number of indices emitted");

      for (unsigned i = 0; i < inIndices.size(); ++i) {
        Value offset = LLVM::linearize(rewriter, loc, inIndices[i], smemShape);
        Value ptr = b.gep(elemPtrTy, llvmElemTy, smemBase, offset);
        b.store(inVals[i], ptr);
      }
    }

    // Cluster barrier
    rewriter.create<triton::nvidia_gpu::ClusterArriveOp>(loc, false);
    rewriter.create<triton::nvidia_gpu::ClusterWaitOp>(loc);

    // Load from remote shared memory
    {
      SmallVector<Value> srcShapePerCTACache;
      for (unsigned i = 0; i < rank; ++i)
        srcShapePerCTACache.push_back(b.i32_val(srcShapePerCTA[i]));

      SmallVector<Value> outVals;
      auto outIndices = emitIndices(loc, rewriter, targetInfo, dstLayout, dstTy,
                                    /*withCTAOffset*/ true);

      for (unsigned i = 0; i < outIndices.size(); ++i) {
        auto coord = outIndices[i];
        assert(coord.size() == rank && "Unexpected rank of index emitted");

        SmallVector<Value> multiDimCTAId, localCoord;
        for (unsigned d = 0; d < rank; ++d) {
          multiDimCTAId.push_back(b.udiv(coord[d], srcShapePerCTACache[d]));
          localCoord.push_back(b.urem(coord[d], srcShapePerCTACache[d]));
        }

        Value remoteCTAId = LLVM::linearize(rewriter, loc, multiDimCTAId,
                                            srcCTAsPerCGA, srcCTAOrder);
        Value localOffset =
            LLVM::linearize(rewriter, loc, localCoord, smemShape);

        Value ptr = b.gep(elemPtrTy, llvmElemTy, smemBase, localOffset);
        outVals.push_back(targetInfo.loadDShared(rewriter, loc, ptr,
                                                 remoteCTAId, llvmElemTy,
                                                 /*pred=*/b.true_val()));
      }

      Value result =
          packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
      rewriter.replaceOp(op, result);
    }

    // Cluster barrier
    rewriter.create<triton::nvidia_gpu::ClusterArriveOp>(loc, false);
    rewriter.create<triton::nvidia_gpu::ClusterWaitOp>(loc);

    return success();
  }

  // Convert from accumulator MMA layout to 8bit dot operand layout.
  // The conversion logic is taken from:
  // https://github.com/ColfaxResearch/cutlass-kernels/blob/a9de6446c1c0415c926025cea284210c799b11f8/src/fmha-pipeline/reg2reg.h#L45
  void
  convertMMAV3To8BitsDotOperand(triton::gpu::ConvertLayoutOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto dstTy = op.getType();
    auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> retVals;
    for (int i = 0; i < vals.size(); i += 8) {
      Value upper = b.undef(vec_ty(i8_ty, 4));
      for (int j = 0; j < 4; j++) {
        upper = b.insert_element(vec_ty(i8_ty, 4), upper, vals[i + j],
                                 b.i32_val(j));
      }
      upper = b.bitcast(upper, i32_ty);
      Value lower = b.undef(vec_ty(i8_ty, 4));
      for (int j = 0; j < 4; j++) {
        lower = b.insert_element(vec_ty(i8_ty, 4), lower, vals[i + 4 + j],
                                 b.i32_val(j));
      }
      lower = b.bitcast(lower, i32_ty);

      Value threadIdMod4 = b.urem(getThreadId(rewriter, loc), b.i32_val(4));
      Value cnd = b.or_(b.icmp_eq(threadIdMod4, b.i32_val(0)),
                        b.icmp_eq(threadIdMod4, b.i32_val(3)));
      Value selectorEx0 = b.select(cnd, b.i32_val(0x3210), b.i32_val(0x7654));
      Value selectorEx1 = b.select(cnd, b.i32_val(0x7654), b.i32_val(0x3210));
      Value selectorEx4 = b.select(cnd, b.i32_val(0x5410), b.i32_val(0x1054));
      Value selectorEx5 = b.select(cnd, b.i32_val(0x7632), b.i32_val(0x3276));

      Value isOne = b.icmp_eq(threadIdMod4, b.i32_val(1));
      Value isTwo = b.icmp_eq(threadIdMod4, b.i32_val(2));
      Value isThree = b.icmp_eq(threadIdMod4, b.i32_val(3));
      Value upperIdx = b.i32_val(0);
      upperIdx = b.select(isOne, b.i32_val(3), upperIdx);
      upperIdx = b.select(isTwo, b.i32_val(1), upperIdx);
      upperIdx = b.select(isThree, b.i32_val(2), upperIdx);

      Value lowerIdx = b.i32_val(1);
      lowerIdx = b.select(isOne, b.i32_val(2), lowerIdx);
      lowerIdx = b.select(isTwo, b.i32_val(0), lowerIdx);
      lowerIdx = b.select(isThree, b.i32_val(3), lowerIdx);

      Value upper0 =
          LLVM::NVIDIA::permute(loc, rewriter, upper, lower, selectorEx0);
      Value lower0 =
          LLVM::NVIDIA::permute(loc, rewriter, upper, lower, selectorEx1);
      Value mask = b.i32_val(0xFFFFFFFF);
      // Set clamp tp shuffle only within 4 lanes.
      Value clamp = b.i32_val(0x1C1F);
      upper0 =
          rewriter.create<NVVM::ShflOp>(loc, i32_ty, mask, upper0, upperIdx,
                                        clamp, NVVM::ShflKind::idx, UnitAttr());
      lower0 =
          rewriter.create<NVVM::ShflOp>(loc, i32_ty, mask, lower0, lowerIdx,
                                        clamp, NVVM::ShflKind::idx, UnitAttr());
      Value upper1 =
          LLVM::NVIDIA::permute(loc, rewriter, upper0, lower0, selectorEx4);
      Value vecVal = b.bitcast(upper1, vec_ty(i8_ty, 4));
      for (int i = 0; i < 4; i++) {
        retVals.push_back(b.extract_element(i8_ty, vecVal, b.i32_val(i)));
      }
      Value lower1 =
          LLVM::NVIDIA::permute(loc, rewriter, upper0, lower0, selectorEx5);
      vecVal = b.bitcast(lower1, vec_ty(i8_ty, 4));
      for (int i = 0; i < 4; i++) {
        retVals.push_back(b.extract_element(i8_ty, vecVal, b.i32_val(i)));
      }
    }
    Value result =
        packLLElements(loc, getTypeConverter(), retVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);
  }

  // mma -> dot_operand
  LogicalResult
  lowerMmaToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    if (matchMmaV3AndDotOperandLayout(srcTy, dstTy)) {
      assert(srcTy.getElementType().getIntOrFloatBitWidth() == 8 &&
             "Unsupported type size.");
      convertMMAV3To8BitsDotOperand(op, adaptor, rewriter);
      return success();
    }
    return failure();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::NVIDIA::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // Give this convertLayoutOpConversion a higher benefit as it only matches
  // optimized or cross CTA cases
  patterns.add<ConvertLayoutOpConversion>(typeConverter, targetInfo,
                                          benefit.getBenefit() + 1);
  mlir::triton::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                                      patterns, benefit);
}
