#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using ::mlir::LLVM::AMD::isUsedByDotScaledOp;
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
struct LocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isa<DotOperandEncodingAttr>(dstLayout) &&
        isa<AMDMfmaEncodingAttr, AMDWmmaEncodingAttr>(
            cast<DotOperandEncodingAttr>(dstLayout).getParent())) {
      return lowerSharedToDotOperand(op, adaptor, getTypeConverter(), rewriter);
    }
    return failure();
  }

private:
  /// Lower ttg.local_load in dot operand layout if the operand parent layout is
  /// MFMA or WMMA.
  ///
  /// \returns value with packed loaded values or empty value if this local_load
  /// is not supproted.
  Value lowerSharedToDotOperandMMA(
      triton::gpu::LocalLoadOp op, triton::gpu::LocalLoadOpAdaptor adaptor,
      const LLVMTypeConverter *typeConverter,
      ConversionPatternRewriter &rewriter,
      const DotOperandEncodingAttr &dotOperandLayout) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto llvmElemTy = typeConverter->convertType(
        cast<MemDescType>(src.getType()).getElementType());

    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    Value res;
    auto dopOpParent = dotOperandLayout.getParent();
    if (isa<AMDMfmaEncodingAttr, AMDWmmaEncodingAttr>(dopOpParent)) {
      auto sharedToDotConvert = isa<AMDMfmaEncodingAttr>(dopOpParent)
                                    ? SharedToDotOperandMFMA::convertLayout
                                    : SharedToDotOperandWMMA::convertLayout;
      res = sharedToDotConvert(dotOperandLayout.getOpIdx(), rewriter, loc, src,
                               dotOperandLayout, smemObj, typeConverter,
                               getThreadId(rewriter, loc));
    } else {
      assert(false && "unsupported layout found");
    }
    return res;
  }

  // shared -> matrix_core_dot_operand
  LogicalResult
  lowerSharedToDotOperand(triton::gpu::LocalLoadOp op,
                          triton::gpu::LocalLoadOpAdaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) const {
    Value dst = op.getResult();
    auto dstTensorTy = cast<RankedTensorType>(dst.getType());
    auto dotOperandLayout =
        cast<DotOperandEncodingAttr>(dstTensorTy.getEncoding());

    Value res = lowerSharedToDotOperandMMA(op, adaptor, typeConverter, rewriter,
                                           dotOperandLayout);
    if (!res)
      return failure();
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct TransLocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  TransLocalLoadOpConversion(const LLVMTypeConverter &converter,
                             const AMD::TargetInfo &targetInfo,
                             PatternBenefit benefit = 2)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();

    if (canUseTransLoad(op, srcTy, dstTy)) {
      assert(checkPerformanceProperties(srcTy, dstTy));
      return lowerSharedToDotOperandTransLL(op, adaptor, getTypeConverter(),
                                            rewriter);
    }
    return failure();
  }

private:
  bool checkLayoutProperties(MemDescType srcTy, RankedTensorType dstTy) const {
    // Verify the layout properties required for using the ds_read_tr
    // instruction. This instruction is used to load non-k contiguous tensors
    // from shared memory into a dot layout with an MFMA layout parent.
    auto dotEnc = llvm::dyn_cast<DotOperandEncodingAttr>(dstTy.getEncoding());
    if (!dotEnc) {
      return false;
    }

    auto mfmaEnc = llvm::dyn_cast<AMDMfmaEncodingAttr>(dotEnc.getParent());
    if (!mfmaEnc) {
      return false;
    }

    auto sharedEnc =
        dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(srcTy.getEncoding());
    if (!sharedEnc)
      return false;

    int rank = dstTy.getRank();
    const int kDim = dotEnc.getOpIdx() == 0 ? rank - 1 : rank - 2;
    return kDim != sharedEnc.getOrder()[0];
  }

  bool checkPerformanceProperties(MemDescType srcTy,
                                  RankedTensorType dstTy) const {
    // Single rate MFMA insts:
    // fp16, bf16: mfma32x32x8, mfma16x16x16
    // fp8, bf8: mfma32x32x16, mfma16x16x32
    // int8: mfma32x32x16, mfma16x16x32
    //
    // Double rate MFMA insts:
    // fp16, bf16: mfma32x32x16, mfma16x16x32
    // fp8, bf8: mfma32x32x64, mfma16x16x128
    // i8: mfma32x32x32, mfma16x16x64
    //
    // Check that double-rate MFMA instructions are used whenever possible.
    // Single rate instructions should only be used if the K block size is not
    // large enough.
    auto dotEnc = llvm::cast<DotOperandEncodingAttr>(dstTy.getEncoding());
    auto mfmaEnc = llvm::cast<AMDMfmaEncodingAttr>(dotEnc.getParent());

    int rank = dstTy.getRank();
    auto bitwidth = typeConverter->convertType(dstTy.getElementType())
                        .getIntOrFloatBitWidth();
    int32_t kWidth = dotEnc.getKWidth();
    const int32_t mDim = mfmaEnc.getMDim();
    assert((mDim == 32 || mDim == 16) && "Invalid MFMA instruction dimension");

    const int kFactor = 16 / bitwidth;
    const int kSizeSingleRateMfma32 = 8 * kFactor;
    const int kSizeSingleRateMfma16 = 16 * kFactor;
    int largeTileThreshold =
        (mDim == 32) ? kSizeSingleRateMfma32 : kSizeSingleRateMfma16;

    // For FP8, wider MFMA instructions (scaled MFMA) have a k-dimension
    // that is four times of regular MFMA instructions.
    if (dstTy.getElementType().isFloat() && bitwidth == 8) {
      largeTileThreshold *= 2;
    }

    const auto shape = dstTy.getShape();
    const int kDim = dotEnc.getOpIdx() == 0 ? rank - 1 : rank - 2;

    const bool isLargeTile = shape[kDim] > largeTileThreshold;
    const int expectedKWidth = (isLargeTile ? 8 : 4) * kFactor;
    return kWidth == expectedKWidth;
  }

  bool checkCurrentLimitation(Operation *localLoad,
                              RankedTensorType dstTy) const {

    auto bitwidth = typeConverter->convertType(dstTy.getElementType())
                        .getIntOrFloatBitWidth();

    // Triton does not natively support the FP4 type, so it is packed and
    // represented as an i8. Currently, the only way to distinguish FP4 from an
    // actual int8 is by checking whether the localLoad is used in a scaled dot
    // operation, as int8 is never used in one.
    bool isFP4 = isUsedByDotScaledOp(localLoad) && bitwidth == 8 &&
                 dstTy.getElementType().isInteger();

    if (isFP4 || (bitwidth != 16 && bitwidth != 8)) {
      return false;
    }

    return true;
  }

  bool canUseTransLoad(Operation *localLoad, MemDescType srcTy,
                       RankedTensorType dstTy) const {
    auto bitwidth = typeConverter->convertType(dstTy.getElementType())
                        .getIntOrFloatBitWidth();

    // 1. Check GPU arch properties.
    if (!targetInfo.canUseLDSTransLoad(bitwidth)) {
      return false;
    }

    // 2. Check layout properties.
    if (!checkLayoutProperties(srcTy, dstTy)) {
      return false;
    }

    // 3. Check current limitations.
    if (!checkCurrentLimitation(localLoad, dstTy)) {
      return false;
    }

    return true;
  }

  LogicalResult
  lowerSharedToDotOperandTransLL(triton::gpu::LocalLoadOp op,
                                 triton::gpu::LocalLoadOpAdaptor adaptor,
                                 const LLVMTypeConverter *typeConverter,
                                 ConversionPatternRewriter &rewriter) const {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    auto dotEnc = cast<DotOperandEncodingAttr>(dstTy.getEncoding());
    auto shape = dstTy.getShape();

    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    auto ldsTransLayout = chooseDsReadB64TrLayout(dotEnc, shape, bitwidth);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    SmallVector<Value> outVals;
    SmallVector<Value> elemsI32;
    mlir::Type retTy = dstTy;
    bool valid = emitTransferBetweenRegistersAndShared(
        ldsTransLayout, srcTy, llvmElemTy,
        /*maxVecElems=*/std::nullopt, smemObj, loc, rewriter, targetInfo,
        [&](VectorType vecTy, Value vecAddr) {
          if (bitwidth == 16) {
            auto dsReadOp =
                rewriter.create<ROCDL::ds_read_tr16_b64>(loc, vecTy, vecAddr);
            Value vecVal = dsReadOp.getResult();
            for (int v = 0; v < vecTy.getNumElements(); v++) {
              outVals.push_back(
                  b.extract_element(llvmElemTy, vecVal, b.i32_val(v)));
            }
          } else {
            // pack elements in i32 vectors
            auto numElems = vecTy.getNumElements();
            auto numElemsI32 = (numElems * bitwidth / 32);
            auto i32VecTy = VectorType::get(numElemsI32, i32_ty);

            auto dsReadOp =
                rewriter.create<ROCDL::ds_read_tr8_b64>(loc, i32VecTy, vecAddr);
            Value vecVal = dsReadOp.getResult();
            for (auto i = 0; i < numElemsI32; ++i) {
              elemsI32.push_back(
                  b.extract_element(i32_ty, vecVal, b.i32_val(i)));
            }
          }
        });

    // unpack i32 vectors and cast to native type
    if (bitwidth != 16) {
      auto numElemsPerVec = 32 / bitwidth;
      auto vecTy = vec_ty(llvmElemTy, numElemsPerVec);
      for (int v = 0; v < static_cast<int>(elemsI32.size()); ++v) {
        auto vec = b.bitcast(elemsI32[v], vecTy);
        for (int i = 0; i < numElemsPerVec; ++i)
          outVals.push_back(b.extract_element(llvmElemTy, vec, b.i32_val(i)));
      }

      retTy = LLVM::LLVMStructType::getLiteral(
          ctx, SmallVector<Type>(outVals.size(), llvmElemTy));
    }
    assert(valid && "Failed to emit LDS transpose load operations");
    Value result = packLLElements(loc, typeConverter, outVals, rewriter, retTy);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::AMD::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  PatternBenefit transBenefit = PatternBenefit(benefit.getBenefit() + 1);
  patterns.add<LocalLoadOpConversion>(typeConverter, benefit);
  patterns.add<TransLocalLoadOpConversion>(typeConverter, targetInfo,
                                           transBenefit);
}
