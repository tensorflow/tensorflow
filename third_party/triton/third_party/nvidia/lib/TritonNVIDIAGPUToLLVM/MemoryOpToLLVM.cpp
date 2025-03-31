#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

struct LocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  LocalLoadOpConversion(const LLVMTypeConverter &converter,
                        const NVIDIA::TargetInfo &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isa<DotOperandEncodingAttr>(dstLayout) &&
        isa<NvidiaMmaEncodingAttr>(
            cast<DotOperandEncodingAttr>(dstLayout).getParent())) {
      auto dotEnc = cast<DotOperandEncodingAttr>(dstLayout);
      auto mmaEnc = cast<NvidiaMmaEncodingAttr>(dotEnc.getParent());
      auto sharedEnc = dyn_cast<SwizzledSharedEncodingAttr>(srcLayout);
      if (!sharedEnc)
        return failure();
      auto bitwidth = dstTy.getElementTypeBitWidth();
      auto vecWidth = 32 / bitwidth;
      auto kWidth = dotEnc.getKWidth();
      auto rank = dstTy.getRank();
      auto kOrder = dotEnc.getOpIdx() == 0 ? rank - 1 : rank - 2;
      auto nonKOrder = dotEnc.getOpIdx() == 0 ? rank - 2 : rank - 1;
      auto needTrans = kOrder != sharedEnc.getOrder()[0];
      // Limitation 1 [TODO: remove]: Check LL bases to verify register and
      // address alignment
      auto canUseLdmatrix = (kWidth == vecWidth);
      canUseLdmatrix &= (sharedEnc.getMaxPhase() == 1) ||
                        (sharedEnc.getVec() * bitwidth >= 8 * 16);
      auto shape = srcTy.getShape();
      // Limitation 2 [TODO: remove]: Only support 2d matrices now but we should
      // be able to support 3D minor changes
      canUseLdmatrix &= (bitwidth == 16 || !needTrans) && shape.size() <= 2;
      // Limitation 3: Minimum tile size (8)x(8x16bits)
      canUseLdmatrix &=
          shape[kOrder] >= (8 * 16 / bitwidth) && shape[nonKOrder] >= 8;
      if (canUseLdmatrix) {
        return lowerSharedToDotOperand(op, adaptor, getTypeConverter(),
                                       rewriter);
      }
    }
    return failure();
  }

private:
  LogicalResult
  lowerSharedToDotOperand(triton::gpu::LocalLoadOp op,
                          triton::gpu::LocalLoadOpAdaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) const {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    auto dotEnc = cast<DotOperandEncodingAttr>(dstTy.getEncoding());
    auto sharedEnc = cast<SwizzledSharedEncodingAttr>(srcTy.getEncoding());
    auto shape = dstTy.getShape();
    auto rank = dstTy.getRank();
    auto kOrder = dotEnc.getOpIdx() == 0 ? rank - 1 : rank - 2;
    auto nonKOrder = dotEnc.getOpIdx() == 0 ? rank - 2 : rank - 1;
    auto needTrans = kOrder != sharedEnc.getOrder()[0];

    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    auto ldmatrixLayout =
        chooseLdMatrixLayout(dotEnc, shape, needTrans, bitwidth);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    // Emit ldmatrix load operations for values packed in i32s
    SmallVector<Value> elemsI32;
    // Typically we load 32x8 to use ldmatrix.x4, but the minimum tile size for
    // opIdx=1 is 16x8. Therefore, we use ldmatrix.x2 instead of
    // ldmatrix.x4 in this case.
    auto shift = dotEnc.getOpIdx() == 1 && shape[kOrder] < (32 * 16 / bitwidth);
    auto maxVecElems = 8 * 16 / bitwidth;
    bool valid = emitTransferBetweenRegistersAndShared(
        ldmatrixLayout, srcTy, llvmElemTy,
        /*maxVecElems=*/maxVecElems, smemObj, loc, rewriter, targetInfo,
        [&](VectorType vecTy, Value vecAddr) {
          auto numElems = vecTy.getNumElements();
          auto numElemsI32 = (numElems * bitwidth / 32) >> shift;
          auto matTy = LLVM::LLVMStructType::getLiteral(
              ctx, SmallVector<Type>(numElemsI32, i32_ty));
          auto ldMatrixOp = rewriter.create<nvgpu::LoadMatrixOp>(
              loc, matTy, vecAddr, /*needTrans=*/needTrans);
          auto res = ldMatrixOp.getResult();
          for (auto i = 0; i < numElemsI32; ++i) {
            elemsI32.push_back(b.extract_val(i32_ty, res, i));
          }
        });
    assert(valid && "Failed to emit ldmatrix load operations");

    // Unpack i32 values to the original type
    SmallVector<Value> elems;
    auto numElemsPerVec = 32 / bitwidth;
    auto vecTy = vec_ty(llvmElemTy, numElemsPerVec);
    for (int v = 0; v < static_cast<int>(elemsI32.size()); ++v) {
      auto vec = b.bitcast(elemsI32[v], vecTy);
      for (int i = 0; i < numElemsPerVec; ++i)
        elems.push_back(b.extract_element(llvmElemTy, vec, b.i32_val(i)));
    }

    auto structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(elems.size(), llvmElemTy));
    auto ret = packLLElements(loc, typeConverter, elems, rewriter, structTy);
    rewriter.replaceOp(op, ret);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};

LogicalResult lowerDistributedToSharedStmatrix(
    Location loc, TypedValue<RankedTensorType> src, MemDescType memDescType,
    Value adaptorSrc, Value smemBase, const TypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter, const TargetInfoBase &targetInfo,
    std::pair<size_t, Type> *const llvmOpCount = nullptr) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto mmaEncoding =
      dyn_cast<triton::gpu::NvidiaMmaEncodingAttr>(src.getType().getEncoding());
  if (!mmaEncoding)
    return failure();
  auto sharedLayout =
      dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(memDescType.getEncoding());
  if (!sharedLayout)
    return failure();
  int swizzleByteSize = sharedLayout.getSwizzlingByteWidth();

  RankedTensorType srcTy = src.getType();
  SmallVector<unsigned> shape =
      convertType<unsigned, int64_t>(srcTy.getShape());
  SmallVector<unsigned> order = sharedLayout.getTransposed()
                                    ? SmallVector<unsigned>({0, 1})
                                    : SmallVector<unsigned>({1, 0});
  if (!targetInfo.canUseStMatrix(srcTy, shape, shape, order, swizzleByteSize)) {
    return failure();
  }

  auto *ctx = rewriter.getContext();

  auto layout =
      chooseStMatrixLayout(rewriter.getContext(), srcTy, swizzleByteSize);
  auto llvmElemTy = typeConverter->convertType(memDescType.getElementType());
  auto smemPtrTy = ptr_ty(ctx, 3);

  auto kRegister = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");

  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);

  auto regBase = applyLinearLayout(loc, rewriter, layout,
                                   {{kRegister, b.i32_val(0)},
                                    {kLane, laneId},
                                    {kWarp, warpId},
                                    {kBlock, b.i32_val(0)}})[0]
                     .second;
  auto srcVals = unpackLLElements(loc, adaptorSrc, rewriter);
  auto srcVec = layout.getNumConsecutiveInOut();
  for (int i = 0; i < srcVals.size(); i += srcVec) {
    auto regIdx =
        layout.apply({{kRegister, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}})[0]
            .second;
    Value offset = b.xor_(regBase, b.i32_val(regIdx));
    auto vecAddr = b.gep(smemPtrTy, llvmElemTy, smemBase, offset);
    vecAddr.setInbounds(true);
    SmallVector<Value> inValsVec;
    for (int j = 0; j < srcVec; j++)
      inValsVec.push_back(srcVals[i + j]);
    Value valsVec = packLLVector(loc, inValsVec, rewriter);
    targetInfo.storeMatrixShared(rewriter, loc, vecAddr, valsVec);
  }
  return success();
}

struct LocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  LocalAllocOpConversion(const LLVMTypeConverter &converter,
                         const NVIDIA::TargetInfo &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getSrc())
      return failure();
    MemDescType memDescType = op.getType();
    RankedTensorType srcTy = op.getSrc().getType();
    Type llvmElemTy = typeConverter->convertType(srcTy.getElementType());
    Value smemBase =
        LLVM::getSharedMemoryBase(op.getLoc(), rewriter, targetInfo, op);

    if (lowerDistributedToSharedStmatrix(op.getLoc(), op.getSrc(), memDescType,
                                         adaptor.getSrc(), smemBase,
                                         typeConverter, rewriter, targetInfo)
            .failed()) {
      return failure();
    }

    auto resultTy = cast<MemDescType>(op.getType());
    auto smemObj = SharedMemoryObject(smemBase, llvmElemTy, resultTy.getRank(),
                                      op.getLoc(), rewriter);
    auto retVal =
        getStructFromSharedMemoryObject(op.getLoc(), smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};

struct LocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp> {
  LocalStoreOpConversion(const LLVMTypeConverter &converter,
                         const NVIDIA::TargetInfo &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type llvmElemTy =
        getTypeConverter()->convertType(op.getDst().getType().getElementType());
    SharedMemoryObject smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getDst(), llvmElemTy, rewriter);
    MemDescType memDescType = op.getDst().getType();
    if (lowerDistributedToSharedStmatrix(
            op.getLoc(), op.getSrc(), memDescType, adaptor.getSrc(),
            smemObj.getBase(), getTypeConverter(), rewriter, targetInfo)
            .failed()) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};
} // namespace

void mlir::triton::NVIDIA::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // Backend optimized memory ops get higher benefit
  patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo,
                                       benefit.getBenefit() + 1);
  patterns.add<LocalStoreOpConversion>(typeConverter, targetInfo,
                                       benefit.getBenefit() + 1);
  patterns.add<LocalLoadOpConversion>(typeConverter, targetInfo,
                                      benefit.getBenefit() + 1);
  mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                               patterns, benefit);
}
