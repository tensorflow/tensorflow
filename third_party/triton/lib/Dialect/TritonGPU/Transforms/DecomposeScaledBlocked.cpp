#include "triton/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.h"

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

SmallVector<int, 2> getTransposeOrder(int rank) {
  assert(rank >= 2);
  auto transOrder = llvm::to_vector<2>(llvm::seq<int>(rank - 2));
  transOrder.push_back(rank - 1);
  transOrder.push_back(rank - 2);
  return transOrder;
}

class DecomposeScaledBlocked : public OpRewritePattern<DotScaledOp> {

public:
  DecomposeScaledBlocked(MLIRContext *context, int benefit)
      : OpRewritePattern<DotScaledOp>(context, benefit) {}

  LogicalResult matchAndRewrite(DotScaledOp scaledDotOp,
                                PatternRewriter &rewriter) const override {
    // Types
    auto computeType = getComputeType(scaledDotOp.getAElemType(),
                                      scaledDotOp.getBElemType(), rewriter);
    auto loc = scaledDotOp.getLoc();

    auto cvtDotOperand = [&](TypedValue<RankedTensorType> v,
                             int opIdx) -> TypedValue<RankedTensorType> {
      auto *ctx = rewriter.getContext();
      auto retEnc = scaledDotOp.getType().getEncoding();
      auto vType = v.getType();
      auto encoding = DotOperandEncodingAttr::get(ctx, opIdx, retEnc,
                                                  vType.getElementType());
      auto retTy = RankedTensorType::get(vType.getShape(),
                                         vType.getElementType(), encoding);
      return rewriter.create<ConvertLayoutOp>(loc, retTy, v);
    };

    auto scaledA = scaleArg(rewriter, scaledDotOp, 0, computeType);
    scaledA = cvtDotOperand(scaledA, 0);
    auto scaledB = scaleArg(rewriter, scaledDotOp, 1, computeType);
    scaledB = cvtDotOperand(scaledB, 1);
    auto newDot = rewriter.create<DotOp>(scaledDotOp.getLoc(), scaledA, scaledB,
                                         scaledDotOp.getC());

    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(scaledDotOp,
                                                 scaledDotOp.getType(), newDot);
    return success();
  }

private:
  FloatType getComputeType(ScaleDotElemType aType, ScaleDotElemType bType,
                           PatternRewriter &rewriter) const {
    if (aType == ScaleDotElemType::FP16 || bType == ScaleDotElemType::FP16)
      return rewriter.getF16Type();
    return rewriter.getBF16Type();
  }

  TypedValue<RankedTensorType> scaleTo16(PatternRewriter &rewriter,
                                         TypedValue<RankedTensorType> scale,
                                         FloatType computeType) const {
    auto loc = scale.getLoc();
    auto scaleTy = scale.getType();
    assert(computeType == rewriter.getBF16Type() ||
           computeType == rewriter.getF16Type());

    // Choose an fp type that can fit the scale value.
    FloatType largeFpType = computeType == rewriter.getF16Type()
                                ? rewriter.getF32Type()
                                : computeType;
    int intWidth = largeFpType.getIntOrFloatBitWidth();
    auto intType = rewriter.getIntegerType(intWidth);

    auto zexted =
        rewriter.create<arith::ExtUIOp>(loc, scaleTy.clone(intType), scale);
    // getFpMantissaWidth() returns the number of bits in the mantissa plus the
    // sign bit!
    int shiftValue = largeFpType.getFPMantissaWidth() - 1;
    auto shiftConst =
        rewriter.create<arith::ConstantIntOp>(loc, shiftValue, intWidth);
    auto shift =
        rewriter.create<SplatOp>(loc, scaleTy.clone(intType), shiftConst);
    auto shlRes = rewriter.create<arith::ShLIOp>(loc, zexted, shift);
    Value scaleFP =
        rewriter.create<BitcastOp>(loc, scaleTy.clone(largeFpType), shlRes);
    if (largeFpType != computeType) {
      scaleFP = rewriter.create<arith::TruncFOp>(
          loc, scaleTy.clone(computeType), scaleFP);
    }
    return cast<TypedValue<RankedTensorType>>(scaleFP);
  }

  TypedValue<RankedTensorType>
  broadcastScale(PatternRewriter &rewriter, DotScaledOp scaledDotOp,
                 ModuleOp mod, TypedValue<RankedTensorType> scale,
                 int dim) const {
    auto *ctx = rewriter.getContext();
    auto loc = scale.getLoc();
    auto scaleTy = scale.getType();
    auto rank = scaleTy.getRank();
    // 2.1) Expand dims along the last dimension
    {
      // 2.1.1) Find default encoding for ExpandDims
      auto shape = to_vector(scaleTy.getShape());
      shape.insert(shape.end(), 1);
      auto nWarps = lookupNumWarps(scaledDotOp);
      auto threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
      auto numCTAs = TritonGPUDialect::getNumCTAs(mod);
      auto blockedEnc = getDefaultBlockedEncoding(ctx, shape, nWarps,
                                                  threadsPerWarp, numCTAs);
      // 2.1.2) Cast scale16 to SliceEncoding
      auto sliceEnc = SliceEncodingAttr::get(ctx, rank, blockedEnc);
      auto sliceType = RankedTensorType::get(
          scaleTy.getShape(), scaleTy.getElementType(), sliceEnc);
      scale = rewriter.create<ConvertLayoutOp>(loc, sliceType, scale);
    }
    auto expandScale = rewriter.create<ExpandDimsOp>(loc, scale, rank);
    // 2.2) Broadcast the dimension to size 32
    auto scaleShape = to_vector(scaleTy.getShape());
    scaleShape.push_back(32);
    auto broadcastScale = rewriter.create<BroadcastOp>(
        loc, expandScale.getType().clone(scaleShape), expandScale);
    // 2.3) Transpose the dimension to the scaled dimension
    auto transposeOrder = llvm::to_vector(llvm::seq<int32_t>(rank));
    transposeOrder.insert(transposeOrder.begin() + dim + 1, rank);
    auto transposedScale =
        rewriter.create<TransOp>(loc, broadcastScale, transposeOrder);
    // 2.4) Reshape to the shape of v
    scaleShape.pop_back();
    scaleShape[dim] *= 32;
    auto reshapeScale =
        rewriter.create<ReshapeOp>(loc, scaleShape, transposedScale);
    return reshapeScale;
  }

  TypedValue<RankedTensorType> maskNan(PatternRewriter &rewriter,
                                       DotScaledOp scaledDotOp, ModuleOp mod,
                                       TypedValue<RankedTensorType> mxfp,
                                       TypedValue<RankedTensorType> scale,
                                       int dim) const {
    // Implement tl.where(scale == 0xFF, float("nan"), mxfp)
    auto loc = scale.getLoc();

    // Scale is NaN
    auto scaleTy = scale.getType();
    auto constFF = rewriter.create<arith::ConstantOp>(
        loc, scaleTy,
        DenseElementsAttr::get(scaleTy,
                               APInt(scaleTy.getElementTypeBitWidth(), 0xff)));
    auto scaleIsNan = cast<TypedValue<RankedTensorType>>(
        rewriter
            .create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, scale,
                                   constFF)
            .getResult());
    auto cond = broadcastScale(rewriter, scaledDotOp, mod, scaleIsNan, dim);
    // Make scale is NaN compatible with mxfp
    auto condTy = cond.getType();
    condTy = RankedTensorType::get(condTy.getShape(), condTy.getElementType(),
                                   mxfp.getType().getEncoding());
    cond = rewriter.create<ConvertLayoutOp>(loc, condTy, cond);

    // Create NaN
    auto mxfpTy = mxfp.getType();
    auto nan = APFloat::getNaN(
        cast<FloatType>(mxfpTy.getElementType()).getFloatSemantics());
    auto constNan = rewriter.create<arith::ConstantOp>(
        loc, mxfpTy, DenseElementsAttr::get(mxfpTy, nan));

    auto result = rewriter.create<arith::SelectOp>(loc, cond, constNan, mxfp);
    return cast<TypedValue<RankedTensorType>>(result.getResult());
  }

  TypedValue<RankedTensorType> scaleArg(PatternRewriter &rewriter,
                                        DotScaledOp scaledDotOp, int opIdx,
                                        FloatType computeType) const {
    auto v = opIdx == 0 ? scaledDotOp.getA() : scaledDotOp.getB();
    auto scale = opIdx == 0 ? scaledDotOp.getAScale() : scaledDotOp.getBScale();
    auto isFp4 =
        ScaleDotElemType::E2M1 ==
        (opIdx == 0 ? scaledDotOp.getAElemType() : scaledDotOp.getBElemType());
    auto fastMath = scaledDotOp.getFastMath();

    auto *ctx = rewriter.getContext();
    auto loc = v.getLoc();
    auto mod = scaledDotOp->getParentOfType<ModuleOp>();
    auto rank = v.getType().getRank();
    auto kDim = opIdx == 0 ? rank - 1 : rank - 2;

    // 0) Upcast value to computeType (fp16/bf16)
    if (isFp4) {
      // We always pack along the fastest moving dimension, kDim
      v = rewriter.create<Fp4ToFpOp>(loc, v, computeType, kDim);
    } else {
      auto vType16 = v.getType().clone(computeType);
      v = cast<TypedValue<RankedTensorType>>(
          rewriter.create<FpToFpOp>(loc, vType16, v).getResult());
    }
    if (!scale)
      return v;

    // For some weird reason, we take the scale with shape as if it were coming
    // from the lhs even when it's the rhs. In a normal world, we should accept
    // this parametre transposed, as we do with the mxfp.
    if (opIdx == 1) {
      auto order = getTransposeOrder(rank);
      scale = rewriter.create<TransOp>(loc, scale, order);
    }

    // 1) Cast scale to compute type (fp16/bf16)
    auto scale16 = scaleTo16(rewriter, scale, computeType);

    // 2) Broadcast scale to the same shape and layout as v
    auto reshapeScale =
        broadcastScale(rewriter, scaledDotOp, mod, scale16, kDim);
    reshapeScale =
        rewriter.create<ConvertLayoutOp>(loc, v.getType(), reshapeScale);

    // 3) Multiply
    auto mxfp = cast<TypedValue<RankedTensorType>>(
        rewriter.create<arith::MulFOp>(loc, v, reshapeScale).getResult());

    // Skip NaN checks if fastMath
    if (fastMath)
      return mxfp;

    // 4) If the scale is NaN, return NaN, else return the scaled value.
    return maskNan(rewriter, scaledDotOp, mod, mxfp, scale, kDim);
  }
};

} // namespace

namespace mlir::triton::gpu {

void populateDecomposeScaledBlockedPatterns(RewritePatternSet &patterns,
                                            int benefit) {
  patterns.add<DecomposeScaledBlocked>(patterns.getContext(), benefit);
}

} // namespace mlir::triton::gpu
