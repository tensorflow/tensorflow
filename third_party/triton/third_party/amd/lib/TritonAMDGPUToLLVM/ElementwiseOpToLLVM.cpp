#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

using mlir::triton::gpu::appendOrGetExternFuncOp;
using mlir::triton::gpu::ElementwiseOpConversionBase;
using mlir::triton::gpu::getElementType;
using mlir::triton::gpu::getFunctionType;
using mlir::triton::gpu::MultipleOperandsRange;

using ConverterT = std::function<SmallVector<Value>(
    Location, ConversionPatternRewriter &, const SmallVector<Value> &)>;

namespace {
//===----------------------------------------------------------------------===//
// Data type conversion utility functions
//===----------------------------------------------------------------------===//
// Convert Ocp Fp8/Bf8 to Fp16/Bf16/Fp32 on CDNA4
template <typename convertOp>
static SmallVector<Value>
cvtScalePkUpcastFromFp8(Location loc, ConversionPatternRewriter &rewriter,
                        Value v0, Value v1) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value fp8x4Vec = b.undef(fp8x4VecTy);
  auto idx0 = b.i32_val(0);
  auto idx1 = b.i32_val(1);
  fp8x4Vec = b.insert_element(fp8x4VecTy, fp8x4Vec, v0, idx0);
  fp8x4Vec = b.insert_element(fp8x4VecTy, fp8x4Vec, v1, idx1);
  auto i32v = b.bitcast(fp8x4Vec, i32_ty);

  auto resType = i32_ty;
  auto dstType = f32_ty;
  if constexpr (std::is_same_v<convertOp, ROCDL::CvtScaleF32PkF32Fp8Op> ||
                std::is_same_v<convertOp, ROCDL::CvtScaleF32PkF32Bf8Op>) {
    resType = i64_ty;
    dstType = f32_ty;
  } else if constexpr (std::is_same_v<convertOp,
                                      ROCDL::CvtScaleF32PkF16Fp8Op> ||
                       std::is_same_v<convertOp,
                                      ROCDL::CvtScaleF32PkF16Bf8Op>) {
    resType = i32_ty;
    dstType = f16_ty;
  } else {
    resType = i32_ty;
    dstType = bf16_ty;
  }
  Value scale = b.f32_val(1);
  Value select = b.false_val();
  auto result = rewriter.create<convertOp>(loc, resType, i32v, scale, select);
  auto retVecTy = vec_ty(dstType, 2);
  auto retVec = b.bitcast(result, retVecTy);
  SmallVector<Value> ret(2);
  ret[0] = b.extract_element(dstType, retVec, idx0);
  ret[1] = b.extract_element(dstType, retVec, idx1);
  return ret;
}

// Convert Fp16/Bf16/Fp32 to OCP Fp8/Bf8 on CDNA4
template <typename convertOp>
static SmallVector<Value>
cvtScalePkDowncastToFp8(Location loc, ConversionPatternRewriter &rewriter,
                        Value v0, Value v1) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type v2I16Ty = vec_ty(i16_ty, 2);
  Value v2I16Vec = b.undef(v2I16Ty);
  Value scale = b.f32_val(1);
  Value select = b.false_val();

  Value result;
  if constexpr (std::is_same_v<convertOp, ROCDL::CvtScaleF32PkFp8F32Op> ||
                std::is_same_v<convertOp, ROCDL::CvtScaleF32PkBf8F32Op>) {
    result = rewriter.create<convertOp>(loc, v2I16Ty, v2I16Vec, v0, v1, scale,
                                        select);
  } else {
    Type v2F16Ty = vec_ty(v0.getType(), 2);
    Value srcVec = b.undef(v2F16Ty);
    auto idx0 = b.i32_val(0);
    auto idx1 = b.i32_val(1);
    srcVec = b.insert_element(v2F16Ty, srcVec, v0, idx0);
    srcVec = b.insert_element(v2F16Ty, srcVec, v1, idx1);
    result = rewriter.create<convertOp>(loc, v2I16Ty, v2I16Vec, srcVec, scale,
                                        select);
  }
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  auto fp8x4Vec = b.bitcast(result, fp8x4VecTy);
  SmallVector<Value> ret(2);
  auto idx0 = b.i32_val(0);
  auto idx1 = b.i32_val(1);
  ret[0] = b.extract_element(i8_ty, fp8x4Vec, idx0);
  ret[1] = b.extract_element(i8_ty, fp8x4Vec, idx1);
  return ret;
}

// Fp16 -> OCP Bf8 (RTNE)

// FP8E5M2 is the open-compute standard FP8E5M2 format. NVIDIA GPU supports it
// natively but we don't have hardware native support on CDNA3.
//
// The SW based downcast with RTNE is not fully functional for the denorm
// values. We need rewrite it if we need to emulate this data type on AMDGPU.
static SmallVector<Value>
Fp16_to_Fp8E5M2_RTNE_SW(Location loc, ConversionPatternRewriter &rewriter,
                        const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  Value fp16x2Vec0 = b.undef(fp16x2VecTy);
  Value fp16x2Vec1 = b.undef(fp16x2VecTy);
  fp16x2Vec0 = b.insert_element(fp16x2VecTy, fp16x2Vec0, v[0], b.i32_val(0));
  fp16x2Vec0 = b.insert_element(fp16x2VecTy, fp16x2Vec0, v[1], b.i32_val(1));
  fp16x2Vec1 = b.insert_element(fp16x2VecTy, fp16x2Vec1, v[2], b.i32_val(0));
  fp16x2Vec1 = b.insert_element(fp16x2VecTy, fp16x2Vec1, v[3], b.i32_val(1));

  Value a0 = b.bitcast(fp16x2Vec0, i32_ty);
  Value a1 = b.bitcast(fp16x2Vec1, i32_ty);

  a0 = b.and_(i32_ty, a0, b.i32_val(0xfffefffe));
  a1 = b.and_(i32_ty, a1, b.i32_val(0xfffefffe));

  a0 = b.add(i32_ty, a0, b.i32_val(0x00800080));
  a1 = b.add(i32_ty, a1, b.i32_val(0x00800080));

  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  a0 = b.bitcast(a0, fp8x4VecTy);
  a1 = b.bitcast(a1, fp8x4VecTy);

  return {b.extract_element(i8_ty, a0, b.i32_val(1)),
          b.extract_element(i8_ty, a0, b.i32_val(3)),
          b.extract_element(i8_ty, a1, b.i32_val(1)),
          b.extract_element(i8_ty, a1, b.i32_val(3))};
}

static SmallVector<Value>
Fp16_to_Fp8E5M2_RTNE_HW(Location loc, ConversionPatternRewriter &rewriter,
                        const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtScalePkDowncastToFp8<ROCDL::CvtScaleF32PkBf8F16Op>(loc, rewriter,
                                                               v[0], v[1]);
}

ConverterT Fp16_to_Fp8E5M2_RTNE(AMD::ISAFamily isaFamily) {
  return isaFamily == AMD::ISAFamily::CDNA4 ? Fp16_to_Fp8E5M2_RTNE_HW
                                            : Fp16_to_Fp8E5M2_RTNE_SW;
}

// Fp16 -> OCP Bf8 (RTZ)
static SmallVector<Value>
Fp16_to_Fp8E5M2_RTZ(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  Value fp16x2Vec0 = b.undef(fp16x2VecTy);
  Value fp16x2Vec1 = b.undef(fp16x2VecTy);
  fp16x2Vec0 = b.insert_element(fp16x2VecTy, fp16x2Vec0, v[0], b.i32_val(0));
  fp16x2Vec0 = b.insert_element(fp16x2VecTy, fp16x2Vec0, v[1], b.i32_val(1));
  fp16x2Vec1 = b.insert_element(fp16x2VecTy, fp16x2Vec1, v[2], b.i32_val(0));
  fp16x2Vec1 = b.insert_element(fp16x2VecTy, fp16x2Vec1, v[3], b.i32_val(1));

  Value a0 = b.bitcast(fp16x2Vec0, i32_ty);
  Value a1 = b.bitcast(fp16x2Vec1, i32_ty);

  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  a0 = b.bitcast(a0, fp8x4VecTy);
  a1 = b.bitcast(a1, fp8x4VecTy);

  return {b.extract_element(i8_ty, a0, b.i32_val(1)),
          b.extract_element(i8_ty, a0, b.i32_val(3)),
          b.extract_element(i8_ty, a1, b.i32_val(1)),
          b.extract_element(i8_ty, a1, b.i32_val(3))};
}

static Value checkIsNan(TritonLLVMOpBuilder &builder, Value v) {
  StringRef intrinsic = "llvm.is.fpclass";
  // bits 0 and 1 indicate signaling Nan and quiet Nan, respectively
  Location loc = builder.loc;
  OpBuilder &rewriter = *builder.builder;
  Value nanBits = builder.i32_val(3);

  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, i1_ty,
                                         ValueRange{v, nanBits})
      ->getResult(0);
}

// Fp16 -> OCP Fp8 (RTNZ)

// Cast FP16 to FP8E4M3FN in saturation and round-to-nearest-even mode.
// According to
// https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1,
// In saturation mode, inf and out-of-range numbers are converted to the largest
// normal number, i.e. ±448. NaNs are converted to NaNs.
static Value
Fp16_to_Fp8E4M3FN_RTNE_oneValue(Location loc,
                                ConversionPatternRewriter &rewriter, Value v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value isNaN = checkIsNan(b, v);
  // Get sign and absolute value
  Value vi16 = b.bitcast(v, i16_ty);
  Value sign =
      b.trunc(i8_ty, b.lshr(b.and_(vi16, b.i16_val(0x8000)), b.i16_val(8)));
  vi16 = b.and_(vi16, b.i16_val(0x7FFF));

  // Rounding to nearest even
  constexpr uint16_t baseRoundingBias = 0x003F; // 1 << (10 - 3 - 1) - 1

  // S.EEEEE.MMMMMMMMMM => 0.00000.00M0000000 => 0.00000.000000000M
  Value remainingMantissaLSB =
      b.lshr(b.and_(vi16, b.i16_val(0x0080)), b.i16_val(7));
  Value roundingBias = b.add(remainingMantissaLSB, b.i16_val(baseRoundingBias));
  Value vFp8 = b.add(vi16, roundingBias);

  // Reduce mantissa to 3 bits
  vFp8 = b.and_(vFp8, b.i16_val(0xFF80)); // 0xFF80 == 1.11111.1110000000

  // 0x2400 is the FP16 representation of 2^{-6}, which is the smallest normal
  // number in FP8E4M3FN. We round numbers smaller than that to 0x2400 to make
  // it easier to handle subnormals
  vFp8 = b.umax(vFp8, b.i16_val(0x2400));

  // Adjust exponent bias
  vFp8 = b.sub(vFp8, b.i16_val(0x2000)); // (15 - 7) << 10

  // Shift right and truncate
  vFp8 = b.trunc(i8_ty, b.lshr(vFp8, b.i16_val(7))); // 10 - 3

  // 0x5F7F == 0.10111.1101111111 is the largest possible normal
  // number(including infinity) after rounding in FP8
  //
  // In saturation mode, numbers larger than the max normal number(including
  // infinity) in FP8 after rounding will be replaced with max_E4M3, i.e. 0x7E
  // === 0.1111.110
  Value isOverflowOrInf = b.icmp_ugt(vi16, b.i16_val(0x5F7F));
  vFp8 = b.select(isOverflowOrInf, b.i8_val(0x7E), vFp8);

  // Round subnormals to nearest even. Ref:
  // https://github.com/openxla/xla/blob/f20c6fe2/xla/service/elemental_ir_emitter.cc#L272
  constexpr size_t lutSize = 8;
  constexpr float halfwayPointsLUT[lutSize] = {0x1400, 0x1A00, 0x1D00, 0x1F00,
                                               0x2080, 0x2180, 0x2280, 0x2380};

  for (int i = lutSize - 1; i >= 0; i--) {
    Value cmp;
    if (i % 2 == 0) {
      cmp = b.icmp_ule(vi16, b.i16_val(halfwayPointsLUT[i]));
    } else {
      cmp = b.icmp_ult(vi16, b.i16_val(halfwayPointsLUT[i]));
    }

    vFp8 = b.select(cmp, b.i8_val(i), vFp8);
  }

  // NaN remains NaN after conversion
  vFp8 = b.select(isNaN, b.i8_val(0x7F), vFp8);

  // Set sign bit
  vFp8 = b.or_(vFp8, sign);

  return vFp8;
}

static SmallVector<Value>
Fp16_to_Fp8E4M3FN_RTNE_SW(Location loc, ConversionPatternRewriter &rewriter,
                          const SmallVector<Value> &v) {
  SmallVector<Value> result(2);
  result[0] = Fp16_to_Fp8E4M3FN_RTNE_oneValue(loc, rewriter, v[0]);
  result[1] = Fp16_to_Fp8E4M3FN_RTNE_oneValue(loc, rewriter, v[1]);
  return result;
}

static SmallVector<Value>
Fp16_to_Fp8E4M3FN_RTNE_HW(Location loc, ConversionPatternRewriter &rewriter,
                          const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtScalePkDowncastToFp8<ROCDL::CvtScaleF32PkFp8F16Op>(loc, rewriter,
                                                               v[0], v[1]);
}

ConverterT Fp16_to_Fp8E4M3FN_RTNE(AMD::ISAFamily isaFamily) {
  return isaFamily == AMD::ISAFamily::CDNA4 ? Fp16_to_Fp8E4M3FN_RTNE_HW
                                            : Fp16_to_Fp8E4M3FN_RTNE_SW;
}

// Fp16 -> Fp32
static Value cvtFp16ToFp32(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v) {

  TritonLLVMOpBuilder b(loc, rewriter);
  return b.fpext(f32_ty, v);
}

// Convert Fp8 to Fp32 on CDNA3
static SmallVector<Value> cvtFp8ToFp32(Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       Value v0, Value v1,
                                       const std::string &fp8_format) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  assert(fp8_format == "fp8" || fp8_format == "bf8");
  std::string ins_str = "v_cvt_pk_f32_" + fp8_format;

  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value fp8x4Vec = b.undef(fp8x4VecTy);
  fp8x4Vec = b.insert_element(fp8x4VecTy, fp8x4Vec, v0, b.i32_val(0));
  fp8x4Vec = b.insert_element(fp8x4VecTy, fp8x4Vec, v1, b.i32_val(1));
  auto i32v = b.bitcast(fp8x4Vec, i32_ty);

  GCNBuilder builder1;
  auto &cvt = *builder1.create(ins_str);
  auto res = builder1.newOperand("=v");
  auto operand = builder1.newOperand(i32v, "v");
  cvt(res, operand);
  auto i64v = builder1.launch(rewriter, loc, i64_ty, false);
  auto fp32x2VecTy = vec_ty(f32_ty, 2);
  auto fp32x2Vec = b.bitcast(i64v, fp32x2VecTy);

  SmallVector<Value> ret(2);
  ret[0] = b.extract_element(f32_ty, fp32x2Vec, b.i32_val(0));
  ret[1] = b.extract_element(f32_ty, fp32x2Vec, b.i32_val(1));

  return ret;
}

// Convert Fp32 to Fp8 on CDNA3
static SmallVector<Value> cvtFp32ToFp8(Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       Value v0, Value v1,
                                       const std::string &fp8_format) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  assert(fp8_format == "fp8" || fp8_format == "bf8");
  std::string ins_str = "v_cvt_pk_" + fp8_format + "_f32";

  GCNBuilder builder;
  auto &cvt = *builder.create(ins_str);
  auto res = builder.newOperand("=v");
  auto operand0 = builder.newOperand(v0, "v");
  auto operand1 = builder.newOperand(v1, "v");
  cvt(res, operand0, operand1);
  auto fp8x4Vec = builder.launch(rewriter, loc, i32_ty, false);

  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  auto a1 = b.bitcast(fp8x4Vec, fp8x4VecTy);

  SmallVector<Value> ret(2);
  ret[0] = b.extract_element(i8_ty, a1, b.i32_val(0));
  ret[1] = b.extract_element(i8_ty, a1, b.i32_val(1));

  return ret;
}

// Convert Fp16 to Fp8 on CDNA3
static SmallVector<Value>
convert_val_Fp16_to_Fp8(Location loc, ConversionPatternRewriter &rewriter,
                        Value v0, Value v1, const std::string &fp8_format) {
  assert(fp8_format == "fp8" || fp8_format == "bf8");
  std::string ins_str = "v_cvt_pk_" + fp8_format + "_f32";

  auto f32_0 = cvtFp16ToFp32(loc, rewriter, v0);
  auto f32_1 = cvtFp16ToFp32(loc, rewriter, v1);

  // Convert fp32 to fp8
  return cvtFp32ToFp8(loc, rewriter, f32_0, f32_1, fp8_format);
}

// Convert Fp8 to Fp16 on CDNA3
static SmallVector<Value>
convert_val_Fp8_to_Fp16(Location loc, ConversionPatternRewriter &rewriter,
                        Value v0, Value v1, const std::string &fp8_format) {
  // Convert fp8 to fp32
  SmallVector<Value> ret = cvtFp8ToFp32(loc, rewriter, v0, v1, fp8_format);

  // Convert fp32 to fp16
  ret[0] = LLVM::AMD::cvtFp32ToFp16(loc, rewriter, ret[0], RoundingMode::RTNE);
  ret[1] = LLVM::AMD::cvtFp32ToFp16(loc, rewriter, ret[1], RoundingMode::RTNE);

  return ret;
}

// Convert OCP Fp8 to Fp32 on CDNA4
static SmallVector<Value> Fp8E4M3FN_to_Fp32(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtScalePkUpcastFromFp8<ROCDL::CvtScaleF32PkF32Fp8Op>(loc, rewriter,
                                                               v[0], v[1]);
}

// Convert OCP Bf8 to Fp32 on CDNA4
static SmallVector<Value> Fp8E5M2_to_Fp32(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtScalePkUpcastFromFp8<ROCDL::CvtScaleF32PkF32Bf8Op>(loc, rewriter,
                                                               v[0], v[1]);
}

// Convert Fp32 to OCP Fp8 on CDNA4
static SmallVector<Value> Fp32_to_Fp8E4M3FN(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtScalePkDowncastToFp8<ROCDL::CvtScaleF32PkFp8F32Op>(loc, rewriter,
                                                               v[0], v[1]);
}

// Convert Fp32 to OCP Bf8 on CDNA4
static SmallVector<Value> Fp32_to_Fp8E5M2(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtScalePkDowncastToFp8<ROCDL::CvtScaleF32PkBf8F32Op>(loc, rewriter,
                                                               v[0], v[1]);
}

// Fp32 -> Nanoo Bf8 on CDNA3
static SmallVector<Value>
Fp32_to_Fp8E5M2FNUZ(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtFp32ToFp8(loc, rewriter, v[0], v[1], "bf8");
}

// Fp32 -> Nanoo Fp8 on CDNA3
static SmallVector<Value>
Fp32_to_Fp8E4M3FNUZ(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtFp32ToFp8(loc, rewriter, v[0], v[1], "fp8");
}

// Nanoo Bf8 -> Fp32 on CDNA3
static SmallVector<Value>
Fp8E5M2FNUZ_to_Fp32(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtFp8ToFp32(loc, rewriter, v[0], v[1], "bf8");
}

// Nanoo Fp8 -> Fp32 on CDNA3
static SmallVector<Value>
Fp8E4M3FNUZ_to_Fp32(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtFp8ToFp32(loc, rewriter, v[0], v[1], "fp8");
}

// Depend on whether we focus more on performance, we may skip
// the processing of submornal values
static Value Fp16_to_Fp8E5M2FNUZ_oneValue(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          Value v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto vi16 = b.bitcast(v, i16_ty);
  auto e = b.and_(i16_ty, vi16, b.int_val(16, 0x7C00));
  auto sign = b.and_(i16_ty, vi16, b.int_val(16, 0x8000));

  // normal value
  auto a = b.and_(i16_ty, vi16, b.int_val(16, 0x7FFFF));
  auto a1 = b.add(i16_ty, a, b.int_val(16, 0x0400));
  auto o1 = b.or_(i16_ty, a1, sign);

  // subnormal value, e is 0
  auto m = b.and_(i16_ty, vi16, b.int_val(16, 0x03FF));
  auto m2 = b.shl(m, b.int_val(16, 1));
  auto o2 = b.or_(i16_ty, sign, b.or_(i16_ty, b.int_val(16, 1), m2));

  auto e_is_zero = b.icmp_eq(e, b.int_val(16, 0));
  auto e_is_all1 = b.icmp_eq(e, b.int_val(16, 0x7C00));

  auto ot = b.select(e_is_zero, o2, o1);
  auto o = b.select(e_is_all1, vi16, ot);
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  auto res = b.bitcast(o, fp8x2VecTy);

  return b.extract_element(i8_ty, res, b.i32_val(1));
}

static SmallVector<Value>
Fp16_to_Fp8E5M2FNUZ_SW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  SmallVector<Value> result(2);
  result[0] = Fp16_to_Fp8E5M2FNUZ_oneValue(loc, rewriter, v[0]);
  result[1] = Fp16_to_Fp8E5M2FNUZ_oneValue(loc, rewriter, v[1]);
  return result;
}

static SmallVector<Value>
Fp16_to_Fp8E5M2FNUZ_HW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  return convert_val_Fp16_to_Fp8(loc, rewriter, v[0], v[1], "bf8");
}

ConverterT Fp16_to_Fp8E5M2FNUZ(AMD::ISAFamily isaFamily) {
  return isaFamily == AMD::ISAFamily::CDNA3 ? Fp16_to_Fp8E5M2FNUZ_HW
                                            : Fp16_to_Fp8E5M2FNUZ_SW;
}

static Value Fp8E4M3FN_to_Fp16_oneValue(Location loc,
                                        ConversionPatternRewriter &rewriter,
                                        Value v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  Value a = b.undef(fp8x2VecTy);
  a = b.insert_element(fp8x2VecTy, a, b.i8_val(0), b.i32_val(0));
  a = b.insert_element(fp8x2VecTy, a, v, b.i32_val(1));
  a = b.bitcast(a, i16_ty);

  // Get sign and absolute value
  Value sign = b.and_(a, b.i16_val(0x8000));
  a = b.and_(a, b.i16_val(0x7FFF));

  // Right shift 1 bit to adjust the positions of exponent and mantissa
  a = b.lshr(a, b.i16_val(1));

  // Adjust exponent, (15 - 7) << 10 === 0x2000
  a = b.add(a, b.i16_val(0x2000));

  // Check NaN
  Value vAbs = b.and_(b.bitcast(v, i8_ty), b.i8_val(0x7F));
  a = b.select(b.icmp_eq(vAbs, b.i8_val(0x7F)), b.i16_val(0x7E00), a);

  // Check denorms and zero
  // Here we use a LUT to map S.0000.000 ~ S.0000.111 to its corresponding fp16
  // value
  constexpr size_t lutSize = 8;
  static constexpr int denormsAndZeroLut[lutSize] = {
      0x0000, 0x1800, 0x1C00, 0x1E00, 0x2000, 0x2100, 0x2200, 0x2300};

  for (int i = 0; i < lutSize; i++) {
    a = b.select(b.icmp_eq(vAbs, b.i8_val(i)), b.i16_val(denormsAndZeroLut[i]),
                 a);
  }

  // Set sign
  a = b.or_(a, sign);
  a = b.bitcast(a, f16_ty);

  return a;
}

// Ocp Fp8->Fp16
static SmallVector<Value>
Fp8E4M3FN_to_Fp16_SW(Location loc, ConversionPatternRewriter &rewriter,
                     const SmallVector<Value> &values) {
  SmallVector<Value> results(2);
  results[0] = Fp8E4M3FN_to_Fp16_oneValue(loc, rewriter, values[0]);
  results[1] = Fp8E4M3FN_to_Fp16_oneValue(loc, rewriter, values[1]);
  return results;
}

static SmallVector<Value>
Fp8E4M3FN_to_Fp16_HW(Location loc, ConversionPatternRewriter &rewriter,
                     const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtScalePkUpcastFromFp8<ROCDL::CvtScaleF32PkF16Fp8Op>(loc, rewriter,
                                                               v[0], v[1]);
}

ConverterT Fp8E4M3FN_to_Fp16(AMD::ISAFamily isaFamily) {
  return isaFamily == AMD::ISAFamily::CDNA4 ? Fp8E4M3FN_to_Fp16_HW
                                            : Fp8E4M3FN_to_Fp16_SW;
}

// Ocp Bf8->Fp16
static SmallVector<Value>
Fp8E5M2_to_Fp16_SW(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = b.undef(fp8x4VecTy);
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(0));
  a0 = b.insert_element(fp8x4VecTy, a0, v[0], b.i32_val(1));
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(2));
  a0 = b.insert_element(fp8x4VecTy, a0, v[1], b.i32_val(3));
  a0 = b.bitcast(a0, i32_ty);
  Value a1 = b.undef(fp8x4VecTy);
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(0));
  a1 = b.insert_element(fp8x4VecTy, a1, v[2], b.i32_val(1));
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(2));
  a1 = b.insert_element(fp8x4VecTy, a1, v[3], b.i32_val(3));
  a1 = b.bitcast(a1, i32_ty);

  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  auto fp16x2Vec0 = b.bitcast(a0, fp16x2VecTy);
  auto fp16x2Vec1 = b.bitcast(a1, fp16x2VecTy);

  return {b.extract_element(f16_ty, fp16x2Vec0, b.i32_val(0)),
          b.extract_element(f16_ty, fp16x2Vec0, b.i32_val(1)),
          b.extract_element(f16_ty, fp16x2Vec1, b.i32_val(0)),
          b.extract_element(f16_ty, fp16x2Vec1, b.i32_val(1))};
}

static SmallVector<Value>
Fp8E5M2_to_Fp16_HW(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtScalePkUpcastFromFp8<ROCDL::CvtScaleF32PkF16Bf8Op>(loc, rewriter,
                                                               v[0], v[1]);
}

ConverterT Fp8E5M2_to_Fp16(AMD::ISAFamily isaFamily) {
  return isaFamily == AMD::ISAFamily::CDNA4 ? Fp8E5M2_to_Fp16_HW
                                            : Fp8E5M2_to_Fp16_SW;
}

static Value convertBf16ToFp32(Location loc,
                               ConversionPatternRewriter &rewriter,
                               const Value &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto as_int16 = b.bitcast(v, i16_ty);
  auto as_int32 = b.zext(i32_ty, as_int16);
  auto shifted = b.shl(i32_ty, as_int32, b.i32_val(16));
  return b.bitcast(shifted, f32_ty);
}

static Value convertFp32ToBf16(Location loc,
                               ConversionPatternRewriter &rewriter,
                               const Value &v, const RoundingMode rounding) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto as_int32 = b.bitcast(v, i32_ty);
  if (rounding == RoundingMode::RTZ) {
    auto shifted = b.lshr(i32_ty, as_int32, b.i32_val(16));
    auto truncated = b.trunc(i16_ty, shifted);
    return b.bitcast(truncated, bf16_ty);
  }

  // This implementation is a faster version for fp32 to bf16 type conversion
  // It is from CK:
  // https://github.com/cgmillette/composable_kernel/commit/24e75bef6aa5
  // It uses less VGPR and less number of instructions compared to the
  // previous implementation
  Value isNan = checkIsNan(b, v);
  Value v16 = b.i32_val(16);
  Value tmp = b.and_(i32_ty, b.lshr(i32_ty, as_int32, v16), b.i32_val(1));

  Value v7FFF = b.i32_val(0x7FFF);
  Value s1 = b.add(as_int32, tmp);
  Value s2 = b.add(s1, v7FFF);

  Value vNan = b.i32_val(0x7FFF0000);
  Value res = b.select(isNan, vNan, s2);

  Value shifted = b.lshr(i32_ty, res, v16);
  Value truncated = b.trunc(i16_ty, shifted);
  return b.bitcast(truncated, bf16_ty);
}

static Value Fp8E5M2FNUZ_to_Fp16_oneValue(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          Value v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  Value a = b.undef(fp8x2VecTy);
  a = b.insert_element(fp8x2VecTy, a, b.int_val(8, 0), b.i32_val(0));
  a = b.insert_element(fp8x2VecTy, a, v, b.i32_val(1));
  a = b.bitcast(a, i16_ty);

  auto e = b.and_(i16_ty, a, b.int_val(16, 0x7C00));
  auto m = b.and_(i16_ty, a, b.int_val(16, 0x0300));
  auto sign = b.and_(i16_ty, a, b.int_val(16, 0x8000));

  // check whether all exponents are zeros
  auto e_is_zero = b.icmp_eq(e, b.int_val(16, 0x0));

  // case 1, e is zero, need to move m right by 1 bit
  auto m1 = b.lshr(i16_ty, m, b.int_val(16, 1));
  auto o0 = b.or_(i16_ty, sign, m1);

  // case 2, e is nonzero, sub exponent by 1
  auto e1 = b.sub(i16_ty, e, b.int_val(16, 0x0400));

  auto e_is_one = b.icmp_eq(e, b.int_val(16, 0x0400));
  auto m2 = b.add(i16_ty, m1, b.int_val(16, 0x0200));

  auto o1 = b.or_(i16_ty, sign, b.or_(i16_ty, m, e1));
  auto o2 = b.or_(i16_ty, sign, m2);

  auto o12 = b.select(e_is_one, o2, o1);
  auto o = b.select(e_is_zero, o0, o12);

  return b.bitcast(o, f16_ty);
}

static SmallVector<Value>
Fp8E5M2FNUZ_to_Fp16_SW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  SmallVector<Value> result(2);
  result[0] = Fp8E5M2FNUZ_to_Fp16_oneValue(loc, rewriter, v[0]);
  result[1] = Fp8E5M2FNUZ_to_Fp16_oneValue(loc, rewriter, v[1]);
  return result;
}

static SmallVector<Value>
Fp8E5M2FNUZ_to_Fp16_HW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  return convert_val_Fp8_to_Fp16(loc, rewriter, v[0], v[1], "bf8");
}

ConverterT Fp8E5M2FNUZ_to_Fp16(AMD::ISAFamily isaFamily) {
  return isaFamily == AMD::ISAFamily::CDNA3 ? Fp8E5M2FNUZ_to_Fp16_HW
                                            : Fp8E5M2FNUZ_to_Fp16_SW;
}

// OCP Bf8 -> Bf16
static SmallVector<Value>
Fp8E5M2_to_Bf16_SW(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = b.undef(fp8x4VecTy);
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(0));
  a0 = b.insert_element(fp8x4VecTy, a0, v[0], b.i32_val(1));
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(2));
  a0 = b.insert_element(fp8x4VecTy, a0, v[1], b.i32_val(3));
  a0 = b.bitcast(a0, i32_ty);

  Value a1 = b.undef(fp8x4VecTy);
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(0));
  a1 = b.insert_element(fp8x4VecTy, a1, v[2], b.i32_val(1));
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(2));
  a1 = b.insert_element(fp8x4VecTy, a1, v[3], b.i32_val(3));
  a1 = b.bitcast(a1, i32_ty);

  Value b0 = b.and_(i32_ty, a0, b.i32_val(0x7fff7fff));
  Value b1 = b.and_(i32_ty, a1, b.i32_val(0x7fff7fff));
  b0 = b.lshr(i32_ty, b0, b.i32_val(3));
  b1 = b.lshr(i32_ty, b1, b.i32_val(3));

  Value c0 = b.shl(i32_ty, b0, b.i32_val(16));
  Value c1 = b.and_(i32_ty, b0, b.i32_val(0xFFFF0000));
  Value c2 = b.shl(i32_ty, b1, b.i32_val(16));
  Value c3 = b.and_(i32_ty, b1, b.i32_val(0xFFFF0000));

  c0 = b.bitcast(c0, f32_ty);
  c1 = b.bitcast(c1, f32_ty);
  c2 = b.bitcast(c2, f32_ty);
  c3 = b.bitcast(c3, f32_ty);

  Value d0 = b.fmul(f32_ty, c0, b.f32_val(0x1p+112));
  Value d1 = b.fmul(f32_ty, c1, b.f32_val(0x1p+112));
  Value d2 = b.fmul(f32_ty, c2, b.f32_val(0x1p+112));
  Value d3 = b.fmul(f32_ty, c3, b.f32_val(0x1p+112));

  d0 = b.bitcast(d0, i32_ty);
  d1 = b.bitcast(d1, i32_ty);
  d2 = b.bitcast(d2, i32_ty);
  d3 = b.bitcast(d3, i32_ty);

  Value out0 = b.or_(i32_ty, b.lshr(i32_ty, d0, b.i32_val(16)), d1);
  Value out1 = b.or_(i32_ty, b.lshr(i32_ty, d2, b.i32_val(16)), d3);

  Value sign0 = b.and_(i32_ty, a0, b.i32_val(0x80008000));
  Value sign1 = b.and_(i32_ty, a1, b.i32_val(0x80008000));

  out0 = b.or_(i32_ty, out0, sign0);
  out1 = b.or_(i32_ty, out1, sign1);

  auto bf16x2VecTy = vec_ty(bf16_ty, 2);
  out0 = b.bitcast(out0, bf16x2VecTy);
  out1 = b.bitcast(out1, bf16x2VecTy);

  return {b.extract_element(bf16_ty, out0, b.i32_val(0)),
          b.extract_element(bf16_ty, out0, b.i32_val(1)),
          b.extract_element(bf16_ty, out1, b.i32_val(0)),
          b.extract_element(bf16_ty, out1, b.i32_val(1))};
}

static SmallVector<Value>
Fp8E5M2_to_Bf16_HW(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtScalePkUpcastFromFp8<ROCDL::CvtScaleF32PkBf16Bf8Op>(loc, rewriter,
                                                                v[0], v[1]);
}

ConverterT Fp8E5M2_to_Bf16(AMD::ISAFamily isaFamily) {
  return isaFamily == AMD::ISAFamily::CDNA4 ? Fp8E5M2_to_Bf16_HW
                                            : Fp8E5M2_to_Bf16_SW;
}

// Bf16 -> OCP Bf8
static SmallVector<Value>
Bf16_to_Fp8E5M2_SW(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto bf16x2VecTy = vec_ty(bf16_ty, 2);
  Value bf16x2Vec0 = b.undef(bf16x2VecTy);
  Value bf16x2Vec1 = b.undef(bf16x2VecTy);
  bf16x2Vec0 = b.insert_element(bf16x2VecTy, bf16x2Vec0, v[0], b.i32_val(0));
  bf16x2Vec0 = b.insert_element(bf16x2VecTy, bf16x2Vec0, v[1], b.i32_val(1));
  bf16x2Vec1 = b.insert_element(bf16x2VecTy, bf16x2Vec1, v[2], b.i32_val(0));
  bf16x2Vec1 = b.insert_element(bf16x2VecTy, bf16x2Vec1, v[3], b.i32_val(1));
  bf16x2Vec0 = b.bitcast(bf16x2Vec0, i32_ty);
  bf16x2Vec1 = b.bitcast(bf16x2Vec1, i32_ty);

  Value sign0 = b.and_(i32_ty, bf16x2Vec0, b.i32_val(0x80008000));
  Value sign1 = b.and_(i32_ty, bf16x2Vec1, b.i32_val(0x80008000));
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value sign = b.undef(fp8x4VecTy);
  sign0 = b.bitcast(sign0, fp8x4VecTy);
  sign1 = b.bitcast(sign1, fp8x4VecTy);
  sign = b.insert_element(fp8x4VecTy, sign,
                          b.extract_element(i8_ty, sign0, b.i32_val(1)),
                          b.i32_val(0));
  sign = b.insert_element(fp8x4VecTy, sign,
                          b.extract_element(i8_ty, sign0, b.i32_val(3)),
                          b.i32_val(1));
  sign = b.insert_element(fp8x4VecTy, sign,
                          b.extract_element(i8_ty, sign1, b.i32_val(1)),
                          b.i32_val(2));
  sign = b.insert_element(fp8x4VecTy, sign,
                          b.extract_element(i8_ty, sign1, b.i32_val(3)),
                          b.i32_val(3));
  sign = b.bitcast(sign, i32_ty);

  Value nosign0 = b.and_(i32_ty, bf16x2Vec0, b.i32_val(0x7fff7fff));
  Value nosign1 = b.and_(i32_ty, bf16x2Vec1, b.i32_val(0x7fff7fff));

  Value nosign_0_0 = b.and_(i32_ty, nosign0, b.i32_val(0xffff0000));
  nosign_0_0 = b.umax(i32_ty, nosign_0_0, b.i32_val(0x38000000));
  nosign_0_0 = b.umin(i32_ty, nosign_0_0, b.i32_val(0x57e00000));
  Value nosign_0_1 = b.and_(i32_ty, nosign0, b.i32_val(0x0000ffff));
  nosign_0_1 = b.umax(i32_ty, nosign_0_1, b.i32_val(0x3800));
  nosign_0_1 = b.umin(i32_ty, nosign_0_1, b.i32_val(0x57e0));
  nosign0 = b.or_(i32_ty, nosign_0_0, nosign_0_1);

  Value nosign_1_0 = b.and_(i32_ty, nosign1, b.i32_val(0xffff0000));
  nosign_1_0 = b.umax(i32_ty, nosign_1_0, b.i32_val(0x38000000));
  nosign_1_0 = b.umin(i32_ty, nosign_1_0, b.i32_val(0x57e00000));
  Value nosign_1_1 = b.and_(i32_ty, nosign1, b.i32_val(0x0000ffff));
  nosign_1_1 = b.umax(i32_ty, nosign_1_1, b.i32_val(0x3800));
  nosign_1_1 = b.umin(i32_ty, nosign_1_1, b.i32_val(0x57e0));
  nosign1 = b.or_(i32_ty, nosign_1_0, nosign_1_1);

  nosign0 = b.add(i32_ty, nosign0, b.i32_val(0x00100010));
  nosign1 = b.add(i32_ty, nosign1, b.i32_val(0x00100010));
  nosign0 = b.sub(i32_ty, nosign0, b.i32_val(0x38003800));
  nosign1 = b.sub(i32_ty, nosign1, b.i32_val(0x38003800));
  nosign0 = b.shl(i32_ty, nosign0, b.i32_val(3));
  nosign1 = b.shl(i32_ty, nosign1, b.i32_val(3));

  nosign0 = b.bitcast(nosign0, fp8x4VecTy);
  nosign1 = b.bitcast(nosign1, fp8x4VecTy);
  Value nosign = b.undef(fp8x4VecTy);
  nosign = b.insert_element(fp8x4VecTy, nosign,
                            b.extract_element(i8_ty, nosign0, b.i32_val(1)),
                            b.i32_val(0));
  nosign = b.insert_element(fp8x4VecTy, nosign,
                            b.extract_element(i8_ty, nosign0, b.i32_val(3)),
                            b.i32_val(1));
  nosign = b.insert_element(fp8x4VecTy, nosign,
                            b.extract_element(i8_ty, nosign1, b.i32_val(1)),
                            b.i32_val(2));
  nosign = b.insert_element(fp8x4VecTy, nosign,
                            b.extract_element(i8_ty, nosign1, b.i32_val(3)),
                            b.i32_val(3));
  nosign = b.bitcast(nosign, i32_ty);

  Value fp8x4Vec = b.or_(i32_ty, nosign, sign);
  fp8x4Vec = b.bitcast(fp8x4Vec, fp8x4VecTy);
  return {b.extract_element(i8_ty, fp8x4Vec, b.i32_val(0)),
          b.extract_element(i8_ty, fp8x4Vec, b.i32_val(1)),
          b.extract_element(i8_ty, fp8x4Vec, b.i32_val(2)),
          b.extract_element(i8_ty, fp8x4Vec, b.i32_val(3))};
}

static SmallVector<Value>
Bf16_to_Fp8E5M2_HW(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtScalePkDowncastToFp8<ROCDL::CvtScaleF32PkBf8Bf16Op>(loc, rewriter,
                                                                v[0], v[1]);
}

static ConverterT Bf16_to_Fp8E5M2(AMD::ISAFamily isaFamily) {
  return isaFamily == AMD::ISAFamily::CDNA4 ? Bf16_to_Fp8E5M2_HW
                                            : Bf16_to_Fp8E5M2_SW;
}
// Bf16 -> OCP Fp8
static SmallVector<Value> Bf16_to_Fp8E4M3FN(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtScalePkDowncastToFp8<ROCDL::CvtScaleF32PkFp8Bf16Op>(loc, rewriter,
                                                                v[0], v[1]);
}

// fp8e4m3fn to bf16
static SmallVector<Value>
Fp8E4M3FN_to_Bf16_SW(Location loc, ConversionPatternRewriter &rewriter,
                     const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = b.undef(fp8x4VecTy);
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(0));
  a0 = b.insert_element(fp8x4VecTy, a0, v[0], b.i32_val(1));
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(2));
  a0 = b.insert_element(fp8x4VecTy, a0, v[1], b.i32_val(3));
  a0 = b.bitcast(a0, i32_ty);

  Value b0 = b.and_(i32_ty, a0, b.i32_val(0x7fff7fff));
  b0 = b.lshr(i32_ty, b0, b.i32_val(4));

  Value c0 = b.shl(i32_ty, b0, b.i32_val(16));
  Value c1 = b.and_(i32_ty, b0, b.i32_val(0xFFFF0000));
  c0 = b.bitcast(c0, f32_ty);
  c1 = b.bitcast(c1, f32_ty);

  Value d0 = b.fmul(f32_ty, c0, b.f32_val(0x1p+120)); // bias 2**(127-7)
  Value d1 = b.fmul(f32_ty, c1, b.f32_val(0x1p+120));
  d0 = b.bitcast(d0, i32_ty);
  d1 = b.bitcast(d1, i32_ty);

  Value out0 = b.or_(i32_ty, b.lshr(i32_ty, d0, b.i32_val(16)), d1);
  Value sign0 = b.and_(i32_ty, a0, b.i32_val(0x80008000));
  out0 = b.or_(i32_ty, out0, sign0);

  auto bf16x2VecTy = vec_ty(bf16_ty, 2);
  out0 = b.bitcast(out0, bf16x2VecTy);
  return {b.extract_element(bf16_ty, out0, b.i32_val(0)),
          b.extract_element(bf16_ty, out0, b.i32_val(1))};
}

static SmallVector<Value>
Fp8E4M3FN_to_Bf16_HW(Location loc, ConversionPatternRewriter &rewriter,
                     const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtScalePkUpcastFromFp8<ROCDL::CvtScaleF32PkBf16Fp8Op>(loc, rewriter,
                                                                v[0], v[1]);
}

ConverterT Fp8E4M3FN_to_Bf16(AMD::ISAFamily isaFamily) {
  return isaFamily == AMD::ISAFamily::CDNA4 ? Fp8E4M3FN_to_Bf16_HW
                                            : Fp8E4M3FN_to_Bf16_SW;
}

// fp8e4m3fnuz to bf16
static SmallVector<Value>
Fp8E4M3FNUZ_to_Bf16(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 2);
  auto ret = cvtFp8ToFp32(loc, rewriter, v[0], v[1], "fp8");
  ret[0] = convertFp32ToBf16(loc, rewriter, ret[0], RoundingMode::RTZ);
  ret[1] = convertFp32ToBf16(loc, rewriter, ret[1], RoundingMode::RTZ);
  return ret;
}

// bf16 to fp8e4m3fnuz
static SmallVector<Value>
Bf16_to_Fp8E4M3FNUZ(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 2);
  auto v0 = convertBf16ToFp32(loc, rewriter, v[0]);
  auto v1 = convertBf16ToFp32(loc, rewriter, v[1]);
  return cvtFp32ToFp8(loc, rewriter, v0, v1, "fp8");
}

// fp8e5m2fnuz to bf16
static SmallVector<Value>
Fp8E5M2FNUZ_to_Bf16(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 2);
  auto ret = cvtFp8ToFp32(loc, rewriter, v[0], v[1], "bf8");
  ret[0] = convertFp32ToBf16(loc, rewriter, ret[0], RoundingMode::RTZ);
  ret[1] = convertFp32ToBf16(loc, rewriter, ret[1], RoundingMode::RTZ);
  return ret;
}

// bf16 to fp8e5m2fnuz
static SmallVector<Value>
Bf16_to_Fp8E5M2FNUZ(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 2);
  auto v0 = convertBf16ToFp32(loc, rewriter, v[0]);
  auto v1 = convertBf16ToFp32(loc, rewriter, v[1]);
  return cvtFp32ToFp8(loc, rewriter, v0, v1, "bf8");
}

static Value Fp8E4M3FNUZ_to_Fp16_oneValue(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          Value v) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  Value a = tb.undef(fp8x2VecTy);
  a = tb.insert_element(fp8x2VecTy, a, tb.int_val(8, 0), tb.i32_val(0));
  a = tb.insert_element(fp8x2VecTy, a, v, tb.i32_val(1));
  a = tb.bitcast(a, i16_ty);

  auto e_mask = tb.int_val(16, 0x7A00);
  auto e = tb.and_(i16_ty, a, e_mask);

  auto m = tb.and_(i16_ty, a, tb.int_val(16, 0x0700));
  auto sign = tb.and_(i16_ty, a, tb.int_val(16, 0x8000));

  // check whether all exponents are zeros
  auto e_is_zero = tb.icmp_eq(e, tb.int_val(16, 0x0));
  auto b = tb.and_(i16_ty, a, tb.int_val(16, 0x7FFF));
  auto b1 = tb.lshr(i16_ty, b, tb.int_val(16, 1));

  // case 1, e is nonzero, add exponent by 6
  auto o0v = tb.add(i16_ty, b1, tb.int_val(16, 0x0C00));
  auto o0 = tb.or_(i16_ty, o0v, sign);

  // case 2, e is nonzero, add exponent by 7
  auto o1v = tb.add(i16_ty, b1, tb.int_val(16, 0x1C00));
  auto o1 = tb.or_(i16_ty, o1v, sign);

  auto io = tb.select(e_is_zero, o0, o1);
  return tb.bitcast(io, f16_ty);
}

static SmallVector<Value>
Fp8E4M3FNUZ_to_Fp16_SW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  SmallVector<Value> result(2);
  result[0] = Fp8E4M3FNUZ_to_Fp16_oneValue(loc, rewriter, v[0]);
  result[1] = Fp8E4M3FNUZ_to_Fp16_oneValue(loc, rewriter, v[1]);
  return result;
}

static SmallVector<Value>
Fp8E4M3FNUZ_to_Fp16_HW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  return convert_val_Fp8_to_Fp16(loc, rewriter, v[0], v[1], "fp8");
}

static ConverterT Fp8E4M3FNUZ_to_Fp16(AMD::ISAFamily isaFamily) {
  return isaFamily == AMD::ISAFamily::CDNA3 ? Fp8E4M3FNUZ_to_Fp16_HW
                                            : Fp8E4M3FNUZ_to_Fp16_SW;
}

// Fp16 -> Fp8E4M3 (packed)
static Value Fp16_to_Fp8E4M3FNUZ_oneValue(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          Value v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto vi16 = b.bitcast(v, i16_ty);
  auto e10 = b.and_(vi16, b.int_val(16, 0x7C00));
  auto e = b.lshr(i16_ty, e10, b.int_val(16, 10));

  auto s = b.and_(i16_ty, vi16, b.int_val(16, 0x8000));

  auto m7 = b.and_(i16_ty, vi16, b.int_val(16, 0x0380));
  auto m = b.shl(i16_ty, m7, b.int_val(16, 1));

  // three cases:
  //  1) e > 21 --> e = 1111,
  //  2) e <= 7 ---> e = 0,
  //  3) others, normal conversion
  auto e1 = b.int_val(16, 0x7800);
  auto e2 = b.int_val(16, 0x0);
  auto e31 = b.sub(i16_ty, e10, b.int_val(16, 0x1C00));
  auto e3 = b.shl(i16_ty, e31, b.int_val(16, 1));

  auto c13 = b.icmp_sgt(e, b.int_val(16, 21));
  auto e13 = b.select(c13, e1, e3);
  auto c23 = b.icmp_sle(e, b.int_val(16, 7));
  auto re = b.select(c23, e2, e13);

  auto r = b.or_(i16_ty, s, b.or_(i16_ty, re, m));
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  auto res = b.bitcast(r, fp8x2VecTy);

  return b.extract_element(i8_ty, res, b.i32_val(1));
}

static SmallVector<Value>
Fp16_to_Fp8E4M3FNUZ_SW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  SmallVector<Value> result(2);
  result[0] = Fp16_to_Fp8E4M3FNUZ_oneValue(loc, rewriter, v[0]);
  result[1] = Fp16_to_Fp8E4M3FNUZ_oneValue(loc, rewriter, v[1]);

  return result;
}

static SmallVector<Value>
Fp16_to_Fp8E4M3FNUZ_HW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  return convert_val_Fp16_to_Fp8(loc, rewriter, v[0], v[1], "fp8");
}

static ConverterT Fp16_to_Fp8E4M3FNUZ(AMD::ISAFamily isaFamily) {
  return isaFamily == AMD::ISAFamily::CDNA3 ? Fp16_to_Fp8E4M3FNUZ_HW
                                            : Fp16_to_Fp8E4M3FNUZ_SW;
}

//===----------------------------------------------------------------------===//
// Data type conversion patterns
//===----------------------------------------------------------------------===//

template <typename SourceOp, typename DestOp>
struct ElementwiseOpConversion
    : public ElementwiseOpConversionBase<
          SourceOp, ElementwiseOpConversion<SourceOp, DestOp>> {
  using Base = ElementwiseOpConversionBase<SourceOp, ElementwiseOpConversion>;
  using OpAdaptor = typename Base::OpAdaptor;

  using Base::Base;

  // An interface to support variant DestOp builder.
  SmallVector<DestOp> createDestOps(SourceOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    Type elemTy, MultipleOperandsRange operands,
                                    Location loc) const {
    return {rewriter.create<DestOp>(loc, elemTy, operands[0],
                                    adaptor.getAttributes().getValue())};
  }
};

// Attempts to use vectorized conversions via inline PTX when possible.
struct FpToFpOpConversion
    : public ElementwiseOpConversionBase<triton::FpToFpOp, FpToFpOpConversion> {
  explicit FpToFpOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              AMD::ISAFamily isaFamily,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        isaFamily(isaFamily) {}

  static Value convertFp16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    return cvtFp16ToFp32(loc, rewriter, v);
  }

  FailureOr<ConverterT>
  getConversionFunc(Type srcTy, Type dstTy,
                    std::optional<RoundingMode> roundingMode) const {
    auto F8E4M3B15TyID = TypeID::get<Float8E4M3B11FNUZType>();
    auto F8E4M3FNUZTyID = TypeID::get<Float8E4M3FNUZType>();
    auto F8E5M2FNUZTyID = TypeID::get<Float8E5M2FNUZType>();
    auto F8E5M2TyID = TypeID::get<Float8E5M2Type>();
    auto F8E4M3FNTyID = TypeID::get<Float8E4M3FNType>();
    auto F16TyID = TypeID::get<Float16Type>();
    auto BF16TyID = TypeID::get<BFloat16Type>();
    auto F32TyID = TypeID::get<Float32Type>();
    auto F64TyID = TypeID::get<Float64Type>();

    auto undefRounding = static_cast<RoundingMode>(-1);

    static DenseMap<std::tuple<TypeID, TypeID, RoundingMode>, ConverterT>
        srcMap = {
            // F8 -> F16
            {{F8E4M3FNUZTyID, F16TyID, undefRounding},
             Fp8E4M3FNUZ_to_Fp16(isaFamily)},
            {{F8E4M3FNTyID, F16TyID, undefRounding},
             Fp8E4M3FN_to_Fp16(isaFamily)},
            {{F8E5M2FNUZTyID, F16TyID, undefRounding},
             Fp8E5M2FNUZ_to_Fp16(isaFamily)},
            {{F8E5M2TyID, F16TyID, undefRounding}, Fp8E5M2_to_Fp16(isaFamily)},
            // F16 -> F8
            {{F16TyID, F8E4M3FNTyID, RoundingMode::RTNE},
             Fp16_to_Fp8E4M3FN_RTNE(isaFamily)},
            {{F16TyID, F8E5M2FNUZTyID, RoundingMode::RTNE},
             Fp16_to_Fp8E5M2FNUZ(isaFamily)},
            {{F16TyID, F8E4M3FNUZTyID, RoundingMode::RTNE},
             Fp16_to_Fp8E4M3FNUZ(isaFamily)},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTNE},
             Fp16_to_Fp8E5M2_RTNE(isaFamily)},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTZ}, Fp16_to_Fp8E5M2_RTZ},
            // F8 -> BF16
            {{F8E5M2TyID, BF16TyID, undefRounding}, Fp8E5M2_to_Bf16(isaFamily)},
            {{F8E5M2FNUZTyID, BF16TyID, undefRounding}, Fp8E5M2FNUZ_to_Bf16},
            {{F8E4M3FNTyID, BF16TyID, undefRounding},
             Fp8E4M3FN_to_Bf16(isaFamily)},
            {{F8E4M3FNUZTyID, BF16TyID, undefRounding}, Fp8E4M3FNUZ_to_Bf16},
            // BF16 -> F8
            {{BF16TyID, F8E5M2TyID, RoundingMode::RTNE},
             Bf16_to_Fp8E5M2(isaFamily)},
            {{BF16TyID, F8E4M3FNTyID, RoundingMode::RTNE}, Bf16_to_Fp8E4M3FN},
            {{BF16TyID, F8E5M2FNUZTyID, RoundingMode::RTNE},
             Bf16_to_Fp8E5M2FNUZ},
            {{BF16TyID, F8E4M3FNUZTyID, RoundingMode::RTNE},
             Bf16_to_Fp8E4M3FNUZ},
            // F32 <-> F8
            {{F32TyID, F8E4M3FNUZTyID, RoundingMode::RTNE},
             Fp32_to_Fp8E4M3FNUZ},
            {{F32TyID, F8E5M2FNUZTyID, RoundingMode::RTNE},
             Fp32_to_Fp8E5M2FNUZ},
            {{F32TyID, F8E4M3FNTyID, RoundingMode::RTNE}, Fp32_to_Fp8E4M3FN},
            {{F32TyID, F8E5M2TyID, RoundingMode::RTNE}, Fp32_to_Fp8E5M2},
            {{F8E4M3FNUZTyID, F32TyID, undefRounding}, Fp8E4M3FNUZ_to_Fp32},
            {{F8E5M2FNUZTyID, F32TyID, undefRounding}, Fp8E5M2FNUZ_to_Fp32},
            {{F8E4M3FNTyID, F32TyID, undefRounding}, Fp8E4M3FN_to_Fp32},
            {{F8E5M2TyID, F32TyID, undefRounding}, Fp8E5M2_to_Fp32},
        };
    std::tuple<TypeID, TypeID, RoundingMode> key = {
        srcTy.getTypeID(), dstTy.getTypeID(),
        roundingMode.value_or(undefRounding)};
    if (srcMap.count(key) == 0) {
      return failure();
    }
    return srcMap.lookup(key);
  }

  SmallVector<Value> createDestOps(triton::FpToFpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcElementType = getElementType(op.getSrc());
    auto dstElementType = getElementType(op.getResult());

    auto roundingMode = op.getRounding();
    if (srcElementType.isF32() && dstElementType.isF16()) {
      assert(roundingMode.has_value() &&
             "rounding mode must be specified for fp32->fp16 conversion");
      SmallVector<Value> outVals;
      outVals.reserve(operands[0].size());
      for (Value v : operands[0]) {
        outVals.push_back(
            LLVM::AMD::cvtFp32ToFp16(loc, rewriter, v, roundingMode.value()));
      }
      return outVals;
    }

    if (srcElementType.isF32() && dstElementType.isBF16()) {
      assert(roundingMode.has_value() &&
             "rounding mode must be specified for fp32->bf16 conversion");
      SmallVector<Value> outVals;
      outVals.reserve(operands[0].size());
      for (Value v : operands[0]) {
        outVals.push_back(
            convertFp32ToBf16(loc, rewriter, v, roundingMode.value()));
      }
      return outVals;
    }

    // numElements = 4 for conversions:
    // ocp bf8->fp32/fp16/bf16 on non-CDNA4, or
    // fp32/bf16/fp16->ocp bf8 on non-CDNA4
    // fp32/bf16/fp16->ocp bf8 (RTZ) on CDNA4
    size_t numElements = 2;
    if ((llvm::isa<Float8E5M2Type>(srcElementType) &&
         isaFamily != AMD::ISAFamily::CDNA4) ||
        (llvm::isa<Float8E5M2Type>(dstElementType) &&
         isaFamily != AMD::ISAFamily::CDNA4) ||
        (llvm::isa<Float8E5M2Type>(dstElementType) &&
         roundingMode != RoundingMode::RTNE &&
         isaFamily == AMD::ISAFamily::CDNA4)) {
      numElements = 4;
    }

    // f32->fp8/bf8, if not nanoo fp8/bf8 on CDNA3 or ocp fp8/bf8 on CDNA4, is
    // done in two steps: f32->fp16 with rtne and fp16->fp8/bf8 with rtne
    bool useFP16IntermediateSrc =
        srcElementType.isF32() &&
        !(isaFamily == AMD::ISAFamily::CDNA4 &&
          (llvm::isa<Float8E4M3FNType, Float8E5M2Type>(dstElementType)) &&
          roundingMode == RoundingMode::RTNE) &&
        !(isaFamily == AMD::ISAFamily::CDNA3 &&
          (llvm::isa<Float8E4M3FNUZType, Float8E5M2FNUZType>(dstElementType)));

    // fp8/bf8->f32, if not nanoo fp8/bf8 on CDNA3 or ocp fp8/bf8 on CDNA4, is
    // done in two steps: fp8/bf8->fp16 and fp16->fp32
    bool isDstFP32 = dstElementType.isF32();
    bool useFP16IntermediateDst =
        (isDstFP32 &&
         !(isaFamily == AMD::ISAFamily::CDNA4 &&
           (llvm::isa<Float8E4M3FNType, Float8E5M2Type>(srcElementType))) &&
         !(isaFamily == AMD::ISAFamily::CDNA3 &&
           (llvm::isa<Float8E4M3FNUZType, Float8E5M2FNUZType>(
               srcElementType))));

    Type srcType = useFP16IntermediateSrc ? f16_ty : srcElementType;
    Type dstType = useFP16IntermediateDst ? f16_ty : dstElementType;
    SmallVector<Value> inVals;
    inVals.reserve(std::min(numElements, operands.size()));
    for (unsigned i = 0; i < std::min(numElements, operands.size()); i++) {
      inVals.push_back(operands[i][0]);
    }
    bool isSrcFP16 = srcElementType.isF16();
    bool isSrcBF16 = srcElementType.isBF16();

    if ((isSrcFP16 || isSrcBF16) && isDstFP32) {
      SmallVector<Value> outVals;
      for (Value &v : inVals) {
        if (isSrcFP16)
          outVals.push_back(convertFp16ToFp32(loc, rewriter, v));
        else
          outVals.push_back(convertBf16ToFp32(loc, rewriter, v));
      }
      return outVals;
    }
    if (useFP16IntermediateSrc)
      for (Value &v : inVals)
        v = LLVM::AMD::cvtFp32ToFp16(loc, rewriter, v,
                                     roundingMode.value_or(RoundingMode::RTNE));
    inVals.resize(numElements, b.undef(typeConverter->convertType(srcType)));
    SmallVector<Value> outVals;
    if (srcType != dstType) {
      auto getCvtFunc = getConversionFunc(srcType, dstType, roundingMode);
      if (failed(getCvtFunc)) {
        std::string rmError;
        if (roundingMode.has_value())
          rmError = std::string(" with rounding mode ") +
                    stringifyRoundingMode(roundingMode.value()).str();
        op->emitError("Unsupported conversion from ")
            << srcType << " to " << dstType << rmError;
        return outVals;
      } else {
        auto cvtFunc = getCvtFunc.value();
        outVals = cvtFunc(loc, rewriter, inVals);
      }
    } else {
      outVals = inVals;
    }

    assert(outVals.size() == inVals.size());
    outVals.resize(std::min(numElements, operands.size()));
    if (useFP16IntermediateDst)
      for (Value &v : outVals)
        v = convertFp16ToFp32(loc, rewriter, v);
    // Pack values
    return outVals;
  }

private:
  AMD::ISAFamily isaFamily;
};

template <typename OP>
Value EmitDualBF16ElementwiseOp(Location loc,
                                ConversionPatternRewriter &rewriter,
                                MultipleOperandsRange operands) {
  auto v0 = convertBf16ToFp32(loc, rewriter, operands[0][0]);
  auto v1 = convertBf16ToFp32(loc, rewriter, operands[0][1]);
  auto result = rewriter.create<OP>(loc, f32_ty, v0, v1);
  return convertFp32ToBf16(loc, rewriter, result, RoundingMode::RTNE);
}

struct FDivOpConversion
    : ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::DivFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // For non-F32 input, it's lowered to LLVM::FDivOp, which is a
    // IEEE-compliant DIV operation.
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {rewriter.create<LLVM::FDivOp>(loc, elemTy, operands[0][0],
                                            operands[0][1])};

    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // The algorithm comes from
    // https://github.com/llvm/llvm-project/blob/bda7aadf/llvm/lib/Target/AMDGPU/AMDGPULegalizerInfo.cpp#L4980-L5065
    // with the Newton-Raphson refinement removed, to perform a faster,
    // approximated DIV operation, aligning with the `div.full.f32` instruction
    // on the NV backend.
    Value &lhs = operands[0][0];
    Value &rhs = operands[0][1];
    MLIRContext *ctx = rewriter.getContext();
    Type divScaleResType = struct_ty({elemTy, i1_ty});

    // The `llvm.amdgcn.div.scale.f32` instruction's signature is
    // (src0, src1, src2) -> (ret0, ret1), where
    //
    // src0: The numerator or lhs of FDivOp.
    // src1: The denominator or rhs of FDivOp.
    // src2: A boolean indicating which operand to scale. If true, lhs is
    // scaled; Otherwise, rhs is scaled.
    //
    // ret0: The scaled operand.
    // ret1: The VCC register indicating whether post-scaling is required.
    auto denominatorScaleOp = LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, "llvm.amdgcn.div.scale.f32", divScaleResType,
        {lhs, rhs, b.false_val()});
    Value denominatorScaled = b.extract_val(denominatorScaleOp.getResult(0), 0);
    auto numeratorScaleOp = LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, "llvm.amdgcn.div.scale.f32", divScaleResType,
        {lhs, rhs, b.true_val()});
    Value numeratorScaled = b.extract_val(numeratorScaleOp.getResult(0), 0);
    Value vcc = b.extract_val(numeratorScaleOp.getResult(0), 1);

    Value rcp =
        LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.amdgcn.rcp.f32",
                                        elemTy, {denominatorScaled})
            .getResult(0);

    Value approxDiv = b.fmul(numeratorScaled, rcp);

    // Since the Newton-Raphson is skipped, we use 0 instead of approximations
    // as the inputs.
    auto fmas = LLVM::createLLVMIntrinsicCallOp(
                    rewriter, loc, "llvm.amdgcn.div.fmas.f32", elemTy,
                    {b.f32_val(0), b.f32_val(0), approxDiv, vcc})
                    .getResult(0);

    return {LLVM::createLLVMIntrinsicCallOp(rewriter, loc,
                                            "llvm.amdgcn.div.fixup.f32", elemTy,
                                            {fmas, rhs, lhs})
                .getResult(0)};
  }
};

struct FMulOpConversion
    : ElementwiseOpConversionBase<arith::MulFOp, FMulOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::MulFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {EmitDualBF16ElementwiseOp<LLVM::FMulOp>(loc, rewriter, operands)};
    } else {
      return {rewriter.create<LLVM::FMulOp>(loc, elemTy, operands[0][0],
                                            operands[0][1])};
    }
  }
};

struct FAddOpConversion
    : ElementwiseOpConversionBase<arith::AddFOp, FAddOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::AddFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {EmitDualBF16ElementwiseOp<LLVM::FAddOp>(loc, rewriter, operands)};
    } else {
      return {rewriter.create<LLVM::FAddOp>(loc, elemTy, operands[0][0],
                                            operands[0][1])};
    }
  }
};

struct FSubOpConversion
    : ElementwiseOpConversionBase<arith::SubFOp, FSubOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::SubFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {EmitDualBF16ElementwiseOp<LLVM::FSubOp>(loc, rewriter, operands)};
    } else {
      return {rewriter.create<LLVM::FSubOp>(loc, elemTy, operands[0][0],
                                            operands[0][1])};
    }
  }
};

static SmallVector<Value> S8_to_Bf16(Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> inValues = {v[0], v[1], v[2], v[3]};
  SmallVector<Value> outValues = {};
  for (Value inVal : inValues) {
    Value i32Val = b.sext(i32_ty, inVal);

    GCNBuilder builder;
    auto &cvt = *builder.create("v_cvt_f32_i32");
    auto res = builder.newOperand("=v");
    auto operand = builder.newOperand(i32Val, "v");
    cvt(res, operand);
    auto f32Val = builder.launch(rewriter, loc, f32_ty, false);

    f32Val = b.bitcast(f32Val, i32_ty);
    auto shifted = b.lshr(i32_ty, f32Val, b.i32_val(16));
    auto truncated = b.trunc(i16_ty, shifted);
    outValues.push_back(b.bitcast(truncated, bf16_ty));
  }
  return outValues;
}

// Uses inline ptx to convert s8/u8 to bf16, since the
struct SIToFPOpConversion
    : ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::SIToFPOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type inElemTy = getElementType(op.getIn());
    Type outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16() && inElemTy.isInteger(8) && operands.size() >= 4) {
      SmallVector<Value> inVals = {operands[0][0], operands[1][0],
                                   operands[2][0], operands[3][0]};
      auto outVals = S8_to_Bf16(loc, rewriter, inVals);
      assert(outVals.size() == 4);
      return outVals;
    } else if (outElemTy.isBF16()) {
      auto value = rewriter.create<LLVM::SIToFPOp>(loc, f32_ty, operands[0][0]);
      return {convertFp32ToBf16(loc, rewriter, value, RoundingMode::RTNE)};
    } else {
      return {rewriter.create<LLVM::SIToFPOp>(loc, elemTy, operands[0][0])};
    }
  }
};

struct FPToSIOpConversion
    : ElementwiseOpConversionBase<arith::FPToSIOp, FPToSIOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::FPToSIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto value = convertBf16ToFp32(loc, rewriter, operands[0][0]);
      return {rewriter.create<LLVM::FPToSIOp>(loc, elemTy, value)};
    } else {
      return {rewriter.create<LLVM::FPToSIOp>(loc, elemTy, operands[0][0])};
    }
  }
};

struct ExtFOpConversion
    : ElementwiseOpConversionBase<arith::ExtFOp, ExtFOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::ExtFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto outElemTy = getElementType(op.getOut());
      assert(outElemTy.isF32() && "unsupported conversion");
      return {convertBf16ToFp32(loc, rewriter, operands[0][0])};
    } else {
      return {rewriter.create<LLVM::FPExtOp>(loc, elemTy, operands[0][0])};
    }
  }
};

struct TruncFOpConversion
    : ElementwiseOpConversionBase<arith::TruncFOp, TruncFOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  explicit TruncFOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              llvm::AMDGPU::GPUKind gpuKind,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        gpuKind(gpuKind) {}

  SmallVector<Value> createDestOps(arith::TruncFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16() && gpuKind != llvm::AMDGPU::GK_GFX950) {
      auto inElemTy = getElementType(op.getIn());
      assert(inElemTy.isF32() && "unsupported conversion");
      return {
          convertFp32ToBf16(loc, rewriter, operands[0][0], RoundingMode::RTNE)};
    } else {
      return {rewriter.create<LLVM::FPTruncOp>(loc, elemTy, operands[0][0])};
    }
  }

private:
  llvm::AMDGPU::GPUKind gpuKind;
};

struct ExpOpConversionApprox
    : ElementwiseOpConversionBase<math::ExpOp, ExpOpConversionApprox> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(math::ExpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // For non-FP32 input, call __ocml_exp_f64 for higher-precision calculation
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    const double log2e = 1.4426950408889634;
    Value prod = b.fmul(f32_ty, operands[0][0], b.f32_val(log2e));

    // Here we use llvm.exp2.f32 instead of math::Exp2Op. The latter
    // flushes denorms by default, but we want to preserve denorms by default
    // for expOp.
    StringRef funcName = "llvm.exp2.f32";
    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);

    return {LLVM::createLLVMCallOp(rewriter, loc, funcOp, prod).getResult()};
  }
};

struct Exp2OpConversion
    : ElementwiseOpConversionBase<math::Exp2Op, Exp2OpConversion> {
  explicit Exp2OpConversion(LLVMTypeConverter &typeConverter,
                            ModuleAxisInfoAnalysis &axisInfoAnalysis, bool ftz,
                            PatternBenefit benefit)
      : ElementwiseOpConversionBase(typeConverter, axisInfoAnalysis, benefit),
        ftz(ftz) {}

  SmallVector<Value> createDestOps(math::Exp2Op op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // For non-FP32 input, call __ocml_exp2_f64 for higher-precision calculation
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    // On AMD backend, both intrinsics are lowered to v_exp_f32 instruction,
    // which flushes input and output denorms. `llvm.amdgcn.exp2.f32` provides
    // direct access to v_exp_f32. For `llvm.exp2.f32`, the LLVM backend inserts
    // instructions to handle denorms iff `allow_flush_denorm` is False.
    StringRef funcName = ftz ? "llvm.amdgcn.exp2.f32" : "llvm.exp2.f32";
    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);

    return {
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]).getResult()};
  }

private:
  bool ftz;
};

struct RsqrtOpConversion
    : ElementwiseOpConversionBase<math::RsqrtOp, RsqrtOpConversion> {
  explicit RsqrtOpConversion(LLVMTypeConverter &typeConverter,
                             ModuleAxisInfoAnalysis &axisInfoAnalysis, bool ftz,
                             PatternBenefit benefit)
      : ElementwiseOpConversionBase(typeConverter, axisInfoAnalysis, benefit),
        ftz(ftz) {}

  SmallVector<Value> createDestOps(math::RsqrtOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // This pass only deals with FP32 input with ftz configuration. Other cases
    // are delegate to MLIR.
    //
    // For FP16/FP64 input, it's lowered to __ocml_rsqrt_f16/__ocml_rsqrt_f64.
    //
    // For FP32 input with non-ftz configuration, it's lowered to
    // __ocml_rsqrt_f32, which will check the ftz/daz settings in the backend
    // dynamically to decide to preserve/flush denorms.
    if (elemTy.getIntOrFloatBitWidth() != 32 || !ftz)
      return {};

    // `llvm.amdgcn.rsq.f32` provides direct access to v_rsq_f32_e32.
    StringRef funcName = "llvm.amdgcn.rsq.f32";

    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);

    return {
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]).getResult()};
  }

private:
  bool ftz;
};

static inline std::pair<Value, Value>
scaleUpIfDenorm(ConversionPatternRewriter &rewriter, Location loc,
                const Value &src, float scaleThreshold, float scaleFactor) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value needScale = b.fcmp_ogt(b.f32_val(scaleThreshold), src);
  Value scaledSrc = b.fmul(f32_ty, src, b.f32_val(scaleFactor));
  Value selectedSrc = b.select(needScale, scaledSrc, src);
  return {needScale, selectedSrc};
}

static inline Value scaleDownIfDenorm(ConversionPatternRewriter &rewriter,
                                      Location loc, const Value &src,
                                      Value needScale, float scaleFactor) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value scaledSrc = b.fmul(f32_ty, src, b.f32_val(scaleFactor));
  return b.select(needScale, scaledSrc, src);
}

struct SqrtOpConversion
    : ElementwiseOpConversionBase<math::SqrtOp, SqrtOpConversion> {
  explicit SqrtOpConversion(LLVMTypeConverter &typeConverter,
                            ModuleAxisInfoAnalysis &axisInfoAnalysis, bool ftz,
                            PatternBenefit benefit)
      : ElementwiseOpConversionBase(typeConverter, axisInfoAnalysis, benefit),
        ftz(ftz) {}

  SmallVector<Value> createDestOps(math::SqrtOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // This function only handles FP32 inputs. Other data types are lowered to
    // LLVM::SqrtOp by MLIR.
    //
    // On the AMDGPU backend, instructions legalized from LLVM::SqrtOp are
    // designed to produce IEEE-compliant results and always preserve denorms.
    // But what we actually need is an approximated SQRT. So we need to manually
    // lower the op.
    //
    // Differences in this approach are
    // 1. Refinement iterations following llvm.amdgcn.sqrt.f32 are removed to
    // improve performance.
    // 2. With ftz enabled, the scaling-up-and-down process is bypassed to
    // ensure denorms are flushed to zero.
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    Value needScale = b.false_val();
    Value scaledSrc = operands[0][0];
    if (!ftz) {
      // For non-ftz cases, if the input value is below 2^{-96}, it needs to be
      // scaled up by a factor of 2^{32}, to prevent it from being flushed by
      // llvm.amdgcn.sqrt.f32.
      //
      // The result is then scaled down afterward to get the correct result.
      // Reference:
      // https://github.com/llvm/llvm-project/blob/0876c11c/llvm/lib/Target/AMDGPU/AMDGPULegalizerInfo.cpp#L5235-L5314.
      std::tie(needScale, scaledSrc) = scaleUpIfDenorm(
          rewriter, loc, operands[0][0], 0x1.0p-96f, 0x1.0p+32f);
    }

    // llvm.amdgcn.sqrt.f32 provides direct access to v_sqrt_f32, which provides
    // 1ULP accuracy and flushs denorms.
    StringRef funcName = "llvm.amdgcn.sqrt.f32";

    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);

    Value intrinsicsOutput =
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]).getResult();

    if (!ftz) {
      // In case of non-ftz, we need to calibrate the results by scaling down by
      // a factor of 2^{-16}.
      return {scaleDownIfDenorm(rewriter, loc, intrinsicsOutput, needScale,
                                0x1.0p-16f)};
    } else {
      return {intrinsicsOutput};
    }
  }

private:
  bool ftz;
};

struct PreciseSqrtOpConversion
    : ElementwiseOpConversionBase<triton::PreciseSqrtOp,
                                  PreciseSqrtOpConversion> {
  explicit PreciseSqrtOpConversion(LLVMTypeConverter &typeConverter,
                                   ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                   bool ftz, PatternBenefit benefit)
      : ElementwiseOpConversionBase(typeConverter, axisInfoAnalysis, benefit),
        ftz(ftz) {}

  SmallVector<Value> createDestOps(triton::PreciseSqrtOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // If the op is neither FP32 nor denorm flushing(ftz), it's directly lowered
    // to LLVM::SqrtOp.
    if (elemTy.getIntOrFloatBitWidth() != 32 || !ftz) {
      return {rewriter.create<LLVM::SqrtOp>(
          loc, elemTy, operands[0], adaptor.getAttributes().getValue())};
    }

    // On the AMDGPU backend, instructions legalized from LLVM::SqrtOp are
    // designed to always preserve denorms, according to
    // https://github.com/llvm/llvm-project/blob/3d6b2d49/llvm/lib/Target/AMDGPU/AMDGPULegalizerInfo.cpp#L5235-L5314.
    //
    // For f32 inputs with ftz enabled, we need to manually lower the op to
    // bypass the scaling-up-and-down process while keeping other parts
    // unchanged. To ensure IEEE-compliant results, we approximate `sqrt(x)`
    // using `x * rsq(x)` and apply extra refinement iterations to correct the
    // result.
    StringRef funcName = "llvm.amdgcn.rsq.f32";

    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);

    Value sqrtR =
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]).getResult();

    Value sqrtX = operands[0][0];
    Value sqrtS = b.fmul(f32_ty, sqrtX, sqrtR);

    // Refine the approximation with Newton iteration
    Value sqrtH = b.fmul(f32_ty, sqrtR, b.f32_val(0.5f));
    Value sqrtE = b.fma(b.neg(f32_ty, sqrtH), sqrtS, b.f32_val(0.5f));
    sqrtH = b.fma(sqrtH, sqrtE, sqrtH);
    sqrtS = b.fma(sqrtS, sqrtE, sqrtS);
    Value sqrtD = b.fma(b.neg(f32_ty, sqrtS), sqrtS, sqrtX);
    sqrtS = b.fma(sqrtD, sqrtH, sqrtS);

    // Handle +0/-0/+inf
    // These flags come from
    // https://github.com/llvm/llvm-project/blob/217e0f39/llvm/include/llvm/ADT/FloatingPointMode.h#L239-L265.
    const unsigned fcPosInf = 0x0200;
    const unsigned fcNegZero = 0x0020;
    const unsigned fcPosZero = 0x0040;
    const unsigned fcZero = fcNegZero | fcPosZero;

    Value isZeroOrPosInf =
        rewriter.create<LLVM::IsFPClass>(loc, i1_ty, sqrtX, fcPosInf | fcZero);
    return {b.select(isZeroOrPosInf, sqrtX, sqrtS)};
  }

private:
  bool ftz;
};

} // namespace

namespace mlir::triton::AMD {
void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns, bool ftz,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    const TargetInfo &targetInfo, PatternBenefit benefit) {

  // fmin (return NaN if either op is NaN)
  patterns.add<ElementwiseOpConversion<arith::MinimumFOp, LLVM::MinimumOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  // fmax (return NaN if either op is NaN)
  patterns.add<ElementwiseOpConversion<arith::MaximumFOp, LLVM::MaximumOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ElementwiseOpConversion<triton::PreciseDivFOp, LLVM::FDivOp>>(
      typeConverter, axisInfoAnalysis, benefit);

  patterns.add<FDivOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FSubOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FAddOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FMulOpConversion>(typeConverter, axisInfoAnalysis, benefit);

  patterns.add<ExtFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<TruncFOpConversion>(typeConverter, axisInfoAnalysis,
                                   targetInfo.getGPUKind(), benefit);
  patterns.add<FPToSIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<SIToFPOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FpToFpOpConversion>(typeConverter, axisInfoAnalysis,
                                   targetInfo.getISAFamily(), benefit);

  // ExpOpConversionApprox will try using __ocml_exp2_f32 if the input type is
  // FP32. For other input types, ExpOpConversionApprox will return failure and
  // later pass will call __ocml_exp_f64 for higher-precision calculation
  patterns.add<ExpOpConversionApprox>(typeConverter, axisInfoAnalysis, benefit);
  // Exp2OpConversion will use llvm.exp2.f32 or llvm.amdgcn.exp2.f32
  // based on the ftz flag if the input type is FP32. For FP64 input,
  // Exp2OpConversion will return failure and later pass will call
  // __ocml_exp2_f64 for higher-precision calculation
  patterns.add<Exp2OpConversion>(typeConverter, axisInfoAnalysis, ftz, benefit);
  patterns.add<RsqrtOpConversion>(typeConverter, axisInfoAnalysis, ftz,
                                  benefit);
  patterns.add<SqrtOpConversion>(typeConverter, axisInfoAnalysis, ftz, benefit);
  patterns.add<PreciseSqrtOpConversion>(typeConverter, axisInfoAnalysis, ftz,
                                        benefit);
  triton::populateElementwiseOpToLLVMPatterns(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
  triton::populateMinMaxFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis,
      /*hwNanPropagationSupported=*/false, benefit);
  triton::populateClampFOpToLLVMPattern(typeConverter, patterns,
                                        axisInfoAnalysis, targetInfo, benefit);
}
} // namespace mlir::triton::AMD
