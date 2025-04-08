#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir::triton::gpu;

namespace mlir::triton {

namespace gpu {
namespace {

/* ----- FP8E5M2 ------ */
// This data-type is the standard FP8E5M2 format

struct Fp8ConversionDesc {
  std::string ptx;
  int inVecWidthBits;
  int outVecWidthBits;
  size_t numElements;
};

static const Fp8ConversionDesc Fp16_to_Fp8E5M2_RTNE(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    ret = {
        "{                            \n"
        ".reg .b32 a<2>;              \n"
        "and.b32 a0, $1, 0x01000100;  \n"    // a0 = $1 & 0x01000100
        "and.b32 a1, $2, 0x01000100;  \n"    // (least significant result bit)
        "shr.b32 a0, a0, 8;           \n"    // a0 >>= 8
        "shr.b32 a1, a1, 8;           \n"    // (shift the lsb)
        "add.u32 a0, a0, 0x007f007f;  \n"    // a0 += 0x007f007f
        "add.u32 a1, a1, 0x007f007f;  \n"    // (add rounding base)
        "add.u32 a0, a0, $1;          \n"    // res = $1 + lsb + 0x7f
        "add.u32 a1, a1, $2;          \n"    // (round to nearest)
        "prmt.b32 $0, a0, a1, 0x7531; \n\t"  // output = a1a0
        "}",
        32, 32, 4};
  } else {
    ret = {"cvt.rn.satfinite.e5m2x2.f16x2 $0, $1; \n\t", 32, 16, 2};
  }
  return ret;
}

const Fp8ConversionDesc Fp16_to_Fp8E5M2_RTZ = {
    "{                            \n"
    ".reg .b32 a<2>;              \n"
    "and.b32 a0, $1, 0xfffefffe;  \n"   // a0 &= 0xfffefffe
    "and.b32 a1, $2, 0xfffefffe;  \n"   // (strip lowest bit)
    "prmt.b32 $0, a0, a1, 0x7531; \n\t" // output = a1a0
    "}",
    32, 32, 4};

static const Fp8ConversionDesc Fp8E5M2_to_Fp16(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    ret = {"{                           \n"
           "prmt.b32 $0, 0, $2, 0x5140; \n\t"
           "prmt.b32 $1, 0, $2, 0x7362; \n\t"
           "}",
           32, 32, 4};
  } else {
    ret = {"cvt.rn.f16x2.e5m2x2 $0, $1; \n\t", 16, 32, 2};
  }
  return ret;
}

static const Fp8ConversionDesc Fp8E5M2_to_Bf16(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    ret = {
        "{                                        \n"
        ".reg .b32 a<2>, b<2>, c<4>, d<4>, e112;  \n" // if input = 0xf1f2f3f4
        "mov.u32 e112, 0x77800000;                \n"
        "prmt.b32 a0, 0, $2, 0x5140;              \n" // a0 = 0xf300f400
        "prmt.b32 a1, 0, $2, 0x7362;              \n" // a1 = 0xf100f200
        "lop3.b32 b0, a0, 0x7fff7fff, 0, 0xc0;    \n" // b0 = a0 & 0x7fff7fff
        "lop3.b32 b1, a1, 0x7fff7fff, 0, 0xc0;    \n" // (strip sign)
        "shr.b32  b0, b0, 3;                      \n" // b0 >>= 3
        "shr.b32  b1, b1, 3;                      \n" // shift into bf16
                                                      // position
        "and.b32 c0, b0, 0xFFFF0000;              \n" // c0 = f3
        "shl.b32 c1, b0, 16;                      \n" // c1 = f4
        "and.b32 c2, b1, 0xFFFF0000;              \n" // c2 = f1
        "shl.b32 c3, b1, 16;                      \n" // c3 = f2
        "mul.f32 d0, c0, e112;                    \n" // d0 = c0 * 0x77800000
        "mul.f32 d1, c1, e112;                    \n" // d1 = c1 * 0x77800000
        "mul.f32 d2, c2, e112;                    \n" // d2 = c2 * 0x77800000
        "mul.f32 d3, c3, e112;                    \n" // d3 = c3 * 0x77800000
        "prmt.b32 b0, d0, d1, 0x3276;             \n" // b0 = 0xd3d4
        "prmt.b32 b1, d2, d3, 0x3276;             \n" // b1 = 0xd1d2
        "lop3.b32 $0, b0, 0x80008000, a0, 0xf8;   \n" // out0 =
                                                      // b0|(0x80008000&a0)
        "lop3.b32 $1, b1, 0x80008000, a1, 0xf8;   \n" // (restore sign)
        "}",
        32, 32, 4};
  } else {
    ret = {
        "{                                       \n"
        ".reg .b32 a<2>, b<2>;                  \n" // if input = 0xf1f2f3f4
        ".reg .b32 e112;                        \n"
        "mov.u32 e112, 0x77807780;              \n" // 2**112 represented as
                                                    // bf16x2
        "prmt.b32 a0, 0, $2, 0x5140;            \n" // a0 = 0xf300f400
        "prmt.b32 a1, 0, $2, 0x7362;            \n" // a1 = 0xf100f200
        "lop3.b32 b0, a0, 0x7fff7fff, 0, 0xc0;  \n" // b0 = a0 & 0x7fff7fff
        "lop3.b32 b1, a1, 0x7fff7fff, 0, 0xc0;  \n" // (strip sign)
        "shr.b32  b0, b0, 3;                    \n" // b0 >>= 3
        "shr.b32  b1, b1, 3;                    \n" // shift into bf16 position
        "lop3.b32 b0, b0, 0x80008000, a0, 0xf8; \n" // out0 = b0|(0x80008000&a0)
        "lop3.b32 b1, b1, 0x80008000, a1, 0xf8; \n" // (restore sign)
        "mul.rn.bf16x2 $0, b0, e112;            \n" // b0.exp += 2**7-2**4
        "mul.rn.bf16x2 $1, b1, e112;            \n" // exponent compensate = 112
        "}",
        32, 32, 4};
  }
  return ret;
}

static const Fp8ConversionDesc Bf16_to_Fp8E5M2(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    ret = {
        "{                                           \n" // bf16=fp8>>3 + 112<<7
        ".reg .u32 sign, sign<2>, nosign, nosign<2>; \n" // fp8_min = 0b00000000
        ".reg .u32 fp8_min, fp8_max, rn_;            \n" // fp8_max = 0b11111111
        "mov.u32 fp8_min, 0x38003800;                \n" // so bf16_min = 0x3800
        "mov.u32 fp8_max, 0x57e057e0;                \n" // so bf16_max = 0x57e0
        "mov.u32 rn_, 0x00100010;                    \n" // round to nearest
        "and.b32 sign0, $1, 0x80008000;              \n" // sign0=in0&0x80008000
        "and.b32 sign1, $2, 0x80008000;              \n" // (store sign)
        "prmt.b32 sign, sign0, sign1, 0x7531;        \n"
        "and.b32 nosign0, $1, 0x7fff7fff;            \n" // nosign0=in0&0x7fff7fff
        "and.b32 nosign1, $2, 0x7fff7fff;            \n" // (strip sign)

        // nosign = clamp(nosign, min, max)
        ".reg .u32 nosign_0_<2>, nosign_1_<2>;       \n"
        "and.b32 nosign_0_0, nosign0, 0xffff0000;    \n"
        "max.u32 nosign_0_0, nosign_0_0, 0x38000000; \n"
        "min.u32 nosign_0_0, nosign_0_0, 0x57e00000; \n"
        "and.b32 nosign_0_1, nosign0, 0x0000ffff;    \n"
        "max.u32 nosign_0_1, nosign_0_1, 0x3800;     \n"
        "min.u32 nosign_0_1, nosign_0_1, 0x57e0;     \n"
        "or.b32 nosign0, nosign_0_0, nosign_0_1;     \n"
        "and.b32 nosign_1_0, nosign1, 0xffff0000;    \n"
        "max.u32 nosign_1_0, nosign_1_0, 0x38000000; \n"
        "min.u32 nosign_1_0, nosign_1_0, 0x57e00000; \n"
        "and.b32 nosign_1_1, nosign1, 0x0000ffff;    \n"
        "max.u32 nosign_1_1, nosign_1_1, 0x3800;     \n"
        "min.u32 nosign_1_1, nosign_1_1, 0x57e0;     \n"
        "or.b32 nosign1, nosign_1_0, nosign_1_1;     \n"

        "add.u32 nosign0, nosign0, rn_;              \n" // nosign0 += rn_
        "add.u32 nosign1, nosign1, rn_;              \n" // (round to nearest)
        "sub.u32 nosign0, nosign0, 0x38003800;       \n" // nosign0-=0x38003800
        "sub.u32 nosign1, nosign1, 0x38003800;       \n" // (compensate offset)
        "shl.b32 nosign0, nosign0, 3;                \n" // nosign0 <<= 3
        "shl.b32 nosign1, nosign1, 3;                \n" // shift into to fp8e4
        "prmt.b32 nosign, nosign0, nosign1, 0x7531;  \n" // nosign0 = 0xf100f200
                                                         // nosign1 = 0xf300f400
                                                         // nosign = 0xf3f4f1f2
        "or.b32 $0, nosign, sign;                    \n" // restore sign
        "}",
        32, 32, 4};
  } else {
    ret = {"{                                       \n"
           ".reg .b16 a<2>;                         \n"
           ".reg .f32 b<2>;                         \n"
           "mov.b32 {a0, a1}, $1;                   \n"
           "cvt.f32.bf16 b0, a0;                    \n"
           "cvt.f32.bf16 b1, a1;                    \n"
           "cvt.rn.satfinite.e5m2x2.f32 $0, b1, b0; \n"
           "}",
           32, 16, 2};
  }
  return ret;
}

// Fp8E4M3 (x2) -> Fp16 (x2) (packed)
static const Fp8ConversionDesc Fp8E4M3Nv_to_Fp16 = {
    "{ \n"
    "cvt.rn.f16x2.e4m3x2 $0, $1; \n"
    "}",
    16, 32, 2};

// Fp16 (x2) -> Fp8E4M3 (x2) (packed)
static const Fp8ConversionDesc Fp16_to_Fp8E4M3Nv = {
    "{ \n"
    "cvt.rn.satfinite.e4m3x2.f16x2 $0, $1; \n"
    "}",
    32, 16, 2};

static const Fp8ConversionDesc Fp8E4M3Nv_to_Bf16(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  // Fp8E4M3 (x2) -> Fp16 (x2) (packed)
  if (!hasNativeFP) {
    ret = {"{                                       \n"
           ".reg .b32 a;                            \n"
           ".reg .f16 a<2>;                         \n"
           ".reg .f32 b<2>;                         \n"
           ".reg .b16 c<2>;                         \n"
           "cvt.rn.f16x2.e4m3x2 a, $1;              \n"
           "mov.b32 {a0, a1}, a;                    \n"
           "cvt.f32.f16 b0, a0;                     \n"
           "cvt.f32.f16 b1, a1;                     \n"
           "cvt.rn.bf16.f32 c0, b0;                 \n"
           "cvt.rn.bf16.f32 c1, b1;                 \n"
           "mov.b32 $0, {c0, c1};                   \n"
           "}",
           16, 32, 2};
  } else {
    ret = {"{                                       \n"
           ".reg .b32 a;                            \n"
           ".reg .f16 a<2>;                         \n"
           ".reg .b16 b<2>;                         \n"
           "cvt.rn.f16x2.e4m3x2 a, $1;              \n"
           "mov.b32 {a0, a1}, a;                    \n"
           "cvt.bf16.f16 b0, a0;                    \n"
           "cvt.bf16.f16 b1, a1;                    \n"
           "mov.b32 $0, {b0, b1};                   \n"
           "}",
           16, 32, 2};
  }
  return ret;
}

// Bf16 (x2) -> Fp8E4M3 (x2) (packed)
static const Fp8ConversionDesc Bf16_to_Fp8E4M3Nv = {
    "{                                       \n"
    ".reg .b16 a<2>;                         \n"
    ".reg .f32 b<2>;                         \n"
    "mov.b32 {a0, a1}, $1;                   \n"
    "cvt.f32.bf16 b0, a0;                    \n"
    "cvt.f32.bf16 b1, a1;                    \n"
    "cvt.rn.satfinite.e4m3x2.f32 $0, b1, b0; \n"
    "}",
    32, 16, 2};

// Fp32 (x2) -> Fp8 (x2) (packed)
static const Fp8ConversionDesc Fp32_to_Fp8E4M3Nv = {
    "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \n", 32, 16, 2};
static const Fp8ConversionDesc Fp32_to_Fp8E5M2 = {
    "cvt.rn.satfinite.e5m2x2.f32 $0, $2, $1; \n", 32, 16, 2};

/* ----- Packed integer to BF16 ------ */
static const std::string S8_to_Bf16 =
    "{                                           \n"
    ".reg .s8 s<4>;                              \n"
    ".reg .f32 f<4>;                             \n"
    "mov.b32 {s0, s1, s2, s3}, $2;               \n" // unpack
    "cvt.rn.f32.s8 f0, s0;                       \n" // no s8->bf16 pre-Hopper
    "cvt.rn.f32.s8 f1, s1;                       \n" // fi[0:15] is always 0
    "cvt.rn.f32.s8 f2, s2;                       \n" //
    "cvt.rn.f32.s8 f3, s3;                       \n" //
    "prmt.b32 $0, f0, f1, 0x7632;                \n" // f32->bf16 + pack
    "prmt.b32 $1, f2, f3, 0x7632;                \n" //
    "}";
// Conversions have low throughput, rely on bit tricks instead of cvt
// instruction on Hopper and later GPUs.
static const std::string S8_to_Bf16_sm90 =
    "{                               \n"
    ".reg .b32 l<3>;                 \n"
    ".reg .b32 h<3>;                 \n"
    "prmt.b32 l0, $2, 0x43, 0x4140;  \n" // Unpack to shifted bf16.
    "prmt.b32 h0, $2, 0x43, 0x4342;  \n"
    "and.b32 l1, l0, 0xff7fff7f;     \n" // Zero the least exp bit.
    "and.b32 h1, h0, 0xff7fff7f;     \n"
    "and.b32 l2, l0, 0xff80ff80;     \n" // Zero the mantissa.
    "and.b32 h2, h0, 0xff80ff80;     \n"
    "sub.bf16x2 $0, l1, l2;          \n" // Subtract the offset.
    "sub.bf16x2 $1, h1, h2;          \n"
    "}";

typedef std::function<SmallVector<Value>(Location, ConversionPatternRewriter &,
                                         const SmallVector<Value> &)>
    ConverterT;

static ConverterT makeConverterFromPtx(const std::string &ptxAsm, Type inType,
                                       Type outType,
                                       const int inVecWidthBits = 32,
                                       const int outVecWidthBits = 32) {
  ConverterT converter =
      [ptxAsm, inType, outType, inVecWidthBits,
       outVecWidthBits](Location loc, ConversionPatternRewriter &rewriter,
                        const SmallVector<Value> &v) -> SmallVector<Value> {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int numElements = v.size();
    assert(numElements == 8 || numElements == 4 ||
           numElements == 2 && "invalid vector size");

    auto ctx = rewriter.getContext();
    int inBitwidth = inType.getIntOrFloatBitWidth();
    int outBitwidth = outType.getIntOrFloatBitWidth();
    // first, we pack `v` into 32-bit ints
    int inVecWidth = inVecWidthBits / inBitwidth;
    auto inVecTy = vec_ty(inType, inVecWidth);
    SmallVector<Value> inPacked(numElements / inVecWidth, b.undef(inVecTy));
    for (size_t i = 0; i < numElements; i++)
      inPacked[i / inVecWidth] = b.insert_element(
          inVecTy, inPacked[i / inVecWidth], v[i], b.i32_val(i % inVecWidth));
    for (size_t i = 0; i < inPacked.size(); i++)
      inPacked[i] = b.bitcast(inPacked[i], int_ty(inVecWidthBits));

    // then, we run the provided inline PTX
    int outVecWidth = outVecWidthBits / outBitwidth;
    int outNums = numElements / outVecWidth;
    PTXBuilder builder;
    SmallVector<PTXBuilder::Operand *> operands;
    auto outConstriant = outVecWidthBits == 16 ? "=h" : "=r";
    auto inConstraint = inVecWidthBits == 16 ? "h" : "r";
    for (int i = 0; i < outNums; i++) {
      operands.push_back(builder.newOperand(outConstriant));
    }

    for (Value inVal : inPacked) {
      operands.push_back(builder.newOperand(inVal, inConstraint));
    }

    auto &ptxOp = *builder.create(ptxAsm);
    ptxOp(operands, /*onlyAttachMLIRArgs=*/true);
    auto outVecTy = vec_ty(outType, outVecWidth);
    SmallVector<Value> outPacked;
    if (outNums == 1)
      outPacked.push_back(builder.launch(rewriter, loc, outVecTy, false));
    else {
      auto outStructTy = struct_ty(SmallVector<Type>(outNums, outVecTy));
      auto outStruct = builder.launch(rewriter, loc, outStructTy, false);
      for (int i = 0; i < outNums; i++)
        outPacked.push_back(b.extract_val(outVecTy, outStruct, i));
    }
    // unpack the output
    SmallVector<Value> ret;
    for (size_t i = 0; i < numElements; i++)
      ret.push_back(b.extract_element(outType, outPacked[i / outVecWidth],
                                      b.i32_val(i % outVecWidth)));
    return ret;
  };
  return converter;
}

// Attempts to use vectorized conversions via inline PTX when possible.
struct FpToFpOpConversion
    : public ElementwiseOpConversionBase<FpToFpOp, FpToFpOpConversion> {
  using ElementwiseOpConversionBase<
      FpToFpOp, FpToFpOpConversion>::ElementwiseOpConversionBase;

  explicit FpToFpOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              int computeCapability,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        computeCapability(computeCapability) {}

  static Value convertFp16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    return rewriter.create<LLVM::FPExtOp>(loc, f32_ty, v);
  }

  static Value convertFp32ToBf16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v, const RoundingMode rounding) {
    StringRef name;
    switch (rounding) {
    case RoundingMode::RTNE:
      name = "llvm.nvvm.f2bf16.rn";
      break;
    case RoundingMode::RTZ:
      name = "llvm.nvvm.f2bf16.rz";
      break;
    default:
      emitError(loc) << "unsupported rounding mode for f32->bf16 conversion: "
                     << stringifyRoundingMode(rounding) << "\n";
      llvm::report_fatal_error(
          "unsupported rounding mode for f32->bf16 conversion: " +
          stringifyRoundingMode(rounding) + "\n");
    }
    return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, name, bf16_ty, {v})
        .getResult(0);
  }

  static Value convertFp32ToFp16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v, const RoundingMode rounding) {
    PTXBuilder builder;
    StringRef ptx;
    switch (rounding) {
    case RoundingMode::RTNE:
      ptx = "cvt.rn.f16.f32";
      break;
    case RoundingMode::RTZ:
      ptx = "cvt.rz.f16.f32";
      break;
    default:
      emitError(loc) << "unsupported rounding mode for f32->f16 conversion: "
                     << stringifyRoundingMode(rounding) << "\n";
      llvm::report_fatal_error(
          "unsupported rounding mode for f32->f16 conversion: " +
          stringifyRoundingMode(rounding) + "\n");
    }
    auto &cvt = *builder.create(ptx.str());
    auto res = builder.newOperand("=h");
    auto operand = builder.newOperand(v, "r");
    cvt(res, operand);
    return builder.launch(rewriter, loc, f16_ty, false);
  }

  std::pair<ConverterT, size_t>
  getConversionFunc(Type srcTy, Type dstTy,
                    std::optional<RoundingMode> roundingMode) const {
    auto F8E4M3TyID = TypeID::get<Float8E4M3FNType>();
    auto F8E5M2TyID = TypeID::get<Float8E5M2Type>();
    auto F16TyID = TypeID::get<Float16Type>();
    auto BF16TyID = TypeID::get<BFloat16Type>();
    auto F32TyID = TypeID::get<Float32Type>();
    auto F64TyID = TypeID::get<Float64Type>();

    auto undefRounding = static_cast<RoundingMode>(-1);

    static DenseMap<std::tuple<TypeID, TypeID, RoundingMode>, Fp8ConversionDesc>
        srcMap = {
            // F8 -> F16
            {{F8E4M3TyID, F16TyID, undefRounding}, Fp8E4M3Nv_to_Fp16},
            {{F8E5M2TyID, F16TyID, undefRounding},
             Fp8E5M2_to_Fp16(computeCapability >= 89)},
            {{F16TyID, F8E4M3TyID, RoundingMode::RTNE}, Fp16_to_Fp8E4M3Nv},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTNE},
             Fp16_to_Fp8E5M2_RTNE(computeCapability >= 89)},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTZ}, Fp16_to_Fp8E5M2_RTZ},
            // F8 -> BF16
            {{F8E5M2TyID, BF16TyID, undefRounding},
             Fp8E5M2_to_Bf16(computeCapability >= 89)},
            {{F8E4M3TyID, BF16TyID, undefRounding},
             Fp8E4M3Nv_to_Bf16(computeCapability >= 89)},
            // BF16 -> F8
            {{BF16TyID, F8E5M2TyID, RoundingMode::RTNE},
             Bf16_to_Fp8E5M2(computeCapability >= 89)},
            {{BF16TyID, F8E4M3TyID, RoundingMode::RTNE}, Bf16_to_Fp8E4M3Nv},
            // F32 -> F8
            {{F32TyID, F8E4M3TyID, RoundingMode::RTNE}, Fp32_to_Fp8E4M3Nv},
            {{F32TyID, F8E5M2TyID, RoundingMode::RTNE}, Fp32_to_Fp8E5M2},
        };
    std::tuple<TypeID, TypeID, RoundingMode> key = {
        srcTy.getTypeID(), dstTy.getTypeID(),
        roundingMode.value_or(undefRounding)};
    if (srcMap.count(key) == 0) {
      llvm::errs() << "Unsupported conversion from " << srcTy << " to "
                   << dstTy;
      if (roundingMode.has_value())
        llvm::errs() << " with rounding mode "
                     << stringifyRoundingMode(roundingMode.value());
      llvm::errs() << "\n";
      llvm::report_fatal_error("Unsupported rounding mode for conversion.");
    }
    if (computeCapability < 89 && (llvm::isa<Float8E4M3FNType>(srcTy) ||
                                   llvm::isa<Float8E4M3FNType>(dstTy))) {
      llvm::report_fatal_error("Conversion from/to f8e4m3nv is only supported "
                               "on compute capability >= 89\n");
    }
    auto convDesc = srcMap.lookup(key);
    return {makeConverterFromPtx(
                convDesc.ptx, getTypeConverter()->convertType(srcTy),
                getTypeConverter()->convertType(dstTy), convDesc.inVecWidthBits,
                convDesc.outVecWidthBits),
            convDesc.numElements};
  }

  SmallVector<Value> createDestOps(FpToFpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcElementType = getElementType(op.getSrc());
    auto dstElementType = getElementType(op.getResult());
    auto roundingMode = op.getRounding();

    if (llvm::isa<Float8E5M2Type, Float8E4M3FNType>(dstElementType)) {
      assert(roundingMode.has_value() &&
             "Rounding mode must be specified for convertsions to fp8");

      // For now only RTNE is supported for conversions from fp16 to fp8
      if (!srcElementType.isF32() &&
          roundingMode.value() != RoundingMode::RTNE) {
        llvm::report_fatal_error(
            "Unsupported rounding mode for conversion to fp8: " +
            stringifyRoundingMode(roundingMode.value()) + "\n");
      }
    }

    if (srcElementType.isF32() && dstElementType.isF16()) {
      assert(roundingMode.has_value() &&
             "rounding mode must be specified for fp32->fp16 conversion");
      SmallVector<Value> outVals;
      for (Value v : operands[0]) {
        outVals.push_back(
            convertFp32ToFp16(loc, rewriter, v, roundingMode.value()));
      }
      return outVals;
    }

    if (srcElementType.isF32() && dstElementType.isBF16()) {
      assert(roundingMode.has_value() &&
             "rounding mode must be specified for fp32->bf16 conversion");
      SmallVector<Value> outVals;
      for (Value v : operands[0]) {
        outVals.push_back(
            convertFp32ToBf16(loc, rewriter, v, roundingMode.value()));
      }
      return outVals;
    }

    bool useFP16IntermediateSrc =
        srcElementType.isF32() &&
        (!(computeCapability >= 90 &&
           (llvm::isa<Float8E4M3FNType, Float8E5M2Type>(dstElementType))) ||
         roundingMode.value() == RoundingMode::RTZ);
    bool isDstFP32 = dstElementType.isF32();
    Type srcType = useFP16IntermediateSrc ? f16_ty : srcElementType;
    Type dstType = isDstFP32 ? f16_ty : dstElementType;
    auto [cvtFunc, numElements] =
        getConversionFunc(srcType, dstType, roundingMode);
    SmallVector<Value> inVals;
    for (unsigned i = 0; i < std::min(numElements, operands.size()); i++) {
      inVals.push_back(operands[i][0]);
    }
    if (useFP16IntermediateSrc)
      for (Value &v : inVals)
        v = convertFp32ToFp16(loc, rewriter, v, RoundingMode::RTZ);
    inVals.resize(numElements, b.undef(typeConverter->convertType(srcType)));
    SmallVector<Value> outVals = cvtFunc(loc, rewriter, inVals);
    assert(outVals.size() == inVals.size());
    outVals.resize(std::min(numElements, operands.size()));
    if (isDstFP32)
      for (Value &v : outVals)
        v = convertFp16ToFp32(loc, rewriter, v);
    // Pack values
    return outVals;
  }

private:
  int computeCapability;
};

struct FDivOpConversion
    : ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::DivFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
    StringRef name;
    Type resultTy;
    if (32 == bitwidth) {
      name = "llvm.nvvm.div.full";
      resultTy = f32_ty;
    } else if (64 == bitwidth) {
      name = "llvm.nvvm.div.rn.d";
      resultTy = f64_ty;
    } else {
      llvm::report_fatal_error("Unsupported bitwidth");
    }
    Value args[] = {operands[0][0], operands[0][1]};
    auto callOp =
        LLVM::createLLVMIntrinsicCallOp(rewriter, loc, name, resultTy, args);
    return {callOp.getResult(0)};
  }
};

// Uses inline ptx to convert s8/u8 to bf16, since the
struct SIToFPOpConversion
    : ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion>;
  using Adaptor = typename Base::OpAdaptor;

  explicit SIToFPOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              int computeCapability,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        computeCapability(computeCapability) {}

  LogicalResult matchAndRewrite(
      arith::SIToFPOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (succeeded(matchAndRewriteInt4ToBf16Conversion(op, rewriter))) {
      return success();
    }
    return Base::matchAndRewrite(op, adaptor, rewriter);
  }

  // Matches subgraph of convert 8xi4 to 8xbf16 and rewrites it to inline PTX.
  LogicalResult matchAndRewriteInt4ToBf16Conversion(
      arith::SIToFPOp op, ConversionPatternRewriter &rewriter) const {
    if (computeCapability < 90) return failure();
    Type inElemTy = getElementType(op.getIn());
    Type outElemTy = getElementType(op.getOut());
    if (!inElemTy.isInteger(8) || !outElemTy.isBF16()) return failure();
    FailureOr<Value> unpack = matchInt4Unpack(op.getIn());
    if (failed(unpack)) return failure();

    Location loc = op.getLoc();
    Value src = rewriter.getRemappedValue(unpack.value());
    auto structTy = dyn_cast<LLVM::LLVMStructType>(src.getType());
    if (!structTy || structTy.getBody().size() % 4 != 0) return failure();
    auto isInt8 = [](Type type) { return type.isInteger(8); };
    if (!all_of(structTy.getBody(), isInt8)) return failure();

    const LLVMTypeConverter *typeConverter = getTypeConverter();
    assert(inElemTy == typeConverter->convertType(inElemTy));
    assert(outElemTy == typeConverter->convertType(outElemTy));

    const std::string S4_to_Bf16_sm90 = R"({
      .reg .b32 r<4>, mi, mf;
      mov.b32 mi, 0x43404340 - 0x00080008;
      mov.b32 mf, 0x43404340;
      // Shift 4-bit inputs to 16-bit boundary.
      shr.u32 r1, $4, 4;
      shr.u32 r2, $4, 8;
      shr.u32 r3, $4, 12;
      // Sign-extend from 4 bits is equivalent to (x ^ 0x8) - 0x8.
      lop3.b32 r0, $4, 0x000f000f, 0x00080008, (0xf0 & 0xcc) ^ 0xaa;
      lop3.b32 r1, r1, 0x000f000f, 0x00080008, (0xf0 & 0xcc) ^ 0xaa;
      lop3.b32 r2, r2, 0x000f000f, 0x00080008, (0xf0 & 0xcc) ^ 0xaa;
      lop3.b32 r3, r3, 0x000f000f, 0x00080008, (0xf0 & 0xcc) ^ 0xaa;
      // Interger-add magic number (minus bias from sign-extend above).
      add.s16x2 r0, r0, mi;
      add.s16x2 r1, r1, mi;
      add.s16x2 r2, r2, mi;
      add.s16x2 r3, r3, mi;
      // Float-subtract magic number.
      sub.bf16x2 r0, r0, mf;
      sub.bf16x2 r1, r1, mf;
      sub.bf16x2 r2, r2, mf;
      sub.bf16x2 r3, r3, mf;
      // Shuffle results into correct order.
      prmt.b32 $0, r1, r0, 0x5410;
      prmt.b32 $1, r3, r2, 0x5410;
      prmt.b32 $2, r1, r0, 0x7632;
      prmt.b32 $3, r3, r2, 0x7632;
    })";

    SmallVector<Value> resultVals;
    SmallVector<Value> unpackedVals = unpackLLElements(loc, src, rewriter);
    auto cvtFunc = makeConverterFromPtx(S4_to_Bf16_sm90, inElemTy, outElemTy);
    for (ValueRange operands = unpackedVals; !operands.empty();
         operands = operands.drop_front(4)) {
      SmallVector<Value> inVals = {
          operands[0], operands[1], operands[2], operands[3],
          // Repeat operands so that cvtFunc produces 8 outputs.
          operands[0], operands[1], operands[2], operands[3]};
      auto outVals = cvtFunc(loc, rewriter, inVals);
      assert(inVals.size() == outVals.size());
      resultVals.append(outVals.begin(), outVals.end());
    }

    resultVals = maybeDeduplicate(op, resultVals);
    Value view =
        packLLElements(loc, typeConverter, resultVals, rewriter, op.getType());
    rewriter.replaceOp(op, view);

    return success();
  }

  // Returns the source if value is the result of an 2xi4 -> 2xi8 unpack
  // sequence.
  static FailureOr<Value> matchInt4Unpack(Value value) {
    auto reshape = value.getDefiningOp<ReshapeOp>();
    if (!reshape) return failure();
    auto join = reshape.getSrc().getDefiningOp<JoinOp>();
    if (!join) return failure();
    auto shrHi = join.getLhs().getDefiningOp<arith::ShRSIOp>();
    if (!shrHi || !isConst4(shrHi.getRhs())) return failure();
    auto shrLo = join.getRhs().getDefiningOp<arith::ShRSIOp>();
    if (!shrLo || !isConst4(shrLo.getRhs())) return failure();
    auto shlLo = shrLo.getLhs().getDefiningOp<arith::ShLIOp>();
    if (!shlLo || !isConst4(shlLo.getRhs())) return failure();
    if (shrHi.getLhs() != shlLo.getLhs()) return failure();
    return shrHi.getLhs();
  }

  // Returns true if the value is equal to 4.
  static bool isConst4(Value v) {
    auto constOp = v.getDefiningOp<arith::ConstantOp>();
    if (!constOp) return false;
    auto attr = mlir::dyn_cast<DenseIntOrFPElementsAttr>(constOp.getValue());
    if (!attr || !attr.isSplat()) return false;
    return attr.getSplatValue<APInt>().getLimitedValue() == 4;
  };

  SmallVector<Value> createDestOps(arith::SIToFPOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type inElemTy = getElementType(op.getIn());
    Type outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16() && inElemTy.isInteger(8) && operands.size() >= 4) {
      auto cvtFunc = makeConverterFromPtx(
          computeCapability >= 90 ? S8_to_Bf16_sm90 : S8_to_Bf16,
          getTypeConverter()->convertType(inElemTy),
          getTypeConverter()->convertType(outElemTy));
      SmallVector<Value> inVals = {operands[0][0], operands[1][0],
                                   operands[2][0], operands[3][0]};
      auto outVals = cvtFunc(loc, rewriter, inVals);
      assert(outVals.size() == 4);
      return outVals;
    } else {
      return {rewriter.create<LLVM::SIToFPOp>(loc, elemTy, operands[0][0])};
    }
  }

private:
  int computeCapability;
};

struct FPToSIOpConversion
    : ElementwiseOpConversionBase<arith::FPToSIOp, FPToSIOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::FPToSIOp, FPToSIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::FPToSIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    return {rewriter.create<LLVM::FPToSIOp>(loc, elemTy, operands[0][0])};
  }
};

struct ExpOpConversionApprox
    : ElementwiseOpConversionBase<math::ExpOp, ExpOpConversionApprox> {
  using Base = ElementwiseOpConversionBase<math::ExpOp, ExpOpConversionApprox>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(math::ExpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // For non-FP32 input, call __nv_expf for higher-precision calculation
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    const double log2e = 1.4426950408889634;
    Value prod = b.fmul(f32_ty, operands[0][0], b.f32_val(log2e));

    Type resultTy = operands[0][0].getType();
    StringRef name = "llvm.nvvm.ex2.approx.f";
    auto callOp =
        LLVM::createLLVMIntrinsicCallOp(rewriter, loc, name, resultTy, {prod});
    return {callOp.getResult(0)};
  }
};

struct ClampFOpConversion
    : ElementwiseOpConversionBase<ClampFOp, ClampFOpConversion> {
  using Base = ElementwiseOpConversionBase<ClampFOp, ClampFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  explicit ClampFOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              int computeCapability,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        computeCapability(computeCapability) {}

  bool isClipPattern(ClampFOp op) const {
    bool xorsignAbsAvailable = (computeCapability >= 90);
    // Pattern matching the sequence of clamp(x, -limit, limit) to generate
    // more efficient PTX code. NOTE: This pattern matching is not general
    // enough, but it is sufficient. We detect only two cases here:
    // 1. where the "-limit" is computed as 0 - limit:
    //   %cst = arith.constant dense<0.000000e+00>
    //   %8 = tt.load %7, %2
    //   %11 = arith.subf %cst, %8
    //   %12 = tt.clamp %5, %11, %8
    // 2. where "-limit" and "limit" are constants.
    //   %cst_6 = arith.constant dense<-6.0000e+00>
    //   %cst_7 = arith.constant dense<6.0000e+00>
    //   %160 = tt.clamp %158, %cst_6, %cst_7
    bool patternFound = false;

    auto getSplatInitializer = [](Value v) -> std::optional<double> {
      if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
        if (auto attr = mlir::dyn_cast<DenseIntOrFPElementsAttr>(
                constOp.getValueAttr())) {
          if (attr.isSplat()) {
            return attr.getSplatValue<APFloat>().convertToDouble();
          }
        }
      }
      return std::nullopt;
    };

    if (xorsignAbsAvailable) {
      if (auto subOp = op.getOperand(1).getDefiningOp<arith::SubFOp>()) {
        if (subOp.getOperand(1) == op.getOperand(2)) {
          auto initializer = getSplatInitializer(subOp.getOperand(0));
          if (initializer.has_value() && initializer.value() == 0.0) {
            patternFound = true;
          }
        }
      } else {
        auto initializer1 = getSplatInitializer(op.getOperand(1));
        auto initializer2 = getSplatInitializer(op.getOperand(2));
        if (initializer1.has_value() && initializer2.has_value() &&
            initializer1.value() == -initializer2.value()) {
          patternFound = true;
        }
      }
    }
    return patternFound;
  }

  SmallVector<Value> emitOptimization(ClampFOp op,
                                      ConversionPatternRewriter &rewriter,
                                      Type elemTy,
                                      MultipleOperandsRange operands,
                                      Location loc) const {
    std::string name = "llvm.nvvm.fmin";
    if (op.getPropagateNan() == PropagateNan::ALL) {
      name += ".nan";
    }
    name += ".xorsign.abs";
    if (elemTy.isF32()) {
      name += ".f";
    } else if (elemTy.isF16()) {
      name += ".f16";
    }

    Type resultTy = operands[0][0].getType();
    Value args[] = {operands[0][0], operands[0][2]};
    auto callOp =
        LLVM::createLLVMIntrinsicCallOp(rewriter, loc, name, resultTy, args);
    return {callOp.getResult(0)};
  }

  SmallVector<Value> createDestOps(ClampFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (isClipPattern(op)) {
      return emitOptimization(op, rewriter, elemTy, operands, loc);
    }
    return {};
  }

private:
  int computeCapability;
};

template <typename TritonOp>
struct OpToExternCallConversion
    : public ElementwiseOpConversionBase<TritonOp,
                                         OpToExternCallConversion<TritonOp>> {
  using Base =
      ElementwiseOpConversionBase<TritonOp, OpToExternCallConversion<TritonOp>>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  explicit OpToExternCallConversion(LLVMTypeConverter &typeConverter,
                                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                                    StringRef externFuncName,
                                    PatternBenefit benefit)
      : Base::ElementwiseOpConversionBase(typeConverter, axisAnalysisPass,
                                          benefit),
        funcName(externFuncName) {}

  SmallVector<Value> createDestOps(TritonOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
    return {
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]).getResult()};
  }

private:
  StringRef funcName;
};
} // namespace
} // namespace gpu

} // namespace mlir::triton

void mlir::triton::NVIDIA::populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  using namespace mlir::triton::gpu;

  patterns.add<OpToExternCallConversion<triton::PreciseSqrtOp>>(
      typeConverter, axisInfoAnalysis, "__nv_fsqrt_rn", benefit);
  patterns.add<OpToExternCallConversion<triton::PreciseDivFOp>>(
      typeConverter, axisInfoAnalysis, "__nv_fdiv_rn", benefit);

  mlir::triton::populateElementwiseOpToLLVMPatterns(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);

#define POPULATE_OP(SRC_OP, DST_OP)                                            \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit)

  POPULATE_OP(arith::SubFOp, LLVM::FSubOp);
  POPULATE_OP(arith::AddFOp, LLVM::FAddOp);
  POPULATE_OP(arith::MulFOp, LLVM::FMulOp);

  POPULATE_OP(arith::ExtFOp, LLVM::FPExtOp);
  POPULATE_OP(arith::TruncFOp, LLVM::FPTruncOp);

#undef POPULATE_OP

  patterns.add<FDivOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FPToSIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<SIToFPOpConversion>(typeConverter, axisInfoAnalysis,
                                   computeCapability, benefit);
  patterns.add<FpToFpOpConversion>(typeConverter, axisInfoAnalysis,
                                   computeCapability, benefit);

  // ExpOpConversionApprox will try using ex2.approx if the input type is
  // FP32. For other input types, ExpOpConversionApprox will return failure and
  // ElementwiseOpConversion<math::ExpOp, math::ExpOp> defined below will call
  // __nv_expf for higher-precision calculation
  patterns.add<ExpOpConversionApprox>(typeConverter, axisInfoAnalysis, benefit);
  bool hwNanPropagationSupported = computeCapability >= 80;
  mlir::triton::populateMinMaxFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis, hwNanPropagationSupported,
      benefit);
  mlir::triton::populateClampFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
}

void mlir::triton::NVIDIA::populateClampFOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
    PatternBenefit benefit) {
  using namespace mlir::triton::gpu;

  patterns.add<ClampFOpConversion>(typeConverter, axisInfoAnalysis,
                                   computeCapability, benefit);
}
