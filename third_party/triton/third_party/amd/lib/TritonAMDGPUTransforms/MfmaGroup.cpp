#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include <tuple>

namespace mlir {
namespace {

//===----------------------------------------------------------------------===//
// MFMA intrinsic query key
//===----------------------------------------------------------------------===//

// The tuple used as key to query MFMA intrinsic map.
using MfmaKey =
    std::tuple<unsigned /*version*/, unsigned /*mDim*/, unsigned /*nDim*/,
               TypeID /*aElemType*/, TypeID /*bElemType*/>;

// Returns a key for querying an MFMA intrinsic for the given parameters.
// Updates the passed-in A/B element type to the chosen MFMA intrinsic's A/B
// element type if the chosen intrinsic is not a direct hit and will require
// emulation.
//
// This function adapts certain parameters so we can be flexible when trying
// to query with "mismatches".
MfmaKey composeMfmaKeyFor(unsigned version, unsigned mDim, unsigned nDim,
                          Type &aElemType, Type &bElemType, bool withScale,
                          bool useTF32) {
  Type aET = aElemType, bET = bElemType;
  Builder b(aElemType.getContext());
  if (withScale) {
    assert(version == 4 && isF8F6F4(aET) && isF8F6F4(bET));
    // For MXFP types, we have the same intrinsic, which uses FP4 as the key
    // in the MFMA map. So adjust to that.
    aET = bET = b.getType<Float4E2M1FNType>();
  } else if (useTF32 && aET.isF32() && bET.isF32()) {
    // In Triton we use fp32 with TF32 input precision to mean TF32 types.
    // In the MFMA map we use the proper TF32 type. So "fix" it here.
    assert(version == 3);
    aET = bET = b.getType<FloatTF32Type>();
  } else if (version <= 3 && isa<Float8E5M2Type>(aET) &&
             isa<Float8E5M2Type>(bET)) {
    // For the OCP FP8 E5M2 type, we can emulate the support for it with FP16.
    aElemType = bElemType = aET = bET = b.getF16Type();
  }
  return {version, mDim, nDim, aET.getTypeID(), bET.getTypeID()};
}

//===----------------------------------------------------------------------===//
// MFMA intrinsic map
//===----------------------------------------------------------------------===//

using MfmaMapValue =
    std::tuple<StringRef /*symbol*/, unsigned /*kDim*/, unsigned /*kBase*/>;
using MfmaMap = llvm::DenseMap<MfmaKey, SmallVector<MfmaMapValue, 2>>;

class MfmaDatabase {
public:
  static const MfmaMap &get(MLIRContext *context) {
    static MfmaDatabase db(context);
    return db.mfmaMap;
  }

private:
  explicit MfmaDatabase(MLIRContext *context);

  MfmaMap mfmaMap;
};

MfmaDatabase::MfmaDatabase(MLIRContext *context) {
// Macro for defining MFMA intrinsics at a specific gfx version.
#define TRITON_MFMA_v(v, m, n, aET, bET, symbol, k, kBase)                     \
  {                                                                            \
    /*key=*/{v, m, n, aET.getTypeID(), bET.getTypeID()}, /*value=*/{           \
      {ROCDL::symbol::getOperationName(), k, kBase},                           \
    }                                                                          \
  }

// For certain architectures, we can have two intrinsics with the same M/N but
// different K. Order matters here: case1 will be preferred to case2.
#define TRITON_MFMA_v_2case(v, m, n, aET, bET, symbol1, k1, kBase1, symbol2,   \
                            k2, kBase2)                                        \
  {                                                                            \
    /*key=*/{v, m, n, aET.getTypeID(), bET.getTypeID()}, /*value=*/{           \
      {ROCDL::symbol1::getOperationName(), k1, kBase1},                        \
          {ROCDL::symbol2::getOperationName(), k2, kBase2},                    \
    }                                                                          \
  }
#define TRITON_MFMA_v4_2case(m, n, aET, bET, symbol1, k1, kBase1, symbol2, k2, \
                             kBase2)                                           \
  TRITON_MFMA_v_2case(4, m, n, aET, bET, symbol1, k1, kBase1, symbol2, k2,     \
                      kBase2)
#define TRITON_MFMA_v2_2case(m, n, aET, bET, symbol1, k1, kBase1, symbol2, k2, \
                             kBase2)                                           \
  TRITON_MFMA_v_2case(2, m, n, aET, bET, symbol1, k1, kBase1, symbol2, k2,     \
                      kBase2)

// Macro for defining MFMA intrinsics existing in multiple gfx versions.
#define TRITON_MFMA_v1to2(m, n, aET, bET, symbol, k, kBase)                    \
  TRITON_MFMA_v(1, m, n, aET, bET, symbol, k, kBase),                          \
      TRITON_MFMA_v(2, m, n, aET, bET, symbol, k, kBase)

#define TRITON_MFMA_v2to3(m, n, aET, bET, symbol, k, kBase)                    \
  TRITON_MFMA_v(2, m, n, aET, bET, symbol, k, kBase),                          \
      TRITON_MFMA_v(3, m, n, aET, bET, symbol, k, kBase)

#define TRITON_MFMA_v3to4(m, n, aET, bET, symbol, k, kBase)                    \
  TRITON_MFMA_v(3, m, n, aET, bET, symbol, k, kBase),                          \
      TRITON_MFMA_v(4, m, n, aET, bET, symbol, k, kBase)

#define TRITON_MFMA_v2to4(m, n, aET, bET, symbol, k, kBase)                    \
  TRITON_MFMA_v(2, m, n, aET, bET, symbol, k, kBase),                          \
      TRITON_MFMA_v3to4(m, n, aET, bET, symbol, k, kBase)

#define TRITON_MFMA_v1to3(m, n, aET, bET, symbol, k, kBase)                    \
  TRITON_MFMA_v(1, m, n, aET, bET, symbol, k, kBase),                          \
      TRITON_MFMA_v2to3(m, n, aET, bET, symbol, k, kBase)

#define TRITON_MFMA_v1to4(m, n, aET, bET, symbol, k, kBase)                    \
  TRITON_MFMA_v(1, m, n, aET, bET, symbol, k, kBase),                          \
      TRITON_MFMA_v2to4(m, n, aET, bET, symbol, k, kBase)

  Builder b(context);
  auto f32T = b.getF32Type();
  auto tf32T = b.getTF32Type();
  auto f16T = b.getF16Type();
  auto bf16T = b.getBF16Type();
  auto i8T = b.getI8Type();
  auto amdFp8T = b.getType<Float8E4M3FNUZType>();
  auto amdBf8T = b.getType<Float8E5M2FNUZType>();
  auto ocpFp8T = b.getType<Float8E4M3FNType>();
  auto ocpBf8T = b.getType<Float8E5M2Type>();
  auto fp4T = b.getType<Float4E2M1FNType>();

  mfmaMap = {
      // f32 inputs
      // mfma_f32_32x32x2f32
      TRITON_MFMA_v1to4(32, 32, f32T, f32T, mfma_f32_32x32x2f32, 2, 1),
      // mfma_f32_16x16x4f32
      TRITON_MFMA_v1to4(16, 16, f32T, f32T, mfma_f32_16x16x4f32, 4, 1),
      // mfma_f32_4x4x1f32 / mfma_f32_4x4x1_16B_f32
      TRITON_MFMA_v1to4(4, 4, f32T, f32T, mfma_f32_4x4x1f32, 16, 1),
      TRITON_MFMA_v1to4(4, 64, f32T, f32T, mfma_f32_4x4x1f32, 1, 1),
      TRITON_MFMA_v1to4(64, 4, f32T, f32T, mfma_f32_4x4x1f32, 1, 1),

      // xf32
      // mfma.xf32.16x16x8xf32
      TRITON_MFMA_v(3, 16, 16, tf32T, tf32T, mfma_f32_16x16x8_xf32, 8, 2),
      // mfma.xf32.32x32x4.xf32
      TRITON_MFMA_v(3, 32, 32, tf32T, tf32T, mfma_f32_32x32x4_xf32, 4, 2),

      // f16 inputs
      // mfma_f32_32x32x16_f16 & mfma_f32_32x32x8f16
      TRITON_MFMA_v4_2case(32, 32, f16T, f16T, mfma_f32_32x32x16_f16, 16, 8,
                           mfma_f32_32x32x8f16, 8, 4),
      // mfma_f32_32x32x8f16
      TRITON_MFMA_v1to3(32, 32, f16T, f16T, mfma_f32_32x32x8f16, 8, 4),
      // mfma_f32_16x16x32_f16 & mfma_f32_16x16x16f16
      TRITON_MFMA_v4_2case(16, 16, f16T, f16T, mfma_f32_16x16x32_f16, 32, 8,
                           mfma_f32_16x16x16f16, 16, 4),
      // mfma_f32_16x16x16f16
      TRITON_MFMA_v1to3(16, 16, f16T, f16T, mfma_f32_16x16x16f16, 16, 4),
      // mfma_f32_4x4x4f16
      TRITON_MFMA_v1to4(4, 4, f16T, f16T, mfma_f32_4x4x4f16, 64, 4),
      TRITON_MFMA_v1to4(4, 64, f16T, f16T, mfma_f32_4x4x4f16, 4, 4),
      TRITON_MFMA_v1to4(64, 4, f16T, f16T, mfma_f32_4x4x4f16, 4, 4),

      // bf16 inputs
      // mfma_f32_32x32x16_bf16 & mfma_f32_32x32x8_bf16_1K
      TRITON_MFMA_v4_2case(32, 32, bf16T, bf16T, mfma_f32_32x32x16_bf16, 16, 8,
                           mfma_f32_32x32x8bf16_1k, 8, 4),
      TRITON_MFMA_v(3, 32, 32, bf16T, bf16T, mfma_f32_32x32x8bf16_1k, 8, 4),
      // mfma_f32_32x32x8_bf16_1K & mfma_f32_32x32x4bf16_1k
      TRITON_MFMA_v2_2case(32, 32, bf16T, bf16T, mfma_f32_32x32x8bf16_1k, 8, 4,
                           mfma_f32_32x32x4bf16_1k, 4, 2),
      // mfma_f32_16x16x32_bf16 & mfma_f32_16x16x16_bf16_1K
      TRITON_MFMA_v4_2case(16, 16, bf16T, bf16T, mfma_f32_16x16x32_bf16, 32, 8,
                           mfma_f32_16x16x16bf16_1k, 16, 4),
      TRITON_MFMA_v(3, 16, 16, bf16T, bf16T, mfma_f32_16x16x16bf16_1k, 16, 4),
      // mfma_f32_16x16x16_bf16_1K & mfma_f32_16x16x8_bf16
      TRITON_MFMA_v2_2case(16, 16, bf16T, bf16T, mfma_f32_16x16x16bf16_1k, 16,
                           4, mfma_f32_16x16x8bf16, 8, 2),
      // mfma_f32_32x32x4_bf16
      TRITON_MFMA_v(1, 32, 32, bf16T, bf16T, mfma_f32_32x32x4bf16, 4, 2),
      // mfma_f32_16x16x8_bf16
      TRITON_MFMA_v(1, 16, 16, bf16T, bf16T, mfma_f32_16x16x8bf16, 8, 2),
      // mfma_f32_4x4x4_bf16_1K
      TRITON_MFMA_v2to4(4, 4, bf16T, bf16T, mfma_f32_4x4x4bf16_1k, 64, 4),
      TRITON_MFMA_v2to4(4, 64, bf16T, bf16T, mfma_f32_4x4x4bf16_1k, 4, 4),
      TRITON_MFMA_v2to4(64, 4, bf16T, bf16T, mfma_f32_4x4x4bf16_1k, 4, 4),
      // mfma_f32_4x4x2_bf16
      TRITON_MFMA_v(1, 4, 4, bf16T, bf16T, mfma_f32_4x4x2bf16, 32, 2),
      TRITON_MFMA_v(1, 4, 64, bf16T, bf16T, mfma_f32_4x4x2bf16, 2, 2),
      TRITON_MFMA_v(1, 64, 4, bf16T, bf16T, mfma_f32_4x4x2bf16, 2, 2),

      // fp8/bf8 inputs
      // mfma_f32_32x32x16_FP8_FP8
      TRITON_MFMA_v(4, 32, 32, ocpFp8T, ocpFp8T, mfma_f32_32x32x16_fp8_fp8, 16,
                    8),
      TRITON_MFMA_v(3, 32, 32, amdFp8T, amdFp8T, mfma_f32_32x32x16_fp8_fp8, 16,
                    8),
      // mfma_f32_32x32x16_FP8_BF8
      TRITON_MFMA_v(4, 32, 32, ocpFp8T, ocpBf8T, mfma_f32_32x32x16_fp8_bf8, 16,
                    8),
      TRITON_MFMA_v(3, 32, 32, amdFp8T, amdBf8T, mfma_f32_32x32x16_fp8_bf8, 16,
                    8),
      // mfma_f32_32x32x16_BF8_FP8
      TRITON_MFMA_v(4, 32, 32, ocpBf8T, ocpFp8T, mfma_f32_32x32x16_bf8_fp8, 16,
                    8),
      TRITON_MFMA_v(3, 32, 32, amdBf8T, amdFp8T, mfma_f32_32x32x16_bf8_fp8, 16,
                    8),
      // mfma_f32_32x32x16_BF8_BF8
      TRITON_MFMA_v(4, 32, 32, ocpBf8T, ocpBf8T, mfma_f32_32x32x16_bf8_bf8, 16,
                    8),
      TRITON_MFMA_v(3, 32, 32, amdBf8T, amdBf8T, mfma_f32_32x32x16_bf8_bf8, 16,
                    8),
      // mfma_f32_16x16x32_FP8_FP8
      TRITON_MFMA_v(4, 16, 16, ocpFp8T, ocpFp8T, mfma_f32_16x16x32_fp8_fp8, 32,
                    8),
      TRITON_MFMA_v(3, 16, 16, amdFp8T, amdFp8T, mfma_f32_16x16x32_fp8_fp8, 32,
                    8),
      // mfma_f32_16x16x32_FP8_BF8
      TRITON_MFMA_v(4, 16, 16, ocpFp8T, ocpBf8T, mfma_f32_16x16x32_fp8_bf8, 32,
                    8),
      TRITON_MFMA_v(3, 16, 16, amdFp8T, amdBf8T, mfma_f32_16x16x32_fp8_bf8, 32,
                    8),
      // mfma_f32_16x16x32_BF8_FP8
      TRITON_MFMA_v(4, 16, 16, ocpBf8T, ocpFp8T, mfma_f32_16x16x32_bf8_fp8, 32,
                    8),
      TRITON_MFMA_v(3, 16, 16, amdBf8T, amdFp8T, mfma_f32_16x16x32_bf8_fp8, 32,
                    8),
      // mfma_f32_16x16x32_BF8_BF8
      TRITON_MFMA_v(4, 16, 16, ocpBf8T, ocpBf8T, mfma_f32_16x16x32_bf8_bf8, 32,
                    8),
      TRITON_MFMA_v(3, 16, 16, amdBf8T, amdBf8T, mfma_f32_16x16x32_bf8_bf8, 32,
                    8),

      // int8 inputs
      // mfma_i32_32x32x32_i8 & mfma_i32_32x32x16i8
      TRITON_MFMA_v4_2case(32, 32, i8T, i8T, mfma_i32_32x32x32_i8, 32, 16,
                           mfma_i32_32x32x16_i8, 16, 8),
      TRITON_MFMA_v(3, 32, 32, i8T, i8T, mfma_i32_32x32x16_i8, 16, 8),
      // mfma_i32_32x32x8i8
      TRITON_MFMA_v1to2(32, 32, i8T, i8T, mfma_i32_32x32x8i8, 8, 4),
      // mfma_i32_16x16x64_i8 & mfma_i32_16x16x32i8
      TRITON_MFMA_v4_2case(16, 16, i8T, i8T, mfma_i32_16x16x64_i8, 64, 16,
                           mfma_i32_16x16x32_i8, 32, 8),
      TRITON_MFMA_v(3, 16, 16, i8T, i8T, mfma_i32_16x16x32_i8, 32, 8),
      // mfma_i32_16x16x16i8
      TRITON_MFMA_v1to2(16, 16, i8T, i8T, mfma_i32_16x16x16i8, 16, 4),
      // mfma_i32_4x4x4i8
      TRITON_MFMA_v1to4(4, 4, i8T, i8T, mfma_i32_4x4x4i8, 64, 4),
      TRITON_MFMA_v1to4(4, 64, i8T, i8T, mfma_i32_4x4x4i8, 4, 4),
      TRITON_MFMA_v1to4(64, 4, i8T, i8T, mfma_i32_4x4x4i8, 4, 4),

      // Scaled mfma f8f6f4
      // mfma_scale_F32_16x16x128_F8F6F4
      TRITON_MFMA_v(4, 16, 16, fp4T, fp4T, mfma_scale_f32_16x16x128_f8f6f4, 128,
                    32),
      // mfma_scale_F32_32x32x64_F8F6F4
      TRITON_MFMA_v(4, 32, 32, fp4T, fp4T, mfma_scale_f32_32x32x64_f8f6f4, 64,
                    32),
  };
}

} // namespace

//===----------------------------------------------------------------------===//
// MFMA intrinsic selection
//===----------------------------------------------------------------------===//

FailureOr<MfmaIntrinsic>
MfmaIntrinsic::selectFor(int version, unsigned mDim, unsigned nDim,
                         unsigned inputKDim, Type aElemType, Type bElemType,
                         bool withScale, bool useTF32) {
  const MfmaMap &mfmaMap = MfmaDatabase::get(aElemType.getContext());
  MfmaKey key = composeMfmaKeyFor(version, mDim, nDim, aElemType, bElemType,
                                  withScale, useTF32);

  auto it = mfmaMap.find(key);
  if (it == mfmaMap.end())
    return failure();

  const SmallVector<MfmaMapValue, 2> &values = it->second;

  // If We have more than one instrinsics, prefer those with a larger K.
  for (const auto [symbol, k, kBase] : llvm::drop_end(values)) {
    if (inputKDim >= k)
      return MfmaIntrinsic(symbol, mDim, nDim, k, kBase, aElemType, bElemType);
  }

  // We always have one choice--the only / smallest-K intrinsic.
  auto [symbol, k, kBase] = values.back();
  return MfmaIntrinsic(symbol, mDim, nDim, k, kBase, aElemType, bElemType);
}
} // namespace mlir
