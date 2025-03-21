/* Copyright 2017 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/elemental_ir_emitter.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "absl/algorithm/container.h"
#include "absl/base/macros.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/algorithm_util.h"
#include "xla/service/float8_fnuz_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_loop.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

using absl::StrCat;
using llvm_ir::IrArray;
using llvm_ir::IrName;
using llvm_ir::SetToFirstInsertPoint;
using xla::float8_fnuz_ir_emitter::EmitF8fnuzToFloating;
using xla::float8_fnuz_ir_emitter::EmitFloatingToF8fnuz;

absl::StatusOr<llvm::Value*> EmitReducePrecisionIR(
    PrimitiveType src_ty, llvm::Value* x, int64_t dest_exponent_bits,
    int64_t dest_mantissa_bits, bool quiet_nans, llvm::IRBuilderBase* b) {
  using llvm::APInt;

  if (!primitive_util::IsFloatingPointType(src_ty)) {
    return Unimplemented(
        "ReducePrecision cannot accept non-floating-point type %s.",
        PrimitiveType_Name(src_ty));
  }

  // Integer and float types for casting and constant generation.
  llvm::Type* float_type = x->getType();
  int64_t nbits = float_type->getPrimitiveSizeInBits();
  llvm::IntegerType* int_type = b->getIntNTy(nbits);

  // SignificandWidth includes the implicit extra bit.
  int src_mantissa_bits = primitive_util::SignificandWidth(src_ty) - 1;
  int src_exponent_bits = nbits - 1 - src_mantissa_bits;

  // Cast the input value to an integer for bitwise manipulation.
  llvm::Value* x_as_int = b->CreateBitCast(x, int_type);

  // Clear the sign bit, it does not participate in rounding and we will restore
  // it later.
  APInt sign_bit_mask(nbits, 1);
  sign_bit_mask <<= nbits - 1;
  llvm::Value* x_abs_bits =
      b->CreateAnd(x_as_int, llvm::ConstantInt::get(int_type, ~sign_bit_mask));

  APInt exp_bits_mask(nbits, 1);
  exp_bits_mask = ((exp_bits_mask << src_exponent_bits) - 1)
                  << src_mantissa_bits;
  auto x_is_nan = b->CreateICmpUGT(
      x_abs_bits, llvm::ConstantInt::get(int_type, exp_bits_mask));

  if (dest_mantissa_bits < src_mantissa_bits) {
    // Last remaining mantissa bit.
    APInt last_mantissa_bit_mask(nbits, 1);
    last_mantissa_bit_mask <<= src_mantissa_bits - dest_mantissa_bits;

    // Compute rounding bias for round-to-nearest with ties to even.  This is
    // equal to a base value of 0111... plus one bit if the last remaining
    // mantissa bit is 1.
    APInt base_rounding_bias = last_mantissa_bit_mask.lshr(1) - 1;
    llvm::Value* x_last_mantissa_bit = b->CreateLShr(
        b->CreateAnd(x_as_int,
                     llvm::ConstantInt::get(int_type, last_mantissa_bit_mask)),
        (src_mantissa_bits - dest_mantissa_bits));
    llvm::Value* x_rounding_bias =
        b->CreateAdd(x_last_mantissa_bit,
                     llvm::ConstantInt::get(int_type, base_rounding_bias));

    // Add rounding bias, and mask out truncated bits.  Note that the case
    // where adding the rounding bias overflows into the exponent bits is
    // correct; the non-masked mantissa bits will all be zero, and the
    // exponent will be incremented by one.
    APInt truncation_mask = ~(last_mantissa_bit_mask - 1);
    llvm::Value* x_rounded = b->CreateAdd(x_as_int, x_rounding_bias);
    x_rounded = b->CreateAnd(x_rounded,
                             llvm::ConstantInt::get(int_type, truncation_mask));
    if (quiet_nans) {
      x_as_int = b->CreateSelect(x_is_nan, x_as_int, x_rounded);
    } else {
      x_as_int = x_rounded;
    }
  }

  if (dest_exponent_bits < src_exponent_bits) {
    // An exponent of 2^(n-1)-1 -- that is, 0111... with the zero in the most-
    // significant bit -- is equal to 1.0f for all exponent sizes.  Adding
    // 2^(n-1)-1 to this gives us the highest non-infinite exponent for a bit-
    // size of n, and subtracting 2^(n-1)-1 from this gives us the lowest'
    // exponent (corresponding to 0.0f).
    //
    // Thus, the f32 exponent corresponding to the highest non-infinite
    // exponent for a bit size of n is (2^7-1) + 2^(n-1)-1, and the f32
    // exponent corresponding to the lowest exponent for a bit size of n is
    // (2^7-1) - 2^(n-1)-1.
    //
    // Note that we have already checked that exponents_bits >= 1.
    APInt exponent_bias(nbits, 1);
    exponent_bias = (exponent_bias << (src_exponent_bits - 1)) - 1;

    APInt reduced_exponent_bias(nbits, 1);
    reduced_exponent_bias =
        (reduced_exponent_bias << (dest_exponent_bits - 1)) - 1;

    APInt reduced_max_exponent = exponent_bias + reduced_exponent_bias;
    APInt reduced_min_exponent = exponent_bias - reduced_exponent_bias;

    // Do we overflow or underflow?
    llvm::Value* x_exponent =
        b->CreateAnd(x_as_int, llvm::ConstantInt::get(int_type, exp_bits_mask));
    llvm::Value* x_overflows = b->CreateICmpUGT(
        x_exponent, llvm::ConstantInt::get(
                        int_type, reduced_max_exponent << src_mantissa_bits));
    llvm::Value* x_underflows = b->CreateICmpULE(
        x_exponent, llvm::ConstantInt::get(
                        int_type, reduced_min_exponent << src_mantissa_bits));

    // Compute appropriately-signed values of zero and infinity.
    llvm::Value* x_signed_zero =
        b->CreateAnd(x_as_int, llvm::ConstantInt::get(int_type, sign_bit_mask));
    llvm::Value* x_signed_inf = b->CreateOr(
        x_signed_zero, llvm::ConstantInt::get(int_type, exp_bits_mask));

    // Force to zero or infinity if overflow or underflow.  (Note that this
    // truncates all denormal values to zero, rather than rounding them.)
    x_as_int = b->CreateSelect(x_overflows, x_signed_inf, x_as_int);
    x_as_int = b->CreateSelect(x_underflows, x_signed_zero, x_as_int);
  }

  // Cast the result back to a floating-point type.
  llvm::Value* result = b->CreateBitCast(x_as_int, float_type);

  // Correct result for NaN inputs.
  //
  // The exponent handling will "normalize" NaN values to infinities, which is
  // undesirable (except in the case with no mantissa bits, in which case it
  // is mandatory).  This logic also handles cases where mantissa-rounding
  // causes a NaN's mantissa to overflow into the exponent bits, which would
  // otherwise create an erroneous zero value.

  if (dest_mantissa_bits > 0) {
    if (quiet_nans) {
      APInt qnan_mask(nbits, 1);
      qnan_mask <<= src_mantissa_bits - 1;
      llvm::Value* x_with_qnan_bit_set =
          b->CreateOr(x_as_int, llvm::ConstantInt::get(int_type, qnan_mask));
      x_with_qnan_bit_set = b->CreateBitCast(x_with_qnan_bit_set, float_type);
      result = b->CreateSelect(x_is_nan, x_with_qnan_bit_set, result);
    } else {
      result = b->CreateSelect(x_is_nan, x, result);
    }
  } else {
    result = b->CreateSelect(x_is_nan,
                             llvm::ConstantFP::getInfinity(float_type), result);
  }

  return result;
}

namespace {

template <PrimitiveType fx_type, int f8_exponent_bits>
llvm::Value* handle_halfway_points_FxToF8(llvm::Value* fx_abs_bits,
                                          llvm::Value* f8_bits,
                                          llvm::IRBuilderBase* b) {
  using llvm::APFloat;
  using llvm::APInt;
  using llvm::Value;
  static_assert(fx_type == F16 || fx_type == F32 || fx_type == F64);
  static_assert(3 <= f8_exponent_bits && f8_exponent_bits <= 4);

  const llvm::fltSemantics* fx_semantics;
  llvm::IntegerType* ix_type;

  if constexpr (fx_type == F16) {
    fx_semantics = &llvm::APFloat::IEEEhalf();
    ix_type = b->getInt16Ty();
  } else if constexpr (fx_type == F32) {
    fx_semantics = &llvm::APFloat::IEEEsingle();
    ix_type = b->getInt32Ty();
  } else if constexpr (fx_type == F64) {
    fx_semantics = &llvm::APFloat::IEEEdouble();
    ix_type = b->getInt64Ty();
  }

  auto ix_const = [fx_semantics, ix_type](APFloat val) {
    bool losesInfo;
    val.convert(*fx_semantics, llvm::RoundingMode::NearestTiesToEven,
                &losesInfo);
    return llvm::ConstantInt::get(ix_type, val.bitcastToAPInt());
  };

  llvm::IntegerType* i8_type = b->getInt8Ty();
  auto i8_const = [i8_type](int val) {
    return llvm::ConstantInt::get(i8_type, val);
  };

  // F16 values that are halfway between denormal F8 values. This is used to
  // determine how to round to denormal F8 values.
  const APFloat halfway_points_e4[8] = {
      APFloat(0x1.0p-10),  // halfway between [0/8 * 2^-6, 1/8 * 2^-6]
      APFloat(0x1.8p-9),   // halfway between [1/8 * 2^-6, 2/8 * 2^-6]
      APFloat(0x1.4p-8),   // halfway between [2/8 * 2^-6, 3/8 * 2^-6]
      APFloat(0x1.Cp-8),   // halfway between [3/8 * 2^-6, 4/8 * 2^-6]
      APFloat(0x1.2p-7),   // halfway between [4/8 * 2^-6, 5/8 * 2^-6]
      APFloat(0x1.6p-7),   // halfway between [5/8 * 2^-6, 6/8 * 2^-6]
      APFloat(0x1.Ap-7),   // halfway between [6/8 * 2^-6, 7/8 * 2^-6]
      APFloat(0x1.Ep-7)    // halfway between [7/8 * 2^-6, 8/8 * 2^-6]
  };

  const APFloat halfway_points_e3[16] = {
      APFloat(0x1.0p-7),  // halfway between [0/16 * 2^-2, 1/16 * 2^-2]
      APFloat(0x1.8p-6),  // halfway between [1/16 * 2^-2, 2/16 * 2^-2]
      APFloat(0x1.4p-5),  // halfway between [2/16 * 2^-2, 3/16 * 2^-2]
      APFloat(0x1.Cp-5),  // halfway between [3/16 * 2^-2, 4/16 * 2^-2]
      APFloat(0x1.2p-4),  // halfway between [4/16 * 2^-2, 5/16 * 2^-2]
      APFloat(0x1.6p-4),  // halfway between [5/16 * 2^-2, 6/16 * 2^-2]
      APFloat(0x1.Ap-4),  // halfway between [6/16 * 2^-2, 7/16 * 2^-2]
      APFloat(0x1.Ep-4),  // halfway between [7/16 * 2^-2, 8/16 * 2^-2]
      APFloat(0x1.1p-3),  // halfway between [8/16 * 2^-2, 9/16 * 2^-2]
      APFloat(0x1.3p-3),  // halfway between [9/16 * 2^-2, 10/16 * 2^-2]
      APFloat(0x1.5p-3),  // halfway between [10/16 * 2^-2, 11/16 * 2^-2]
      APFloat(0x1.7p-3),  // halfway between [11/16 * 2^-2, 12/16 * 2^-2]
      APFloat(0x1.9p-3),  // halfway between [12/16 * 2^-2, 13/16 * 2^-2]
      APFloat(0x1.Bp-3),  // halfway between [13/16 * 2^-2, 14/16 * 2^-2]
      APFloat(0x1.Dp-3),  // halfway between [14/16 * 2^-2, 15/16 * 2^-2]
      APFloat(0x1.Fp-3),  // halfway between [15/16 * 2^-2, 16/16 * 2^-2]
  };

  const APFloat* halfway_points;
  int arr_sz;
  if constexpr (f8_exponent_bits == 4) {
    halfway_points = halfway_points_e4;
    arr_sz = 8;
  } else if constexpr (f8_exponent_bits == 3) {
    halfway_points = halfway_points_e3;
    arr_sz = 16;
  }

  // Handle case where output is denormal. If we're rounding to a denormal
  // value, ignore the current value of f8_bits and set it to the correct
  // denormal value. We emit the equivalent of the following:
  //
  //   if (f16_abs_bits <= halfway_points[0]) {
  //     f8_bits = 0;
  //   } else if (f16_abs_bits < halfway_points[1]) {
  //     f8_bits = 1;
  //   } else if (f16_abs_bits <= halfway_points[2]) {
  //   ...  // More if-else statements. The comparisons alternate between <=
  //   ...  // and < to handle round-to-even properly.
  //   } else if (f16_abs_bits < halfway_points[7])  {
  //     f8_bits = 7;
  //   }
  for (int i = arr_sz - 1; i >= 0; i--) {
    Value* comparison;
    llvm::Constant* half_way_point = ix_const(halfway_points[i]);

    if (i % 2 == 0) {
      comparison = b->CreateICmpULE(fx_abs_bits, half_way_point);
    } else {
      comparison = b->CreateICmpULT(fx_abs_bits, half_way_point);
    }

    f8_bits = b->CreateSelect(comparison, i8_const(i), f8_bits);
  }

  return f8_bits;
}

absl::StatusOr<llvm::Value*> EmitF16ToF8e5m2(llvm::Value* f16_value,
                                             llvm::IRBuilderBase* b) {
  TF_ASSIGN_OR_RETURN(
      llvm::Value * reduced_precision,
      EmitReducePrecisionIR(
          /*src_ty=*/F16, f16_value,
          /*dest_exponent_bits=*/primitive_util::ExponentWidth(F8E5M2),
          /*dest_mantissa_bits=*/primitive_util::SignificandWidth(F8E5M2) - 1,
          /*quiet_nans=*/true, b));
  llvm::Value* as_int16 = b->CreateBitCast(reduced_precision, b->getInt16Ty());
  llvm::Value* shifted = b->CreateLShr(as_int16, 8);
  llvm::Value* truncated = b->CreateTrunc(shifted, b->getInt8Ty());
  return b->CreateBitCast(truncated, b->getInt8Ty());
}

llvm::Value* EmitF8e5m2ToF16(llvm::Value* f8_value, llvm::IRBuilderBase* b) {
  llvm::Value* as_int8 = b->CreateBitCast(f8_value, b->getInt8Ty());
  llvm::Value* as_int16 = b->CreateZExt(as_int8, b->getInt16Ty());
  llvm::Value* shifted = b->CreateShl(as_int16, 8);
  return b->CreateBitCast(shifted, b->getHalfTy());
}

// Convert a float "fx_value" of type "fx_type" to an F8e "f8_exponent_bits"
// bits wide.
template <PrimitiveType fx_type, int f8_exponent_bits>
absl::StatusOr<llvm::Value*> EmitFxToF8e(llvm::Value* fx_value,
                                         llvm::IRBuilderBase* b) {
  static_assert(fx_type == F16 || fx_type == F32 || fx_type == F64);
  static_assert(3 <= f8_exponent_bits && f8_exponent_bits <= 4);

  constexpr int f8_mantissa_bits = 7 - f8_exponent_bits;
  constexpr int f8_bias = (1 << (f8_exponent_bits - 1)) - 1;

  const uint64_t fx_width = primitive_util::BitWidth(fx_type);
  const uint64_t fx_bias = primitive_util::ExponentBias(fx_type);
  const uint64_t fx_mantissa_bits =
      primitive_util::SignificandWidth(fx_type) - 1;

  const uint64_t exponent_bias_difference = fx_bias - f8_bias;

  using llvm::APInt;
  using llvm::Value;

  const llvm::fltSemantics* fx_semantics;
  llvm::IntegerType* ix_type;

  if constexpr (fx_type == F16) {
    ix_type = b->getInt16Ty();
    fx_semantics = &llvm::APFloat::IEEEhalf();
  } else if constexpr (fx_type == F32) {
    ix_type = b->getInt32Ty();
    fx_semantics = &llvm::APFloat::IEEEsingle();
  } else if constexpr (fx_type == F64) {
    ix_type = b->getInt64Ty();
    fx_semantics = &llvm::APFloat::IEEEdouble();
  }

  auto ix_const = [ix_type](uint64_t val) {
    return llvm::ConstantInt::get(ix_type, val);
  };

  llvm::IntegerType* i8_type = b->getInt8Ty();
  llvm::Constant* infinity = llvm::ConstantInt::get(
      ix_type, llvm::APFloat::getInf(*fx_semantics).bitcastToAPInt());
  llvm::ConstantInt* nosign_mask =
      ix_const(ix_type->getBitMask() ^ ix_type->getSignBit());
  llvm::ConstantInt* sign_mask = ix_const(ix_type->getSignBit());
  llvm::ConstantInt* sign_shift = ix_const(fx_width - 8);
  llvm::ConstantInt* fx_exponent_bias_difference =
      ix_const(exponent_bias_difference << fx_mantissa_bits);
  llvm::ConstantInt* fx_doubled_exponent_bias_difference =
      ix_const(exponent_bias_difference << (fx_mantissa_bits + 1));
  llvm::ConstantInt* mantissa_bits_difference =
      ix_const(fx_mantissa_bits - f8_mantissa_bits);
  llvm::ConstantInt* min_normal_value =
      ix_const((exponent_bias_difference + 1) << fx_mantissa_bits);

  // Cast the input value to an integer for bitwise manipulation. Get the
  // absolute value of the input value.
  //   fx_as_int = bitcast(fx_value, int)
  //   fx_abs_bits = fx_as_int & nosign_mask
  Value* fx_as_int = b->CreateBitCast(fx_value, ix_type);
  llvm::Value* fx_abs_bits = b->CreateAnd(fx_as_int, nosign_mask);

  // Get the sign.
  //   f8_sign = (fx_as_int & sign_mask) >> sign_shift
  Value* fx_sign = b->CreateAnd(fx_as_int, sign_mask);
  fx_sign = b->CreateLShr(fx_sign, sign_shift);
  Value* f8_sign = b->CreateTrunc(fx_sign, i8_type);

  // Truncate the mantissa to f8 mantissa bits and exponent to f8 exponent bits
  // Denormal values are not handled properly here and are
  // dealt with later in this function.
  absl::StatusOr<Value*> fx_reduced_statusor = EmitReducePrecisionIR(
      /*src_ty=*/fx_type, fx_value,
      /*dest_exponent_bits=*/f8_exponent_bits,
      /*dest_mantissa_bits=*/f8_mantissa_bits,
      /*quiet_nans=*/true, b);
  CHECK_OK(fx_reduced_statusor.status());  // Crash OK
  Value* fx_reduced = fx_reduced_statusor.value();
  fx_reduced = b->CreateBitCast(fx_reduced, ix_type);

  // Remove the sign bit.
  //   fx_reduced = fx_reduced & nosign_mask
  fx_reduced = b->CreateAnd(fx_reduced, nosign_mask);

  // Round values smaller than the smallest F8 normal value up to the smallest
  // F8 normal value. The case where we round to a denormal value is handled
  // later.
  //    fx_reduced = max(fx_reduced, min_normal_value)
  fx_reduced = b->CreateSelect(b->CreateICmpULT(fx_reduced, min_normal_value),
                               min_normal_value, fx_reduced);

  // Adjust the exponent by subtracting the difference in exponent bias:
  //   fx_reduced -= (exponent_bias_difference << fx_mantissa_bits)
  // For infinity/NaN values, subtract twice the difference in exponent bias
  // to ensure the leading exponent bit(s) of fx_reduced are set to zero.
  fx_reduced = b->CreateSub(
      fx_reduced, b->CreateSelect(b->CreateICmpULT(fx_reduced, infinity),
                                  fx_exponent_bias_difference,
                                  fx_doubled_exponent_bias_difference));

  // Shift to convert to F8.
  //   fx_reduced = fx_reduced >> mantissa_bits_difference;
  fx_reduced = b->CreateLShr(fx_reduced, mantissa_bits_difference);

  Value* f8_bits = b->CreateTrunc(fx_reduced, i8_type);

  // Handle Fx values that are halfway between denormal F8 values.
  f8_bits = handle_halfway_points_FxToF8<fx_type, f8_exponent_bits>(fx_abs_bits,
                                                                    f8_bits, b);

  // Set the sign bit.
  //   f8_bits |= f8_sign
  f8_bits = b->CreateOr(f8_bits, f8_sign);
  return f8_bits;
}

template <int f8_exponent_bits>
llvm::Value* EmitToF16F8e(llvm::Value* f8_value, llvm::IRBuilderBase* b) {
  using llvm::APInt;
  using llvm::Value;
  static_assert(3 <= f8_exponent_bits && f8_exponent_bits <= 4);

  llvm::IntegerType* i8_type = b->getInt8Ty();
  llvm::IntegerType* i16_type = b->getInt16Ty();
  auto i8_const = [i8_type](int val) {
    return llvm::ConstantInt::get(i8_type, val);
  };
  auto i16_const = [i16_type](int val) {
    return llvm::ConstantInt::get(i16_type, val);
  };

  // Map from F8 denormal value to F16 value.
  const int f8_denormal_to_f16_e4[8] = {
      0x0000,  // 0
      0x1800,  // 1/8 * 2^-6
      0x1C00,  // 2/8 * 2^-6
      0x1E00,  // 3/8 * 2^-6
      0x2000,  // 4/8 * 2^-6
      0x2100,  // 5/8 * 2^-6
      0x2200,  // 6/8 * 2^-6
      0x2300,  // 7/8 * 2^-6
  };

  // Map from F8 denormal value to F16 value.
  const int f8_denormal_to_f16_e3[16] = {
      0x0000,  //  0
      0x2400,  //  1/16 * 2^-2
      0x2800,  //  2/16 * 2^-2
      0x2A00,  //  3/16 * 2^-2
      0x2C00,  //  4/16 * 2^-2
      0x2D00,  //  5/16 * 2^-2
      0x2E00,  //  6/16 * 2^-2
      0x2F00,  //  7/16 * 2^-2
      0x3000,  //  8/16 * 2^-2
      0x3080,  //  9/16 * 2^-2
      0x3100,  // 10/16 * 2^-2
      0x3180,  // 11/16 * 2^-2
      0x3200,  // 12/16 * 2^-2
      0x3280,  // 13/16 * 2^-2
      0x3300,  // 14/16 * 2^-2
      0x3380,  // 15/16 * 2^-2
  };

  // Cast the input value to an integer for bitwise manipulation. Get the
  // absolute value of the input value.
  //   f8_as_int = bitcast(f16_value, int)
  //   f8_abs_bits = f8_as_int & 0x7F
  Value* f8_as_int = b->CreateBitCast(f8_value, i8_type);
  Value* f8_abs_bits = b->CreateAnd(f8_as_int, i8_const(0x7F));

  // We assume below that the value is neither NaN nor denormal. If it NaN or
  // denormal, the output is set to NaN or zero at the end using Select
  // instructions.

  // Get the sign:
  //   f16_sign = (f8_as_int & 0x80) << 8
  Value* f8_sign = b->CreateAnd(f8_as_int, i8_const(0x80));
  Value* f16_sign = b->CreateZExt(f8_sign, i16_type);
  f16_sign = b->CreateShl(f16_sign, i16_const(8));

  int exponent_mask;
  const int* f8_denormal_to_f16;
  int f8_denormal_size;
  if constexpr (f8_exponent_bits == 4) {
    exponent_mask = 0x78;
    f8_denormal_to_f16 = f8_denormal_to_f16_e4;
    f8_denormal_size = 8;
  } else if constexpr (f8_exponent_bits == 3) {
    exponent_mask = 0x70;
    f8_denormal_to_f16 = f8_denormal_to_f16_e3;
    f8_denormal_size = 16;
  }
  constexpr int f8_bias = (1 << (f8_exponent_bits - 1)) - 1;
  constexpr int exponent_bias_difference = 15 - f8_bias;
  constexpr int f16_mantissa_bits = 10;  // e5m10
  constexpr int f8_mantissa_bits = 7 - f8_exponent_bits;
  constexpr int mantissa_bits_difference = f16_mantissa_bits - f8_mantissa_bits;
  constexpr int f8_mantissa_mask = (1 << f8_mantissa_bits) - 1;

  // Get the exponent:
  //   f8_exponent = (f8_as_int & exponent_mask) >> f8_mantissa_bits
  Value* f8_exponent_bits_v = b->CreateAnd(f8_as_int, i8_const(exponent_mask));
  Value* f8_exponent =
      b->CreateLShr(f8_exponent_bits_v, i8_const(f8_mantissa_bits));

  // Adjust the exponent by adding the difference in exponent bias:
  //   f16_exponent = (f8_exponent + exponent_bias_difference)
  //                  << f16_mantissa_bits
  Value* f16_exponent =
      b->CreateAdd(f8_exponent, i8_const(exponent_bias_difference));
  f16_exponent = b->CreateZExt(f16_exponent, i16_type);
  f16_exponent = b->CreateShl(f16_exponent, i16_const(f16_mantissa_bits));

  // Set output exponent to 11111 if input exponent is 111 (Inf or NaN)
  // 0.11111.0000000000 is 0x7C00
  Value* is_exp_1111 =
      b->CreateICmpEQ(f8_exponent_bits_v, i8_const(exponent_mask));
  f16_exponent = b->CreateSelect(is_exp_1111, i16_const(0x7C00), f16_exponent);

  // Get the mantissa:
  //   f16_mantissa = (f8_mantissa & f8_mantissa_mask)
  //                  << mantissa_bits_difference
  Value* f8_mantissa = b->CreateAnd(f8_as_int, i8_const(f8_mantissa_mask));
  Value* f16_mantissa = b->CreateZExt(f8_mantissa, i16_type);
  f16_mantissa =
      b->CreateShl(f16_mantissa, i16_const(mantissa_bits_difference));

  // Combine the exponent and mantissa:
  //   f16_as_int = f16_exponent | f16_mantissa
  Value* f16_as_int = b->CreateOr(f16_exponent, f16_mantissa);

  // If the F8 value is denormal, use the map above to determine the correct F16
  // value.
  //    if (f8_abs_bits < 8) { f16_as_int = f8_denormal_to_f16[f8_abs_bits]; }
  for (int i = 0; i < f8_denormal_size; i++) {
    Value* is_denormal_value = b->CreateICmpEQ(f8_abs_bits, i8_const(i));
    f16_as_int = b->CreateSelect(is_denormal_value,
                                 i16_const(f8_denormal_to_f16[i]), f16_as_int);
  }

  // Set the sign bit.
  //   f16_as_int |= f16_sign
  f16_as_int = b->CreateOr(f16_as_int, f16_sign);
  return b->CreateBitCast(f16_as_int, b->getHalfTy());
}

llvm::Value* EmitF16ToF8e4m3fn(llvm::Value* f16_value, llvm::IRBuilderBase* b) {
  using llvm::APInt;
  using llvm::Value;

  llvm::IntegerType* i8_type = b->getInt8Ty();
  llvm::IntegerType* i16_type = b->getInt16Ty();
  auto i8_const = [i8_type](int val) {
    return llvm::ConstantInt::get(i8_type, val);
  };
  auto i16_const = [i16_type](int val) {
    return llvm::ConstantInt::get(i16_type, val);
  };

  // Cast the input value to an integer for bitwise manipulation. Get the
  // absolute value of the input value.
  //   f16_as_int = bitcast(f16_value, int)
  //   f16_abs_bits = f16_as_int & 0x7FFF
  Value* f16_as_int = b->CreateBitCast(f16_value, i16_type);
  llvm::Value* f16_abs_bits = b->CreateAnd(f16_as_int, i16_const(0x7FFF));

  // Get the sign.
  //   f8_sign = (f16_as_int & 0x8000) >> 8
  Value* f16_sign = b->CreateAnd(f16_as_int, i16_const(0x8000));
  f16_sign = b->CreateLShr(f16_sign, i16_const(8));
  Value* f8_sign = b->CreateTrunc(f16_sign, i8_type);

  // Truncate the mantissa to 3 bits. ReducePrecision cannot deal with
  // f8E4M3FN's NaN representations, so don't use ReducePrecision to handle
  // exponent reduction. Denormal values are not handled properly here and are
  // dealt with later in this function.
  absl::StatusOr<Value*> f16_reduced_statusor = EmitReducePrecisionIR(
      /*src_ty=*/F16, f16_value,
      /*dest_exponent_bits=*/5,
      /*dest_mantissa_bits=*/3,
      /*quiet_nans=*/false, b);
  CHECK_OK(f16_reduced_statusor.status());  // Crash OK
  Value* f16_reduced = f16_reduced_statusor.value();
  f16_reduced = b->CreateBitCast(f16_reduced, i16_type);

  // Remove the sign bit.
  //   f16_reduced = f16_reduced & 0x7FFF
  f16_reduced = b->CreateAnd(f16_reduced, i16_const(0x7FFF));

  // Bits of the F16 representation of the smallest F8 normal value.
  constexpr int min_normal_value = 0x2400;

  // Round values smaller than the smallest F8 normal value up to the smallest
  // F8 normal value. The case where we round to a denormal value is handled
  // later.
  //    f16_reduced = max(f16_reduced, min_normal_value)
  f16_reduced = b->CreateSelect(
      b->CreateICmpULT(f16_reduced, i16_const(min_normal_value)),
      i16_const(min_normal_value), f16_reduced);

  constexpr int exponent_bias_difference = 15 - 7;
  constexpr int f8_exponent_bits = 4;
  constexpr int f16_mantissa_bits = 10;
  constexpr int f8_mantissa_bits = 3;
  constexpr int mantissa_bits_difference = f16_mantissa_bits - f8_mantissa_bits;

  // Adjust the exponent by subtracting the difference in exponent bias.
  //   f16_reduced -= (exponent_bias_difference << f16_mantissa_bits)
  f16_reduced = b->CreateSub(
      f16_reduced, i16_const(exponent_bias_difference << f16_mantissa_bits));

  // Shift to convert to F8.
  //   f8_bits = f16_reduced >> mantissa_bits_difference;
  Value* f8_bits =
      b->CreateLShr(f16_reduced, i16_const(mantissa_bits_difference));
  f8_bits = b->CreateTrunc(f8_bits, i8_type);

  // Bits of the highest F16 value that gets converted to a finite F8 value.
  // In binary: 0 10111 1101111111
  constexpr int max_finite_value = 0x5F7F;

  // If we're above the maximum F8 value, output NaN.
  //   f8_bits = f16_abs_bits > max_finite_value ? 0x7F : f8_bits
  f8_bits = b->CreateSelect(
      b->CreateICmpUGT(f16_abs_bits, i16_const(max_finite_value)),
      i8_const(0x7F), f8_bits);

  // Handle F16 values that are halfway between denormal F8 values.
  f8_bits = handle_halfway_points_FxToF8<F16, f8_exponent_bits>(f16_abs_bits,
                                                                f8_bits, b);

  // Set the sign bit.
  //   f8_bits |= f8_sign
  f8_bits = b->CreateOr(f8_bits, f8_sign);
  return f8_bits;
}

llvm::Value* EmitF8e4m3fnToF16(llvm::Value* f8_value, llvm::IRBuilderBase* b) {
  using llvm::APInt;
  using llvm::Value;

  llvm::IntegerType* i8_type = b->getInt8Ty();
  llvm::IntegerType* i16_type = b->getInt16Ty();
  auto i8_const = [i8_type](int val) {
    return llvm::ConstantInt::get(i8_type, val);
  };
  auto i16_const = [i16_type](int val) {
    return llvm::ConstantInt::get(i16_type, val);
  };

  // Cast the input value to an integer for bitwise manipulation. Get the
  // absolute value of the input value.
  //   f8_as_int = bitcast(f16_value, int)
  //   f8_abs_bits = f8_as_int & 0x7F
  Value* f8_as_int = b->CreateBitCast(f8_value, i8_type);
  Value* f8_abs_bits = b->CreateAnd(f8_as_int, i8_const(0x7F));

  // We assume below that the value is neither NaN nor denormal. If it NaN or
  // denormal, the output is set to NaN or zero at the end using Select
  // instructions.

  // Get the sign:
  //   f16_sign = (f8_as_int & 0x80) << 8
  Value* f8_sign = b->CreateAnd(f8_as_int, i8_const(0x80));
  Value* f16_sign = b->CreateZExt(f8_sign, i16_type);
  f16_sign = b->CreateShl(f16_sign, i16_const(8));

  constexpr int exponent_bias_difference = 15 - 7;
  constexpr int f16_mantissa_bits = 10;
  constexpr int f8_mantissa_bits = 3;
  constexpr int mantissa_bits_difference = f16_mantissa_bits - f8_mantissa_bits;
  constexpr int f8_mantissa_mask = (1 << f8_mantissa_bits) - 1;

  // Get the exponent:
  //   f8_exponent = (f8_as_int & 0x78) >> f8_mantissa_bits
  Value* f8_exponent_bits = b->CreateAnd(f8_as_int, i8_const(0x78));
  Value* f8_exponent =
      b->CreateLShr(f8_exponent_bits, i8_const(f8_mantissa_bits));

  // Adjust the exponent by adding the difference in exponent bias:
  //   f16_exponent = (f8_exponent + exponent_bias_difference)
  //                  << f16_mantissa_bits
  Value* f16_exponent =
      b->CreateAdd(f8_exponent, i8_const(exponent_bias_difference));
  f16_exponent = b->CreateZExt(f16_exponent, i16_type);
  f16_exponent = b->CreateShl(f16_exponent, i16_const(f16_mantissa_bits));

  // Get the mantissa:
  //   f16_mantissa = (f8_mantissa & f8_mantissa_mask)
  //                  << mantissa_bits_difference
  Value* f8_mantissa = b->CreateAnd(f8_as_int, i8_const(f8_mantissa_mask));
  Value* f16_mantissa = b->CreateZExt(f8_mantissa, i16_type);
  f16_mantissa =
      b->CreateShl(f16_mantissa, i16_const(mantissa_bits_difference));

  // Combine the exponent and mantissa:
  //   f16_as_int = f16_exponent | f16_mantissa
  Value* f16_as_int = b->CreateOr(f16_exponent, f16_mantissa);

  // Set output to NaN if input is NaN
  //   f16_as_int = f8_abs_bits == 0x7F ? 0x7E00 : f16_as_int
  Value* is_nan = b->CreateICmpEQ(f8_abs_bits, i8_const(0x7F));
  f16_as_int = b->CreateSelect(is_nan, i16_const(0x7E00), f16_as_int);

  // Map from F8 denormal value to F16 value.
  int f8_denormal_to_f16[8] = {
      0x0000,  // 0
      0x1800,  // 1/8 * 2^-6
      0x1C00,  // 2/8 * 2^-6
      0x1E00,  // 3/8 * 2^-6
      0x2000,  // 4/8 * 2^-6
      0x2100,  // 5/8 * 2^-6
      0x2200,  // 6/8 * 2^-6
      0x2300,  // 7/8 * 2^-6
  };

  // If the F8 value is denormal, use the map above to determine the correct F16
  // value.
  //    if (f8_abs_bits < 8) { f16_as_int = f8_denormal_to_f16[f8_abs_bits]; }
  for (int i = 0; i < ABSL_ARRAYSIZE(f8_denormal_to_f16); i++) {
    Value* is_denormal_value = b->CreateICmpEQ(f8_abs_bits, i8_const(i));
    f16_as_int = b->CreateSelect(is_denormal_value,
                                 i16_const(f8_denormal_to_f16[i]), f16_as_int);
  }

  // Set the sign bit.
  //   f16_as_int |= f16_sign
  f16_as_int = b->CreateOr(f16_as_int, f16_sign);
  return b->CreateBitCast(f16_as_int, b->getHalfTy());
}

llvm::Value* EmitF16ToF8e4m3b11fnuz(llvm::Value* f16_value,
                                    llvm::IRBuilderBase* b) {
  using llvm::APInt;
  using llvm::Value;

  llvm::IntegerType* i8_type = b->getInt8Ty();
  auto i8_const = [i8_type](int val) {
    return llvm::ConstantInt::get(i8_type, val);
  };
  auto type = f16_value->getType();
  auto f16_abs_value = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs,
                                                    {f16_value}, {type}, b);
  auto f16_zero_or_underflow = llvm::ConstantFP::get(type, 0x1.004p-14);
  auto is_zero = b->CreateFCmpOLT(f16_abs_value, f16_zero_or_underflow);
  auto f8_overflow_threshold = llvm::ConstantFP::get(type, 0x1.fp+4);
  auto no_overflow = b->CreateFCmpOLT(f16_abs_value, f8_overflow_threshold);

  // Re-scale the f16, then convert as-if it were e4m3fn.
  f16_value = b->CreateFMul(
      f16_value, llvm::ConstantFP::get(f16_value->getType(), 1 << (11 - 7)));
  auto* f8_value = EmitF16ToF8e4m3fn(f16_value, b);

  // e4m3b11 overflows to NaN.
  f8_value = b->CreateSelect(no_overflow, f8_value, i8_const(0x80));
  // e4m3b11 has no negative zero.
  f8_value = b->CreateSelect(is_zero, i8_const(0x00), f8_value);
  return f8_value;
}

llvm::Value* EmitF8e4m3b11fnuzToF16(llvm::Value* f8_value,
                                    llvm::IRBuilderBase* b) {
  using llvm::APInt;
  using llvm::Value;

  llvm::IntegerType* i8_type = b->getInt8Ty();
  llvm::IntegerType* i16_type = b->getInt16Ty();
  auto i8_const = [i8_type](int val) {
    return llvm::ConstantInt::get(i8_type, val);
  };

  Value* f8_as_int = b->CreateBitCast(f8_value, i8_type);
  Value* is_nan = b->CreateICmpEQ(f8_as_int, i8_const(0x80));

  Value* f8_abs_bits = b->CreateAnd(f8_as_int, i8_const(0x7F));
  Value* is_max = b->CreateICmpEQ(f8_abs_bits, i8_const(0x7F));
  Value* f8_sign_bit = b->CreateAnd(f8_as_int, i8_const(0x80));
  Value* f16_sign_bit =
      b->CreateShl(b->CreateZExt(f8_sign_bit, i16_type), 16 - 8);

  auto* f16_value = EmitF8e4m3fnToF16(f8_value, b);
  f16_value = b->CreateFMul(
      f16_value,
      llvm::ConstantFP::get(f16_value->getType(), 1.0 / (1 << (11 - 7))));
  f16_value = b->CreateSelect(
      is_nan,
      llvm::ConstantFP::get(f16_value->getType(),
                            std::numeric_limits<double>::quiet_NaN()),
      f16_value);

  llvm::Value* max_like_sign = llvm::ConstantFP::get(
      f16_value->getType(),
      static_cast<float>(std::numeric_limits<tsl::float8_e4m3b11fnuz>::max()));
  max_like_sign = b->CreateBitCast(max_like_sign, f16_sign_bit->getType());
  max_like_sign = b->CreateOr(max_like_sign, f16_sign_bit);
  max_like_sign = b->CreateBitCast(max_like_sign, f16_value->getType());
  f16_value = b->CreateSelect(is_max, max_like_sign, f16_value);

  return f16_value;
}

absl::StatusOr<llvm::Value*> EmitF16ToF4e2m1fn(llvm::Value* f16_value,
                                               llvm::IRBuilderBase* b) {
  auto i8_const = [&](int val) {
    return llvm::ConstantInt::get(b->getInt8Ty(), val);
  };
  auto i16_const = [&](int val) {
    return llvm::ConstantInt::get(b->getInt16Ty(), val);
  };
  constexpr int mantissa_diff = 9;  // 10 for F16, 1 for F4
  constexpr int bias_diff = 14;     // 15 for F16, 1 for F4

  // Cast the input value to an integer for bitwise manipulation.
  // Get the absolute value of the input (discard the sign).
  //   f16_bits = bitcast(f16_value, int)
  //   f16_abs_bits = f16_bits & 0x7FFF
  llvm::Value* f16_bits = b->CreateBitCast(f16_value, b->getInt16Ty());
  llvm::Value* f16_abs_bits = b->CreateAnd(f16_bits, i16_const(0x7FFF));

  // If the input absolute value is >= 7.0 or an infinity, the result saturates
  // to max value (6.0). If (0.75 <= input < 1), the result is rounded to 1.0.
  // If (0 <= input <= 0.25), the result is rounded to 0.0.
  // If the input is NaN, the result is undefined (implemented as minus zero).
  // The rest of the cases are handled by the "happy path".
  //   is_overflow = f16_abs_bits >= 0x1.Cp2
  //   is_one = f16_abs_bits >= 0x1.8p-1  (used only if exponent underflows)
  //   is_zero = f16_abs_bits <= 0x1p-2   (used only if exponent underflows)
  //   is_nan = f16_abs_bits > 0x7C00  (F16 NaN threshold)
  llvm::Value* is_overflow =
      b->CreateICmpUGE(f16_abs_bits, i16_const(0x4700));  // 7.0
  llvm::Value* is_one =
      b->CreateICmpUGE(f16_abs_bits, i16_const(0x3A00));  // 0.75
  llvm::Value* is_zero =
      b->CreateICmpULE(f16_abs_bits, i16_const(0x3400));  // 0.25
  llvm::Value* is_nan =
      b->CreateICmpUGT(f16_abs_bits, i16_const(0x7C00));  // inf

  // Truncate the mantissa to 1 bit and the exponent to 3 bits (not 2 bits, as
  // the type doesn't have Inf/NaN and can represent unbiased exponent 2).
  // This case, as well as the denormal, is handled below.
  TF_ASSIGN_OR_RETURN(
      llvm::Value * reduced_precision,
      EmitReducePrecisionIR(
          /*src_ty=*/F16, f16_value,
          /*dest_exponent_bits=*/primitive_util::ExponentWidth(F4E2M1FN) + 1,
          /*dest_mantissa_bits=*/primitive_util::SignificandWidth(F4E2M1FN) - 1,
          /*quiet_nans=*/false, b));

  // Cast the reduced precision value to an integer for bitwise manipulation.
  // Discard the least significant (9) mantissa bits leaving 1 bit.
  // Truncate to
  //   as_int16 = bitcast(reduced_precision, int)
  //   as_int8 = as_int16 >> (f16_mantissa - f4_mantissa)
  llvm::Value* as_int16 = b->CreateBitCast(reduced_precision, b->getInt16Ty());
  llvm::Value* as_int8 =
      b->CreateTrunc(b->CreateLShr(as_int16, mantissa_diff), b->getInt8Ty());

  // Get the sign (0 or 1).
  //   f4_sign = as_int8 >> 6
  llvm::Value* f4_sign = b->CreateLShr(as_int8, 6);

  // Get exponent and mantissa bits without the sign.
  // Important: the mask is 0x3F (not 0x7F), discard bit #6.
  //   f4_bits = as_int8 & 0x3F
  llvm::Value* f4_bits = b->CreateAnd(as_int8, i8_const(0x3F));

  // Convert F16 exponent to F4 exponent by readjusting the exponent bias.
  // This produces the "normal" result, i.e. not Inf or NaN or denormal.
  //   f4_normal = f4_bits - ((f16_bias - f4_bias) << f4_mantissa)
  constexpr int f4_exponent_offset = bias_diff << 1;
  llvm::Value* f4_normal = b->CreateSub(f4_bits, i8_const(f4_exponent_offset));

  // If the rounding resulted in zero exponent, the value is incorrect.
  // This happens when the input is < 1.0
  //   is_underflow = f4_normal <= 1
  llvm::Value* is_underflow = b->CreateICmpSLE(f4_normal, i8_const(1));

  // Chain of selects that handles the special cases.
  //   f4_result =
  //     is_underflow ? (is_one ? 1.0 : (is_zero ? 0.0 : 0.5)) :
  //     is_overflow ? (is_nan ? -0.0 : 6.0) :
  //     f4_normal
  llvm::Value* f4_result = b->CreateSelect(
      is_underflow,
      // If underflow, the input is < 1.0; the result is either 0.0, 0.5 or 1.0
      b->CreateSelect(is_one, i8_const(0x2),
                      b->CreateSelect(is_zero, i8_const(0x0), i8_const(0x1))),
      // If overflow, the input is >= 7.0 or infinity or NaN.
      b->CreateSelect(is_overflow,
                      b->CreateSelect(is_nan, i8_const(0x8), i8_const(0x7)),
                      f4_normal));

  // Add sign to the resulting value.
  //   f4_signed_result = (f4_sign << 3) | f4_result
  return b->CreateOr(f4_result, b->CreateShl(f4_sign, 3));
}

llvm::Value* EmitF4e2m1fnToF16(llvm::Value* f8_value, llvm::IRBuilderBase* b) {
  auto i16_const = [&](int val) {
    return llvm::ConstantInt::get(b->getInt16Ty(), val);
  };
  constexpr int mantissa_diff = 9;  // 10 for F16, 1 for F4
  constexpr int bias_diff = 14;     // 15 for F16, 1 for F4

  // The input value is a 8-bit integer, extend it to 16-bit integer.
  //   as_int16 = bitcast(f8_value, int)
  llvm::Value* as_int16 = b->CreateZExt(f8_value, b->getInt16Ty());

  // Get the sign and shift it to F16 position.
  //   f4_sign = as_int16 >> 3
  //   f16_sign_bit = f4_sign << 15
  llvm::Value* f4_sign = b->CreateLShr(as_int16, 3);
  llvm::Value* f16_sign_bit = b->CreateShl(f4_sign, 15);

  // Get exponent and mantissa bits without the sign.
  //   f4_bits = as_int16 & 0x7
  //   f16_bits = f4_bits << (f16_mantissa - f4_mantissa)
  llvm::Value* f4_bits = b->CreateAnd(as_int16, i16_const(0x7));
  llvm::Value* f16_bits = b->CreateShl(f4_bits, mantissa_diff);

  // Convert F16 exponent to F4 exponent by readjusting the exponent bias.
  //   f4_normal = f4_bits - ((f16_bias - f4_bias) << f4_mantissa)
  constexpr int f16_exponent_offset = bias_diff << 10;
  llvm::Value* f16_normal =
      b->CreateAdd(f16_bits, i16_const(f16_exponent_offset));

  // For denormal and zero, the exponent is different. Handle these cases
  // separately below.
  //   is_denorm_or_zero = f4_bits <= 1
  //   is_zero = f4_bits == 0
  llvm::Value* is_denorm_or_zero = b->CreateICmpULE(f4_bits, i16_const(1));
  llvm::Value* is_zero = b->CreateICmpEQ(f4_bits, i16_const(0));

  // Chain of selects that handles the special cases.
  //   f16_result = is_denorm_or_zero ? (is_zero ? 0.0 : 0.5) : f16_normal
  llvm::Value* f16_result = b->CreateSelect(
      is_denorm_or_zero,
      b->CreateSelect(is_zero, i16_const(0x0000), i16_const(0x3800)),
      f16_normal);

  // Add sign to the resulting value.
  //   f16_signed_result = f16_sign_bit | f16_result
  llvm::Value* f16_signed_result = b->CreateOr(f16_result, f16_sign_bit);
  return b->CreateBitCast(f16_signed_result, b->getHalfTy());
}

llvm::Value* EmitF32ToF8e8m0fnu(llvm::Value* f32_value,
                                llvm::IRBuilderBase* b) {
  auto i32_const = [&](int val) {
    return llvm::ConstantInt::get(b->getInt32Ty(), val);
  };

  // Cast the input value to an integer for bitwise manipulation.
  //   as_int32 = bitcast(f32_value, int)
  llvm::Value* as_int32 = b->CreateBitCast(f32_value, b->getInt32Ty());

  // Check if the input is zero, negative, overflow, infinity or NaN.
  // All of these cases cannot be represented in the E8M0 format.
  //   is_zero_or_negative = as_int32 <= 0
  //   is_overflow_or_nan = as_int32 >= 0x1.8p127
  //   is_nan = is_zero_or_negative | is_overflow_or_nan
  llvm::Value* is_zero_or_negative = b->CreateICmpSLE(as_int32, i32_const(0));
  llvm::Value* is_overflow_or_nan =
      b->CreateICmpSGE(as_int32, i32_const(0x7F400000));  // 1.5 * 2^127
  llvm::Value* is_nan = b->CreateOr(is_zero_or_negative, is_overflow_or_nan);

  // Check if the input is a denormal which should round to the minimum value
  // (2^-127), as there is no zero value.
  //   is_denorm = as_int32 <= 0x1p-127
  llvm::Value* is_denorm =
      b->CreateICmpULE(as_int32, i32_const(0x400000));  // 1.0 * 2^-127

  // Round the value (always up) and discard the mantissa.
  //   rounded = as_int32 + 0x1p-127
  //   f8_normal = as_int32 >> f32_mantissa
  llvm::Value* rounded =
      b->CreateAdd(as_int32, i32_const(0x400000));  // 1.0 * 2^-127
  llvm::Value* f8_normal = b->CreateAShr(rounded, 23);

  // Chain of selects that handles the special cases.
  //   f8_result = is_nan ? 0xFF : (is_denorm ? 0x00 : f8_normal)
  llvm::Value* f8_result =
      b->CreateSelect(is_nan, i32_const(0xFF),
                      b->CreateSelect(is_denorm, i32_const(0x00), f8_normal));

  // Truncate to the result type.
  return b->CreateTrunc(f8_result, b->getInt8Ty());
}

llvm::Value* EmitF8e8m0fnuToF32(llvm::Value* f8_value, llvm::IRBuilderBase* b) {
  auto i32_const = [&](int val) {
    return llvm::ConstantInt::get(b->getInt32Ty(), val);
  };

  // The input value is a 8-bit integer, extend it to 32-bit integer.
  //   as_int32 = bitcast(f8_value, int)
  llvm::Value* as_int32 = b->CreateZExt(f8_value, b->getInt32Ty());

  // Check if the input is a denormal or NaN.
  //   is_zero = as_int32 == 0x00
  //   is_nan = as_int32 == 0xFF
  llvm::Value* is_zero = b->CreateICmpEQ(as_int32, i32_const(0));
  llvm::Value* is_nan = b->CreateICmpEQ(as_int32, i32_const(0xFF));

  // Shift exponent to the left for the normal case.
  //   f32_normal = as_int32 << mantissa_diff
  llvm::Value* f32_normal = b->CreateShl(as_int32, 23);

  // Chain of selects that handles the special cases.
  //   f32_result = is_nan ? 0x7FC00000 : (is_zero ? 0x1p-127 : f32_normal)
  llvm::Value* f32_result = b->CreateSelect(
      is_nan, i32_const(0x7FC00000),
      b->CreateSelect(is_zero, i32_const(0x400000), f32_normal));

  // Bitcast integer bits to the result type.
  return b->CreateBitCast(f32_result, b->getFloatTy());
}

llvm::Value* EmitIntegralToFloating(llvm::Value* integer_value,
                                    PrimitiveType from_type,
                                    PrimitiveType to_type, llvm::Module* module,
                                    llvm::IRBuilderBase* b) {
  if (primitive_util::IsSignedIntegralType(from_type)) {
    return b->CreateSIToFP(integer_value, llvm_ir::PrimitiveTypeToIrType(
                                              to_type, module->getContext()));
  } else {
    CHECK(primitive_util::IsUnsignedIntegralType(from_type) ||
          from_type == PRED);
    return b->CreateUIToFP(integer_value, llvm_ir::PrimitiveTypeToIrType(
                                              to_type, module->getContext()));
  }
}

}  // namespace

/*static*/ bool ElementalIrEmitter::OpInvalidatesCache(
    const HloInstruction* hlo) {
  switch (hlo->opcode()) {
    // This list of ops was created by inspecting the code. There is no
    // guarantee that it is complete.
    case HloOpcode::kConcatenate:
    case HloOpcode::kDot:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kPad:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
      return true;
    default:
      return false;
  }
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitUnaryOp(
    const HloInstruction* op, llvm::Value* operand_value) {
  if (ShapeUtil::ElementIsIntegral(op->operand(0)->shape()) ||
      op->operand(0)->shape().element_type() == PRED) {
    return EmitIntegerUnaryOp(op, operand_value);
  } else if (ShapeUtil::ElementIsComplex(op->operand(0)->shape())) {
    return EmitComplexUnaryOp(op, operand_value);
  } else {
    return EmitFloatUnaryOp(op, operand_value);
  }
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitIntegerUnaryOp(
    const HloInstruction* op, llvm::Value* operand_value) {
  switch (op->opcode()) {
    case HloOpcode::kConvert: {
      PrimitiveType from_type = op->operand(0)->shape().element_type();
      PrimitiveType to_type = op->shape().element_type();
      CHECK(primitive_util::IsIntegralType(from_type) || from_type == PRED)
          << from_type;
      if (from_type == to_type) {
        return operand_value;
      }
      if (to_type == PRED) {
        return b_->CreateZExt(
            ICmpNE(operand_value,
                   llvm::ConstantInt::get(operand_value->getType(), 0)),
            llvm_ir::PrimitiveTypeToIrType(PRED, module_->getContext()));
      }
      if (primitive_util::IsIntegralType(to_type)) {
        return IntCast(
            operand_value,
            llvm_ir::PrimitiveTypeToIrType(to_type, module_->getContext()),
            primitive_util::IsSignedIntegralType(from_type));
      }
      if (primitive_util::IsFloatingPointType(to_type)) {
        if (to_type == F8E5M2) {
          return EmitF16ToF8e5m2(
              EmitIntegralToFloating(operand_value, from_type, F16, module_,
                                     b_),
              b_);
        }
        if (to_type == F8E4M3) {
          return EmitFxToF8e<F16, 4>(
              EmitIntegralToFloating(operand_value, from_type, F16, module_,
                                     b_),
              b_);
        }
        if (to_type == F8E4M3FN) {
          return EmitF16ToF8e4m3fn(
              EmitIntegralToFloating(operand_value, from_type, F16, module_,
                                     b_),
              b_);
        }
        if (to_type == F8E4M3B11FNUZ) {
          return EmitF16ToF8e4m3b11fnuz(
              EmitIntegralToFloating(operand_value, from_type, F16, module_,
                                     b_),
              b_);
        }
        if (to_type == F4E2M1FN) {
          return EmitF16ToF4e2m1fn(
              EmitIntegralToFloating(operand_value, from_type, F16, module_,
                                     b_),
              b_);
        }
        if (to_type == F8E8M0FNU) {
          return EmitF32ToF8e8m0fnu(
              EmitIntegralToFloating(operand_value, from_type, F32, module_,
                                     b_),
              b_);
        }
        if (to_type == F8E5M2FNUZ || to_type == F8E4M3FNUZ) {
          return EmitFloatingToF8fnuz(
              F16,
              EmitIntegralToFloating(operand_value, from_type, F16, module_,
                                     b_),
              to_type, b_);
        }
        if (to_type == F8E3M4) {
          return EmitFxToF8e<F16, 3>(
              EmitIntegralToFloating(operand_value, from_type, F16, module_,
                                     b_),
              b_);
        }
        return EmitIntegralToFloating(operand_value, from_type, to_type,
                                      module_, b_);
      }
      if (primitive_util::IsComplexType(to_type)) {
        auto to_ir_component_type = llvm_ir::PrimitiveTypeToIrType(
            primitive_util::ComplexComponentType(to_type),
            module_->getContext());
        if (primitive_util::IsSignedIntegralType(from_type)) {
          return EmitComposeComplex(
              op, SIToFP(operand_value, to_ir_component_type), nullptr);
        }
        if (primitive_util::IsUnsignedIntegralType(from_type) ||
            from_type == PRED) {
          return EmitComposeComplex(
              op, UIToFP(operand_value, to_ir_component_type), nullptr);
        }
      }
      return Unimplemented("conversion from primitive type %s to %s",
                           PrimitiveType_Name(from_type),
                           PrimitiveType_Name(to_type));
    }
    case HloOpcode::kBitcastConvert: {
      PrimitiveType from_type = op->operand(0)->shape().element_type();
      PrimitiveType to_type = op->shape().element_type();
      CHECK(primitive_util::IsIntegralType(from_type));
      if (from_type == to_type) {
        return operand_value;
      }
      if (primitive_util::BitWidth(from_type) ==
          primitive_util::BitWidth(to_type)) {
        return BitCast(operand_value, llvm_ir::PrimitiveTypeToIrType(
                                          to_type, module_->getContext()));
      }
      return InvalidArgument(
          "bitcast conversion from primitive type %s to %s with unequal "
          "bit-widths (%u versus %u) ",
          PrimitiveType_Name(from_type), PrimitiveType_Name(to_type),
          primitive_util::BitWidth(from_type),
          primitive_util::BitWidth(to_type));
    }
    case HloOpcode::kAbs: {
      bool is_signed =
          primitive_util::IsSignedIntegralType(op->shape().element_type());
      if (is_signed) {
        auto type = llvm_ir::PrimitiveTypeToIrType(op->shape().element_type(),
                                                   module_->getContext());
        auto cmp = ICmpSGE(operand_value, GetZero(type));
        return Select(cmp, operand_value, Neg(operand_value));
      } else {
        return operand_value;
      }
    }
    case HloOpcode::kClz: {
      auto is_zero_undef = b_->getFalse();
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::ctlz,
                                          {operand_value, is_zero_undef},
                                          {operand_value->getType()}, b_);
    }
    case HloOpcode::kSign: {
      CHECK(primitive_util::IsSignedIntegralType(op->shape().element_type()))
          << op->shape().element_type();
      auto type = llvm_ir::PrimitiveTypeToIrType(op->shape().element_type(),
                                                 module_->getContext());
      auto cmp = ICmpEQ(operand_value, GetZero(type));
      auto ashr = AShr(operand_value, type->getIntegerBitWidth() - 1);
      return Select(cmp, GetZero(type), Or(ashr, 1));
    }
    case HloOpcode::kNegate:
      return Neg(operand_value);
    case HloOpcode::kNot: {
      auto type = op->shape().element_type();
      if (type == PRED) {
        // It is not sufficient to just call CreateNot() here because a PRED
        // is represented as an i8 and the truth value is stored only in the
        // bottom bit.
        return b_->CreateZExt(
            Not(Trunc(operand_value, b_->getInt1Ty())),
            llvm_ir::PrimitiveTypeToIrType(PRED, module_->getContext()));
      } else if (primitive_util::IsIntegralType(type)) {
        return Not(operand_value);
      }
      return Unimplemented("unary op Not is not defined for type '%d'", type);
    }
    case HloOpcode::kPopulationCount: {
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::ctpop,
                                          {operand_value},
                                          {operand_value->getType()}, b_);
    }
    default:
      return Unimplemented("unary integer op '%s'",
                           HloOpcodeString(op->opcode()));
  }
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitFloatUnaryOp(
    const HloInstruction* op, llvm::Value* operand_value) {
  switch (op->opcode()) {
    case HloOpcode::kConvert: {
      PrimitiveType from_type = op->operand(0)->shape().element_type();
      PrimitiveType to_type = op->shape().element_type();
      CHECK(primitive_util::IsFloatingPointType(from_type)) << from_type;

      // LLVM optimizes away `fpcast` and `fpext` operations and optimized
      // LLVM IR has arithmetic operations on `bfloat16` that are not natively
      // supported on any of the CPUs, and LLVM inserts very expensive calls to
      // fp conversion functions around bf16 operations. To avoid this, we use
      // bitcasts and shifts to convert bf16 to f32 and back using truncation
      // with rounding, and suppress LLVM optimizations that hurt performance.
      // This is enabled explicitly by a flag only for XLA:CPU backend.
      if (options_.xla_cpu_use_truncate_f32_to_bf16_conversion) {
        if (from_type == F32 && to_type == BF16) {
          // This implementation is based on Eigen `float_to_bfloat16_rtne` with
          // a special case for nans.
          auto* i32 = b_->CreateBitCast(operand_value, b_->getInt32Ty());

          // Rounding bias for non-nan values.
          auto* lsb =
              b_->CreateAnd(b_->CreateLShr(i32, 16),
                            llvm::ConstantInt::get(b_->getInt32Ty(), 1));
          auto* rounding_bias = b_->CreateAdd(
              llvm::ConstantInt::get(b_->getInt32Ty(), 0x7fff), lsb);

          // For NaNs, we set all of them to quiet NaNs by masking the mantissa
          // so that only the MSB is 1, then simply truncate the original value
          // to retain the sign.
          auto* is_nan =
              b_->createIsFPClass(operand_value, llvm::FPClassTest::fcNan);
          auto* nan_mask = llvm::ConstantInt::get(b_->getInt32Ty(), 0xFFC00000);
          auto* msb = llvm::ConstantInt::get(b_->getInt32Ty(), 0x00400000);
          auto* quiet_nan = b_->CreateOr(b_->CreateAnd(i32, nan_mask), msb);
          auto* i16 = b_->CreateTrunc(
              b_->CreateLShr(
                  b_->CreateSelect(is_nan, quiet_nan,
                                   b_->CreateAdd(i32, rounding_bias)),
                  16),
              b_->getInt16Ty());

          return b_->CreateBitCast(i16, b_->getBFloatTy());
        }
        if (from_type == BF16 && to_type == F32) {
          auto* i16 = b_->CreateBitCast(operand_value, b_->getInt16Ty());
          auto* i32 = b_->CreateZExt(i16, b_->getInt32Ty());
          auto* i32s = b_->CreateShl(i32, 16);
          auto* f32 = b_->CreateBitCast(i32s, b_->getFloatTy());
          return f32;
        }
      }

      if (from_type == to_type) {
        return operand_value;
      }
      if (from_type == BF16) {
        TF_RET_CHECK(to_type != BF16);
        // The code below expects the source type to be F32.
        operand_value = b_->CreateFPExt(operand_value, b_->getFloatTy());
        from_type = F32;
        if (from_type == to_type) {
          return operand_value;
        }
      }
      if (from_type == F8E5M2) {
        TF_RET_CHECK(to_type != F8E5M2);
        operand_value = EmitF8e5m2ToF16(operand_value, b_);
        from_type = F16;
        if (from_type == to_type) {
          return operand_value;
        }
      }
      if (from_type == F8E4M3) {
        TF_RET_CHECK(to_type != F8E4M3);
        operand_value = EmitToF16F8e<4>(operand_value, b_);
        from_type = F16;
        if (from_type == to_type) {
          return operand_value;
        }
      }
      if (from_type == F8E4M3FN) {
        TF_RET_CHECK(to_type != F8E4M3FN);
        operand_value = EmitF8e4m3fnToF16(operand_value, b_);
        from_type = F16;
        if (from_type == to_type) {
          return operand_value;
        }
      }
      if (from_type == F8E4M3B11FNUZ) {
        TF_RET_CHECK(to_type != F8E4M3B11FNUZ);
        operand_value = EmitF8e4m3b11fnuzToF16(operand_value, b_);
        from_type = F16;
        if (from_type == to_type) {
          return operand_value;
        }
      }
      if (from_type == F4E2M1FN) {
        TF_RET_CHECK(to_type != F4E2M1FN);
        operand_value = EmitF4e2m1fnToF16(operand_value, b_);
        from_type = F16;
        if (from_type == to_type) {
          return operand_value;
        }
      }
      if (from_type == F8E8M0FNU) {
        TF_RET_CHECK(to_type != F8E8M0FNU);
        operand_value = EmitF8e8m0fnuToF32(operand_value, b_);
        from_type = F32;
        if (from_type == to_type) {
          return operand_value;
        }
      }
      if (from_type == F8E5M2FNUZ || from_type == F8E4M3FNUZ) {
        TF_RET_CHECK(to_type != from_type);
        PrimitiveType cast_type =
            primitive_util::IsFloatingPointType(to_type) ? to_type : F16;
        if (to_type == F8E8M0FNU || to_type == F4E2M1FN) {
          cast_type = F32;
        }
        TF_ASSIGN_OR_RETURN(operand_value,
                            EmitF8fnuzToFloating(from_type, operand_value,
                                                 cast_type, b_, module_));
        from_type = cast_type;
        if (from_type == to_type) {
          return operand_value;
        }
      }
      if (from_type == F8E3M4) {
        TF_RET_CHECK(to_type != F8E3M4);
        operand_value = EmitToF16F8e<3>(operand_value, b_);
        from_type = F16;
        if (from_type == to_type) {
          return operand_value;
        }
      }
      if (primitive_util::IsComplexType(to_type)) {
        PrimitiveType to_component_type =
            primitive_util::ComplexComponentType(to_type);
        if (from_type == to_component_type) {
          return EmitComposeComplex(op, operand_value, nullptr);
        }
        return EmitComposeComplex(
            op,
            FPCast(operand_value,
                   llvm_ir::PrimitiveTypeToIrType(to_component_type,
                                                  module_->getContext())),
            nullptr);
      }
      if (to_type == BF16) {
        // F16 to BF16 has to go through an intermediate F32.
        if (from_type == F16) {
          operand_value = b_->CreateFPExt(operand_value, b_->getFloatTy());
        }
        return FPCast(operand_value, b_->getBFloatTy());
      }
      if (to_type == F8E5M2) {
        // Cast to F16 first. Casts to F8E5M2 must be from F16.
        if (from_type != F16) {
          operand_value = b_->CreateFPCast(
              operand_value,
              llvm_ir::PrimitiveTypeToIrType(F16, module_->getContext()));
        }
        return EmitF16ToF8e5m2(operand_value, b_);
      }
      if (to_type == F8E4M3) {
        switch (from_type) {
          case F16:
            return EmitFxToF8e<F16, 4>(operand_value, b_);
          case F32:
            return EmitFxToF8e<F32, 4>(operand_value, b_);
          case F64:
            return EmitFxToF8e<F64, 4>(operand_value, b_);
          case BF16:
            operand_value = b_->CreateFPCast(
                operand_value,
                llvm_ir::PrimitiveTypeToIrType(F16, module_->getContext()));
            return EmitFxToF8e<F16, 4>(operand_value, b_);
          default:
            return InvalidArgument("Unsupported conversion from %s to %s",
                                   PrimitiveType_Name(from_type),
                                   PrimitiveType_Name(to_type));
        }
      }
      if (to_type == F8E4M3FN) {
        // Cast to F16 first. Casts to F8E4M3FN must be from F16.
        if (from_type != F16) {
          operand_value = b_->CreateFPCast(
              operand_value,
              llvm_ir::PrimitiveTypeToIrType(F16, module_->getContext()));
        }
        return EmitF16ToF8e4m3fn(operand_value, b_);
      }
      if (to_type == F8E4M3B11FNUZ) {
        // Cast to F16 first. Casts to F8E4M3B11FNUZ must be from F16.
        if (from_type != F16) {
          operand_value = b_->CreateFPCast(
              operand_value,
              llvm_ir::PrimitiveTypeToIrType(F16, module_->getContext()));
        }
        return EmitF16ToF8e4m3b11fnuz(operand_value, b_);
      }
      if (to_type == F4E2M1FN) {
        // Cast to F16 first. Casts to F4E2M1FN must be from F16.
        if (from_type != F16) {
          operand_value = b_->CreateFPCast(
              operand_value,
              llvm_ir::PrimitiveTypeToIrType(F16, module_->getContext()));
        }
        return EmitF16ToF4e2m1fn(operand_value, b_);
      }
      if (to_type == F8E8M0FNU) {
        // Cast to F32 first. Casts to F8E8M0FNU must be from F32.
        if (from_type != F32) {
          operand_value = b_->CreateFPCast(
              operand_value,
              llvm_ir::PrimitiveTypeToIrType(F32, module_->getContext()));
        }
        return EmitF32ToF8e8m0fnu(operand_value, b_);
      }
      if (to_type == F8E5M2FNUZ || to_type == F8E4M3FNUZ) {
        return EmitFloatingToF8fnuz(from_type, operand_value, to_type, b_);
      }
      if (to_type == F8E3M4) {
        switch (from_type) {
          case F16:
            return EmitFxToF8e<F16, 3>(operand_value, b_);
          case F32:
            return EmitFxToF8e<F32, 3>(operand_value, b_);
          case F64:
            return EmitFxToF8e<F64, 3>(operand_value, b_);
          case BF16:
            operand_value = b_->CreateFPCast(
                operand_value,
                llvm_ir::PrimitiveTypeToIrType(F16, module_->getContext()));
            return EmitFxToF8e<F16, 3>(operand_value, b_);
          default:
            return InvalidArgument("Unsupported conversion from %s to %s",
                                   PrimitiveType_Name(from_type),
                                   PrimitiveType_Name(to_type));
        }
      }
      if (to_type == PRED) {
        return b_->CreateZExt(
            FCmpUNE(operand_value,
                    llvm::ConstantFP::get(operand_value->getType(), 0.0)),
            llvm_ir::PrimitiveTypeToIrType(PRED, module_->getContext()));
      }
      auto* to_ir_type =
          llvm_ir::PrimitiveTypeToIrType(to_type, module_->getContext());
      if (primitive_util::IsFloatingPointType(to_type)) {
        return FPCast(operand_value, to_ir_type);
      }
      auto* from_ir_type =
          llvm_ir::PrimitiveTypeToIrType(from_type, module_->getContext());
      int to_width = primitive_util::BitWidth(to_type);
      if (primitive_util::IsSignedIntegralType(to_type)) {
        int64_t min_int = llvm::minIntN(to_width);
        int64_t max_int = llvm::maxIntN(to_width);
        auto zero_int = llvm::ConstantInt::get(to_ir_type, 0);
        auto min_value_int = llvm::ConstantInt::get(to_ir_type, min_int);
        auto max_value_int = llvm::ConstantInt::get(to_ir_type, max_int);
        auto min_value_float = llvm::ConstantFP::get(from_ir_type, min_int);
        auto max_value_float = llvm::ConstantFP::get(from_ir_type, max_int);
        auto clamped = FPToSI(
            operand_value,
            llvm_ir::PrimitiveTypeToIrType(to_type, module_->getContext()));
        // x <= static_cast<float>(INT_MIN) ? INT_MIN : ...
        clamped = Select(FCmpOLE(operand_value, min_value_float), min_value_int,
                         clamped);
        // x >= static_cast<float>(INT_MAX) ? INT_MAX : ...
        clamped = Select(FCmpOGE(operand_value, max_value_float), max_value_int,
                         clamped);
        // isnan(x) ? 0 : ...
        clamped =
            Select(FCmpUNO(operand_value, operand_value), zero_int, clamped);
        return clamped;
      }
      if (primitive_util::IsUnsignedIntegralType(to_type)) {
        uint64_t min_int = 0;
        uint64_t max_int = llvm::maxUIntN(to_width);
        auto min_value_int = llvm::ConstantInt::get(to_ir_type, min_int);
        auto max_value_int = llvm::ConstantInt::get(to_ir_type, max_int);
        auto min_value_float = llvm::ConstantFP::get(from_ir_type, min_int);
        auto max_value_float = llvm::ConstantFP::get(from_ir_type, max_int);
        auto clamped = FPToUI(
            operand_value,
            llvm_ir::PrimitiveTypeToIrType(to_type, module_->getContext()));
        // (x <= 0.0 || isnan(x)) ? 0 : ...
        clamped = Select(FCmpULE(operand_value, min_value_float), min_value_int,
                         clamped);
        // x >= static_cast<float>(UINT_MAX) ? UINT_MAX : ...
        clamped = Select(FCmpOGE(operand_value, max_value_float), max_value_int,
                         clamped);
        return clamped;
      }
      return Unimplemented("unhandled conversion operation: %s => %s",
                           PrimitiveType_Name(from_type),
                           PrimitiveType_Name(to_type));
    }
    case HloOpcode::kBitcastConvert: {
      PrimitiveType from_type = op->operand(0)->shape().element_type();
      PrimitiveType to_type = op->shape().element_type();
      CHECK(primitive_util::IsFloatingPointType(from_type));
      if (from_type == to_type) {
        return operand_value;
      }
      if (primitive_util::BitWidth(from_type) ==
          primitive_util::BitWidth(to_type)) {
        return BitCast(operand_value, llvm_ir::PrimitiveTypeToIrType(
                                          to_type, module_->getContext()));
      }
      return InvalidArgument(
          "bitcast conversion from primitive type %s to %s with unequal "
          "bit-widths (%u versus %u) ",
          PrimitiveType_Name(from_type), PrimitiveType_Name(to_type),
          primitive_util::BitWidth(from_type),
          primitive_util::BitWidth(to_type));
    }
    case HloOpcode::kErf:
      return EmitErf(op->shape().element_type(), operand_value);
    case HloOpcode::kExp:
      return EmitExp(op->shape().element_type(), operand_value, "");
    case HloOpcode::kExpm1:
      return EmitExpm1(op->shape().element_type(), operand_value);
    case HloOpcode::kLog:
      return EmitLog(op->shape().element_type(), operand_value);
    case HloOpcode::kLog1p:
      return EmitLog1p(op->shape().element_type(), operand_value);
    case HloOpcode::kCos:
      return EmitCos(op->shape().element_type(), operand_value);
    case HloOpcode::kSin:
      return EmitSin(op->shape().element_type(), operand_value);
    case HloOpcode::kTan:
      return EmitTan(op->shape().element_type(), operand_value);
    case HloOpcode::kTanh:
      return EmitTanh(op->shape().element_type(), operand_value);
    case HloOpcode::kSqrt:
      return EmitSqrt(op->shape().element_type(), operand_value);
    case HloOpcode::kRsqrt:
      return EmitRsqrt(op->shape().element_type(), operand_value);
    case HloOpcode::kCbrt:
      return EmitCbrt(op->shape().element_type(), operand_value);
    case HloOpcode::kFloor:
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::floor,
                                          {operand_value},
                                          {operand_value->getType()}, b_);
    case HloOpcode::kCeil:
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::ceil,
                                          {operand_value},
                                          {operand_value->getType()}, b_);
    case HloOpcode::kAbs:
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs,
                                          {operand_value},
                                          {operand_value->getType()}, b_);
    case HloOpcode::kRoundNearestAfz:
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::round,
                                          {operand_value},
                                          {operand_value->getType()}, b_);
    // TODO(b/238238423): llvm::Intrinsic::nearbyint is equivalent to roundeven
    // as TF and JAX default to FE_TONEAREST. Call llvm::Intrinsic::roundeven
    // instead once GPU emitter supports lowering LLVM.
    case HloOpcode::kRoundNearestEven:
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::nearbyint,
                                          {operand_value},
                                          {operand_value->getType()}, b_);
    case HloOpcode::kSign: {
      auto type = operand_value->getType();
      auto zero = llvm::ConstantFP::get(type, 0.0);
      auto ne0_i1 = FCmpONE(operand_value, zero);
      auto ne0_float = UIToFP(ne0_i1, type);
      llvm::Value* result = llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::copysign, {ne0_float, operand_value},
          {operand_value->getType()}, b_);
      auto is_nan = FCmpUNO(operand_value, operand_value);
      result = Select(is_nan, operand_value, result);
      return result;
    }
    case HloOpcode::kIsFinite: {
      // abs(x) o!= inf, this works because the comparison returns false if
      // either operand is NaN.
      auto type = operand_value->getType();
      auto abs_value = llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::fabs, {operand_value}, {type}, b_);
      auto infinity = llvm::ConstantFP::getInfinity(type);
      auto not_infinite = FCmpONE(abs_value, infinity);
      return b_->CreateZExt(not_infinite, llvm_ir::PrimitiveTypeToIrType(
                                              PRED, module_->getContext()));
    }
    case HloOpcode::kNegate:
      return FNeg(operand_value);
    case HloOpcode::kReal:
      return operand_value;
    case HloOpcode::kImag:
      return llvm::ConstantFP::get(operand_value->getType(), 0.0);
    default:
      return Unimplemented("unary floating-point op '%s'",
                           HloOpcodeString(op->opcode()));
  }
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitComplexUnaryOp(
    const HloInstruction* op, llvm::Value* operand_value) {
  PrimitiveType input_type = op->operand(0)->shape().element_type();
  PrimitiveType component_type =
      primitive_util::IsComplexType(input_type)
          ? primitive_util::ComplexComponentType(input_type)
          : input_type;
  switch (op->opcode()) {
    case HloOpcode::kLog: {
      return EmitComplexLog(op, operand_value);
    }
    case HloOpcode::kLog1p: {
      //  log1p(a+bi) = .5*log((a+1)^2+b^2) + i*atan2(b, a + 1)
      //  log((a+1)+bi) = .5*log(a*a + 2*a + 1 + b*b) + i*atan2(b, a+1)
      //   log((a+1)+bi) = .5*log1p(a*a + 2*a + b*b) + i*atan2(b, a+1)
      //
      // that is accurate only when |a| is relatively small while
      // large |a| and |b| lead to multiplication overflow in the real
      // part.
      //
      // The following expression for the real part:
      //
      // log1p(a+bi).real = log(hypot(a+1, b))
      //                  = log(max(|a+1|, |b|) * sqrt(1 + (min(|a+1|, |b|) /
      //                  max(|a+1|, b))^2)) [to fix overflow for maximal values
      //                  of |a+1| and |b|] = log(max(|a+1|, |b|)) + log(sqrt(1
      //                  + (min(|a+1|, |b|) / max(|a+1|, b))^2)) =
      //                  log(max(|a+1|, |b|)) + 0.5*log1p((min(|a+1|, |b|) /
      //                  max(|a+1|, b))^2) [to fix inaccuracies for small a,
      //                  we'll use log1p] = log1p((1 + a > |b| ? a : max(|a+1|,
      //                  |b|) - 1) + 0.5*log1p((min(|a+1|, |b|) / max(|a+1|,
      //                  b))^2)
      //
      // is accurate on the whole complex plane except when |b| is
      // small and a is very close to -|b|^2/2 that leads to
      // substraction errors when adding the two log1p values as in
      //   log1p(-|b|^2) + log1p(|b|^2)
      // TODO: improve the accuracy for the case above.

      auto a = EmitExtractReal(operand_value);
      auto b = EmitExtractImag(operand_value);
      llvm::Type* llvm_ty = a->getType();
      auto one = llvm::ConstantFP::get(llvm_ty, 1.0);
      auto half = llvm::ConstantFP::get(llvm_ty, 0.5);

      auto a1 = FAdd(a, one);
      auto abs_a1 = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs, {a1},
                                                 {llvm_ty}, b_);
      auto abs_b = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs, {b},
                                                {llvm_ty}, b_);

      auto max_abs_of_a1_and_b = EmitFloatMax(abs_a1, abs_b, "");
      auto min_abs_of_a1_and_b = EmitFloatMin(abs_a1, abs_b, "");

      auto max_abs_of_a1_and_b_minus_one =
          Select(FCmpOGT(a1, abs_b), a, FSub(max_abs_of_a1_and_b, one));
      auto min_max_ratio = FDiv(min_abs_of_a1_and_b, max_abs_of_a1_and_b);

      TF_ASSIGN_OR_RETURN(
          auto log_of_max_abs_of_a1_and_b,
          EmitLog1p(component_type, max_abs_of_a1_and_b_minus_one));
      TF_ASSIGN_OR_RETURN(
          auto log_of_sqrt_part,
          EmitLog1p(component_type, FMul(min_max_ratio, min_max_ratio)));

      auto r = FAdd(FMul(half, log_of_sqrt_part), log_of_max_abs_of_a1_and_b);
      auto real_part = Select(FCmpUNO(r, r), min_abs_of_a1_and_b,
                              r);  // handles nan and inf values correctly

      TF_ASSIGN_OR_RETURN(auto imag_part, EmitAtan2(component_type, b, a1, ""));
      return EmitComposeComplex(op, real_part, imag_part);
    }
    case HloOpcode::kConvert: {
      PrimitiveType from_type = op->operand(0)->shape().element_type();
      TF_RET_CHECK(primitive_util::IsComplexType(from_type));
      PrimitiveType to_type = op->shape().element_type();
      TF_RET_CHECK(primitive_util::IsComplexType(to_type));
      if (from_type == to_type) {
        return operand_value;
      }
      PrimitiveType to_component_type =
          primitive_util::ComplexComponentType(to_type);
      auto to_ir_component_type = llvm_ir::PrimitiveTypeToIrType(
          to_component_type, module_->getContext());
      return EmitComposeComplex(
          op, FPCast(EmitExtractReal(operand_value), to_ir_component_type),
          FPCast(EmitExtractImag(operand_value), to_ir_component_type));
    }
    case HloOpcode::kExp: {
      // e^(a+bi) = e^a*(cos(b)+sin(b)i)
      TF_ASSIGN_OR_RETURN(
          auto exp_a,
          EmitExp(component_type, EmitExtractReal(operand_value), ""));
      TF_ASSIGN_OR_RETURN(
          auto cos_b, EmitCos(component_type, EmitExtractImag(operand_value)));
      TF_ASSIGN_OR_RETURN(
          auto sin_b, EmitSin(component_type, EmitExtractImag(operand_value)));
      return EmitComposeComplex(op, FMul(exp_a, cos_b), FMul(exp_a, sin_b));
    }
    case HloOpcode::kExpm1: {
      // e^(a+bi)-1 = (e^a*cos(b)-1)+e^a*sin(b)i
      //            [handle inaccuracies when a and/or b are small]
      //            = ((e^a - 1) * cos(b) + cos(b) - 1) + e^a*sin(b)i
      //            = (expm1(a) * cos(b) + cosm1(b)) + e^a*sin(b)i
      auto a = EmitExtractReal(operand_value);
      auto b = EmitExtractImag(operand_value);
      auto zero = llvm::ConstantFP::get(b->getType(), 0.0);
      auto one = llvm::ConstantFP::get(b->getType(), 1.0);
      auto b_is_zero = FCmpOEQ(b, zero);
      TF_ASSIGN_OR_RETURN(auto expm1_a, EmitExpm1(component_type, a));
      auto exp_a = FAdd(expm1_a, one);
      TF_ASSIGN_OR_RETURN(auto sin_b, EmitSin(component_type, b));
      TF_ASSIGN_OR_RETURN(auto cos_b_minus_one, EmitCosm1(component_type, b));
      auto cos_b = FAdd(cos_b_minus_one, one);
      auto real_result = FAdd(FMul(expm1_a, cos_b), cos_b_minus_one);
      auto imag_result = Select(b_is_zero, zero, FMul(exp_a, sin_b));
      return EmitComposeComplex(op, real_result, imag_result);
    }
    case HloOpcode::kCos:
    case HloOpcode::kSin: {
      // If the argument is z = x + i*y, let
      //   sinh(y) = (exp(y) - exp(-y)) / 2
      //   cosh(y) = (exp(y) + exp(-y)) / 2 ,
      // then
      //   sin(x + i*y) = sin(x)*cosh(y) + i*cos(x)*sinh(y)
      //   cos(x + i*y) = cos(x)*cosh(y) - i*sin(x)*sinh(y)
      auto x = EmitExtractReal(operand_value);
      auto y = EmitExtractImag(operand_value);
      auto type = y->getType();
      TF_ASSIGN_OR_RETURN(auto exp_y, EmitExp(component_type, y, ""));
      auto half_exp_y = FMul(llvm::ConstantFP::get(type, 0.5), exp_y);
      auto half_exp_neg_y = FDiv(llvm::ConstantFP::get(type, 0.5), exp_y);
      TF_ASSIGN_OR_RETURN(auto sin_x, EmitSin(component_type, x));
      TF_ASSIGN_OR_RETURN(auto cos_x, EmitCos(component_type, x));
      auto sinh_y = FSub(half_exp_y, half_exp_neg_y);
      auto cosh_y = FAdd(half_exp_y, half_exp_neg_y);
      llvm::Value* real_result = nullptr;
      llvm::Value* imag_result = nullptr;
      if (op->opcode() == HloOpcode::kSin) {
        real_result = FMul(sin_x, cosh_y);
        imag_result = FMul(cos_x, sinh_y);
      }
      if (op->opcode() == HloOpcode::kCos) {
        real_result = FMul(cos_x, cosh_y);
        imag_result = FNeg(FMul(sin_x, sinh_y));
      }
      return EmitComposeComplex(op, real_result, imag_result);
    }
    case HloOpcode::kTan:
      // tan(x+yi) = -i*tanh(-y + xi)
    case HloOpcode::kTanh: {
      /*
      tanh=(exp(x)-exp(-x)) / (exp(x)+exp(-x))
      e^(a+bi) = e^a*(cos(b)+sin(b)i)
      so tanh=(((cos(b)+sin(b)i)e^a - (cos(-b)+sin(-b)i)e^-a)) /
              (((cos(b)+sin(b)i)e^a + (cos(-b)+sin(-b)i)e^-a))
      cos(b)=cos(-b), sin(-b)=-sin(b)
      so tanh=(((cos(b)+sin(b)i)e^a - (cos(b)-sin(b)i)e^-a)) /
              (((cos(b)+sin(b)i)e^a + (cos(b)-sin(b)i)e^-a))
             =(cos(b)e^a+i*sin(b)e^a + cos(b)(-e^-a)+i*sin(b)e^-a) /
              (cos(b)e^a+i*sin(b)e^a + cos(b)e^-a+i*sin(b)(-e^-a))
             =(cos(b)(e^a-e^-a) + i*sin(b)(e^a+e^-a)) /
              (cos(b)(e^a+e^-a) + i*sin(b)(e^a-e^-a))
      This is a complex division, so we can multiply by denom_conj/denom_conj
             =(cos(b)(e^a-e^-a) + i*sin(b)(e^a+e^-a)) *
              (cos(b)(e^a+e^-a) - i*sin(b)(e^a-e^-a)) /
              ((cos(b)(e^a+e^-a))^2 + (sin(b)(e^a-e^-a))^2)
             =(cos(b)^2(e^(2a)-e^(-2a)) + sin(b)^2(e^(2a)-e^(-2a)) +
               i*(cos(b)sin(b)(e^a+e^-a)^2 - cos(b)sin(b)(e^a-e^-a)^2)) /
              ((cos(b)(e^a+e^-a))^2 + (sin(b)(e^a-e^-a))^2)
             =(e^(2a)-e^(-2a) +
               i*[cos(b)sin(b)(e^(2a)+2+e^(-2a))-cos(b)sin(b)(e^(2a)-2+e^(2a)))]
               / (cos(b)^2*(e^(2a)+2+e^(-2a)) + sin(b)^2*(e^(2a)-2+e^(2a))
             =(e^(2a)-e^(-2a) +
               i*cos(b)sin(b)*[e^(2a)+2+e^(-2a)-e^(2a)+2-e^(-2a)]) /
               ([cos(b)^2 + sin(b)^2][e^(2a)+e^(-2a)])+2*[cos(b)^2 - sin(b)^2])
             =(e^(2a)-e^(-2a) + i*cos(b)sin(b)*4) /
              (e^(2a)+e^(-2a)+2*[cos(b)^2 - sin(b)^2])
             =(e^(2a)-e^(-2a) + i*[sin(2b)/2]*4) /
              (e^(2a)+e^(-2a)+2*[cos(2b)])
             =(e^(2a)-e^(-2a) + i*2*sin(2b)) / (e^(2a) + e^(-2a) + 2*cos(2b))
      */
      bool op_is_tan = op->opcode() == HloOpcode::kTan;
      llvm::Value* a0 = EmitExtractReal(operand_value);
      llvm::Value* b0 = EmitExtractImag(operand_value);
      llvm::Type* type = a0->getType();
      llvm::Value* neg_one = llvm::ConstantFP::get(type, -1.0);

      auto a = op_is_tan ? FMul(neg_one, b0) : a0;
      auto b = op_is_tan ? a0 : b0;

      llvm::Value* two_a = FAdd(a, a);
      llvm::Value* neg_2a = FMul(neg_one, two_a);

      // When we are calculating the real numerator, e^(2a)-e^(-2a), for small
      // values of `a`, we will get a ULP of 2^-23 using the exp function. Using
      // expm1 to calculate e^(2a)-e^(-2a) = [e^(2a)-1] - [e^(-2a)-1] allows our
      // ULP to be arbitrarily small. For larger values of `a`, calculating the
      // numerator as Exp(2a)-Exp(-2a) vs Expm1(2a)-Expm1(-2a) return virtually
      // identical results.
      TF_ASSIGN_OR_RETURN(llvm::Value * exp_2a_m1,
                          EmitExpm1(component_type, two_a));
      TF_ASSIGN_OR_RETURN(llvm::Value * exp_neg_2a_m1,
                          EmitExpm1(component_type, neg_2a));
      llvm::Value* real_numerator = FSub(exp_2a_m1, exp_neg_2a_m1);

      // We can use the identity cos(2b)+1 = cos(b)^2-sin(b)^2+cos(b)^2+sin(b)^2
      // = 2cos(b)^2. This gives us the ability to be more precise when the
      // denominator is close to zero.
      TF_ASSIGN_OR_RETURN(llvm::Value * cos_b, EmitCos(component_type, b));
      llvm::Value* four = llvm::ConstantFP::get(type, 4.0);
      llvm::Value* cos_b_sq = FMul(cos_b, cos_b);
      llvm::Value* two_cos_2b_p2 = FMul(cos_b_sq, four);

      // Similarly we can compute sin(2b) with the formula sin(2b) =
      // 2*sin(b)*cos(b).
      TF_ASSIGN_OR_RETURN(llvm::Value * sin_b, EmitSin(component_type, b));
      llvm::Value* imag_numerator = FMul(four, FMul(cos_b, sin_b));

      // About "x^2 is a better approximation than Expm1(x) + Expm1(x)
      // for small values of x": this statement is not
      // accurate. Previously, Expm1(x) implementation had accuracy
      // issues for small x (where it was supposed to stand out in
      // accuracy!), but after resolving these issues (see
      // openxla/xla#10376), using precomputed exp_2a_m1 and
      // exp_neg_2a_m1 is accurate enough and we'll save a few
      // instructions.

      auto exp_sum_m2 = FAdd(exp_2a_m1, exp_neg_2a_m1);
      llvm::Value* denom = FAdd(exp_sum_m2, two_cos_2b_p2);

      // As `a` grows toward +inf and -inf, the real numerator will grow towards
      // +inf and -inf respectively, while the denominator will always grow
      // towards +inf. The result is real_numerator/denom = NaN, when it should
      // equal +1 and -1 respectively. Therefore, if our denominator is +inf,
      // we just hardcode the limits for the real numbers.
      llvm::Value* inf = llvm::ConstantFP::getInfinity(type);
      llvm::Value* is_inf = FCmpOEQ(exp_sum_m2, inf);
      llvm::Value* real_limit = llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::copysign, {neg_one, a}, {type}, b_);

      llvm::Value* real =
          Select(is_inf, real_limit, FDiv(real_numerator, denom));
      llvm::Value* imag = FDiv(imag_numerator, denom);

      // The complex tanh functions have a few corner cases:
      // 1. (+0, +0) => (+0, +0)        - Handled normally
      // 2. (x, +Inf) => (NaN, NaN)     - See below
      // 3. (x, NaN) => (NaN, NaN)      - See below
      // 4. (+inf, y) => (1, +0)        - Handled normally
      // 5. (+Inf, +Inf) => (1, +/-0)   - See below
      // 6. (+Inf, NaN) => (1, +/-0)    - See below
      // 7. (NaN, +0) => (NaN, +0)      - See below
      // 8. (NaN, y) => (NaN, NaN)      - Handled normally
      // 9. (NaN, NaN) => (NaN, NaN)    - Handled normally
      //
      // For the cases that aren't handled normally:
      // 2/3) Part of the calculation we do is that if exp(a) + exp(-a) = +inf,
      //      then we return (+/-1, +/-0). However, this is only true if we
      //      assume that a is infinity or b is finite. In the event that both a
      //      is finite and b is either +/-Inf or NaN, then our normal
      //      calculation would end up returing (+/-1, NaN), as opposed to (NaN,
      //      NaN).
      // 5/6) We always calculate the imaginary value as sin(2b)/denominator.
      //      When the denominator is infinity, this assures us that the zero is
      //      the correct sign. However if our imaginary input results in
      //      sin(2b) = NaN, we calculate our imaginary result as NaN.
      // 7)   In the event that a is NaN, the denominator will be NaN.
      //      Therefore, the normal calculation gives (NaN, NaN) while we need
      //      (NaN, +0).
      if (!(b_->getFastMathFlags().noNaNs() &&
            b_->getFastMathFlags().noInfs())) {
        llvm::Value* abs_a = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs,
                                                          {a}, {type}, b_);
        llvm::Value* zero = llvm::ConstantFP::get(type, 0.0);
        llvm::Value* nan = llvm::ConstantFP::getNaN(type);

        llvm::Value* a_is_inf = FCmpOEQ(abs_a, inf);
        llvm::Value* b_is_zero = FCmpOEQ(b, zero);

        // imag_numerator = 2sin(2b), so sin(2b) is NaN if and only if
        // imag_numerator is NaN.
        llvm::Value* sin_2b_is_nan =
            b_->CreateFCmpUNO(imag_numerator, imag_numerator);

        llvm::Value* real_is_nan =
            b_->CreateAnd(sin_2b_is_nan, b_->CreateNot(a_is_inf));
        llvm::Value* imag_is_zero =
            b_->CreateOr(b_is_zero, b_->CreateAnd(a_is_inf, sin_2b_is_nan));

        real = Select(real_is_nan, nan, real);
        imag = Select(imag_is_zero, zero, imag);
      }

      return op_is_tan ? EmitComposeComplex(op, imag, FMul(neg_one, real))
                       : EmitComposeComplex(op, real, imag);
    }
    case HloOpcode::kAbs: {
      return EmitComplexAbs(component_type, operand_value);
    }
    case HloOpcode::kSign: {  // Sign(c) = c / |c|
      TF_ASSIGN_OR_RETURN(auto cplx_abs,
                          EmitComplexAbs(component_type, operand_value));
      auto type = cplx_abs->getType();
      auto zero = llvm::ConstantFP::get(type, 0.0);
      auto oeq = FCmpOEQ(cplx_abs, zero);
      return Select(
          oeq, EmitComposeComplex(op, zero, zero),
          EmitComposeComplex(op, FDiv(EmitExtractReal(operand_value), cplx_abs),
                             FDiv(EmitExtractImag(operand_value), cplx_abs)));
    }
    case HloOpcode::kSqrt: {
      return EmitComplexSqrt(op, component_type, operand_value);
    }
    case HloOpcode::kRsqrt: {
      return EmitComplexRsqrt(op, component_type, operand_value);
    }
    case HloOpcode::kNegate:
      return EmitComposeComplex(op, FNeg(EmitExtractReal(operand_value)),
                                FNeg(EmitExtractImag(operand_value)));
    case HloOpcode::kReal:
      return EmitExtractReal(operand_value);
    case HloOpcode::kImag:
      return EmitExtractImag(operand_value);
    default:
      return Unimplemented("unary complex op '%s'",
                           HloOpcodeString(op->opcode()));
  }
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
  PrimitiveType operand_type = op->operand(0)->shape().element_type();
  if (operand_type == PRED) {
    return EmitPredBinaryOp(op, lhs_value, rhs_value);
  } else if (ShapeUtil::ElementIsIntegral(op->operand(0)->shape())) {
    return EmitIntegerBinaryOp(
        op, lhs_value, rhs_value,
        primitive_util::IsSignedIntegralType(operand_type));
  } else if (primitive_util::IsComplexType(operand_type)) {
    return EmitComplexBinaryOp(op, lhs_value, rhs_value);
  } else {
    return EmitFloatBinaryOp(op, lhs_value, rhs_value);
  }
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitFloatBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
  switch (op->opcode()) {
    case HloOpcode::kComplex:
      return EmitComposeComplex(op, lhs_value, rhs_value);
    case HloOpcode::kAdd:
      return FAdd(lhs_value, rhs_value, op->name());
    case HloOpcode::kSubtract:
      return FSub(lhs_value, rhs_value, op->name());
    case HloOpcode::kMultiply:
      return FMul(lhs_value, rhs_value, op->name());
    case HloOpcode::kDivide:
      return FDiv(lhs_value, rhs_value, op->name());
    case HloOpcode::kRemainder:
      return FRem(lhs_value, rhs_value, op->name());
    // LLVM comparisons can be "unordered" (U) or "ordered" (O) -- ordered
    // comparisons always return false when one of the operands is NaN, whereas
    // unordered comparisons return true.
    //
    // We use ordered comparisons for everything except kNe, where we use an
    // unordered comparison.  This makes x != y equivalent to !(x == y), and
    // matches C++'s semantics.
    case HloOpcode::kCompare: {
      PrimitiveType operand_type = op->operand(0)->shape().element_type();
      if (operand_type == F8E5M2) {
        lhs_value = EmitF8e5m2ToF16(lhs_value, b_);
        rhs_value = EmitF8e5m2ToF16(rhs_value, b_);
      } else if (operand_type == F8E4M3) {
        lhs_value = EmitToF16F8e<4>(lhs_value, b_);
        rhs_value = EmitToF16F8e<4>(rhs_value, b_);
      } else if (operand_type == F8E4M3FN) {
        lhs_value = EmitF8e4m3fnToF16(lhs_value, b_);
        rhs_value = EmitF8e4m3fnToF16(rhs_value, b_);
      } else if (operand_type == F4E2M1FN) {
        lhs_value = EmitF4e2m1fnToF16(lhs_value, b_);
        rhs_value = EmitF4e2m1fnToF16(rhs_value, b_);
      } else if (operand_type == F8E8M0FNU) {
        lhs_value = EmitF8e8m0fnuToF32(lhs_value, b_);
        rhs_value = EmitF8e8m0fnuToF32(rhs_value, b_);
      } else if (operand_type == F8E5M2FNUZ || operand_type == F8E4M3FNUZ) {
        TF_ASSIGN_OR_RETURN(
            lhs_value,
            EmitF8fnuzToFloating(operand_type, lhs_value, F16, b_, module_));
        TF_ASSIGN_OR_RETURN(
            rhs_value,
            EmitF8fnuzToFloating(operand_type, rhs_value, F16, b_, module_));
      } else if (operand_type == F8E3M4) {
        lhs_value = EmitToF16F8e<3>(lhs_value, b_);
        rhs_value = EmitToF16F8e<3>(rhs_value, b_);
      }
      switch (op->comparison_direction()) {
        case ComparisonDirection::kEq:
          return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OEQ, lhs_value,
                                         rhs_value, b_, op->name());
        case ComparisonDirection::kNe:
          return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_UNE, lhs_value,
                                         rhs_value, b_, op->name());
        case ComparisonDirection::kLt:
          return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OLT, lhs_value,
                                         rhs_value, b_, op->name());
        case ComparisonDirection::kGt:
          return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OGT, lhs_value,
                                         rhs_value, b_, op->name());
        case ComparisonDirection::kLe:
          return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OLE, lhs_value,
                                         rhs_value, b_, op->name());
        case ComparisonDirection::kGe:
          return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OGE, lhs_value,
                                         rhs_value, b_, op->name());
      }
    }
    case HloOpcode::kMaximum:
      return EmitFloatMax(lhs_value, rhs_value, op->name());
    case HloOpcode::kMinimum:
      return EmitFloatMin(lhs_value, rhs_value, op->name());
    case HloOpcode::kPower:
      return EmitPow(op->shape().element_type(), lhs_value, rhs_value,
                     op->name());
    case HloOpcode::kAtan2:
      return EmitAtan2(op->shape().element_type(), lhs_value, rhs_value,
                       op->name());
    default:
      return Unimplemented("binary floating point op '%s'",
                           HloOpcodeString(op->opcode()));
  }
}

// Using sqrt(a^2 + b^2) can cause overflow errors. Therefore we can use
// sqrt(a^2 + b^2) = sqrt(a^2 * (1 + b^2/a^2))
//                 = |a| * sqrt(1 + (b/a)^2)
// With the assumption that |a| >= |b|. This assumption is enforced by swapping
// a and b during the computation if necessary, since the final result is the
// same.
//
// This method returns the min, max, and sqrt term for this calculation. This is
// done to prevent potential overflow errors that can occur from multiplying the
// max with the sqrt term. (i.e. when calculating the sqrt of the absolute
// value, we can take the sqrt of the max and the sqrt term before multiplying
// them together.) If return_sqrt is false, it returns 1 + (min/max)^2 instead
// of sqrt(1 + (min/max)^2).
//
// Note that the precision of this computation can be improved by implementing
// another algorithm:
//   Carlos F. Borges - An Improved Algorithm for hypot(a,b):
//   https://arxiv.org/pdf/1904.09481.pdf
absl::StatusOr<std::tuple<llvm::Value*, llvm::Value*, llvm::Value*>>
ElementalIrEmitter::EmitComplexAbsHelper(PrimitiveType prim_type,
                                         llvm::Value* real, llvm::Value* imag,
                                         bool return_sqrt) {
  llvm::Value* abs_real = llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::fabs, {real}, {real->getType()}, b_);
  llvm::Value* abs_imag = llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::fabs, {imag}, {imag->getType()}, b_);
  llvm::Value* max = EmitFloatMax(abs_real, abs_imag, "");
  llvm::Value* min = EmitFloatMin(abs_real, abs_imag, "");

  llvm::Value* div = FDiv(min, max);
  llvm::Value* div_sq = FMul(div, div);
  llvm::Value* one = llvm::ConstantFP::get(max->getType(), 1);
  llvm::Value* one_p_div_sq = FAdd(one, div_sq);
  TF_ASSIGN_OR_RETURN(llvm::Value * sqrt, EmitSqrt(prim_type, one_p_div_sq));
  return std::make_tuple(min, max, return_sqrt ? sqrt : one_p_div_sq);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitComplexAbs(
    PrimitiveType prim_type, llvm::Value* operand_value) {
  llvm::Value* min;
  llvm::Value* max;
  llvm::Value* sqrt;
  llvm::Value* real = EmitExtractReal(operand_value);
  llvm::Value* imag = EmitExtractImag(operand_value);
  TF_ASSIGN_OR_RETURN(
      std::tie(min, max, sqrt),
      EmitComplexAbsHelper(prim_type, real, imag, /*return_sqrt=*/true));
  llvm::Value* result = FMul(max, sqrt);
  // When (min, max) are (0, 0), (inf, inf), or (NaN, ...), `result` is NaN.
  // In such cases, we return `min` instead of `result`.
  return Select(FCmpUNO(result, result), min, result);
}

// Calculates ComplexAbs in the same way, except using:
// sqrt(|a| * sqrt(1 + (b/a)^2)) = sqrt(|a|) * pow(1 + (b/a)^2, .25)
absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitSqrtComplexAbs(
    PrimitiveType prim_type, llvm::Value* operand_value) {
  llvm::Value* min;
  llvm::Value* max;
  llvm::Value* one_p_div_sq;
  llvm::Value* real = EmitExtractReal(operand_value);
  llvm::Value* imag = EmitExtractImag(operand_value);
  TF_ASSIGN_OR_RETURN(
      std::tie(min, max, one_p_div_sq),
      EmitComplexAbsHelper(prim_type, real, imag, /*return_sqrt=*/false));
  TF_ASSIGN_OR_RETURN(llvm::Value * sqrt_max, EmitSqrt(prim_type, max));
  TF_ASSIGN_OR_RETURN(llvm::Value * pow,
                      EmitPow(prim_type, one_p_div_sq,
                              llvm::ConstantFP::get(max->getType(), .25), ""));
  llvm::Value* result = FMul(sqrt_max, pow);
  // When (min, max) are (0, 0), (inf, inf), or (NaN, ...), `result` is NaN.
  // In such cases, we return `min` instead of `result`.
  return Select(FCmpUNO(result, result), min, result);
}

// Calculates ComplexAbs in the same way, except using:
// rsqrt(|a| * sqrt(1 + (b/a)^2)) = rsqrt(|a|) * rsqrt(sqrt(1 + (b/a)^2))
absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitRsqrtComplexAbs(
    PrimitiveType prim_type, llvm::Value* operand_value) {
  llvm::Value* min;
  llvm::Value* max;
  llvm::Value* sqrt;
  llvm::Value* real = EmitExtractReal(operand_value);
  llvm::Value* imag = EmitExtractImag(operand_value);
  TF_ASSIGN_OR_RETURN(
      std::tie(min, max, sqrt),
      EmitComplexAbsHelper(prim_type, real, imag, /*return_sqrt=*/true));
  TF_ASSIGN_OR_RETURN(llvm::Value * rsqrt_max, EmitRsqrt(prim_type, max));
  TF_ASSIGN_OR_RETURN(llvm::Value * rsqrt_sqrt, EmitRsqrt(prim_type, sqrt));
  llvm::Value* result = FMul(rsqrt_max, rsqrt_sqrt);
  TF_ASSIGN_OR_RETURN(llvm::Value * rsqrt_min, EmitRsqrt(prim_type, min));
  // When (min, max) are (0, 0), (inf, inf), or (NaN, ...), `result` is NaN.
  // In such cases, we return rsqrt(min) instead of `result`.
  return Select(FCmpUNO(result, result), rsqrt_min, result);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitComplexAdd(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
  return EmitComposeComplex(
      op, FAdd(EmitExtractReal(lhs_value), EmitExtractReal(rhs_value)),
      FAdd(EmitExtractImag(lhs_value), EmitExtractImag(rhs_value)));
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitComplexSubtract(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
  return EmitComposeComplex(
      op, FSub(EmitExtractReal(lhs_value), EmitExtractReal(rhs_value)),
      FSub(EmitExtractImag(lhs_value), EmitExtractImag(rhs_value)));
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitComplexMultiply(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
  return EmitComposeComplex(
      op,
      FSub(FMul(EmitExtractReal(lhs_value), EmitExtractReal(rhs_value)),
           FMul(EmitExtractImag(lhs_value), EmitExtractImag(rhs_value))),
      FAdd(FMul(EmitExtractReal(lhs_value), EmitExtractImag(rhs_value)),
           FMul(EmitExtractImag(lhs_value), EmitExtractReal(rhs_value))));
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitComplexDivide(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
  // Division of complex numbers is implemented here, taking into account
  // over/underflow, NaN and Inf values.
  auto a_r = EmitExtractReal(lhs_value);
  auto a_i = EmitExtractImag(lhs_value);
  auto b_r = EmitExtractReal(rhs_value);
  auto b_i = EmitExtractImag(rhs_value);
  auto type = a_r->getType();

  // Smith's algorithm to divide complex numbers. It is just a bit smarter
  // way to compute the following formula:
  //  (a_r + a_i * i) / (b_r + b_i * i)
  //    = (a_r + a_i * i) (b_r - b_i * i) / ((b_r + b_i * i)(b_r - b_i * i))
  //    = ((a_r * b_r + a_i * b_i) + (a_i * b_r - a_r * b_i) * i) / ||b||^2
  //
  // Depending on whether |b_r| < |b_i| we compute either
  //   b_r_b_i_ratio = b_r / b_i
  //   b_r_b_i_denom = b_i + b_r * b_r_b_i_ratio
  //   c_r = (a_r * b_r_b_i_ratio + a_i ) / b_r_b_i_denom
  //   c_i = (a_i * b_r_b_i_ratio - a_r ) / b_r_b_i_denom
  //
  // or
  //
  //   b_i_b_r_ratio = b_i / b_r
  //   b_i_b_r_denom = b_r + b_i * b_i_b_r_ratio
  //   c_r = (a_r + a_i * b_i_b_r_ratio ) / b_i_b_r_denom
  //   c_i = (a_i - a_r * b_i_b_r_ratio ) / b_i_b_r_denom
  //
  // See https://dl.acm.org/citation.cfm?id=368661 for more details.
  auto b_r_b_i_ratio = FDiv(b_r, b_i);
  auto b_r_b_i_denom = FAdd(b_i, FMul(b_r_b_i_ratio, b_r));
  auto b_i_b_r_ratio = FDiv(b_i, b_r);
  auto b_i_b_r_denom = FAdd(b_r, FMul(b_i_b_r_ratio, b_i));

  auto b_r_abs =
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs, {b_r}, {type}, b_);
  auto b_i_abs =
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs, {b_i}, {type}, b_);
  auto b_r_lt_b_i = FCmpOLT(b_r_abs, b_i_abs);
  auto c_r = Select(b_r_lt_b_i,
                    FDiv(FAdd(FMul(b_r_b_i_ratio, a_r), a_i), b_r_b_i_denom),
                    FDiv(FAdd(FMul(b_i_b_r_ratio, a_i), a_r), b_i_b_r_denom));
  auto c_i = Select(b_r_lt_b_i,
                    FDiv(FSub(FMul(b_r_b_i_ratio, a_i), a_r), b_r_b_i_denom),
                    FDiv(FSub(a_i, FMul(b_i_b_r_ratio, a_r)), b_i_b_r_denom));
  auto result = EmitComposeComplex(op, c_r, c_i);

  // Consider corner cases, if the result is (NaN, NaN).
  auto zero = llvm::ConstantFP::get(type, 0.0);
  auto one = llvm::ConstantFP::get(type, 1.0);
  auto inf = llvm::ConstantFP::getInfinity(type);

  // Case 1. Zero denominator.
  auto zero_denominator =
      And(And(FCmpOEQ(b_r_abs, zero), FCmpOEQ(b_i_abs, zero)),
          Or(Not(FCmpUNO(a_r, zero)), Not(FCmpUNO(a_i, zero))));
  auto inf_with_sign_of_b_r = llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::copysign, {inf, b_r}, {type}, b_);
  auto zero_denominator_result = EmitComposeComplex(
      op, FMul(inf_with_sign_of_b_r, a_r), FMul(inf_with_sign_of_b_r, a_i));

  // Case 2. Infinite numerator, finite denominator.
  auto b_r_finite = FCmpONE(b_r_abs, inf);
  auto b_i_finite = FCmpONE(b_i_abs, inf);
  auto a_r_abs =
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs, {a_r}, {type}, b_);
  auto a_i_abs =
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs, {a_i}, {type}, b_);
  auto a_r_infinite = FCmpOEQ(a_r_abs, inf);
  auto a_i_infinite = FCmpOEQ(a_i_abs, inf);
  auto inf_num_finite_denom =
      And(Or(a_r_infinite, a_i_infinite), And(b_r_finite, b_i_finite));

  auto a_r_inf_with_sign = llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::copysign, {Select(a_r_infinite, one, zero), a_r}, {type},
      b_);
  auto a_i_inf_with_sign = llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::copysign, {Select(a_i_infinite, one, zero), a_i}, {type},
      b_);
  auto inf_num_finite_denom_result = EmitComposeComplex(
      op,
      FMul(inf,
           FAdd(FMul(a_r_inf_with_sign, b_r), FMul(a_i_inf_with_sign, b_i))),
      FMul(inf,
           FSub(FMul(a_i_inf_with_sign, b_r), FMul(a_r_inf_with_sign, b_i))));

  // Case 3. Finite numerator, infinite denominator.
  auto a_r_finite = FCmpONE(a_r_abs, inf);
  auto a_i_finite = FCmpONE(a_i_abs, inf);
  auto b_r_infinite = FCmpOEQ(b_r_abs, inf);
  auto b_i_infinite = FCmpOEQ(b_i_abs, inf);
  auto finite_num_inf_denom =
      And(Or(b_r_infinite, b_i_infinite), And(a_r_finite, a_i_finite));

  auto b_r_inf_with_sign = llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::copysign, {Select(b_r_infinite, one, zero), b_r}, {type},
      b_);
  auto b_i_inf_with_sign = llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::copysign, {Select(b_i_infinite, one, zero), b_i}, {type},
      b_);
  auto finite_num_inf_denom_result = EmitComposeComplex(
      op,
      FMul(zero,
           FAdd(FMul(a_r, b_r_inf_with_sign), FMul(a_i, b_i_inf_with_sign))),
      FMul(zero,
           FSub(FMul(a_i, b_r_inf_with_sign), FMul(a_r, b_i_inf_with_sign))));

  auto c_nan = And(FCmpUNO(c_r, zero), FCmpUNO(c_i, zero));
  return Select(c_nan,
                Select(zero_denominator, zero_denominator_result,
                       Select(inf_num_finite_denom, inf_num_finite_denom_result,
                              Select(finite_num_inf_denom,
                                     finite_num_inf_denom_result, result))),
                result);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitComplexLog(
    const HloInstruction* op, llvm::Value* operand_value) {
  // log(a+bi) = log(abs(a+bi)) + i*atan2(b,a)
  PrimitiveType component_type =
      primitive_util::ComplexComponentType(op->shape().element_type());
  auto a = EmitExtractReal(operand_value);
  auto b = EmitExtractImag(operand_value);
  TF_ASSIGN_OR_RETURN(llvm::Value * angle, EmitAtan2(component_type, b, a, ""));
  TF_ASSIGN_OR_RETURN(llvm::Value * abs,
                      EmitComplexAbs(component_type, operand_value));
  TF_ASSIGN_OR_RETURN(llvm::Value * log_abs, EmitLog(component_type, abs));
  return EmitComposeComplex(op, log_abs, angle);
}

// Using our EmitComplexPower formula, but setting c=0.5 and d=0, we get:
//   e^[ln(r)*c - t*d] * [cos(ln(r)*d + t*c) + i*sin(ln(r)*d + t*c)]
// = e^[ln(r)*0.5] * [cos(t*0.5) + i*sin(t*0.5)]
// = r^0.5 * [cos(t/2) + i*sin(t/2)]
// = sqrt(r) * [cos(t/2) + i*sin(t/2)]
// where r = |a+bi| and t = atan2(b,a)
// TODO(bixia): See doc for implementation without atan2.
absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitComplexSqrt(
    const HloInstruction* op, PrimitiveType prim_type,
    llvm::Value* operand_value) {
  llvm::Type* type = static_cast<llvm::StructType*>(operand_value->getType())
                         ->getElementType(0);

  TF_ASSIGN_OR_RETURN(llvm::Value * r,
                      EmitSqrtComplexAbs(prim_type, operand_value));

  llvm::Value* a = EmitExtractReal(operand_value);
  llvm::Value* b = EmitExtractImag(operand_value);
  TF_ASSIGN_OR_RETURN(llvm::Value * t, EmitAtan2(prim_type, b, a, ""));

  llvm::Value* c = llvm::ConstantFP::get(type, 0.5);
  llvm::Value* angle = FMul(t, c);
  TF_ASSIGN_OR_RETURN(llvm::Value * cos, EmitCos(prim_type, angle));
  TF_ASSIGN_OR_RETURN(llvm::Value * sin, EmitSin(prim_type, angle));

  llvm::Value* real_part;
  llvm::Value* imag_part;

  llvm::Value* zero = llvm::ConstantFP::get(type, 0);

  if (!(b_->getFastMathFlags().noNaNs() && b_->getFastMathFlags().noInfs())) {
    llvm::Value* inf = llvm::ConstantFP::getInfinity(type);
    llvm::Value* neg_inf = llvm::ConstantFP::getInfinity(type, true);
    llvm::Value* nan = llvm::ConstantFP::getNaN(type);
    llvm::Value* abs_b = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs,
                                                      {b}, {b->getType()}, b_);

    real_part = Select(Or(FCmpOEQ(abs_b, inf), FCmpOEQ(a, inf)), inf,
                       Select(And(FCmpOEQ(a, neg_inf), FCmpONE(abs_b, inf)),
                              zero, FMul(r, cos)));

    llvm::Value* b_signed_inf = llvm_ir::EmitCallToIntrinsic(
        llvm::Intrinsic::copysign, {inf, b}, {b->getType()}, b_);
    imag_part =
        Select(Or(FCmpOEQ(abs_b, inf), FCmpOEQ(a, neg_inf)), b_signed_inf,
               Select(FCmpUNO(r, r), nan,
                      Select(FCmpOEQ(sin, zero), sin, FMul(r, sin))));
  } else {
    real_part = FMul(r, cos);
    imag_part = Select(FCmpOEQ(sin, zero), sin, FMul(r, sin));
  }

  return Select(FCmpOEQ(r, zero), EmitComposeComplex(op, zero, zero),
                EmitComposeComplex(op, real_part, imag_part));
}

// Similar to Sqrt, we can use our EmitComplexPower formula, but set
// c=-0.5 and d=0. We get:
//   e^[ln(r)*c - t*d] * [cos(ln(r)*d + t*c) + i*sin(ln(r)*d + t*c)]
// = e^[ln(r)*-0.5] * [cos(t*-0.5) + i*sin(t*-0.5)]
// = r^(-0.5) * [cos(-t/2) + i*sin(-t/2)]
// = rsqrt(r) * [cos(-t/2) + i*sin(-t/2)]
// where r = |a+bi| and t = atan2(b,a).
absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitComplexRsqrt(
    const HloInstruction* op, PrimitiveType prim_type,
    llvm::Value* operand_value) {
  llvm::Type* type = static_cast<llvm::StructType*>(operand_value->getType())
                         ->getElementType(0);

  TF_ASSIGN_OR_RETURN(llvm::Value * r,
                      EmitRsqrtComplexAbs(prim_type, operand_value));

  llvm::Value* a = EmitExtractReal(operand_value);
  llvm::Value* b = EmitExtractImag(operand_value);
  TF_ASSIGN_OR_RETURN(llvm::Value * t, EmitAtan2(prim_type, b, a, ""));

  llvm::Value* c = llvm::ConstantFP::get(type, -0.5);
  llvm::Value* angle = FMul(t, c);
  TF_ASSIGN_OR_RETURN(llvm::Value * cos, EmitCos(prim_type, angle));
  TF_ASSIGN_OR_RETURN(llvm::Value * sin, EmitSin(prim_type, angle));

  llvm::Value* real_part = FMul(r, cos);
  llvm::Value* imag_part = FMul(r, sin);

  if (!(b_->getFastMathFlags().noNaNs() && b_->getFastMathFlags().noInfs())) {
    llvm::Value* zero = llvm::ConstantFP::get(type, 0);
    llvm::Value* neg_one = llvm::ConstantFP::get(type, -1);
    llvm::Value* inf = llvm::ConstantFP::getInfinity(type);
    llvm::Value* nan = llvm::ConstantFP::getNaN(type);
    llvm::Value* a_signed_zero = llvm_ir::EmitCallToIntrinsic(
        llvm::Intrinsic::copysign, {zero, a}, {a->getType()}, b_);
    llvm::Value* b_signed_zero = llvm_ir::EmitCallToIntrinsic(
        llvm::Intrinsic::copysign, {zero, b}, {b->getType()}, b_);
    llvm::Value* neg_b_signed_zero = FMul(b_signed_zero, neg_one);

    llvm::Value* abs_a = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs,
                                                      {a}, {a->getType()}, b_);
    llvm::Value* abs_b = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs,
                                                      {b}, {b->getType()}, b_);

    llvm::Value* is_zero_zero = And(FCmpOEQ(b, zero), FCmpOEQ(a, zero));
    real_part = Select(
        is_zero_zero, inf,
        Select(Or(And(FCmpOEQ(abs_b, inf), FCmpUNO(a, a)), FCmpOEQ(abs_a, inf)),
               a_signed_zero, FMul(r, cos)));
    imag_part = Select(
        is_zero_zero, nan,
        Select(Or(And(FCmpOEQ(abs_b, inf), FCmpUNO(a, a)), FCmpOEQ(abs_a, inf)),
               neg_b_signed_zero, FMul(r, sin)));
  } else {
    llvm::Value* zero = llvm::ConstantFP::get(type, 0);
    llvm::Value* inf = llvm::ConstantFP::getInfinity(type);
    llvm::Value* nan = llvm::ConstantFP::getNaN(type);

    llvm::Value* is_zero_zero = And(FCmpOEQ(b, zero), FCmpOEQ(a, zero));
    real_part = Select(is_zero_zero, inf, FMul(r, cos));
    imag_part = Select(is_zero_zero, nan, FMul(r, sin));
  }

  return EmitComposeComplex(op, real_part, imag_part);
}

//   lhs_value^rhs_value
// = (a+bi)^(c+di)
// = (a*a+b*b)^(0.5c) * exp(-d*atan2(b,a)) * (cos(q) + i*sin(q)),
//    where q = c*atan2(b,a)+0.5d*ln(a*a+b*b)
// = |lhs|^c * exp(-d*atan2(b,a)) * (cos(q) + i*sin(q))
//   where q = c*atan2(b,a)+d*ln(|lhs|)
absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitComplexPower(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
  PrimitiveType component_type =
      primitive_util::ComplexComponentType(op->shape().element_type());
  auto a = EmitExtractReal(lhs_value);
  auto b = EmitExtractImag(lhs_value);
  auto c = EmitExtractReal(rhs_value);
  auto d = EmitExtractImag(rhs_value);

  TF_ASSIGN_OR_RETURN(auto abs, EmitComplexAbs(component_type, lhs_value));
  TF_ASSIGN_OR_RETURN(auto abs_to_c, EmitPow(component_type, abs, c, ""));

  auto neg_d = FNeg(d);
  TF_ASSIGN_OR_RETURN(auto arg_lhs, EmitAtan2(component_type, b, a, ""));
  auto neg_d_arg_lhs = FMul(neg_d, arg_lhs);
  TF_ASSIGN_OR_RETURN(auto e_to_neg_d_arg_lhs,
                      EmitExp(component_type, neg_d_arg_lhs, ""));
  auto coeff = FMul(abs_to_c, e_to_neg_d_arg_lhs);
  TF_ASSIGN_OR_RETURN(auto ln_abs, EmitLog(component_type, abs));
  auto q = FAdd(FMul(c, arg_lhs), FMul(d, ln_abs));
  TF_ASSIGN_OR_RETURN(auto cos_q, EmitCos(component_type, q));
  TF_ASSIGN_OR_RETURN(auto sin_q, EmitSin(component_type, q));

  llvm::Value* inf = llvm::ConstantFP::getInfinity(a->getType());
  auto zero = llvm::ConstantFP::get(a->getType(), 0);
  auto one = llvm::ConstantFP::get(a->getType(), 1);

  // Case 0:
  // d^c is 0 if d is 0 and c > 0. 0^0 is defined to be 1.0, see
  // Branch Cuts for Complex Elementary Functions or Much Ado About
  // Nothing's Sign Bit, W. Kahan, Section 10.
  auto cutoff_0 =
      Select(And(And(FCmpOEQ(abs, zero), FCmpOEQ(d, zero)), FCmpOLE(zero, c)),
             EmitComposeComplex(op, Select(FCmpOEQ(zero, c), one, zero), zero),
             EmitComposeComplex(op, FMul(coeff, cos_q), FMul(coeff, sin_q)));

  // Case 1:
  // x^0 is defined to be 1 for any x, see
  // Branch Cuts for Complex Elementary Functions or Much Ado About
  // Nothing's Sign Bit, W. Kahan, Section 10.
  auto cutoff_1 = Select(And(FCmpOEQ(zero, c), FCmpOEQ(d, zero)),
                         EmitComposeComplex(op, one, zero), cutoff_0);

  // Case 2:
  // 1^(c + d*i) = 1 + 0*i
  auto cutoff_2 = Select(And(FCmpOEQ(a, one), FCmpOEQ(b, zero)),
                         EmitComposeComplex(op, one, zero), cutoff_1);

  // Case 3:
  // inf^(c + 0*i) = inf + 0*i, c > 0
  auto cutoff_3 = Select(
      And(FCmpOEQ(a, inf),
          And(FCmpOEQ(b, zero), And(FCmpOEQ(d, zero), FCmpOGT(c, zero)))),
      EmitComposeComplex(op, inf, zero), cutoff_2);

  // Case 4:
  // inf^(c + 0*i) = 0 + 0*i, c < 0
  auto cutoff_4 = Select(
      And(FCmpOEQ(a, inf),
          And(FCmpOEQ(b, zero), And(FCmpOEQ(d, zero), FCmpOLT(c, zero)))),
      EmitComposeComplex(op, zero, zero), cutoff_3);

  return cutoff_4;
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitComplexBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
  switch (op->opcode()) {
    case HloOpcode::kAdd:
      return EmitComplexAdd(op, lhs_value, rhs_value);
    case HloOpcode::kSubtract:
      return EmitComplexSubtract(op, lhs_value, rhs_value);
    case HloOpcode::kMultiply:
      return EmitComplexMultiply(op, lhs_value, rhs_value);
    case HloOpcode::kDivide: {
      return EmitComplexDivide(op, lhs_value, rhs_value);
    }
    // LLVM comparisons can be "unordered" (U) or "ordered" (O) -- ordered
    // comparisons always return false when one of the operands is NaN, whereas
    // unordered comparisons return true.
    //
    // We use ordered comparisons for everything except kNe, where we use an
    // unordered comparison.  This makes x != y equivalent to !(x == y), and
    // matches C++'s semantics.
    case HloOpcode::kCompare: {
      switch (op->comparison_direction()) {
        case ComparisonDirection::kEq:
          return And(llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OEQ,
                                             EmitExtractReal(lhs_value),
                                             EmitExtractReal(rhs_value), b_),
                     llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OEQ,
                                             EmitExtractImag(lhs_value),
                                             EmitExtractImag(rhs_value), b_));
        case ComparisonDirection::kNe:
          return Or(llvm_ir::EmitComparison(llvm::CmpInst::FCMP_UNE,
                                            EmitExtractReal(lhs_value),
                                            EmitExtractReal(rhs_value), b_),
                    llvm_ir::EmitComparison(llvm::CmpInst::FCMP_UNE,
                                            EmitExtractImag(lhs_value),
                                            EmitExtractImag(rhs_value), b_));
        default:
          return Unimplemented(
              "complex comparison '%s'",
              ComparisonDirectionToString(op->comparison_direction()));
      }
    }
    case HloOpcode::kPower: {
      return EmitComplexPower(op, lhs_value, rhs_value);
    }
    case HloOpcode::kAtan2: {
      // atan2(y,x) = -i * log((x + i * y)/sqrt(x**2+y**2))
      auto y = lhs_value;
      auto x = rhs_value;
      TF_ASSIGN_OR_RETURN(auto x_squared, EmitComplexMultiply(op, x, x));
      TF_ASSIGN_OR_RETURN(auto y_squared, EmitComplexMultiply(op, y, y));
      TF_ASSIGN_OR_RETURN(auto x_squared_plus_y_squared,
                          EmitComplexAdd(op, x_squared, y_squared));
      auto component_type =
          primitive_util::ComplexComponentType(op->shape().element_type());
      TF_ASSIGN_OR_RETURN(
          auto sqrt_x_squared_plus_y_squared,
          EmitComplexSqrt(op, component_type, x_squared_plus_y_squared));
      auto type =
          llvm_ir::PrimitiveTypeToIrType(component_type, module_->getContext());
      auto zero = llvm::ConstantFP::get(type, 0.0);
      auto one = llvm::ConstantFP::get(type, 1.0);
      auto i = EmitComposeComplex(op, zero, one);
      TF_ASSIGN_OR_RETURN(auto i_times_y, EmitComplexMultiply(op, i, y));
      TF_ASSIGN_OR_RETURN(auto x_plus_iy, EmitComplexAdd(op, x, i_times_y));
      TF_ASSIGN_OR_RETURN(
          auto div_result,
          EmitComplexDivide(op, x_plus_iy, sqrt_x_squared_plus_y_squared));
      TF_ASSIGN_OR_RETURN(auto log_result, EmitComplexLog(op, div_result));
      auto negative_one = llvm::ConstantFP::get(type, -1.0);
      auto negative_i = EmitComposeComplex(op, zero, negative_one);
      return EmitComplexMultiply(op, negative_i, log_result);
    }
    default:
      return Unimplemented("binary complex op '%s'",
                           HloOpcodeString(op->opcode()));
  }
}

llvm::Value* ElementalIrEmitter::EmitFloatMax(llvm::Value* lhs_value,
                                              llvm::Value* rhs_value,
                                              absl::string_view name) {
  return llvm_ir::EmitFloatMax(lhs_value, rhs_value, b_, fast_min_max(), name);
}

llvm::Value* ElementalIrEmitter::EmitFloatMin(llvm::Value* lhs_value,
                                              llvm::Value* rhs_value,
                                              absl::string_view name) {
  return llvm_ir::EmitFloatMin(lhs_value, rhs_value, b_, fast_min_max(), name);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitLog(
    PrimitiveType prim_type, llvm::Value* value) {
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::log, {value},
                                      {value->getType()}, b_);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitLog1p(
    PrimitiveType prim_type, llvm::Value* value) {
  auto x = value;
  auto type = llvm_ir::PrimitiveTypeToIrType(prim_type, module_->getContext());
  auto one = llvm::ConstantFP::get(type, 1.0);
  auto negative_half = llvm::ConstantFP::get(type, -0.5);
  // When x is large, the naive evaluation of ln(x + 1) is more
  // accurate than the Taylor series.
  TF_ASSIGN_OR_RETURN(auto for_large_x, EmitLog(prim_type, FAdd(x, one)));
  // When x is small, (defined to be less than sqrt(2) / 2), use a rational
  // approximation. The approximation below is based on one from the Cephes
  // Mathematical Library.
  //
  // sqrt(2) - 1.
  const auto kAntilogarithmIsSmallThreshold = 0.41421356237309504880;

  static const std::array<double, 7> kDenominatorCoeffs{
      1.,
      1.5062909083469192043167E1,
      8.3047565967967209469434E1,
      2.2176239823732856465394E2,
      3.0909872225312059774938E2,
      2.1642788614495947685003E2,
      6.0118660497603843919306E1,
  };

  static const std::array<double, 7> kNumeratorCoeffs{
      4.5270000862445199635215E-5, 4.9854102823193375972212E-1,
      6.5787325942061044846969E0,  2.9911919328553073277375E1,
      6.0949667980987787057556E1,  5.7112963590585538103336E1,
      2.0039553499201281259648E1,
  };

  auto x_squared = FMul(x, x);
  TF_ASSIGN_OR_RETURN(auto denominator,
                      EvaluatePolynomial(type, x, kDenominatorCoeffs));
  TF_ASSIGN_OR_RETURN(auto numerator,
                      EvaluatePolynomial(type, x, kNumeratorCoeffs));
  auto for_small_x = FDiv(numerator, denominator);
  for_small_x = FMul(FMul(x, x_squared), for_small_x);
  for_small_x = FAdd(FMul(negative_half, x_squared), for_small_x);
  for_small_x = FAdd(x, for_small_x);

  auto abs_x =
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs, {value}, {type}, b_);
  auto x_is_small = FCmpOLT(
      abs_x, llvm::ConstantFP::get(type, kAntilogarithmIsSmallThreshold));
  return Select(x_is_small, for_small_x, for_large_x);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitSqrt(PrimitiveType,
                                                          llvm::Value* value) {
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::sqrt, {value},
                                      {value->getType()}, b_);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitRsqrt(
    PrimitiveType prim_type, llvm::Value* value) {
  TF_ASSIGN_OR_RETURN(auto sqrt, EmitSqrt(prim_type, value));
  return FDiv(llvm::ConstantFP::get(sqrt->getType(), 1.0), sqrt);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitSin(
    PrimitiveType prim_type, llvm::Value* value) {
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::sin, {value},
                                      {value->getType()}, b_);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitCos(
    PrimitiveType prim_type, llvm::Value* value) {
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::cos, {value},
                                      {value->getType()}, b_);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitCosm1(
    PrimitiveType prim_type, llvm::Value* value) {
  auto x = value;
  auto type = llvm_ir::PrimitiveTypeToIrType(prim_type, module_->getContext());
  auto negative_half = llvm::ConstantFP::get(type, -0.5);
  auto negative_one = llvm::ConstantFP::get(type, -1.0);

  // Algorithm copied from cephes cosm1:
  //   cosm1(x) = -0.5 * x^2 + x^4 * P(x^2);
  // that is suitable when abs(x) < pi/4, otherwise we'll use cos(x)-1.
  //
  // This is an alternative algorithm
  //   cosm1(x) = -2 * sin(x/2)^2
  // that is only slightly less accurate around abs(x) == 0.1 but
  // otherwise equivalent accuracy-wise compared to cephes cosm1.
  // However, we are not using it because it is notably less
  // performant than cephes cosm1.

  // TODO: define cosm1(x) as cosm1(x mod (2*pi)) to increase accuracy
  // for large x values that are close to 2*pi*n where n is some integer.
  static const std::array<double, 7> kCoeffs{
      4.7377507964246204691685E-14, -1.1470284843425359765671E-11,
      2.0876754287081521758361E-9,  -2.7557319214999787979814E-7,
      2.4801587301570552304991E-5,  -1.3888888888888872993737E-3,
      4.1666666666666666609054E-2,
  };
  TF_ASSIGN_OR_RETURN(auto cos_x, EmitCos(prim_type, x));
  auto for_large_x = FAdd(cos_x, negative_one);

  auto xx = FMul(x, x);
  auto xxxx = FMul(xx, xx);
  TF_ASSIGN_OR_RETURN(auto poly, EvaluatePolynomial(type, xx, kCoeffs));
  auto for_small_x = FAdd(FMul(xxxx, poly), FMul(negative_half, xx));

  // (pi/4)^2 is approximately 0.61685
  return Select(FCmpOGT(xx, llvm::ConstantFP::get(type, 0.61685)), for_large_x,
                for_small_x);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitExp(
    PrimitiveType prim_type, llvm::Value* value, absl::string_view name) {
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::exp, {value},
                                      {value->getType()}, b_, name);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitExpm1(
    PrimitiveType prim_type, llvm::Value* value) {
  auto x = value;
  auto type = llvm_ir::PrimitiveTypeToIrType(prim_type, module_->getContext());
  auto one = llvm::ConstantFP::get(type, 1.0);
  auto half = llvm::ConstantFP::get(type, 0.5);
  auto zero = llvm::ConstantFP::get(type, 0.0);

  // expm1(x) == tanh(x/2)*(exp(x)+1)
  // x/2 can underflow, if it does we approximate expm1 with x.
  auto x_over_two = FMul(x, half);
  auto x_over_two_is_zero = FCmpOEQ(x_over_two, zero);
  auto abs_x =
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs, {x}, {type}, b_);
  // Use a naive exp(x)-1 calculation if |x| is > 0.5
  auto x_magnitude_is_large = FCmpOGT(abs_x, half);
  TF_ASSIGN_OR_RETURN(auto tanh_of_x_over_two, EmitTanh(prim_type, x_over_two));
  TF_ASSIGN_OR_RETURN(auto exp_of_x, EmitExp(prim_type, x, ""));
  auto exp_of_x_plus_one = FAdd(exp_of_x, one);
  auto exp_of_x_minus_one = FSub(exp_of_x, one);
  auto expm1_of_x = FMul(tanh_of_x_over_two, exp_of_x_plus_one);
  expm1_of_x = Select(x_magnitude_is_large, exp_of_x_minus_one, expm1_of_x);
  expm1_of_x = Select(x_over_two_is_zero, x, expm1_of_x);
  return expm1_of_x;
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitPow(
    PrimitiveType prim_type, llvm::Value* lhs, llvm::Value* rhs,
    absl::string_view name) {
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::pow, {lhs, rhs},
                                      {lhs->getType()}, b_, name);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitCbrt(
    PrimitiveType prim_type, llvm::Value* value) {
  auto type = llvm_ir::PrimitiveTypeToIrType(prim_type, module_->getContext());
  auto third = llvm::ConstantFP::get(type, 1.0 / 3.0);
  auto abs_value =
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs, {value}, {type}, b_);
  TF_ASSIGN_OR_RETURN(llvm::Value * abs_res,
                      EmitPow(prim_type, abs_value, third, ""));
  auto signed_res = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::copysign,
                                                 {abs_res, value}, {type}, b_);
  return signed_res;
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitAtan2(
    PrimitiveType prim_type, llvm::Value* lhs, llvm::Value* /*rhs*/,
    absl::string_view /*name*/) {
  return Unimplemented("atan2");
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitTanh(
    PrimitiveType prim_type, llvm::Value* value) {
  return Unimplemented("tanh");
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitErf(
    PrimitiveType prim_type, llvm::Value* value) {
  return Unimplemented("erf");
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitTan(
    PrimitiveType prim_type, llvm::Value* value) {
  auto sin_x = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::sin, {value},
                                            {value->getType()}, b_);
  auto cos_x = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::cos, {value},
                                            {value->getType()}, b_);
  return FDiv(sin_x, cos_x);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitReducePrecision(
    const HloInstruction* hlo, llvm::Value* x) {
  return EmitReducePrecisionIR(
      /*src_ty=*/hlo->operand(0)->shape().element_type(), x,
      /*dest_exponent_bits=*/hlo->exponent_bits(),
      /*dest_mantissa_bits=*/hlo->mantissa_bits(),
      /*quiet_nans=*/false, b_);
}

static llvm::Value* SaturateShiftIfNecessary(llvm::IRBuilderBase* b,
                                             llvm::Value* lhs, llvm::Value* rhs,
                                             llvm::Value* shift_result,
                                             bool saturate_to_sign_bit) {
  llvm::IntegerType* integer_type =
      llvm::cast<llvm::IntegerType>(lhs->getType());
  unsigned integer_bitsize = integer_type->getBitWidth();
  llvm::ConstantInt* integer_bitsize_constant =
      llvm::ConstantInt::get(integer_type, integer_bitsize);
  llvm::ConstantInt* zero = llvm::ConstantInt::get(integer_type, 0);
  llvm::ConstantInt* minus_one = llvm::ConstantInt::get(integer_type, -1);
  llvm::Value* saturated_value;
  if (saturate_to_sign_bit) {
    saturated_value =
        b->CreateSelect(b->CreateICmpSLT(lhs, zero), minus_one, zero);
  } else {
    saturated_value = zero;
  }
  llvm::Value* shift_amt_in_range =
      b->CreateICmpULT(rhs, integer_bitsize_constant, "shft.chk");
  return b->CreateSelect(shift_amt_in_range, shift_result, saturated_value);
}

llvm::Value* ElementalIrEmitter::GetOne(llvm::Type* type) {
  return llvm::ConstantInt::get(llvm::cast<llvm::IntegerType>(type), 1);
}

llvm::Value* ElementalIrEmitter::GetZero(llvm::Type* type) {
  return llvm::ConstantInt::get(llvm::cast<llvm::IntegerType>(type), 0);
}

llvm::Value* ElementalIrEmitter::GetIntSMin(llvm::Type* type) {
  auto* integer_type = llvm::cast<llvm::IntegerType>(type);
  return llvm::ConstantInt::get(integer_type, llvm::APInt::getSignedMinValue(
                                                  integer_type->getBitWidth()));
}

llvm::Value* ElementalIrEmitter::GetMinusOne(llvm::Type* type) {
  auto* integer_type = llvm::cast<llvm::IntegerType>(type);
  return llvm::ConstantInt::get(
      integer_type, llvm::APInt::getAllOnes(integer_type->getBitWidth()));
}

llvm::Value* ElementalIrEmitter::IsZero(llvm::Value* v) {
  return ICmpEQ(v, llvm::ConstantInt::get(v->getType(), 0));
}

llvm::Value* ElementalIrEmitter::IsIntMinDivisionOverflow(llvm::Value* lhs,
                                                          llvm::Value* rhs) {
  return And(ICmpEQ(lhs, GetIntSMin(lhs->getType())),
             ICmpEQ(rhs, GetMinusOne(rhs->getType())));
}

llvm::Value* ElementalIrEmitter::EmitIntegerDivide(llvm::Value* lhs,
                                                   llvm::Value* rhs,
                                                   bool is_signed) {
  // Integer division overflow behavior:
  //
  // X / 0 == -1
  // INT_SMIN /s -1 = INT_SMIN

  if (!is_signed) {
    llvm::Value* udiv_is_unsafe = IsZero(rhs);
    llvm::Value* safe_rhs = Select(udiv_is_unsafe, GetOne(lhs->getType()), rhs);
    llvm::Value* safe_div = UDiv(lhs, safe_rhs);
    return Select(udiv_is_unsafe, GetMinusOne(lhs->getType()), safe_div);
  }

  llvm::Value* has_zero_divisor = IsZero(rhs);
  llvm::Value* has_int_min_overflow = IsIntMinDivisionOverflow(lhs, rhs);
  llvm::Value* sdiv_is_unsafe = Or(has_int_min_overflow, has_zero_divisor);
  llvm::Value* safe_rhs = Select(sdiv_is_unsafe, GetOne(lhs->getType()), rhs);
  llvm::Value* safe_div = SDiv(lhs, safe_rhs);

  return Select(
      has_zero_divisor, GetMinusOne(lhs->getType()),
      Select(has_int_min_overflow, GetIntSMin(lhs->getType()), safe_div));
}

llvm::Value* ElementalIrEmitter::EmitIntegerRemainder(llvm::Value* lhs,
                                                      llvm::Value* rhs,
                                                      bool is_signed) {
  // Integer remainder overflow behavior:
  //
  // X % 0 == X
  // INT_SMIN %s -1 = 0

  if (!is_signed) {
    llvm::Value* urem_is_unsafe = IsZero(rhs);
    llvm::Value* safe_rhs = Select(urem_is_unsafe, GetOne(lhs->getType()), rhs);
    llvm::Value* safe_rem = URem(lhs, safe_rhs);
    return Select(urem_is_unsafe, lhs, safe_rem);
  }

  llvm::Value* has_zero_divisor = IsZero(rhs);
  llvm::Value* has_int_min_overflow = IsIntMinDivisionOverflow(lhs, rhs);
  llvm::Value* srem_is_unsafe = Or(has_int_min_overflow, has_zero_divisor);
  llvm::Value* safe_rhs = Select(srem_is_unsafe, GetOne(lhs->getType()), rhs);
  llvm::Value* safe_rem = SRem(lhs, safe_rhs);

  return Select(
      has_zero_divisor, lhs,
      Select(has_int_min_overflow, GetZero(lhs->getType()), safe_rem));
}

llvm::Value* ElementalIrEmitter::EmitIntegerPow(llvm::Value* base,
                                                llvm::Value* exponent,
                                                bool is_signed) {
  // Exponentiation by squaring:
  // https://en.wikipedia.org/wiki/Exponentiation_by_squaring;
  int bits = 6;  // Everything else would overflow for any exponent > 1, as 2^64
                 // is the larget possible exponent for a 64-bit integer, and
                 // that's 1 << 6.
  llvm::Value* accumulator = llvm::ConstantInt::get(base->getType(), 1);
  llvm::Value* one = llvm::ConstantInt::get(exponent->getType(), 1);
  llvm::Value* zero = llvm::ConstantInt::get(exponent->getType(), 0);
  llvm::Value* original_base = base;
  llvm::Value* original_exponent = exponent;

  // Unroll the loop at compile time.
  for (int i = 0; i < bits; i++) {
    accumulator =
        b_->CreateSelect(b_->CreateICmpEQ(b_->CreateAnd(exponent, one), one),
                         b_->CreateMul(accumulator, base), accumulator);
    base = b_->CreateMul(base, base);
    exponent = b_->CreateLShr(exponent, 1);
  }
  return b_->CreateSelect(
      b_->CreateICmpSGE(original_exponent, zero), accumulator,
      b_->CreateSelect(b_->CreateICmpEQ(original_base, one), one, zero));
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitPredBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
  // Per the reference interpreter, pred arithmetic should behave like
  // `int8_t(x) OP int8_t(y) != 0`.  For most permitted ops, we can just emit
  // the underlying i8 op to achieve this (e.g. kAnd, kOr, kXor, kMultiply).  In
  // the case of kAdd, we would need to insert a comparison instruction after
  // the addition, but it's both easier and faster to emit a bitwise or
  // instruction instead.
  //
  // For several of these ops, a faster bitwise implementation is available, but
  // LLVM is unlikely to be able to see it, since it gets IR that e.g. loads i8s
  // from memory, multiplies them, and writes the result back, without any
  // indication that the inputs were assumed to be 0 or 1.  So, just in case,
  // help it out by choosing the faster instruction to begin with.
  switch (op->opcode()) {
    case HloOpcode::kCompare:
    case HloOpcode::kXor:
      return EmitIntegerBinaryOp(op, lhs_value, rhs_value, false);

    // zext(i1 x) + zext(i1 y) != 0 === or(x, y)
    // max(zext(i1 x), zext(i1 y)) != 0 === or(x, y)
    case HloOpcode::kAdd:
    case HloOpcode::kMaximum:
    case HloOpcode::kOr:
      return Or(lhs_value, rhs_value);

    // zext(i1 x) * zext(i1 y) != 0 === and(x, y)
    // min(zext(i1 x), zext(i1 y)) != 0 === and(x, y)
    case HloOpcode::kMultiply:
    case HloOpcode::kMinimum:
    case HloOpcode::kAnd:
      return And(lhs_value, rhs_value);

    // These opcodes are rejected by shape-inference for PRED elements; calling
    // them out here serves more as documentation than a necessary check.
    case HloOpcode::kDivide:
    case HloOpcode::kRemainder:
    case HloOpcode::kPower:
    case HloOpcode::kSubtract:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
      return Internal("Invalid binary op '%s' for pred",
                      HloOpcodeString(op->opcode()));

    default:
      return Unimplemented("binary pred op '%s'",
                           HloOpcodeString(op->opcode()));
  }
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitIntegerBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value,
    bool is_signed) {
  switch (op->opcode()) {
    // TODO(jingyue): add the "nsw" attribute for signed types.
    case HloOpcode::kAdd:
      return Add(lhs_value, rhs_value);
    case HloOpcode::kSubtract:
      return Sub(lhs_value, rhs_value);
    case HloOpcode::kMultiply:
      return Mul(lhs_value, rhs_value);
    case HloOpcode::kDivide:
      return EmitIntegerDivide(lhs_value, rhs_value, is_signed);
    case HloOpcode::kRemainder:
      return EmitIntegerRemainder(lhs_value, rhs_value, is_signed);
    case HloOpcode::kCompare: {
      switch (op->comparison_direction()) {
        case ComparisonDirection::kEq:
          return llvm_ir::EmitComparison(llvm::CmpInst::ICMP_EQ, lhs_value,
                                         rhs_value, b_);
        case ComparisonDirection::kNe:
          return llvm_ir::EmitComparison(llvm::CmpInst::ICMP_NE, lhs_value,
                                         rhs_value, b_);
        case ComparisonDirection::kLt:
          return llvm_ir::EmitComparison(
              is_signed ? llvm::CmpInst::ICMP_SLT : llvm::CmpInst::ICMP_ULT,
              lhs_value, rhs_value, b_);
        case ComparisonDirection::kGt:
          return llvm_ir::EmitComparison(
              is_signed ? llvm::CmpInst::ICMP_SGT : llvm::CmpInst::ICMP_UGT,
              lhs_value, rhs_value, b_);
        case ComparisonDirection::kLe:
          return llvm_ir::EmitComparison(
              is_signed ? llvm::CmpInst::ICMP_SLE : llvm::CmpInst::ICMP_ULE,
              lhs_value, rhs_value, b_);
        case ComparisonDirection::kGe:
          return llvm_ir::EmitComparison(
              is_signed ? llvm::CmpInst::ICMP_SGE : llvm::CmpInst::ICMP_UGE,
              lhs_value, rhs_value, b_);
      }
    }
    case HloOpcode::kMinimum:
      return EmitIntegralMin(lhs_value, rhs_value, is_signed);
    case HloOpcode::kMaximum:
      return EmitIntegralMax(lhs_value, rhs_value, is_signed);
    case HloOpcode::kAnd:
      return And(lhs_value, rhs_value);
    case HloOpcode::kOr:
      return Or(lhs_value, rhs_value);
    case HloOpcode::kPower:
      return EmitIntegerPow(lhs_value, rhs_value, is_signed);
    case HloOpcode::kXor:
      return Xor(lhs_value, rhs_value);

    // Shifting out bits >= the number of bits in the type being shifted
    // produces a poison value in LLVM which is basically "deferred undefined
    // behavior" -- doing something observable with such a value precipitates
    // UB.  We replace the poison value with a constant to avoid this deferred
    // UB.
    case HloOpcode::kShiftRightArithmetic:
      return SaturateShiftIfNecessary(b_, lhs_value, rhs_value,
                                      AShr(lhs_value, rhs_value),
                                      /*saturate_to_sign_bit=*/true);
    case HloOpcode::kShiftLeft:
      return SaturateShiftIfNecessary(b_, lhs_value, rhs_value,
                                      Shl(lhs_value, rhs_value),
                                      /*saturate_to_sign_bit=*/false);
    case HloOpcode::kShiftRightLogical:
      return SaturateShiftIfNecessary(b_, lhs_value, rhs_value,
                                      LShr(lhs_value, rhs_value),
                                      /*saturate_to_sign_bit=*/false);
    default:
      return Unimplemented("binary integer op '%s'",
                           HloOpcodeString(op->opcode()));
  }
}

llvm::Value* ElementalIrEmitter::EmitIntegralMax(llvm::Value* lhs_value,
                                                 llvm::Value* rhs_value,
                                                 bool is_signed) {
  return Select(b_->CreateICmp(is_signed ? llvm::ICmpInst::ICMP_SGE
                                         : llvm::ICmpInst::ICMP_UGE,
                               lhs_value, rhs_value),
                lhs_value, rhs_value);
}

llvm::Value* ElementalIrEmitter::EmitIntegralMin(llvm::Value* lhs_value,
                                                 llvm::Value* rhs_value,
                                                 bool is_signed) {
  return Select(b_->CreateICmp(is_signed ? llvm::ICmpInst::ICMP_SLE
                                         : llvm::ICmpInst::ICMP_ULE,
                               lhs_value, rhs_value),
                lhs_value, rhs_value);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalSelect(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& index) {
  TF_ASSIGN_OR_RETURN(llvm::Value * pred_value,
                      operand_to_generator.at(hlo->operand(0))(index));
  TF_ASSIGN_OR_RETURN(llvm::Value * on_true_value,
                      operand_to_generator.at(hlo->operand(1))(index));
  TF_ASSIGN_OR_RETURN(llvm::Value * on_false_value,
                      operand_to_generator.at(hlo->operand(2))(index));
  return Select(Trunc(pred_value, b_->getInt1Ty()), on_true_value,
                on_false_value);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalClamp(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& index) {
  TF_ASSIGN_OR_RETURN(llvm::Value * min_value,
                      operand_to_generator.at(hlo->operand(0))(index));
  TF_ASSIGN_OR_RETURN(llvm::Value * arg_value,
                      operand_to_generator.at(hlo->operand(1))(index));
  TF_ASSIGN_OR_RETURN(llvm::Value * max_value,
                      operand_to_generator.at(hlo->operand(2))(index));
  PrimitiveType prim_type = hlo->shape().element_type();
  if (primitive_util::IsFloatingPointType(prim_type)) {
    return EmitFloatMin(max_value, EmitFloatMax(min_value, arg_value, ""), "");
  } else if (primitive_util::IsIntegralType(prim_type)) {
    bool is_signed = primitive_util::IsSignedIntegralType(prim_type);
    return EmitIntegralMin(
        max_value, EmitIntegralMax(min_value, arg_value, is_signed), is_signed);
  } else {
    return Unimplemented("Clamp unimplemented for %s",
                         PrimitiveType_Name(prim_type));
  }
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalConcatenate(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& source_index) {
  const int64_t concat_dim = hlo->dimensions(0);
  llvm::BasicBlock* init_block = b_->GetInsertBlock();

  llvm::BasicBlock* exit_block;
  if (b_->GetInsertPoint() != init_block->end()) {
    // Inserting into the middle.
    CHECK(init_block->getTerminator());
    exit_block =
        init_block->splitBasicBlock(b_->GetInsertPoint(), IrName(hlo, "merge"));
    init_block->getTerminator()->eraseFromParent();
  } else {
    // Inserting at the end.
    CHECK(!init_block->getTerminator());
    exit_block = llvm_ir::CreateBasicBlock(
        /*insert_before=*/nullptr, IrName(hlo, "merge"), b_);
  }

  llvm_ir::SetToFirstInsertPoint(exit_block, b_);
  llvm::PHINode* output =
      b_->CreatePHI(llvm_ir::PrimitiveTypeToIrType(hlo->shape().element_type(),
                                                   module_->getContext()),
                    hlo->operands().size());
  auto prior_insert_point = b_->GetInsertPoint();

  b_->SetInsertPoint(init_block);

  // Assign a unique id for each *different* operand, and count how often each
  // operand is used. If all operands are different, the usage count will be 1
  // for each operand.
  absl::flat_hash_map<const HloInstruction*, int64_t> to_unique_operand_id;
  std::vector<int64_t> operand_usage_count;
  for (const HloInstruction* operand : hlo->operands()) {
    if (to_unique_operand_id.contains(operand)) {
      ++operand_usage_count[to_unique_operand_id[operand]];
    } else {
      int64_t unique_operand_id = to_unique_operand_id.size();
      to_unique_operand_id[operand] = unique_operand_id;
      operand_usage_count.push_back(1);
    }
  }

  // To avoid that we emit the same operand more than once, we create one basic
  // block for each *different* operand with a PHI node for the different source
  // index inputs.
  std::vector<llvm::BasicBlock*> emit_operand_blocks(
      to_unique_operand_id.size(), nullptr);
  std::vector<llvm::PHINode*> source_index_phis(to_unique_operand_id.size(),
                                                nullptr);
  for (const HloInstruction* operand : hlo->operands()) {
    int64_t operand_id = to_unique_operand_id[operand];
    if (emit_operand_blocks[operand_id] != nullptr) {
      continue;
    }

    emit_operand_blocks[operand_id] = llvm_ir::CreateBasicBlock(
        exit_block, StrCat("concat_index_from_operand_id", operand_id), b_);
    auto saved_insert_point = b_->GetInsertPoint();
    llvm_ir::SetToFirstInsertPoint(emit_operand_blocks[operand_id], b_);
    source_index_phis[operand_id] =
        b_->CreatePHI(source_index.GetType(), operand_usage_count[operand_id]);
    std::vector<llvm::Value*> operand_multi_index = source_index.multidim();
    operand_multi_index[concat_dim] = b_->CreateNSWSub(
        operand_multi_index[concat_dim], source_index_phis[operand_id]);

    // Create the terminator of the block before calling operand generators,
    // because they require non-degenerate basic blocks.
    b_->SetInsertPoint(llvm::BranchInst::Create(
        exit_block, /*InsertAtEnd=*/emit_operand_blocks[operand_id]));
    llvm_ir::IrArray::Index operand_index(source_index.GetType());
    // If we are concatenating the fastest varying dimension, we can reuse the
    // linear index.
    if (source_index.linear() != nullptr &&
        operand->shape().dimensions_size() > 1 &&
        concat_dim == operand->shape().layout().minor_to_major(0)) {
      llvm::Value* linear_without_concat_dim = b_->CreateUDiv(
          source_index.linear(), source_index.GetConstantWithIndexType(
                                     hlo->shape().dimensions(concat_dim)));
      llvm::Value* adjusted_linear_base =
          b_->CreateMul(linear_without_concat_dim,
                        source_index.GetConstantWithIndexType(
                            operand->shape().dimensions(concat_dim)));
      llvm::Value* adjusted_linear =
          b_->CreateAdd(adjusted_linear_base, operand_multi_index[concat_dim]);
      operand_index = llvm_ir::IrArray::Index(
          adjusted_linear, operand_multi_index, operand->shape(), b_);
    } else {
      operand_index = llvm_ir::IrArray::Index(
          operand_multi_index, operand->shape(), source_index.GetType());
    }

    TF_ASSIGN_OR_RETURN(llvm::Value * value,
                        operand_to_generator.at(operand)(operand_index));
    output->addIncoming(value, b_->GetInsertBlock());
    b_->SetInsertPoint(init_block, saved_insert_point);
  }

  // We use bisection to select the input operand.
  int64_t current_offset = 0;

  // Offset for every operand.
  std::vector<std::pair<int64_t, const HloInstruction*>> cases;

  cases.reserve(hlo->operand_count());
  for (const HloInstruction* operand : hlo->operands()) {
    cases.emplace_back(current_offset, operand);
    current_offset += operand->shape().dimensions(concat_dim);
  }
  CHECK_EQ(current_offset, hlo->shape().dimensions(concat_dim));

  std::function<llvm::BasicBlock*(
      absl::Span<const std::pair<int64_t, const HloInstruction*>> operands)>
      emit_tree =
          [&](absl::Span<const std::pair<int64_t, const HloInstruction*>>
                  operands) {
            llvm::IRBuilder<>::InsertPointGuard guard(*b_);
            size_t mid = operands.size() / 2;
            const std::pair<int64_t, const HloInstruction*>& pivot =
                operands[mid];
            llvm::BasicBlock* block = llvm_ir::CreateBasicBlock(
                exit_block,
                absl::StrCat("concatenate.pivot.", pivot.first, "."), b_);
            b_->SetInsertPoint(block);

            // If there's only one element we're done. The range is contiguous
            // so we can just jump to the block for it.
            if (operands.size() == 1) {
              const std::pair<int64_t, const HloInstruction*>& operand =
                  operands.back();
              int64_t operand_id = to_unique_operand_id[operand.second];

              source_index_phis[operand_id]->addIncoming(
                  source_index.GetConstantWithIndexType(operand.first),
                  b_->GetInsertBlock());
              b_->CreateBr(emit_operand_blocks[operand_id]);
              return block;
            }

            // Take the middle element and recurse.
            llvm::Constant* pivot_const = llvm::ConstantInt::get(
                source_index[concat_dim]->getType(), pivot.first);
            llvm::Value* comp =
                b_->CreateICmpULT(source_index[concat_dim], pivot_const);

            llvm::BasicBlock* left_block = emit_tree(operands.subspan(0, mid));
            llvm::BasicBlock* right_block = emit_tree(operands.subspan(mid));

            b_->CreateCondBr(comp, left_block, right_block);
            return block;
          };

  Br(emit_tree(cases));

  b_->SetInsertPoint(exit_block, prior_insert_point);
  return output;
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalDynamicSlice(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& index) {
  // Emit IR to read dynamic start indices from hlo->operand(1).
  const HloInstruction* input_hlo = hlo->operand(0);
  const int64_t rank = input_hlo->shape().dimensions_size();
  // Use the same index type for all tensor accesses in the same kernel.
  llvm::Type* index_type = index.GetType();
  std::vector<llvm::Value*> slice_start_multi_index(rank);
  for (int64_t i = 0; i < rank; ++i) {
    auto index_typed_const = [&](uint64_t c) -> llvm::Constant* {
      return llvm::ConstantInt::get(index_type, c);
    };
    llvm_ir::IrArray::Index zero_index(index_type);
    TF_ASSIGN_OR_RETURN(
        llvm::Value * start_index_value,
        operand_to_generator.at(hlo->operand(1 + i))(zero_index));

    // Clamp the start index so that the sliced portion fits in the operand:
    // start_index = clamp(start_index, 0, operand_dim_size - output_dim_size)
    bool is_signed = ShapeUtil::ElementIsSigned(hlo->operand(1)->shape());
    start_index_value =
        b_->CreateIntCast(start_index_value, index_type, is_signed);
    int64_t largest_valid_start_index =
        input_hlo->shape().dimensions(i) - hlo->shape().dimensions(i);
    CHECK_GE(largest_valid_start_index, 0);

    start_index_value = EmitIntegralMin(
        index_typed_const(largest_valid_start_index),
        EmitIntegralMax(index_typed_const(0), start_index_value, is_signed),
        is_signed);

    start_index_value->setName(IrName(hlo, StrCat("start_idx", i)));
    slice_start_multi_index[i] = start_index_value;
  }

  std::vector<llvm::Value*> input_multi_index(rank);
  for (int64_t i = 0; i < rank; ++i) {
    // Emit IR which computes:
    //   input_index = start_index + offset_index
    input_multi_index[i] = Add(slice_start_multi_index[i], index[i]);
  }
  llvm_ir::IrArray::Index input_index(input_multi_index, input_hlo->shape(),
                                      index_type);
  return operand_to_generator.at(input_hlo)(input_index);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalGather(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& index) {
  const Shape& operand_shape = hlo->operand(0)->shape();
  const Shape& indices_shape = hlo->operand(1)->shape();
  const Shape& output_shape = hlo->shape();

  const GatherDimensionNumbers& dim_numbers = hlo->gather_dimension_numbers();

  const llvm_ir::ElementGenerator& operand_generator =
      operand_to_generator.at(hlo->operand(0));
  const llvm_ir::ElementGenerator& indices_generator =
      operand_to_generator.at(hlo->operand(1));

  llvm::Type* index_type = index.GetType();
  // This is the index into `operand` that holds the element we want to
  // generate.
  std::vector<llvm::Value*> operand_multi_index;

  // First copy in the window indices to operand_index. Also collect a mapping
  // from operand dimension to output window dimension. Elided window dimensions
  // map to -1.
  std::vector<int64_t> operand_to_output_dim(operand_shape.dimensions_size(),
                                             -1);
  for (int64_t i = 0, e = operand_shape.dimensions_size(),
               operand_index_dim = 0;
       i < e; i++) {
    if (absl::c_binary_search(dim_numbers.collapsed_slice_dims(), i)) {
      operand_multi_index.push_back(index.GetConstantWithIndexType(0));
    } else {
      int64_t output_window_dim = dim_numbers.offset_dims(operand_index_dim++);
      operand_to_output_dim[i] = output_window_dim;
      operand_multi_index.push_back(index[output_window_dim]);
    }
  }

  // This is the index of the index vector in the start_indices tensor.
  std::vector<llvm::Value*> gather_index_index_components;
  {
    for (int64_t i = 0, e = output_shape.dimensions_size(); i < e; i++) {
      if (!absl::c_binary_search(dim_numbers.offset_dims(), i)) {
        gather_index_index_components.push_back(index[i]);
      }
    }

    if (gather_index_index_components.size() !=
        indices_shape.dimensions_size()) {
      gather_index_index_components.insert(
          gather_index_index_components.begin() +
              dim_numbers.index_vector_dim(),
          nullptr);
    }
  }

  auto add_to_operand_index = [&](llvm::Value* index_component, int64_t dim) {
    auto index_component_type = index_component->getType();
    auto extended_type = index_component_type->getScalarSizeInBits() >=
                                 index_type->getScalarSizeInBits()
                             ? index_component_type
                             : index_type;
    // Possibly extend the value at the beginning to ensure clamping logic stays
    // in bounds.
    bool is_signed = ShapeUtil::ElementIsSigned(indices_shape);
    auto maybe_extended_index =
        index_component_type != extended_type
            ? b_->CreateIntCast(index_component, extended_type, is_signed)
            : index_component;
    int64_t operand_dim = dim_numbers.start_index_map(dim);
    int64_t output_dim = operand_to_output_dim[operand_dim];
    // If 'output_dim' is -1, it means 'operand_dim' is an elided window dim.
    // This means we set the iteration index to 0, so for the purpose of the
    // following calculations we can consider the output dimension size to be 1.
    int64_t output_dim_size =
        output_dim == -1 ? 1 : output_shape.dimensions(output_dim);
    int64_t largest_valid_start_index =
        operand_shape.dimensions(operand_dim) - output_dim_size;
    CHECK_GE(largest_valid_start_index, 0);

    // Clamp the gather index so that the gather region fits in the operand.
    // clamped_index =
    //     clamp(gather_dim_component_extended, 0, largest_valid_start_index);
    auto clamped_index = EmitIntegralMin(
        llvm::ConstantInt::get(extended_type, largest_valid_start_index),
        EmitIntegralMax(llvm::ConstantInt::get(extended_type, 0),
                        maybe_extended_index, is_signed),
        is_signed);
    // Truncate at the end to the optimized index size
    auto maybe_truncated_clamped_index = extended_type != index_type
                                             ? Trunc(clamped_index, index_type)
                                             : clamped_index;

    operand_multi_index[operand_dim] =
        Add(operand_multi_index[operand_dim], maybe_truncated_clamped_index);
  };

  if (indices_shape.dimensions_size() == dim_numbers.index_vector_dim()) {
    IrArray::Index gather_index_index(gather_index_index_components,
                                      indices_shape, index_type);
    TF_ASSIGN_OR_RETURN(llvm::Value * gather_dim_component,
                        indices_generator(gather_index_index));
    add_to_operand_index(gather_dim_component, 0);
  } else {
    int64_t index_vector_size =
        indices_shape.dimensions(dim_numbers.index_vector_dim());
    for (int64_t i = 0; i < index_vector_size; i++) {
      gather_index_index_components[dim_numbers.index_vector_dim()] =
          index.GetConstantWithIndexType(i);
      IrArray::Index gather_index_index(gather_index_index_components,
                                        indices_shape, index_type);
      TF_ASSIGN_OR_RETURN(llvm::Value * gather_dim_component,
                          indices_generator(gather_index_index));
      add_to_operand_index(gather_dim_component, i);
    }
  }
  IrArray::Index operand_index(operand_multi_index, operand_shape, index_type);
  return operand_generator(operand_index);
}

absl::StatusOr<llvm::Value*>
ElementalIrEmitter::EmitElementalDynamicUpdateSlice(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& index) {
  const HloInstruction* input_hlo = hlo->operand(0);
  const HloInstruction* update_hlo = hlo->operand(1);
  const HloInstruction* start_hlo = hlo->operand(2);
  // Calculate slice start/end indices.
  const int64_t rank = input_hlo->shape().dimensions_size();
  std::vector<llvm::Value*> slice_start_multi_index(rank);
  std::vector<llvm::Value*> slice_limit_multi_index(rank);
  // Slice intersection gathers (ANDs) conditions on all ranks for which
  // 'input' is set to 'update'
  llvm::Value* slice_intersection = b_->getTrue();

  for (int64_t i = 0; i < rank; ++i) {
    llvm::Type* index_type = index[0]->getType();
    auto index_typed_const = [&](uint64_t c) -> llvm::Constant* {
      return llvm::ConstantInt::get(index_type, c);
    };

    llvm_ir::IrArray::Index zero_index(index_type);
    TF_ASSIGN_OR_RETURN(
        llvm::Value * start_index_value,
        operand_to_generator.at(hlo->operand(2 + i))(zero_index));

    // Clamp the start index so that the update region fits in the operand.
    // start_index = clamp(start_index, 0, input_dim_size - update_dim_size)
    bool is_signed = ShapeUtil::ElementIsSigned(start_hlo->shape());
    start_index_value =
        b_->CreateIntCast(start_index_value, index_type, is_signed);
    llvm::Value* update_dim_size =
        index_typed_const(update_hlo->shape().dimensions(i));
    int64_t largest_valid_start_index =
        input_hlo->shape().dimensions(i) - update_hlo->shape().dimensions(i);
    CHECK_GE(largest_valid_start_index, 0);

    start_index_value = EmitIntegralMin(
        index_typed_const(largest_valid_start_index),
        EmitIntegralMax(index_typed_const(0), start_index_value, is_signed),
        is_signed);

    start_index_value->setName(IrName(hlo, StrCat("start_idx", i)));
    slice_start_multi_index[i] = start_index_value;
    slice_limit_multi_index[i] =
        Add(slice_start_multi_index[i], update_dim_size);

    slice_intersection =
        And(slice_intersection, ICmpSGE(index[i], slice_start_multi_index[i]),
            "slice_intersection");
    slice_intersection =
        And(slice_intersection, ICmpSLT(index[i], slice_limit_multi_index[i]),
            "slice_intersection");
  }

  // Emit:
  // if (slice_intersection) -> return data from 'update'.
  // else                    -> return data from 'input'.
  llvm::AllocaInst* ret_value_addr = llvm_ir::EmitAllocaAtFunctionEntry(
      llvm_ir::PrimitiveTypeToIrType(hlo->shape().element_type(),
                                     module_->getContext()),
      "ret_value_addr", b_);
  llvm_ir::LlvmIfData if_data =
      llvm_ir::EmitIfThenElse(slice_intersection, "slice_intersection", b_);

  // Handle true BB (return data from 'update')
  SetToFirstInsertPoint(if_data.true_block, b_);
  // Compute update index for intersection case.
  std::vector<llvm::Value*> update_multi_index(rank);
  for (int64_t i = 0; i < rank; ++i) {
    update_multi_index[i] = Sub(index[i], slice_start_multi_index[i]);
  }
  llvm_ir::IrArray::Index update_index(update_multi_index, update_hlo->shape(),
                                       index.GetType());
  TF_ASSIGN_OR_RETURN(llvm::Value * true_value,
                      operand_to_generator.at(update_hlo)(update_index));
  Store(true_value, ret_value_addr);

  // Handle false BB (return data from 'input')
  SetToFirstInsertPoint(if_data.false_block, b_);
  TF_ASSIGN_OR_RETURN(llvm::Value * false_value,
                      operand_to_generator.at(input_hlo)(index));
  Store(false_value, ret_value_addr);

  SetToFirstInsertPoint(if_data.after_block, b_);
  return Load(ret_value_addr->getAllocatedType(), ret_value_addr);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalPad(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& padded_index) {
  std::vector<llvm::Value*> multi_index = padded_index.multidim();
  llvm::Value* in_bounds = b_->getTrue();
  for (size_t i = 0; i < multi_index.size(); ++i) {
    auto index_typed_const = [=](int64_t n) {
      return padded_index.GetConstantWithIndexType(n);
    };
    const auto& pad_dim = hlo->padding_config().dimensions(i);
    multi_index[i] =
        Sub(multi_index[i], index_typed_const(pad_dim.edge_padding_low()));
    in_bounds = And(in_bounds, ICmpSGE(multi_index[i], index_typed_const(0)),
                    "in_bounds");
    in_bounds =
        And(in_bounds,
            ICmpEQ(index_typed_const(0),
                   URem(multi_index[i],
                        index_typed_const(pad_dim.interior_padding() + 1))),
            "in_bounds");
    multi_index[i] =
        SDiv(multi_index[i], index_typed_const(pad_dim.interior_padding() + 1));
    in_bounds =
        And(in_bounds,
            ICmpSLT(multi_index[i],
                    index_typed_const(hlo->operand(0)->shape().dimensions(i))),
            "in_bounds");
  }

  // if (in_bounds) {
  //   ret_value = operand0[index];  // source
  // } else {
  //   ret_value = *operand1;        // padding
  // }
  llvm::AllocaInst* ret_value_addr = llvm_ir::EmitAllocaAtFunctionEntry(
      llvm_ir::PrimitiveTypeToIrType(hlo->shape().element_type(),
                                     module_->getContext()),
      "pad_result_addr", b_);
  llvm_ir::LlvmIfData if_data =
      llvm_ir::EmitIfThenElse(in_bounds, "in_bounds", b_);
  SetToFirstInsertPoint(if_data.true_block, b_);
  llvm_ir::IrArray::Index index(multi_index, hlo->operand(0)->shape(),
                                padded_index.GetType());
  TF_ASSIGN_OR_RETURN(llvm::Value * operand_value,
                      operand_to_generator.at(hlo->operand(0))(index));
  Store(operand_value, ret_value_addr);

  SetToFirstInsertPoint(if_data.false_block, b_);
  TF_ASSIGN_OR_RETURN(llvm::Value * padding_value,
                      operand_to_generator.at(hlo->operand(1))(
                          IrArray::Index(index.GetType())));
  Store(padding_value, ret_value_addr);

  SetToFirstInsertPoint(if_data.after_block, b_);
  // Don't create phi(operand_value, padding_value) here, because invoking
  // operand_to_generator may create new basic blocks, making the parent
  // of operand_value or padding_value no longer a predecessor of
  // if_data.after_block.
  return Load(ret_value_addr->getAllocatedType(), ret_value_addr);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalDot(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& dot_result_index) {
  if (!algorithm_util::IsSupportedByElementalIrEmitter(
          hlo->precision_config().algorithm())) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Algorithm not supported by the ElementalIrEmitter: %s",
        PrecisionConfig::Algorithm_Name(hlo->precision_config().algorithm())));
  }
  const HloDotInstruction* dot = Cast<HloDotInstruction>(hlo);
  if (dot->sparse_operands()) {
    return Unimplemented("Sparse dot is supported by Triton emitter only.");
  }

  auto lhs_generator = operand_to_generator.at(hlo->operand(0));
  auto rhs_generator = operand_to_generator.at(hlo->operand(1));

  const DotDimensionNumbers& dim_numbers = hlo->dot_dimension_numbers();
  int64_t lhs_contracting_dim = dim_numbers.lhs_contracting_dimensions(0);
  int64_t rhs_contracting_dim = dim_numbers.rhs_contracting_dimensions(0);

  int64_t contracted_dim_size =
      hlo->operand(0)->shape().dimensions(lhs_contracting_dim);
  int64_t lhs_dims = hlo->operand(0)->shape().dimensions_size();
  int64_t rhs_dims = hlo->operand(1)->shape().dimensions_size();

  llvm::Type* index_type = dot_result_index.GetType();
  auto index_typed_const = [&](uint64_t c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_type, c);
  };

  std::unique_ptr<llvm_ir::ForLoop> inner_loop = llvm_ir::ForLoop::EmitForLoop(
      IrName(hlo, "inner"), index_typed_const(0),
      index_typed_const(contracted_dim_size), index_typed_const(1), b_);

  SetToFirstInsertPoint(inner_loop->GetPreheaderBasicBlock(), b_);
  PrimitiveType primitive_type = hlo->shape().element_type();
  llvm::Type* primitive_type_llvm =
      llvm_ir::PrimitiveTypeToIrType(primitive_type, module_->getContext());
  if (primitive_type == BF16) {
    primitive_type_llvm =
        llvm_ir::PrimitiveTypeToIrType(F32, module_->getContext());
  }
  llvm::AllocaInst* accumulator_alloca =
      llvm_ir::EmitAllocaAtFunctionEntry(primitive_type_llvm, "dot_acc", b_);
  Store(llvm::Constant::getNullValue(primitive_type_llvm), accumulator_alloca);

  SetToFirstInsertPoint(inner_loop->GetBodyBasicBlock(), b_);

  // This is the inner reduction loop for a dot operation that produces
  // one element in the output.  If the operands to the dot operation have
  // shapes [A,B,C,T] and [D,T,E], the result has a shape [A,B,C,D,E].
  // Given an output index [a,b,c,d,e] in the result, we compute:
  //   sum(lhs[a,b,c,t]*rhs[d,t,e] for t in [0, T))
  // Note that due to dot dimension numbers, the operands can have the
  // dimensions in a different order, so when constructing the index for the
  // operands, we need to take that into account. We can assume though that the
  // non-contracting dimensions are in the same order as in the output.

  std::vector<llvm::Value*> lhs_multi_index(lhs_dims, nullptr),
      rhs_multi_index(rhs_dims, nullptr);
  int64_t num_batch_dims = dim_numbers.lhs_batch_dimensions_size();
  DCHECK_EQ(num_batch_dims, dim_numbers.rhs_batch_dimensions_size());

  for (int64_t i = 0; i < num_batch_dims; ++i) {
    lhs_multi_index[dim_numbers.lhs_batch_dimensions(i)] = dot_result_index[i];
    rhs_multi_index[dim_numbers.rhs_batch_dimensions(i)] = dot_result_index[i];
  }
  lhs_multi_index[lhs_contracting_dim] = inner_loop->GetIndVarValue();
  rhs_multi_index[rhs_contracting_dim] = inner_loop->GetIndVarValue();
  // There are lhs_dims - 1 - num_batch_dims non-contracting dimensions for the
  // lhs operand. We can assume they have the same relative order as in the
  // output.
  for (int64_t i = num_batch_dims, j = 0; i < lhs_dims - 1; ++i, ++j) {
    // Skip the positions which have already been filled with contracting
    // dimension and batch dimensions.
    while (j < lhs_dims && lhs_multi_index[j] != nullptr) {
      ++j;
    }
    DCHECK_LT(j, lhs_dims);
    lhs_multi_index[j] = dot_result_index[i];
  }
  IrArray::Index lhs_index(lhs_multi_index, hlo->operand(0)->shape(),
                           index_type);

  // There are rhs_dims - 1 - num_batch_dims non-contracting dimensions for the
  // rhs operand. We can assume they have the same relative order as in the
  // output.
  DCHECK_EQ(hlo->shape().dimensions_size(),
            lhs_dims + rhs_dims - 2 - num_batch_dims);
  for (int64_t i = lhs_dims - 1, j = 0; i < hlo->shape().dimensions_size();
       ++i, ++j) {
    // Skip the positions which have already been filled with contracting
    // dimension and batch dimensions.
    while (j < rhs_dims && rhs_multi_index[j] != nullptr) {
      ++j;
    }
    DCHECK_LT(j, rhs_dims);
    rhs_multi_index[j] = dot_result_index[i];
  }
  IrArray::Index rhs_index(rhs_multi_index, hlo->operand(1)->shape(),
                           index_type);

  llvm::Value* current_accumulator =
      Load(accumulator_alloca->getAllocatedType(), accumulator_alloca);
  TF_ASSIGN_OR_RETURN(llvm::Value * lhs_value, lhs_generator(lhs_index));
  TF_ASSIGN_OR_RETURN(llvm::Value * rhs_value, rhs_generator(rhs_index));

  if (primitive_type == BF16) {
    lhs_value = b_->CreateFPExt(lhs_value, b_->getFloatTy());
    rhs_value = b_->CreateFPExt(rhs_value, b_->getFloatTy());
  }

  llvm::Value* next_accumulator =
      EmitMulAdd(lhs_value, rhs_value, current_accumulator,
                 primitive_type == BF16 ? F32 : primitive_type);
  Store(next_accumulator, accumulator_alloca);

  SetToFirstInsertPoint(inner_loop->GetExitBasicBlock(), b_);
  llvm::Value* result =
      Load(accumulator_alloca->getAllocatedType(), accumulator_alloca);

  return primitive_type == BF16 ? FPTrunc(result, b_->getBFloatTy()) : result;
}

absl::StatusOr<std::vector<llvm::Value*>>
ElementalIrEmitter::EmitThreadLocalCall(
    const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
    absl::string_view name, bool is_reducer) {
  return Unimplemented("EmitThreadLocalCall is not implemented");
}

llvm_ir::ElementGenerator ElementalIrEmitter::MakeElementGenerator(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator) {
  switch (hlo->opcode()) {
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kCeil:
    case HloOpcode::kClz:
    case HloOpcode::kConvert:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCos:
    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kReal:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kCbrt:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> absl::StatusOr<llvm::Value*> {
        TF_ASSIGN_OR_RETURN(llvm::Value * operand_value,
                            operand_to_generator.at(hlo->operand(0))(index));
        return EmitUnaryOp(hlo, operand_value);
      };
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kAtan2:
    case HloOpcode::kCompare:
    case HloOpcode::kComplex:
    case HloOpcode::kDivide:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSubtract:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> absl::StatusOr<llvm::Value*> {
        const HloInstruction* lhs = hlo->operand(0);
        const HloInstruction* rhs = hlo->operand(1);
        TF_ASSIGN_OR_RETURN(llvm::Value * lhs_value,
                            operand_to_generator.at(lhs)(index));
        TF_ASSIGN_OR_RETURN(llvm::Value * rhs_value,
                            operand_to_generator.at(rhs)(index));
        return EmitBinaryOp(hlo, lhs_value, rhs_value);
      };
    case HloOpcode::kSelect:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> absl::StatusOr<llvm::Value*> {
        return EmitElementalSelect(hlo, operand_to_generator, index);
      };
    case HloOpcode::kClamp:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> absl::StatusOr<llvm::Value*> {
        return EmitElementalClamp(hlo, operand_to_generator, index);
      };
    case HloOpcode::kReducePrecision:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> absl::StatusOr<llvm::Value*> {
        TF_ASSIGN_OR_RETURN(llvm::Value * operand_value,
                            operand_to_generator.at(hlo->operand(0))(index));
        return EmitReducePrecision(hlo, operand_value);
      };
    case HloOpcode::kConcatenate:
      return
          [this, hlo, &operand_to_generator](const IrArray::Index target_index)
              -> absl::StatusOr<llvm::Value*> {
            return EmitElementalConcatenate(hlo, operand_to_generator,
                                            target_index);
          };
    case HloOpcode::kReverse:
      return [this, hlo,
              &operand_to_generator](const IrArray::Index& target_index)
                 -> absl::StatusOr<llvm::Value*> {
        const HloInstruction* operand = hlo->operand(0);
        std::vector<llvm::Value*> source_multi_index = target_index.multidim();
        for (int64_t dim : hlo->dimensions()) {
          source_multi_index[dim] = Sub(target_index.GetConstantWithIndexType(
                                            hlo->shape().dimensions(dim) - 1),
                                        target_index[dim]);
        }
        llvm_ir::IrArray::Index source_index(
            source_multi_index, operand->shape(), target_index.GetType());
        return operand_to_generator.at(operand)(source_index);
      };
    case HloOpcode::kBroadcast:
      return
          [this, hlo, &operand_to_generator](const IrArray::Index& target_index)
              -> absl::StatusOr<llvm::Value*> {
            const HloInstruction* operand = hlo->operand(0);
            // The `dimensions` member of the broadcast instruction maps from
            // input dimensions to output dimensions.
            return operand_to_generator.at(operand)(
                target_index.SourceIndexOfBroadcast(
                    hlo->shape(), operand->shape(), hlo->dimensions(), b_));
          };
    case HloOpcode::kIota:
      return [this, hlo](const IrArray::Index& target_index)
                 -> absl::StatusOr<llvm::Value*> {
        auto* iota = Cast<HloIotaInstruction>(hlo);
        PrimitiveType element_type = iota->shape().element_type();
        IrArray::Index elem_index =
            iota->shape().dimensions_size() > 1
                ? target_index.SourceIndexOfBroadcast(
                      iota->shape(),
                      ShapeUtil::MakeShapeWithDescendingLayout(
                          element_type,
                          {iota->shape().dimensions(iota->iota_dimension())}),
                      {iota->iota_dimension()}, b_)
                : target_index;
        llvm::Value* elem_index_linear = elem_index.linear();
        if (elem_index_linear == nullptr) {
          std::vector<int64_t> iota_bound = {
              iota->shape().dimensions(iota->iota_dimension())};
          elem_index_linear = elem_index.Linearize(iota_bound, b_);
        }
        Shape component_shape =
            ShapeUtil::ElementIsComplex(iota->shape())
                ? ShapeUtil::ComplexComponentShape(iota->shape())
                : iota->shape();
        PrimitiveType component_element_type = component_shape.element_type();
        llvm::Value* iota_result;
        if (primitive_util::IsIntegralType(component_element_type)) {
          iota_result = b_->CreateIntCast(
              elem_index_linear,
              llvm_ir::PrimitiveTypeToIrType(component_element_type,
                                             module_->getContext()),
              /*isSigned=*/false);
        } else {
          TF_RET_CHECK(
              primitive_util::IsFloatingPointType(component_element_type))
              << component_element_type;
          llvm::Type* float_ir_type;
          if (component_element_type == F8E4M3FNUZ ||
              component_element_type == F8E5M2FNUZ) {
            float_ir_type =
                llvm_ir::PrimitiveTypeToIrType(F16, module_->getContext());
          } else {
            float_ir_type = llvm_ir::PrimitiveTypeToIrType(
                component_element_type, module_->getContext());
          }
          llvm::Value* float_val =
              b_->CreateUIToFP(elem_index_linear, float_ir_type);
          if (component_element_type == F8E4M3FNUZ ||
              component_element_type == F8E5M2FNUZ) {
            TF_ASSIGN_OR_RETURN(
                iota_result, EmitFloatingToF8fnuz(F16, float_val,
                                                  component_element_type, b_));
          } else {
            iota_result = float_val;
          }
        }
        if (ShapeUtil::ElementIsComplex(iota->shape())) {
          return EmitComposeComplex(iota, iota_result, nullptr);
        } else {
          return iota_result;
        }
      };
    case HloOpcode::kSlice:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> absl::StatusOr<llvm::Value*> {
        IrArray::Index sliced_index = index.SourceIndexOfSlice(
            /*operand_shape=*/hlo->operand(0)->shape(),
            /*starts=*/hlo->slice_starts(),
            /*strides=*/hlo->slice_strides(), /*builder=*/b_);
        return operand_to_generator.at(hlo->operand(0))(sliced_index);
      };
    case HloOpcode::kDynamicSlice:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> absl::StatusOr<llvm::Value*> {
        return EmitElementalDynamicSlice(hlo, operand_to_generator, index);
      };

    case HloOpcode::kGather:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> absl::StatusOr<llvm::Value*> {
        return EmitElementalGather(hlo, operand_to_generator, index);
      };
    case HloOpcode::kDynamicUpdateSlice:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> absl::StatusOr<llvm::Value*> {
        return EmitElementalDynamicUpdateSlice(hlo, operand_to_generator,
                                               index);
      };
    case HloOpcode::kBitcast:
      CHECK_EQ(ShapeUtil::ElementsIn(hlo->shape()),
               ShapeUtil::ElementsIn(hlo->operand(0)->shape()));
      return [this, hlo, &operand_to_generator](const IrArray::Index& index) {
        const HloInstruction* operand = hlo->operand(0);
        IrArray::Index operand_index = GetSourceIndexOfBitcast(index, hlo);
        return operand_to_generator.at(operand)(operand_index);
      };
    case HloOpcode::kReshape:
      CHECK_EQ(ShapeUtil::ElementsIn(hlo->shape()),
               ShapeUtil::ElementsIn(hlo->operand(0)->shape()));
      return [this, hlo, &operand_to_generator](const IrArray::Index& index) {
        const HloInstruction* operand = hlo->operand(0);
        return operand_to_generator.at(operand)(
            index.SourceIndexOfReshape(hlo->shape(), operand->shape(), b_));
      };
    case HloOpcode::kCopy:
      return [hlo, &operand_to_generator](const IrArray::Index& target_index)
                 -> absl::StatusOr<llvm::Value*> {
        IrArray::Index source_index(target_index.multidim(),
                                    hlo->operand(0)->shape(),
                                    target_index.GetType());
        TF_ASSIGN_OR_RETURN(
            llvm::Value * operand_value,
            operand_to_generator.at(hlo->operand(0))(source_index));
        return operand_value;
      };
    case HloOpcode::kTranspose:
      return [hlo, &operand_to_generator](const IrArray::Index& target_index) {
        return operand_to_generator.at(hlo->operand(0))(
            target_index.SourceIndexOfTranspose(
                hlo->shape(), hlo->operand(0)->shape(), hlo->dimensions()));
      };
    case HloOpcode::kPad:
      return
          [this, hlo, &operand_to_generator](const IrArray::Index& padded_index)
              -> absl::StatusOr<llvm::Value*> {
            return EmitElementalPad(hlo, operand_to_generator, padded_index);
          };

    case HloOpcode::kDot:
      return [this, hlo,
              &operand_to_generator](const IrArray::Index& dot_result_index)
                 -> absl::StatusOr<llvm::Value*> {
        return EmitElementalDot(hlo, operand_to_generator, dot_result_index);
      };
    case HloOpcode::kMap:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> absl::StatusOr<llvm::Value*> {
        std::vector<llvm::Value*> operands;
        for (int i = 0; i < hlo->operand_count(); i++) {
          TF_ASSIGN_OR_RETURN(llvm::Value * operand_value,
                              operand_to_generator.at(hlo->operand(i))(index));
          operands.push_back(operand_value);
        }
        return EmitElementalMap(Cast<HloMapInstruction>(hlo), operands);
      };
    case HloOpcode::kReduceWindow:
      return [this, hlo, &operand_to_generator](const IrArray::Index& index) {
        auto reduce_window_instr = Cast<HloReduceWindowInstruction>(hlo);
        std::vector<llvm_ir::ElementGenerator> input_generators;
        for (const HloInstruction* instr : reduce_window_instr->inputs()) {
          input_generators.push_back(operand_to_generator.at(instr));
        }

        std::vector<llvm_ir::ElementGenerator> initial_value_generators;
        for (const HloInstruction* instr : reduce_window_instr->init_values()) {
          initial_value_generators.push_back(operand_to_generator.at(instr));
        }
        return EmitElementalReduceWindow(
            Cast<HloReduceWindowInstruction>(hlo), std::move(input_generators),
            std::move(initial_value_generators), index);
      };
    case HloOpcode::kReduce:
      return [this, hlo, &operand_to_generator](const IrArray::Index& index) {
        auto reduce_instr = Cast<HloReduceInstruction>(hlo);
        std::vector<llvm_ir::ElementGenerator> input_generators;
        for (const HloInstruction* instr : reduce_instr->inputs()) {
          input_generators.push_back(operand_to_generator.at(instr));
        }

        std::vector<llvm_ir::ElementGenerator> initial_value_generators;
        for (const HloInstruction* instr : reduce_instr->init_values()) {
          initial_value_generators.push_back(operand_to_generator.at(instr));
        }
        return EmitElementalReduce(reduce_instr, std::move(input_generators),
                                   std::move(initial_value_generators), index);
      };
    case HloOpcode::kConvolution:
      return [this, hlo, &operand_to_generator](const IrArray::Index& index) {
        return EmitConvolution(hlo, operand_to_generator, index);
      };
    default:
      return [hlo](const IrArray::Index& index) {
        return Unimplemented("Unhandled opcode for elemental IR emission: %s",
                             HloOpcodeString(hlo->opcode()));
      };
  }
}

llvm::Value* ElementalIrEmitter::EmitExtractReal(llvm::Value* value) {
  return ExtractValue(value, {0});
}

llvm::Value* ElementalIrEmitter::EmitExtractImag(llvm::Value* value) {
  return ExtractValue(value, {1});
}

llvm::Value* ElementalIrEmitter::EmitComposeComplex(const HloInstruction* op,
                                                    llvm::Value* real,
                                                    llvm::Value* imag) {
  auto cplx_type = llvm_ir::PrimitiveTypeToIrType(op->shape().element_type(),
                                                  module_->getContext());
  auto complex =
      InsertValue(llvm::ConstantAggregateZero::get(cplx_type), real, {0});
  if (imag != nullptr) {
    complex = InsertValue(complex, imag, {1});
  }
  return complex;
}

llvm::Value* ElementalIrEmitter::EmitMulAdd(llvm::Value* lhs, llvm::Value* rhs,
                                            llvm::Value* accumulator,
                                            xla::PrimitiveType primitive_type) {
  if (primitive_util::IsComplexType(primitive_type)) {
    llvm::Value* product_real =
        FSub(FMul(EmitExtractReal(lhs), EmitExtractReal(rhs)),
             FMul(EmitExtractImag(lhs), EmitExtractImag(rhs)));
    llvm::Value* product_imag =
        FAdd(FMul(EmitExtractReal(lhs), EmitExtractImag(rhs)),
             FMul(EmitExtractImag(lhs), EmitExtractReal(rhs)));
    llvm::Value* next_accumulator = InsertValue(
        accumulator, FAdd(EmitExtractReal(accumulator), product_real), {0});
    return InsertValue(next_accumulator,
                       FAdd(EmitExtractImag(accumulator), product_imag), {1});
  } else if (primitive_util::IsFloatingPointType(primitive_type)) {
    llvm::FastMathFlags reassoc_flag = b_->getFastMathFlags();
    reassoc_flag.setAllowReassoc(true);

    llvm::Value* cast_mul = FPCast(FMul(lhs, rhs), accumulator->getType());
    return b_->CreateFAddFMF(accumulator, cast_mul, reassoc_flag);
  } else if (primitive_type == PRED) {
    return Or(accumulator, And(lhs, rhs));
  }
  return Add(accumulator, Mul(lhs, rhs));
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalMap(
    const HloMapInstruction* map_instr,
    absl::Span<llvm::Value* const> elemental_operands) {
  TF_ASSIGN_OR_RETURN(
      std::vector<llvm::Value*> values,
      EmitThreadLocalCall(*map_instr->to_apply(), elemental_operands,
                          llvm_ir::IrName(map_instr), /*is_reducer=*/false));
  CHECK_EQ(values.size(), 1);
  return values[0];
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalReduceWindow(
    const HloReduceWindowInstruction* reduce_window,
    std::vector<llvm_ir::ElementGenerator> input_generators,
    std::vector<llvm_ir::ElementGenerator> initial_value_generators,
    const llvm_ir::IrArray::Index& index) {
  // Pseudocode:
  // for each index I in output
  //   value = init_value
  //   for each index W in window
  //     for each dimension i from 0 to rank - 1
  //       (input index I)[i] = O[i] * stride[i] + W[i] - pad_low[i]
  //     if I in bounds of input
  //       value = function(value, input[I])
  //     output[O] = value
  int64_t input_count = reduce_window->input_count();
  std::vector<PrimitiveType> operand_element_types;
  std::vector<llvm::Type*> accum_types;
  std::vector<llvm::Value*> accum_ptrs;
  for (int64_t operand_index = 0; operand_index < input_count;
       ++operand_index) {
    auto operand = reduce_window->inputs()[operand_index];
    PrimitiveType operand_element_type = operand->shape().element_type();
    operand_element_types.push_back(operand_element_type);
    llvm::Type* llvm_type = llvm_ir::PrimitiveTypeToIrType(
        operand_element_type, module_->getContext());
    accum_types.push_back(llvm_type);
    llvm::Value* accum_ptr = llvm_ir::EmitAllocaAtFunctionEntry(
        llvm_ir::PrimitiveTypeToIrType(operand_element_type,
                                       module_->getContext()),
        "reduce_window_accum_ptr", b_);
    accum_ptrs.push_back(accum_ptr);
    {
      auto initial_value_generator = initial_value_generators[operand_index];
      TF_ASSIGN_OR_RETURN(
          llvm::Value* const init_value,
          initial_value_generator(llvm_ir::IrArray::Index(index.GetType())));
      Store(init_value, accum_ptr);
    }
  }

  llvm::Type* index_type = index.GetType();
  auto index_typed_const = [&](uint64_t c) -> llvm::Constant* {
    return index.GetConstantWithIndexType(c);
  };

  const Window& window = reduce_window->window();
  llvm_ir::ForLoopNest loops(IrName(reduce_window), b_, index_type);
  std::vector<int64_t> window_size;
  const auto& dimensions = window.dimensions();
  window_size.reserve(dimensions.size());
  for (const auto& dim : dimensions) {
    window_size.push_back(dim.size());
  }
  const IrArray::Index window_index = loops.AddLoopsForShape(
      ShapeUtil::MakeShape(operand_element_types[0], window_size), "window");
  CHECK_EQ(window_index.size(), index.size());

  SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), b_);

  std::vector<llvm::Value*> input_multi_index(index.size());
  llvm::Value* in_bounds = b_->getInt1(true);
  for (size_t i = 0; i < index.size(); ++i) {
    llvm::Value* stridden_index =
        NSWMul(index[i], index_typed_const(window.dimensions(i).stride()));
    input_multi_index[i] = NSWSub(
        NSWAdd(
            stridden_index,
            NSWMul(window_index[i],
                   index_typed_const(window.dimensions(i).window_dilation()))),
        index_typed_const(window.dimensions(i).padding_low()));

    // We need to verify that we are not in the dilated base area.
    llvm::Value* dilation_condition =
        ICmpEQ(SRem(input_multi_index[i],
                    index_typed_const(window.dimensions(i).base_dilation())),
               index_typed_const(0));
    in_bounds = And(in_bounds, dilation_condition);

    // Apply base dilation to the index.
    input_multi_index[i] =
        SDiv(input_multi_index[i],
             index_typed_const(window.dimensions(i).base_dilation()));

    // We must check whether 0 <= input_multi_index[i] < bound, as
    // otherwise we are in the pad and so can skip the computation. This
    // comparison is equivalent to the unsigned comparison
    // input_multi_index[i] < bound, as a negative value wraps to a large
    // positive value.
    in_bounds =
        And(in_bounds,
            ICmpULT(input_multi_index[i],
                    index_typed_const(
                        reduce_window->inputs()[0]->shape().dimensions(i))));
  }

  llvm_ir::LlvmIfData if_data =
      llvm_ir::EmitIfThenElse(in_bounds, "in_bounds", b_);
  SetToFirstInsertPoint(if_data.true_block, b_);

  // We are not in pad, so do the computation.
  std::vector<llvm::Value*> input_values(reduce_window->operand_count());
  IrArray::Index input_index(input_multi_index,
                             reduce_window->inputs()[0]->shape(), index_type);
  for (int64_t operand_idx = 0; operand_idx < input_count; ++operand_idx) {
    TF_ASSIGN_OR_RETURN(llvm::Value * input_value,
                        input_generators[operand_idx](input_index));
    input_values[input_count + operand_idx] = input_value;
    input_values[operand_idx] =
        Load(llvm::cast<llvm::AllocaInst>(accum_ptrs[operand_idx])
                 ->getAllocatedType(),
             accum_ptrs[operand_idx]);
  }
  TF_ASSIGN_OR_RETURN(std::vector<llvm::Value*> accum_values,
                      EmitThreadLocalCall(*reduce_window->to_apply(),
                                          input_values, "reducer_function",
                                          /*is_reducer=*/true));

  for (int64_t operand_idx = 0; operand_idx < accum_values.size();
       ++operand_idx) {
    Store(accum_values[operand_idx], accum_ptrs[operand_idx]);
  }

  SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), b_);
  return EmitAccumResult(accum_ptrs, accum_types,
                         reduce_window->shape().IsTuple());
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalReduce(
    const HloReduceInstruction* reduce,
    std::vector<llvm_ir::ElementGenerator> input_generators,
    std::vector<llvm_ir::ElementGenerator> initial_value_generators,
    const llvm_ir::IrArray::Index& index) {
  const Shape& out_shape = reduce->shape();
  bool is_variadic = !out_shape.IsArray();
  int accumulators_count = 1;
  if (is_variadic) {
    CHECK(out_shape.IsTuple());
    accumulators_count = out_shape.tuple_shapes_size();
  }

  absl::Span<const int64_t> reduced_dimensions(reduce->dimensions());

  std::vector<llvm::Value*> accumulator_addrs;
  std::vector<llvm::Type*> accumulator_types;
  llvm::Type* index_type = index.GetType();
  for (int i = 0; i < accumulators_count; i++) {
    const Shape& element_shape =
        is_variadic ? out_shape.tuple_shapes(i) : out_shape;
    PrimitiveType accumulator_type = element_shape.element_type();
    llvm::Type* accumulator_llvm_type =
        llvm_ir::PrimitiveTypeToIrType(accumulator_type, module_->getContext());
    accumulator_types.push_back(accumulator_llvm_type);

    // Initialize an accumulator with init_value.
    llvm::AllocaInst* accumulator_addr = llvm_ir::EmitAllocaAtFunctionEntry(
        accumulator_llvm_type, "accumulator_" + std::to_string(i), b());
    TF_ASSIGN_OR_RETURN(
        llvm::Value* const init_value,
        initial_value_generators[i](llvm_ir::IrArray::Index(index_type)));
    Store(init_value, accumulator_addr);
    accumulator_addrs.push_back(accumulator_addr);
  }

  // The enclosing loops go over all the target elements. Now we have to compute
  // the actual target element. For this, we build a new loop nest to iterate
  // over all the reduction dimensions in the argument.
  // AddLoopsForShapeOnDimensions will return an Index where induction Value*s
  // are placed for each dimension in dimensions, and all the rest are nullptrs.
  llvm_ir::ForLoopNest loops(IrName(reduce, "inner"), b(), index_type);
  const HloInstruction* arg = reduce->operand(0);
  std::vector<llvm::Value*> input_multi_index =
      loops.AddLoopsForShapeOnDimensions(arg->shape(), reduced_dimensions,
                                         "reduction_dim");

  SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), b());

  // Build a full index for the input argument, using input_multi_index as the
  // base. In input_multi_index only the reduction dimensions are filled in. We
  // fill in the rest of the dimensions with induction Value*s taken from
  // 'index' which iterates over the target array.  See the high-level
  // description in the XLA documentation for details.
  auto it = index.begin();

  for (auto& i : input_multi_index) {
    if (i == nullptr) {
      i = *it++;
    }
  }
  CHECK(index.end() == it);
  llvm_ir::IrArray::Index input_index(input_multi_index, arg->shape(),
                                      index_type);

  std::vector<llvm::Value*> reduction_operands;
  for (llvm::Value* accum : accumulator_addrs) {
    llvm::Value* accum_value =
        Load(llvm::cast<llvm::AllocaInst>(accum)->getAllocatedType(), accum);
    reduction_operands.push_back(accum_value);
  }

  for (int i = 0; i < accumulators_count; i++) {
    TF_ASSIGN_OR_RETURN(llvm::Value* const input_element,
                        input_generators[i](input_index));
    reduction_operands.push_back(input_element);
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<llvm::Value*> results,
      EmitThreadLocalCall(*reduce->to_apply(), reduction_operands,
                          "reduce_function", /*is_reducer=*/true));

  CHECK(results.size() == accumulators_count);
  for (int i = 0; i < accumulators_count; i++) {
    Store(results[i], accumulator_addrs[i]);
  }
  SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), b());
  return EmitAccumResult(accumulator_addrs, accumulator_types, is_variadic);
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitAccumResult(
    absl::Span<llvm::Value* const> accumulator_addrs,
    llvm::ArrayRef<llvm::Type*> accumulator_types, bool is_variadic) {
  TF_RET_CHECK(accumulator_addrs.size() == accumulator_types.size());
  if (is_variadic) {
    // Emit a structure, as that what the LoopEmitter expects.
    llvm::Value* returned_structure = llvm::UndefValue::get(
        llvm::StructType::get(b()->getContext(), accumulator_types));
    for (int64_t i = 0; i < accumulator_addrs.size(); i++) {
      llvm::Value* accumulator_value =
          Load(accumulator_types[i], accumulator_addrs[i]);
      returned_structure =
          b()->CreateInsertValue(returned_structure, accumulator_value, i);
    }
    return returned_structure;
  } else {
    CHECK_EQ(accumulator_addrs.size(), 1);
    return Load(accumulator_types[0], accumulator_addrs[0]);
  }
}

absl::StatusOr<llvm::Value*> ElementalIrEmitter::EmitConvolution(
    const HloInstruction* convolution,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& index) {
  TF_RET_CHECK(convolution->batch_group_count() == 1);
  const HloInstruction* lhs = convolution->operand(0);
  const auto& input_generator = operand_to_generator.at(lhs);
  const HloInstruction* rhs = convolution->operand(1);
  const auto& kernel_generator = operand_to_generator.at(rhs);
  const Window& window = convolution->window();

  const ConvolutionDimensionNumbers& dnums =
      convolution->convolution_dimension_numbers();
  int num_spatial_dims = dnums.output_spatial_dimensions_size();
  std::vector<llvm::Value*> output_spatial(num_spatial_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    output_spatial[i] = index[dnums.output_spatial_dimensions(i)];
  }
  llvm::Value* output_feature = index[dnums.output_feature_dimension()];
  llvm::Value* batch = index[dnums.output_batch_dimension()];

  // We will accumulate the products into this sum to calculate the output entry
  // at the given index.
  PrimitiveType lhs_element_type = lhs->shape().element_type();
  llvm::Type* lhs_llvm_type =
      llvm_ir::PrimitiveTypeToIrType(lhs_element_type, module_->getContext());
  // Upcast the accumulator to F32 from F16 for increased precision.
  llvm::Type* accumulator_type =
      lhs_element_type == F16 ? b_->getFloatTy() : lhs_llvm_type;
  llvm::AllocaInst* sum_address = llvm_ir::EmitAllocaAtFunctionEntry(
      accumulator_type, "convolution_sum_address", b_);
  llvm::Value* constant_zero = llvm::Constant::getNullValue(accumulator_type);
  Store(constant_zero, sum_address);

  llvm_ir::ForLoopNest loops(IrName(convolution, "inner"), b_);
  std::vector<llvm::Value*> kernel_spatial(num_spatial_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    kernel_spatial[i] =
        loops
            .AddLoop(
                0, rhs->shape().dimensions(dnums.kernel_spatial_dimensions(i)),
                absl::StrCat("k", i))
            ->GetIndVarValue();
  }
  const int64_t input_group_size =
      rhs->shape().dimensions(dnums.kernel_input_feature_dimension());
  const int64_t feature_group_count = convolution->feature_group_count();
  const int64_t output_group_size =
      rhs->shape().dimensions(dnums.kernel_output_feature_dimension()) /
      feature_group_count;
  llvm::Value* input_feature =
      loops.AddLoop(0, input_group_size, "iz")->GetIndVarValue();

  SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), b_);

  llvm::Value* group_id = SDiv(output_feature, b_->getInt64(output_group_size));
  llvm::Value* lhs_input_feature =
      NSWAdd(input_feature, NSWMul(group_id, b_->getInt64(input_group_size)));

  // Calculate the spatial index in the input array, taking striding, dilation
  // and padding into account. An index in the padding will be out of the bounds
  // of the array.
  const auto calculate_input_index = [this](llvm::Value* output_index,
                                            llvm::Value* kernel_index,
                                            const WindowDimension& window_dim) {
    llvm::Value* strided_index =
        NSWMul(output_index, b_->getInt64(window_dim.stride()));
    llvm::Value* dilated_kernel_index =
        NSWMul(kernel_index, b_->getInt64(window_dim.window_dilation()));
    return NSWSub(NSWAdd(strided_index, dilated_kernel_index),
                  b_->getInt64(window_dim.padding_low()));
  };
  std::vector<llvm::Value*> input_spatial(num_spatial_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    input_spatial[i] = calculate_input_index(
        output_spatial[i], kernel_spatial[i], window.dimensions(i));
  }

  // We need to check if 0 <= input dim < bound, as otherwise we are in the
  // padding so that we can skip the computation. That is equivalent to input
  // dim < bound as an *unsigned* comparison, since a negative value will wrap
  // to a large positive value. The input dim is dilated, so we need to dilate
  // the bound as well to match.

  // Also need to check that the input coordinates are not in one of the
  // holes created by base dilation.
  const auto not_in_hole = [&](llvm::Value* input_index,
                               int64_t base_dilation) {
    llvm::Value* remainder = SRem(input_index, b_->getInt64(base_dilation));
    return ICmpEQ(remainder, b_->getInt64(0));
  };

  llvm::Value* in_bounds_condition = b_->getInt1(true);
  for (int i = 0; i < num_spatial_dims; ++i) {
    llvm::ConstantInt* input_bound = b_->getInt64(window_util::DilatedBound(
        lhs->shape().dimensions(dnums.input_spatial_dimensions(i)),
        window.dimensions(i).base_dilation()));
    llvm::Value* dim_in_bound = ICmpULT(input_spatial[i], input_bound);
    llvm::Value* dim_not_in_hole =
        not_in_hole(input_spatial[i], window.dimensions(i).base_dilation());
    llvm::Value* dim_ok = And(dim_in_bound, dim_not_in_hole);
    in_bounds_condition = And(in_bounds_condition, dim_ok);
  }

  // Now we need to map the dilated base coordinates back to the actual
  // data indices on the lhs.
  const auto undilate = [&](llvm::Value* input_index, int64_t base_dilation) {
    return SDiv(input_index, b_->getInt64(base_dilation));
  };
  for (int i = 0; i < num_spatial_dims; ++i) {
    input_spatial[i] =
        undilate(input_spatial[i], window.dimensions(i).base_dilation());
  }

  llvm_ir::LlvmIfData if_data =
      llvm_ir::EmitIfThenElse(in_bounds_condition, "in-bounds", b_);
  SetToFirstInsertPoint(if_data.true_block, b_);

  // We are not in the padding, so carry out the computation.
  int num_dims = num_spatial_dims + 2;
  std::vector<llvm::Value*> input_multi_index(num_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    input_multi_index[dnums.input_spatial_dimensions(i)] = input_spatial[i];
  }
  input_multi_index[dnums.input_feature_dimension()] = lhs_input_feature;
  input_multi_index[dnums.input_batch_dimension()] = batch;

  std::vector<llvm::Value*> kernel_multi_index(num_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    kernel_multi_index[dnums.kernel_spatial_dimensions(i)] =
        window.dimensions(i).window_reversal()
            ? NSWSub(b_->getInt64(window.dimensions(i).size() - 1),
                     kernel_spatial[i])
            : kernel_spatial[i];
  }

  kernel_multi_index[dnums.kernel_input_feature_dimension()] = input_feature;
  kernel_multi_index[dnums.kernel_output_feature_dimension()] = output_feature;

  llvm_ir::IrArray::Index input_index(input_multi_index, lhs->shape(),
                                      b_->getInt64Ty());
  TF_ASSIGN_OR_RETURN(llvm::Value* const input_value,
                      input_generator(input_index));
  llvm_ir::IrArray::Index kernel_index(kernel_multi_index, rhs->shape(),
                                       b_->getInt64Ty());
  TF_ASSIGN_OR_RETURN(llvm::Value* const kernel_value,
                      kernel_generator(kernel_index));
  llvm::Value* sum =
      EmitMulAdd(input_value, kernel_value,
                 Load(sum_address->getAllocatedType(), sum_address),
                 convolution->shape().element_type());
  Store(sum, sum_address);

  SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), b_);
  return FPCast(Load(sum_address->getAllocatedType(), sum_address),
                lhs_llvm_type);
}

// Evaluate polynomial using Horner's method.
absl::StatusOr<llvm::Value*> ElementalIrEmitter::EvaluatePolynomial(
    llvm::Type* type, llvm::Value* x, absl::Span<const double> coefficients) {
  llvm::Value* poly = llvm::ConstantFP::get(type, 0.0);
  for (const double c : coefficients) {
    poly = FAdd(FMul(poly, x), llvm::ConstantFP::get(type, c));
  }
  return poly;
}

bool ElementalIrEmitter::fast_min_max() {
  LOG(FATAL) << "not implemented";
  return false;
}

}  // namespace xla
