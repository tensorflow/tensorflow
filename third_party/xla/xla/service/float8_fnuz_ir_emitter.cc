/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/float8_fnuz_ir_emitter.h"

#include <string>

#include "llvm/ADT/APFloat.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Intrinsics.h"
#include "xla/primitive_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace float8_fnuz_ir_emitter {

using primitive_util::BitWidth;
using primitive_util::ExponentBias;
using primitive_util::ExponentWidth;
using primitive_util::OverflowExponent;
using primitive_util::SignificandWidth;
using primitive_util::UnderflowExponent;

namespace {

absl::StatusOr<const llvm::fltSemantics*> PrimitiveTypeToAPFloatSemantics(
    PrimitiveType type) {
  switch (type) {
    case F8E3M4:
      return &llvm::APFloat::Float8E3M4();
    case F8E4M3:
      return &llvm::APFloat::Float8E4M3();
    case F8E4M3B11FNUZ:
      return &llvm::APFloat::Float8E4M3B11FNUZ();
    case F8E4M3FN:
      return &llvm::APFloat::Float8E4M3FN();
    case F8E4M3FNUZ:
      return &llvm::APFloat::Float8E4M3FNUZ();
    case F8E5M2:
      return &llvm::APFloat::Float8E5M2();
    case F8E5M2FNUZ:
      return &llvm::APFloat::Float8E5M2FNUZ();
    case BF16:
      return &llvm::APFloat::BFloat();
    case F16:
      return &llvm::APFloat::IEEEhalf();
    case F32:
      return &llvm::APFloat::IEEEsingle();
    case F64:
      return &llvm::APFloat::IEEEdouble();
    default:
      return Unimplemented(
          "PrimitiveTypeToAPFloatSemantics has no semantics for %s.",
          PrimitiveType_Name(type));
  }
}

absl::StatusOr<llvm::Type*> PrimitiveTypeToLLVMType(llvm::IRBuilderBase* b,
                                                    PrimitiveType type) {
  switch (type) {
    case F8E3M4:
    case F8E4M3:
    case F8E4M3B11FNUZ:
    case F8E4M3FN:
    case F8E4M3FNUZ:
    case F8E5M2:
    case F8E5M2FNUZ:
      return b->getInt8Ty();
    case BF16:
      return b->getBFloatTy();
    case F16:
      return b->getHalfTy();
    case F32:
      return b->getFloatTy();
    case F64:
      return b->getDoubleTy();
    default:
      return Unimplemented("PrimitiveTypeToLLVMType has no LLVM type for %s.",
                           PrimitiveType_Name(type));
  }
}

// Compute the maximum value in the input type that is a finite value when
// converted to the output type. This takes into account rounding. This
// supports floating point types, and assumes the input type is wider than
// the output type.
//
// The result is provided as a uint64_t containing the bit encoding of the
// maximum value.
absl::StatusOr<uint64_t> ComputeMaximumValue(PrimitiveType input_type,
                                             PrimitiveType output_type,
                                             llvm::IRBuilderBase* b) {
  // Sanity check inputs.
  TF_RET_CHECK(primitive_util::IsFloatingPointType(input_type));
  TF_RET_CHECK(primitive_util::IsFloatingPointType(output_type));
  TF_RET_CHECK(BitWidth(input_type) > BitWidth(output_type));

  TF_ASSIGN_OR_RETURN(auto output_semantics,
                      PrimitiveTypeToAPFloatSemantics(output_type));

  TF_ASSIGN_OR_RETURN(auto input_semantics,
                      PrimitiveTypeToAPFloatSemantics(input_type));

  // Compute the largest number of the output type and convert it to the input
  // type.
  bool losesInfo;
  llvm::APFloat largest_output_value =
      llvm::APFloat::getLargest(*output_semantics);
  largest_output_value.convert(
      *input_semantics, llvm::RoundingMode::NearestTiesToEven, &losesInfo);

  llvm::APInt maximum_value = largest_output_value.bitcastToAPInt();

  // The maximum value in the input type that converts to a finite value in the
  // output type has the suffix 0b0111... after the last 1 in the encoding.
  // This is the maximum input value that will round down to the maximum finite
  // output value.
  //
  // To find where to put that suffix, count the trailing zeros. Subtract 1
  // from the trailing zero count to ensure there is a 0 between the current
  // encoding and the new suffix.
  const int trailing_zeros = maximum_value.countTrailingZeros() - 1;

  // Create the 1s that will go in the suffix.
  const uint64_t lower_bits = (0x1ull << trailing_zeros) - 1;

  // Or the suffix into the maximum value.
  return maximum_value.getZExtValue() | lower_bits;
}

// Tests whether the input value can be represented in the output type as a
// finite value. This takes into account rounding.
absl::StatusOr<llvm::Value*> IsInputOutsideOutputRange(
    PrimitiveType input_type, llvm::Value* value, PrimitiveType output_type,
    llvm::IRBuilderBase* b) {
  const uint64_t shift = BitWidth(input_type) - 1;
  const uint64_t bit_mask = (0x1ull << shift) - 1;

  // Ignore the sign bit.
  llvm::Value* non_sign_bits = b->CreateAnd(value, bit_mask);

  TF_ASSIGN_OR_RETURN(uint64_t maximum_value,
                      ComputeMaximumValue(input_type, output_type, b));

  // Compare against the maximum value.
  llvm::Type* uint_type = b->getIntNTy(BitWidth(input_type));
  return b->CreateICmpUGT(non_sign_bits,
                          llvm::ConstantInt::get(uint_type, maximum_value));
}

llvm::Value* IsZero(PrimitiveType type, llvm::Value* value,
                    llvm::IRBuilderBase* b) {
  const uint64_t shift = BitWidth(type) - 1;
  const uint64_t bit_mask = (0x1ull << shift) - 1;

  // Assuming the input is finite, so we can ignore the sign bit.
  llvm::Value* non_sign_bits = b->CreateAnd(value, bit_mask);

  llvm::Type* uint_type = b->getIntNTy(BitWidth(type));
  return b->CreateICmpEQ(non_sign_bits,
                         llvm::ConstantInt::get(uint_type, 0x0u));
}

llvm::Value* IsNormalNumber(PrimitiveType type, llvm::Value* value,
                            llvm::IRBuilderBase* b) {
  const uint64_t width = ExponentWidth(type);
  const uint64_t position = SignificandWidth(type) - 1;
  const uint64_t exponent_bit_mask = ((0x1ull << width) - 0x1ull) << position;

  llvm::Value* exponent_bits = b->CreateAnd(value, exponent_bit_mask);
  llvm::Type* uint_type = b->getIntNTy(BitWidth(type));
  return b->CreateICmpNE(exponent_bits,
                         llvm::ConstantInt::get(uint_type, 0x0u));
}

llvm::Value* IsOutputNormal(PrimitiveType input_type, llvm::Value* exponent,
                            PrimitiveType output_type, llvm::IRBuilderBase* b) {
  const uint64_t denorm_exponent = UnderflowExponent(output_type) - 1;

  llvm::Type* uint_type = b->getIntNTy(BitWidth(input_type));
  return b->CreateICmpSGE(exponent,
                          llvm::ConstantInt::get(uint_type, denorm_exponent));
}

llvm::Value* Max(llvm::Type* type, llvm::Value* x, uint64_t y,
                 llvm::IRBuilderBase* b) {
  return b->CreateBinaryIntrinsic(llvm::Intrinsic::smax, x,
                                  llvm::ConstantInt::get(type, y));
}

llvm::Value* Min(llvm::Type* type, llvm::Value* x, uint64_t y,
                 llvm::IRBuilderBase* b) {
  return b->CreateBinaryIntrinsic(llvm::Intrinsic::smin, x,
                                  llvm::ConstantInt::get(type, y));
}

// Returns the sign bit of the input value shifted down to the least
// significant bit.
llvm::Value* ExtractSign(PrimitiveType type, llvm::Value* value,
                         bool preserve_signed_zero, llvm::IRBuilderBase* b) {
  const uint64_t shift = BitWidth(type) - 1;
  const uint64_t sign_bit_mask = 0x1ull << shift;

  llvm::Value* sign = b->CreateAnd(value, sign_bit_mask);

  llvm::Type* uint_type = b->getIntNTy(BitWidth(type));
  sign = b->CreateLShr(sign, llvm::ConstantInt::get(uint_type, shift));

  if (preserve_signed_zero) {
    return sign;
  }

  llvm::Value* is_zero_pred = IsZero(type, value, b);
  return b->CreateSelect(is_zero_pred, llvm::ConstantInt::get(uint_type, 0x0u),
                         sign);
}

// Returns the exponent of the input value shifted down to the least
// significant bits and without any bias.
llvm::Value* ExtractExponent(PrimitiveType type, llvm::Value* value,
                             llvm::IRBuilderBase* b) {
  const uint64_t shift = BitWidth(type) - 1;
  const uint64_t bit_mask = (0x1ull << shift) - 0x1ull;
  llvm::Type* uint_type = b->getIntNTy(BitWidth(type));

  // Mask out sign bit.
  llvm::Value* exponent = b->CreateAnd(value, bit_mask);

  // Shift the mantissa bits away, leaving the exponent.
  exponent = b->CreateLShr(
      exponent, llvm::ConstantInt::get(uint_type, SignificandWidth(type) - 1));

  // Subtract the exponent bias.
  exponent = b->CreateSub(
      exponent, llvm::ConstantInt::get(uint_type, ExponentBias(type)));

  // If the input number is not a normal number, return the subnormal exponent.
  llvm::Value* input_normal_pred = IsNormalNumber(type, value, b);
  return b->CreateSelect(
      input_normal_pred, exponent,
      llvm::ConstantInt::get(uint_type, UnderflowExponent(type) - 1));
}

// Returns the mantissa of the input value with all bits explicitly
// represented. For normal numbers, the implicit leading 1 is in the
// returned value.
llvm::Value* ExtractMantissa(PrimitiveType type, llvm::Value* value,
                             llvm::IRBuilderBase* b) {
  const uint64_t shift = SignificandWidth(type) - 1;
  const uint64_t mantissa_bit_mask = (0x1ull << shift) - 0x1ull;

  llvm::Value* mantissa = b->CreateAnd(value, mantissa_bit_mask);

  llvm::Value* input_normal_pred = IsNormalNumber(type, value, b);
  llvm::Value* mantissa_normal = b->CreateOr(mantissa, (0x1ull << shift));

  return b->CreateSelect(input_normal_pred, mantissa_normal, mantissa);
}

// Identifies the index of the last bit of the input that can be represented
// in the output type. The index starts from the least significant bit
// (index 0) to the most significant bit (bit n-1). This takes into account
// whether the input value is a normal number.
//
// Example 1:
// input_type  = F16
// output_type = F8E5M2FNUZ
// value = 0.00002664
//       = 0x1.BFp-16
//       = 0x01BF
//       = 0b0|00000|0110111111
//       = 0b0.0110111111 * 2^(-14)
// Given the input and output is a denorm, we are looking for the bit the
// corresponds to the smallest non-zero value. For F8E5M2FNUZ that's 2^(-17).
// ExtractMantissa(value) = 0b0000000110111111
//                                    ^- 2^(-17) is here at bit 7.
// result = LastMantissaBit(F16, 0.00002664, F8E5M2FNUZ, b) = 7
//
// Example 2:
// input_type  = BF16
// output_type = F8E4M3FNUZ
// value = 247.0
//       = 0x1.EEp7
//       = 0x4377
//       = 0b0|10000110|1110111
//       = 0b1.1110111 * 2^7
// Given the input and output is a normal number, we are looking for the bit
// the corresponds to the third bit of the mantissa.
// ExtractMantissa(value) = 0b0000000011110111
//                                       ^- third mantissa bit is at bit 4.
// result = LastMantissaBit(BF16, 247.0, F8E4M3FNUZ, b) = 4
absl::StatusOr<llvm::Value*> LastMantissaBit(PrimitiveType input_type,
                                             llvm::Value* value,
                                             PrimitiveType output_type,
                                             llvm::IRBuilderBase* b) {
  const int src_mantissa_bits = SignificandWidth(input_type) - 1;
  const int dest_mantissa_bits = SignificandWidth(output_type) - 1;
  llvm::Type* int_type = b->getIntNTy(BitWidth(input_type));

  llvm::Value* exponent = ExtractExponent(input_type, value, b);

  // The index when the input/output is normal.
  llvm::Value* last_bit_index =
      llvm::ConstantInt::get(int_type, src_mantissa_bits - dest_mantissa_bits);

  // Increase the index if the output will be denormal given the exponent.
  llvm::Value* denormal_shift = b->CreateSub(
      llvm::ConstantInt::get(int_type, UnderflowExponent(output_type) - 1),
      exponent);
  denormal_shift = Max(int_type, denormal_shift, 0, b);
  last_bit_index = b->CreateAdd(last_bit_index, denormal_shift);

  // Check the output type exponent bias is not greater than the input type by
  // more than 1.
  TF_RET_CHECK(ExponentBias(input_type) >= (ExponentBias(output_type) - 1));

  // The log_2(x) of the smallest denorm value. This gives us the exponent n
  // that produces that number. This corresponds to the encoding with a single
  // bit set in the last significant bit (0b0000...01).
  const int input_log_minimum =
      UnderflowExponent(input_type) - SignificandWidth(input_type);
  const int output_log_minimum =
      UnderflowExponent(output_type) - SignificandWidth(output_type);

  // Alternatively, the input might be a denorm. This directly computes the
  // last mantissa bit when the input is a denorm.  Suppose we have 2.664E-5
  // (0x1.BFp-16) as the input number. The bit encoding for this in F16 is:
  //   S|EEEEE|MMMMMMMMMM
  // 0b0|00000|0110111111 = 2^(-14) * 0b0.0110111111
  //
  // To cast this to F8E5M2FNUZ, we would find the smallest denorm encoding and
  // find the corresponding bit in the input.
  //   S|MMMMM|MM
  // 0b0|00000|01 = 2^(-15) * 0b0.01
  // We can see the "last" bit of an output denorm represents 2^(-17), so we
  // find that corresponding bit in the input. In this F16 example it is
  // highlighted here:
  //   S|EEEEE|MMMMMMMMMM
  // 0b0|00000|0110111111
  //             ^-- last mantissa bit
  llvm::Value* denorm_last_mantissa_bit =
      llvm::ConstantInt::get(int_type, output_log_minimum - input_log_minimum);

  // Select the last mantissa bit based on whether the input is a normal number.
  llvm::Value* normal_pred = IsNormalNumber(input_type, value, b);

  // For the purposes of this function, consider zero a normal number.
  normal_pred = b->CreateOr(normal_pred, IsZero(input_type, value, b));

  // Select the normal or denorm case.
  llvm::Value* last_mantissa_bit =
      b->CreateSelect(normal_pred, last_bit_index, denorm_last_mantissa_bit);

  // Ensure the last_mantissa_bit is a valid bit in the input mantissa.
  // last_mantissa_bit is allowed to be the "2s" bit of the exponent.
  // This means the maximum value is 0b10.000...0. This corresponds to the
  // maximum possible rounding.
  return Min(int_type, last_mantissa_bit, src_mantissa_bits + 1, b);
}

// Compute the rounding bias for round-to-nearest-even for the input value.
// This takes into account whether the input value is a normal number and
// whether it will map to a normal number in the output type.
absl::StatusOr<llvm::Value*> DynamicRoundingBias(PrimitiveType input_type,
                                                 llvm::Value* value,
                                                 PrimitiveType output_type,
                                                 llvm::IRBuilderBase* b) {
  llvm::Type* int_type = b->getIntNTy(BitWidth(input_type));

  // Find the bit position of the last mantissa bit.
  TF_ASSIGN_OR_RETURN(llvm::Value * shift,
                      LastMantissaBit(input_type, value, output_type, b));

  // Compute the mask to select that bit.
  llvm::Value* last_mantissa_bit_mask =
      b->CreateShl(llvm::ConstantInt::get(int_type, 0x1u), shift);

  // Given the mantissa bit mask, compute the rounding bias bits.
  llvm::Value* base_rounding_bias = b->CreateLShr(last_mantissa_bit_mask, 0x1u);
  base_rounding_bias =
      b->CreateSub(base_rounding_bias, llvm::ConstantInt::get(int_type, 0x1u));

  // Select the last mantissa bit, and shift it down to the lsb.
  llvm::Value* mantissa = ExtractMantissa(input_type, value, b);
  llvm::Value* x_last_mantissa_bit =
      b->CreateLShr(b->CreateAnd(mantissa, last_mantissa_bit_mask), shift);

  // Add the last mantissa lsb into the rounding bias.
  return b->CreateAdd(x_last_mantissa_bit, base_rounding_bias);
}

// Given an unbiased exponent and mantissa with no implicit bits, returns the
// mantissa for the output type. The exponent is expected to be unbiased and
// the mantissa should have all bits explicitly represented, including the
// normally implicit leading 1 for a normal number.
llvm::Value* BuildOutputMantissa(PrimitiveType input_type,
                                 llvm::Value* exponent, llvm::Value* mantissa,
                                 PrimitiveType output_type,
                                 llvm::IRBuilderBase* b) {
  llvm::Type* input_int_type = b->getIntNTy(BitWidth(input_type));

  // Count the number of leading zeros, excluding the bits that would contain
  // the exponent.
  llvm::Value* zero_count =
      b->CreateBinaryIntrinsic(llvm::Intrinsic::ctlz, mantissa,
                               llvm::ConstantInt::get(b->getInt1Ty(), 0x0u));
  zero_count = b->CreateSub(
      zero_count,
      llvm::ConstantInt::get(input_int_type, ExponentWidth(input_type)));

  // The amount to shift the normal mantissa down.
  llvm::Value* shift = b->CreateSub(
      llvm::ConstantInt::get(input_int_type, SignificandWidth(input_type) -
                                                 SignificandWidth(output_type)),
      zero_count);

  // Shift the mantissa into its "normal" position.
  mantissa = b->CreateLShr(mantissa, shift);
  exponent = b->CreateSub(exponent, zero_count);

  // Additional shifting required to account for the denorm exponent.
  shift = b->CreateSub(llvm::ConstantInt::get(
                           input_int_type, UnderflowExponent(output_type) - 1),
                       exponent);

  // Avoid shift > BitWidth(input_type) which is UB for lshr. This can happen
  // with a large negative input exponent. This will shift all the bits out,
  // which is equivalent.
  shift = Min(input_int_type, shift, BitWidth(input_type) - 1, b);
  llvm::Value* mantissa_denorm = b->CreateLShr(mantissa, shift);

  // Test whether the output will be a normal number.
  llvm::Value* output_normal_pred =
      IsOutputNormal(input_type, exponent, output_type, b);

  // Select the normal or subnormal mantissa.
  mantissa = b->CreateSelect(output_normal_pred, mantissa, mantissa_denorm);

  // Mask out any additional bits. This includes the now implicit leading 1.
  const uint64_t mantissa_bit_mask =
      (0x1ull << (SignificandWidth(output_type) - 1)) - 1;
  return b->CreateAnd(mantissa, mantissa_bit_mask);
}

// Given an unbiased exponent and mantissa with no implicit bits, returns the
// exponent for the output type. The result is shifted into the correct
// position.
llvm::Value* BuildOutputExponent(PrimitiveType input_type,
                                 llvm::Value* exponent, llvm::Value* mantissa,
                                 PrimitiveType output_type,
                                 llvm::IRBuilderBase* b) {
  llvm::Type* input_int_type = b->getIntNTy(BitWidth(input_type));

  // Count the number of leading zeros, excluding the bits that would contain
  // the exponent.
  llvm::Value* zero_count =
      b->CreateBinaryIntrinsic(llvm::Intrinsic::ctlz, mantissa,
                               llvm::ConstantInt::get(b->getInt1Ty(), 0x0u));
  zero_count = b->CreateSub(
      zero_count,
      llvm::ConstantInt::get(input_int_type, ExponentWidth(input_type)));

  // Lower the exponent value by the number of additional leading zeros.
  exponent = b->CreateSub(exponent, zero_count);

  // Check whether this would lead to a normal number output.
  llvm::Value* output_normal_pred =
      IsOutputNormal(input_type, exponent, output_type, b);

  // If this would lead to a subnormal output, use the subnormal exponent.
  exponent = b->CreateSelect(
      output_normal_pred, exponent,
      llvm::ConstantInt::get(input_int_type, -OverflowExponent(output_type)));

  // Bias the exponent.
  exponent = b->CreateAdd(
      exponent,
      llvm::ConstantInt::get(input_int_type, OverflowExponent(output_type)));

  // Shift the exponent into the appropriate position.
  return b->CreateShl(exponent, SignificandWidth(output_type) - 1);
}

// Returns the sign for the output type. The result is shifted into the correct
// position.
llvm::Value* BuildOutputSign(llvm::Value* sign, PrimitiveType output_type,
                             llvm::IRBuilderBase* b) {
  // Shift the sign bit into the msb.
  return b->CreateShl(sign, BitWidth(output_type) - 1);
}

absl::StatusOr<uint64_t> GetQNaN(PrimitiveType type) {
  TF_ASSIGN_OR_RETURN(auto semantics, PrimitiveTypeToAPFloatSemantics(type));

  return llvm::APFloat::getQNaN(*semantics).bitcastToAPInt().getZExtValue();
}
}  // namespace

absl::StatusOr<llvm::Value*> EmitFloatingToF8fnuz(PrimitiveType input_type,
                                                  llvm::Value* input_value,
                                                  PrimitiveType output_type,
                                                  llvm::IRBuilderBase* b) {
  // Sanity check for supported types.
  TF_RET_CHECK(input_type == BF16 || input_type == F16 || input_type == F32 ||
               input_type == F64);
  TF_RET_CHECK(output_type == F8E4M3FNUZ || output_type == F8E5M2FNUZ);

  llvm::IntegerType* input_int_type = b->getIntNTy(BitWidth(input_type));
  llvm::Value* input_uint = b->CreateBitCast(input_value, input_int_type);

  TF_ASSIGN_OR_RETURN(
      llvm::Value * out_of_range_pred,
      IsInputOutsideOutputRange(input_type, input_uint, output_type, b));
  // We may now assume there won't be any further overflow issues. They will be
  // handled in the final select.

  // Compute rounding bias for round-to-nearest with ties to even.
  TF_ASSIGN_OR_RETURN(
      llvm::Value * input_rounding_bias,
      DynamicRoundingBias(input_type, input_uint, output_type, b));

  // Apply the rounding bias to the input. This won't carry into the sign bit.
  llvm::Value* input_uint_rounded =
      b->CreateAdd(input_uint, input_rounding_bias);

  // The input value is broken down and in a canonical form. Appropriate
  // rounding has been applied, exponent is not biased, and there are no
  // implicit bits in the mantissa.
  llvm::Value* sign =
      ExtractSign(input_type, input_uint, /*preserve_signed_zero=*/false, b);
  llvm::Value* exponent = ExtractExponent(input_type, input_uint_rounded, b);
  llvm::Value* mantissa = ExtractMantissa(input_type, input_uint_rounded, b);

  // The component parts of the output value.
  llvm::Value* output_sign = BuildOutputSign(sign, output_type, b);
  llvm::Value* output_exponent =
      BuildOutputExponent(input_type, exponent, mantissa, output_type, b);
  llvm::Value* output_mantissa =
      BuildOutputMantissa(input_type, exponent, mantissa, output_type, b);

  // Bitwise or the output components together.
  llvm::Value* result = b->CreateOr(output_exponent, output_mantissa);

  // Check for output underflow before adding a sign bit. There's no -0 in
  // fnuz types.
  llvm::Value* is_zero_pred = IsZero(input_type, result, b);
  output_sign = b->CreateSelect(
      is_zero_pred, llvm::ConstantInt::get(input_int_type, 0x0u), output_sign);

  // Bitwise or the sign bit into the result.
  result = b->CreateOr(result, output_sign);

  // Truncate down to int8.
  result = b->CreateTrunc(result, b->getInt8Ty());

  // Select based on whether the value was in range.
  TF_ASSIGN_OR_RETURN(const uint64_t output_qnan, GetQNaN(output_type));
  return b->CreateSelect(out_of_range_pred,
                         llvm::ConstantInt::get(b->getInt8Ty(), output_qnan),
                         result);
}

absl::StatusOr<llvm::Value*> EmitF8fnuzToFloating(PrimitiveType input_type,
                                                  llvm::Value* f8_value,
                                                  PrimitiveType output_type,
                                                  llvm::IRBuilderBase* b,
                                                  llvm::Module* module) {
  // Sanity check for supported types.
  TF_RET_CHECK(input_type == F8E4M3FNUZ || input_type == F8E5M2FNUZ);
  TF_RET_CHECK(primitive_util::IsFloatingPointType(output_type));

  const int output_type_bit_width = BitWidth(output_type);
  llvm::IntegerType* output_int_type = b->getIntNTy(output_type_bit_width);

  llvm::ArrayType* result_lut_array_type =
      llvm::ArrayType::get(output_int_type, 128);

  const std::string lut_name = PrimitiveType_Name(input_type) + "To" +
                               PrimitiveType_Name(output_type) + "LUT";
  TF_ASSIGN_OR_RETURN(auto input_semantics,
                      PrimitiveTypeToAPFloatSemantics(input_type));
  TF_ASSIGN_OR_RETURN(auto output_semantics,
                      PrimitiveTypeToAPFloatSemantics(output_type));

  llvm::Constant* global_result_lut_array = module->getOrInsertGlobal(
      lut_name, result_lut_array_type, [&]() -> llvm::GlobalVariable* {
        // Since the function range is only 2^8 and symmetric on the sign bit,
        // this is implemented as a table lookup.
        llvm::Constant* result_lut[128];

        // Populate the table with values computed using llvm APFloat.
        for (uint8_t i = 0; i < 128; ++i) {
          llvm::APFloat value(*input_semantics, llvm::APInt(8, i));

          bool losesInfo;
          value.convert(*output_semantics, llvm::APFloat::rmNearestTiesToEven,
                        &losesInfo);

          result_lut[i] = llvm::ConstantInt::get(
              output_int_type, value.bitcastToAPInt().getZExtValue());
        }

        llvm::Constant* result_lut_array =
            llvm::ConstantArray::get(result_lut_array_type, result_lut);

        return new llvm::GlobalVariable(
            /*M=*/*module,
            /*Ty=*/result_lut_array_type,
            /*isConstant=*/true,
            /*Linkage=*/llvm::GlobalValue::PrivateLinkage,
            /*Initializer=*/result_lut_array,
            /*Name=*/lut_name);
      });

  // Check for NaN, since it's a special case.
  TF_ASSIGN_OR_RETURN(const uint64_t input_qnan, GetQNaN(input_type));
  llvm::Value* nan_pred = b->CreateICmpEQ(
      f8_value, llvm::ConstantInt::get(b->getInt8Ty(), input_qnan));

  // Extract the sign, which will be added back to the result of the table
  // lookup.
  llvm::Value* sign = b->CreateAnd(f8_value, 0x80);

  // The lower 7 bits used s the index for the table lookup.
  llvm::Value* f8_abs = b->CreateAnd(f8_value, 0x7F);

  // Fetch the value from the lookup table.
  llvm::Value* result_abs =
      b->CreateGEP(output_int_type, global_result_lut_array, f8_abs);
  result_abs = b->CreateLoad(output_int_type, result_abs);

  // Never output a negative zero.
  llvm::Value* is_output_zero_pred = IsZero(output_type, result_abs, b);
  sign = b->CreateSelect(is_output_zero_pred,
                         llvm::ConstantInt::get(b->getInt8Ty(), 0x0u), sign);

  // Bitwise or the sign bit back in.
  sign = b->CreateZExt(sign, output_int_type);
  sign = b->CreateShl(sign, output_type_bit_width - BitWidth(input_type));
  llvm::Value* result = b->CreateOr(sign, result_abs);

  // Bitcast to the output type.
  TF_ASSIGN_OR_RETURN(auto type, PrimitiveTypeToLLVMType(b, output_type));
  TF_ASSIGN_OR_RETURN(const uint64_t output_qnan, GetQNaN(output_type));
  return b->CreateBitCast(
      b->CreateSelect(nan_pred,
                      llvm::ConstantInt::get(output_int_type, output_qnan),
                      result),
      type);
}

}  // namespace float8_fnuz_ir_emitter
}  // namespace xla
