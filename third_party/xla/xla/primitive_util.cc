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

#include "xla/primitive_util.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/logging.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace primitive_util {

int SignificandWidth(PrimitiveType type) {
  return FloatingPointTypeSwitch(
      [&](auto constant_type) -> int {
        return std::numeric_limits<NativeTypeOf<constant_type>>::digits;
      },
      type);
}

int ExponentWidth(PrimitiveType type) {
  // Per the IEEE-754 standard: a floating point type is stored as a sign bit, a
  // biased exponent and a trailing significand field.
  int total_bit_width = BitWidth(type);
  // This field contains all bits in the significand other than the leading
  // digit which is implied by the exponent.
  int trailing_significand_field_width = SignificandWidth(type) - 1;
  // The sign is encoded with a single bit.
  int kSignBitWidth = 1;
  // The remaining bits are used for encoding the biased exponent.
  return total_bit_width - (trailing_significand_field_width + kSignBitWidth);
}

int UnderflowExponent(PrimitiveType type) {
  // |std::numeric_limits<float>::min_exponent| is defined as: "minimum negative
  // integer such that radix raised to the power one less than that integer is a
  // normalized floating-point number." as such it does not actually yield the
  // minimum exponent but one above the minimum exponent that a normalized
  // number can have.
  return FloatingPointTypeSwitch(
      [&](auto constant_type) -> int {
        return std::numeric_limits<NativeTypeOf<constant_type>>::min_exponent;
      },
      type);
}

int OverflowExponent(PrimitiveType type) {
  // |std::numeric_limits<float>::max_exponent| is defined as: "Maximum positive
  // integer such that radix raised to the power one less than that integer is a
  // representable finite floating-point number." as such it does not actually
  // yield the maximum exponent but the exponent of the first integer which
  // overflows.
  return FloatingPointTypeSwitch(
      [&](auto constant_type) -> int {
        return std::numeric_limits<NativeTypeOf<constant_type>>::max_exponent;
      },
      type);
}

int ExponentBias(PrimitiveType type) {
  return (1 - UnderflowExponent(type)) + 1;
}

bool HasInfinity(PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsFloatingPointType(type))) {
    return FloatingPointTypeSwitch(
        [&](auto constant_type) -> bool {
          return std::numeric_limits<NativeTypeOf<constant_type>>::has_infinity;
        },
        type);
  }
  return false;
}

bool HasNaN(PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsFloatingPointType(type))) {
    return FloatingPointTypeSwitch(
        [&](auto constant_type) -> bool {
          return std::numeric_limits<
              NativeTypeOf<constant_type>>::has_quiet_NaN;
        },
        type);
  }
  return false;
}

bool HasNegativeZero(PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsFloatingPointType(type))) {
    return FloatingPointTypeSwitch(
        [&](auto constant_type) -> bool {
          return has_negative_zero_v<NativeTypeOf<constant_type>>;
        },
        type);
  }
  return false;
}

xla::PrimitiveType SignedIntegralTypeForBitWidth(int64_t src_bitwidth) {
  switch (src_bitwidth) {
    case 2:
      return xla::S2;
    case 4:
      return xla::S4;
    case 8:
      return xla::S8;
    case 16:
      return xla::S16;
    case 32:
      return xla::S32;
    case 64:
      return xla::S64;
    default:
      return xla::PRIMITIVE_TYPE_INVALID;
  }
}

bool CastPreservesValues(const PrimitiveType from_type,
                         const PrimitiveType to_type) {
  // * -> *
  if (from_type == to_type) {
    return true;
  }
  // * -> F8E8M0FNU is not possible because zero cannot be represented.
  if (to_type == F8E8M0FNU) {
    return false;
  }
  // PRED -> *
  if (from_type == PRED) {
    return true;
  }
  // ~PRED -> PRED is not safe because it drops almost all numbers.
  if (to_type == PRED) {
    return false;
  }
  // * -> C is safe if the components of * and C can be safely converted.
  if (IsComplexType(to_type)) {
    auto from_component_type =
        IsComplexType(from_type) ? ComplexComponentType(from_type) : from_type;
    auto to_component_type = ComplexComponentType(to_type);
    return CastPreservesValues(from_component_type, to_component_type);
  }
  // ~C -> C is not safe because it drops imaginary components.
  if (IsComplexType(from_type)) {
    return false;
  }
  // F -> F is safe if the exponent/significand are preserved and `to_type`
  // preserves infinities/nans/unsigned zero in `from_type`.
  if (IsFloatingPointType(from_type) && IsFloatingPointType(to_type)) {
    return
        // Target mantissa should be large enough.
        SignificandWidth(from_type) <= SignificandWidth(to_type) &&
        // Target exponent should be large enough.
        ExponentWidth(from_type) <= ExponentWidth(to_type) &&
        // HasInfinity check.
        (!HasInfinity(from_type) || HasInfinity(to_type)) &&
        // HasNaN check.
        (!HasNaN(from_type) || HasNaN(to_type)) &&
        // HasNegativeZero check.
        (!HasNegativeZero(from_type) || HasNegativeZero(to_type)) &&
        // Minimum denormal should be representable by target type.
        (UnderflowExponent(from_type) - SignificandWidth(from_type)) >=
            (UnderflowExponent(to_type) - SignificandWidth(to_type)) &&
        // Maximum exponent may be larger with custom bias (e.g. F8E4M3B11FNUZ).
        OverflowExponent(from_type) <= OverflowExponent(to_type);
  }
  // F -> I is not safe because it drops fractional numbers.
  if (!IsIntegralType(from_type)) {
    return false;
  }
  // An n-bit unsigned integer takes on values from [0, 2^n - 1].
  // An n-bit signed integer takes on values from [-2^(n-1), 2^(n-1) - 1].
  // from_bits/to_bits considers the number of non-sign bits.
  const int from_bits = IsSignedIntegralType(from_type)
                            ? BitWidth(from_type) - 1
                            : BitWidth(from_type);
  const int to_bits =
      IsSignedIntegralType(to_type) ? BitWidth(to_type) - 1 : BitWidth(to_type);
  // I -> F is safe if the integer can be represented exactly.
  if (IsFloatingPointType(to_type)) {
    // In both cases, we need to handle an exponent of n-1.
    // However, the significand needed to represent signed two's complement
    // numbers is smaller by one bit because it will only have a non-zero
    // trailing significand field when the exponent is smaller than n-1.
    return from_bits <= SignificandWidth(to_type) &&
           BitWidth(from_type) - 1 < OverflowExponent(to_type);
  }
  // S -> U is not safe because it drops negative numbers.
  if (IsSignedIntegralType(from_type) && IsUnsignedIntegralType(to_type)) {
    return false;
  }
  // I -> I is safe if the integer can be represented exactly; we've already
  // ensured that signed to unsigned conversions won't happen here.
  CHECK(IsIntegralType(to_type));
  return from_bits <= to_bits;
}

// Class to memoize the computation of
//   absl::AsciiStrToLower(PrimitiveType_Name(p))
// for all PrimitiveType values "p"
//
// xla::OPAQUE_TYPE canonically maps to the string "opaque" -- the only reason
// it's called OPAQUE_TYPE is to avoid clashing with a windows.h macro.
class PrimitiveTypeNameGenerator {
 public:
  PrimitiveTypeNameGenerator() {
    for (size_t idx = 0; idx < std::size(lowercase_name_); ++idx) {
      PrimitiveType t = static_cast<PrimitiveType>(idx + PrimitiveType_MIN);
      if (t == OPAQUE_TYPE) {
        lowercase_name_[idx] = "opaque";
      } else if (PrimitiveType_IsValid(t)) {
        lowercase_name_[idx] = absl::AsciiStrToLower(PrimitiveType_Name(t));
      }
    }
  }
  const std::string& LowercaseName(PrimitiveType t) {
    CHECK_GE(t, PrimitiveType_MIN);
    CHECK_LE(t, PrimitiveType_MAX);
    CHECK(PrimitiveType_IsValid(t))
        << "Invalid PrimitiveType: " << static_cast<int>(t);
    return lowercase_name_[t - PrimitiveType_MIN];
  }

 private:
  std::string lowercase_name_[PrimitiveType_MAX - PrimitiveType_MIN + 1];
};

const std::string& LowercasePrimitiveTypeName(PrimitiveType s) {
  static auto* const gen = new PrimitiveTypeNameGenerator();
  return gen->LowercaseName(s);
}

namespace {

// Returns a map from lower-case primitive type name to primitive type.
//
// Due to Postel's Law considerations, both "opaque" and "opaque_type" map to
// the xla::OPAQUE_TYPE enumerator.
const absl::flat_hash_map<std::string, PrimitiveType>&
LowerCaseNameToPrimitiveType() {
  static absl::flat_hash_map<std::string, PrimitiveType>* const name_to_type =
      [] {
        static auto* const map =
            new absl::flat_hash_map<std::string, PrimitiveType>;
        for (int i = 0; i < PrimitiveType_ARRAYSIZE; i++) {
          if (PrimitiveType_IsValid(i) && i != PRIMITIVE_TYPE_INVALID) {
            auto value = static_cast<PrimitiveType>(i);
            (*map)[LowercasePrimitiveTypeName(value)] = value;
          }
        }
        (*map)["opaque"] = OPAQUE_TYPE;
        return map;
      }();
  return *name_to_type;
}

}  // namespace

absl::StatusOr<PrimitiveType> StringToPrimitiveType(
    absl::string_view lower_name) {
  const auto& map = LowerCaseNameToPrimitiveType();
  auto found = map.find(lower_name);
  if (found == map.end()) {
    return InvalidArgument("Invalid element type string: \"%s\".", lower_name);
  }
  return found->second;
}

bool IsPrimitiveTypeName(absl::string_view name) {
  const auto& map = LowerCaseNameToPrimitiveType();
  auto found = map.find(name);
  return found != map.end();
}

}  // namespace primitive_util
}  // namespace xla
