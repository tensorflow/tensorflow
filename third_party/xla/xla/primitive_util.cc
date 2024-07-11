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

#include <cstdint>
#include <limits>
#include <string>

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace primitive_util {

int SignificandWidth(PrimitiveType type) {
  return FloatingPointTypeSwitch<int>(
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
  return FloatingPointTypeSwitch<int>(
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
  return FloatingPointTypeSwitch<int>(
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
    return FloatingPointTypeSwitch<bool>(
        [&](auto constant_type) -> bool {
          return std::numeric_limits<NativeTypeOf<constant_type>>::has_infinity;
        },
        type);
  }
  return false;
}

bool HasNegativeZero(PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsFloatingPointType(type))) {
    return FloatingPointTypeSwitch<bool>(
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

// Class to memoize the computation of
//   absl::AsciiStrToLower(PrimitiveType_Name(p))
// for all PrimitiveType values "p"
//
// xla::OPAQUE_TYPE canonically maps to the string "opaque" -- the only reason
// it's called OPAQUE_TYPE is to avoid clashing with a windows.h macro.
class PrimitiveTypeNameGenerator {
 public:
  PrimitiveTypeNameGenerator() {
    for (int i = 0; i < PrimitiveType_ARRAYSIZE; i++) {
      if (i == static_cast<int>(OPAQUE_TYPE)) {
        lowercase_name_[i] = "opaque";
      } else if (PrimitiveType_IsValid(i)) {
        lowercase_name_[i] = absl::AsciiStrToLower(
            PrimitiveType_Name(static_cast<PrimitiveType>(i)));
      }
    }
  }
  const std::string& LowercaseName(PrimitiveType t) {
    CHECK_LT(t, PrimitiveType_ARRAYSIZE);
    return lowercase_name_[static_cast<int>(t)];
  }

 private:
  std::string lowercase_name_[PrimitiveType_ARRAYSIZE];
};

const std::string& LowercasePrimitiveTypeName(PrimitiveType s) {
  static auto* gen = new PrimitiveTypeNameGenerator();
  return gen->LowercaseName(s);
}

namespace {

// Returns a map from lower-case primitive type name to primitive type.
//
// Due to Postel's Law considerations, both "opaque" and "opaque_type" map to
// the xla::OPAQUE_TYPE enumerator.
const absl::flat_hash_map<std::string, PrimitiveType>&
GetPrimitiveTypeStringMap() {
  static absl::flat_hash_map<std::string, PrimitiveType>* name_to_type = [] {
    static auto* map = new absl::flat_hash_map<std::string, PrimitiveType>;
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

absl::StatusOr<PrimitiveType> StringToPrimitiveType(absl::string_view name) {
  const auto& map = GetPrimitiveTypeStringMap();
  auto found = map.find(name);
  if (found == map.end()) {
    return InvalidArgument("Invalid element type string: \"%s\".", name);
  }
  return found->second;
}

bool IsPrimitiveTypeName(absl::string_view name) {
  const auto& map = GetPrimitiveTypeStringMap();
  auto found = map.find(name);
  return found != map.end();
}

}  // namespace primitive_util
}  // namespace xla
