/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/primitive_util.h"

#include <limits>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {
namespace primitive_util {

int SignificandWidth(PrimitiveType type) {
  switch (type) {
    case F32:
      return std::numeric_limits<float>::digits;
    case F64:
      return std::numeric_limits<double>::digits;
    case BF16:
      return std::numeric_limits<bfloat16>::digits;
    case F16:
      return std::numeric_limits<half>::digits;
    case F8E5M2:
      return std::numeric_limits<tsl::float8_e5m2>::digits;
    case F8E4M3FN:
      return std::numeric_limits<tsl::float8_e4m3fn>::digits;
    case F8E4M3B11FNUZ:
      return std::numeric_limits<tsl::float8_e4m3b11>::digits;
    default:
      LOG(FATAL) << "Not a floating data type " << type;
  }
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
  // minimum exponent but the exponent of the first integer which overflows.
  switch (type) {
    case F32:
      return std::numeric_limits<float>::min_exponent;
    case F64:
      return std::numeric_limits<double>::min_exponent;
    case BF16:
      return std::numeric_limits<bfloat16>::min_exponent;
    case F16:
      return std::numeric_limits<half>::min_exponent;
    case F8E5M2:
      return std::numeric_limits<tsl::float8_e5m2>::min_exponent;
    case F8E4M3FN:
      return std::numeric_limits<tsl::float8_e4m3fn>::min_exponent;
    case F8E4M3B11FNUZ:
      return std::numeric_limits<tsl::float8_e4m3b11>::min_exponent;
    default:
      LOG(FATAL) << "Not a floating data type " << type;
  }
}

int OverflowExponent(PrimitiveType type) {
  // |std::numeric_limits<float>::max_exponent| is defined as: "Maximum positive
  // integer such that radix raised to the power one less than that integer is a
  // representable finite floating-point number." as such it does not actually
  // yield the maximum exponent but the exponent of the first integer which
  // overflows.
  switch (type) {
    case F32:
      return std::numeric_limits<float>::max_exponent;
    case F64:
      return std::numeric_limits<double>::max_exponent;
    case BF16:
      return std::numeric_limits<bfloat16>::max_exponent;
    case F16:
      return std::numeric_limits<half>::max_exponent;
    case F8E5M2:
      return std::numeric_limits<tsl::float8_e5m2>::max_exponent;
    case F8E4M3FN:
      return std::numeric_limits<tsl::float8_e4m3fn>::max_exponent;
    case F8E4M3B11FNUZ:
      return std::numeric_limits<tsl::float8_e4m3b11>::max_exponent;
    default:
      LOG(FATAL) << "Not a floating data type " << type;
  }
}

bool IsFloatingPointType(PrimitiveType type) {
  return type == F16 || type == F32 || type == F64 || type == BF16 ||
         type == F8E5M2 || type == F8E4M3FN || type == F8E4M3B11FNUZ;
}

bool IsComplexType(PrimitiveType type) { return type == C64 || type == C128; }

bool IsSignedIntegralType(PrimitiveType type) {
  return type == S4 || type == S8 || type == S16 || type == S32 || type == S64;
}

bool IsUnsignedIntegralType(PrimitiveType type) {
  return type == U4 || type == U8 || type == U16 || type == U32 || type == U64;
}

bool IsIntegralType(PrimitiveType type) {
  return IsUnsignedIntegralType(type) || IsSignedIntegralType(type);
}

xla::PrimitiveType UnsignedIntegralTypeForBitWidth(int64_t src_bitwidth) {
  switch (src_bitwidth) {
    case 4:
      return xla::U4;
    case 8:
      return xla::U8;
    case 16:
      return xla::U16;
    case 32:
      return xla::U32;
    case 64:
      return xla::U64;
    default:
      return xla::PRIMITIVE_TYPE_INVALID;
  }
}

xla::PrimitiveType SignedIntegralTypeForBitWidth(int64_t src_bitwidth) {
  switch (src_bitwidth) {
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

PrimitiveType ComplexComponentType(PrimitiveType complex_type) {
  switch (complex_type) {
    case C64:
      return F32;
    case C128:
      return F64;
    default:
      LOG(FATAL) << "Primitive type is not complex: "
                 << PrimitiveType_Name(complex_type);
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

StatusOr<PrimitiveType> StringToPrimitiveType(absl::string_view name) {
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
