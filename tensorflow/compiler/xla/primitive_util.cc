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

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace primitive_util {

int SignificandWidth(PrimitiveType type) {
  switch (type) {
    case F32:
      return std::numeric_limits<float>::digits;
    case F64:
      return std::numeric_limits<double>::digits;
    case BF16:
      return kBFloat16MantissaBits + 1;
    case F16:
      return 11;
    default:
      LOG(FATAL) << "Not a floating data type " << type;
  }
}

bool IsFloatingPointType(PrimitiveType type) {
  return type == F16 || type == F32 || type == F64 || type == BF16;
}

bool IsComplexType(PrimitiveType type) { return type == C64 || type == C128; }

bool IsSignedIntegralType(PrimitiveType type) {
  return type == S8 || type == S16 || type == S32 || type == S64;
}

bool IsUnsignedIntegralType(PrimitiveType type) {
  return type == U8 || type == U16 || type == U32 || type == U64;
}

bool IsIntegralType(PrimitiveType type) {
  return IsUnsignedIntegralType(type) || IsSignedIntegralType(type);
}

int BitWidth(PrimitiveType type) {
  switch (type) {
    case PRED:
      return 1;

    case S8:
    case U8:
      return 8;

    case S16:
    case U16:
    case F16:
    case BF16:
      return 16;

    case U32:
    case S32:
    case F32:
      return 32;

    case U64:
    case S64:
    case F64:
    case C64:
      return 64;

    case C128:
      return 128;

    case TUPLE:
      LOG(FATAL) << "TUPLE is an invalid type for BitWidth";

    case OPAQUE_TYPE:
      LOG(FATAL) << "OPAQUE_TYPE is an invalid type for BitWidth";

    default:
      LOG(FATAL) << "Unhandled primitive type " << type;
  }
}

xla::PrimitiveType UnsignedIntegralTypeForBitWidth(int64 src_bitwidth) {
  switch (src_bitwidth) {
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

bool IsArrayType(PrimitiveType primitive_type) {
  return primitive_type != PRIMITIVE_TYPE_INVALID && primitive_type != TUPLE &&
         primitive_type != OPAQUE_TYPE && primitive_type != TOKEN;
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
  const string& LowercaseName(PrimitiveType t) {
    return lowercase_name_[static_cast<int>(t)];
  }

 private:
  string lowercase_name_[PrimitiveType_ARRAYSIZE];
};

const string& LowercasePrimitiveTypeName(PrimitiveType s) {
  static auto* gen = new PrimitiveTypeNameGenerator();
  return gen->LowercaseName(s);
}

namespace {

// Returns a map from lower-case primitive type name to primitive type.
//
// Due to Postel's Law considerations, both "opaque" and "opaque_type" map to
// the xla::OPAQUE_TYPE enumerator.
const std::unordered_map<string, PrimitiveType>& GetPrimitiveTypeStringMap() {
  static std::unordered_map<string, PrimitiveType>* name_to_type = [] {
    static auto* map = new std::unordered_map<string, PrimitiveType>;
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
  auto found = map.find(string(name));
  if (found == map.end()) {
    return InvalidArgument("Invalid element type string: \"%s\".", name);
  }
  return found->second;
}

bool IsPrimitiveTypeName(absl::string_view name) {
  const auto& map = GetPrimitiveTypeStringMap();
  auto found = map.find(string(name));
  return found != map.end();
}

}  // namespace primitive_util
}  // namespace xla
