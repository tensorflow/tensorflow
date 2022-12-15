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

// Utilities for dealing with XLA primitive types.

#ifndef TENSORFLOW_COMPILER_XLA_PRIMITIVE_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_PRIMITIVE_UTIL_H_

#include <string>
#include <tuple>
#include <type_traits>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace primitive_util {

// Returns the count of significand (mantissa) bits for float datatypes.
// For non-float datatypes, results in a LOG(FATAL).
int SignificandWidth(PrimitiveType type);

// Returns the count of exponent bits for float datatypes.
// For non-float datatypes, results in a LOG(FATAL).
int ExponentWidth(PrimitiveType type);

// Returns the exponent of the smallest number which cannot be represented.
// For non-float datatypes, results in a LOG(FATAL).
int OverflowExponent(PrimitiveType type);

// Returns the XLA primitive type (eg, F32) corresponding to the given
// template parameter native type (eg, float).
template <typename NativeT>
PrimitiveType NativeToPrimitiveType() {
  // Make the expression depend on the template parameter NativeT so
  // that this compile-time error only appears if this function is
  // instantiated with some concrete type that is not specialized
  // below.
  static_assert(!std::is_same<NativeT, NativeT>::value,
                "Cannot map native type to primitive type.");
  return PRIMITIVE_TYPE_INVALID;
}

// Declarations of specializations for each native type which correspond to a
// XLA primitive type.  As an optimization, these are declared inline in the
// header.
template <>
inline PrimitiveType NativeToPrimitiveType<bool>() {
  return PRED;
}

// Unsigned integer
template <>
inline PrimitiveType NativeToPrimitiveType<uint8_t>() {
  return U8;
}

template <>
inline PrimitiveType NativeToPrimitiveType<uint16_t>() {
  return U16;
}

template <>
inline PrimitiveType NativeToPrimitiveType<uint32_t>() {
  return U32;
}

template <>
inline PrimitiveType NativeToPrimitiveType<uint64_t>() {
  return U64;
}

// Signed integer
template <>
inline PrimitiveType NativeToPrimitiveType<int8_t>() {
  return S8;
}

template <>
inline PrimitiveType NativeToPrimitiveType<int16_t>() {
  return S16;
}

template <>
inline PrimitiveType NativeToPrimitiveType<int32_t>() {
  return S32;
}

template <>
inline PrimitiveType NativeToPrimitiveType<int64_t>() {
  return S64;
}

// Floating point
template <>
inline PrimitiveType NativeToPrimitiveType<float>() {
  return F32;
}

template <>
inline PrimitiveType NativeToPrimitiveType<double>() {
  return F64;
}

template <>
inline PrimitiveType NativeToPrimitiveType<half>() {
  return F16;
}

template <>
inline PrimitiveType NativeToPrimitiveType<bfloat16>() {
  return BF16;
}

// Complex
template <>
inline PrimitiveType NativeToPrimitiveType<complex64>() {
  return C64;
}

template <>
inline PrimitiveType NativeToPrimitiveType<complex128>() {
  return C128;
}

bool IsFloatingPointType(PrimitiveType type);

bool IsComplexType(PrimitiveType type);

bool IsSignedIntegralType(PrimitiveType type);

bool IsUnsignedIntegralType(PrimitiveType type);

bool IsIntegralType(PrimitiveType type);

// Returns true if values of the given primitive type are held in array shapes.
inline constexpr bool IsArrayType(PrimitiveType primitive_type) {
  return primitive_type != PRIMITIVE_TYPE_INVALID && primitive_type != TUPLE &&
         primitive_type != OPAQUE_TYPE && primitive_type != TOKEN;
}

// Returns the number of bits in the representation for a given type.
ABSL_ATTRIBUTE_ALWAYS_INLINE inline int BitWidth(PrimitiveType type) {
  switch (type) {
    case PRED:
      return 1;

    case S8:
    case U8:
    case F8E5M2:
    case F8E4M3FN:
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

// Returns the number of bytes in the representation for a given type.
ABSL_ATTRIBUTE_ALWAYS_INLINE inline int ByteWidth(PrimitiveType type) {
  switch (type) {
    case PRED:
      return 1;

    case S8:
    case U8:
    case F8E5M2:
    case F8E4M3FN:
      return 1;

    case S16:
    case U16:
    case F16:
    case BF16:
      return 2;

    case U32:
    case S32:
    case F32:
      return 4;

    case U64:
    case S64:
    case F64:
    case C64:
      return 8;

    case C128:
      return 16;

    case TUPLE:
      LOG(FATAL) << "TUPLE is an invalid type for ByteWidth";

    case OPAQUE_TYPE:
      LOG(FATAL) << "OPAQUE_TYPE is an invalid type for ByteWidth";

    default:
      LOG(FATAL) << "Unhandled primitive type " << type;
  }
}

PrimitiveType UnsignedIntegralTypeForBitWidth(int64_t src_bitwidth);

PrimitiveType SignedIntegralTypeForBitWidth(int64_t src_bitwidth);

// Returns the real, imag component type underlying the given complex type.
// LOG(FATAL)'s if complex_type is not complex.
PrimitiveType ComplexComponentType(PrimitiveType complex_type);

// Returns the higher-precision element type if a and b are both floating
// point types; otherwise, checks that they have the same element type
// and returns it.
inline PrimitiveType HigherPrecisionType(PrimitiveType a, PrimitiveType b) {
  // Returns a tuple where the elements are lexicographically ordered in terms
  // of importance.
  auto type_properties = [](PrimitiveType type) {
    auto component_type =
        IsComplexType(type) ? ComplexComponentType(type) : type;
    return std::make_tuple(
        // Prefer complex types over non-complex types.
        IsComplexType(type),
        // Prefer floating point types with more range over other
        // floating-point types or non-floating point types.
        IsFloatingPointType(component_type) ? OverflowExponent(component_type)
                                            : -1,
        // Prefer floating point types with more precision over less precise
        // types.
        IsFloatingPointType(component_type) ? SignificandWidth(component_type)
                                            : -1,
        // Prefer wider types over narrower types.
        BitWidth(component_type),
        // Prefer signed integer types over unsigned integer types.
        IsSignedIntegralType(component_type));
  };
  auto a_properties = type_properties(a);
  auto b_properties = type_properties(b);
  if (a_properties > b_properties) {
    return a;
  }
  if (b_properties > a_properties) {
    return b;
  }
  CHECK_EQ(a, b);
  return a;
}

// Returns true if a convert from from_type to to_type loses no precision.
inline bool CastPreservesValues(PrimitiveType from_type,
                                PrimitiveType to_type) {
  // * -> *
  if (from_type == to_type) {
    return true;
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
  if (primitive_util::IsComplexType(to_type)) {
    auto from_component_type =
        primitive_util::IsComplexType(from_type)
            ? primitive_util::ComplexComponentType(from_type)
            : from_type;
    auto to_component_type = primitive_util::ComplexComponentType(to_type);
    return CastPreservesValues(from_component_type, to_component_type);
  }
  // ~C -> C is not safe because it drops imaginary components.
  if (primitive_util::IsComplexType(from_type)) {
    return false;
  }
  // F -> F is safe if the exponent and significand are preserved.
  if (primitive_util::IsFloatingPointType(from_type) &&
      primitive_util::IsFloatingPointType(to_type)) {
    return primitive_util::SignificandWidth(from_type) <=
               primitive_util::SignificandWidth(to_type) &&
           primitive_util::ExponentWidth(from_type) <=
               primitive_util::ExponentWidth(to_type) &&
           primitive_util::OverflowExponent(from_type) <=
               primitive_util::OverflowExponent(to_type);
  }
  // F -> I is not safe because it drops fractional numbers.
  if (!primitive_util::IsIntegralType(from_type)) {
    return false;
  }
  // An n-bit unsigned integer takes on values from [0, 2^n - 1].
  // An n-bit signed integer takes on values from [-2^(n-1), 2^(n-1) - 1].
  // from_bits/to_bits considers the number of non-sign bits.
  const int from_bits = primitive_util::IsSignedIntegralType(from_type)
                            ? primitive_util::BitWidth(from_type) - 1
                            : primitive_util::BitWidth(from_type);
  const int to_bits = primitive_util::IsSignedIntegralType(to_type)
                          ? primitive_util::BitWidth(to_type) - 1
                          : primitive_util::BitWidth(to_type);
  // I -> F is safe if the integer can be represented exactly.
  if (primitive_util::IsFloatingPointType(to_type)) {
    // In both cases, we need to handle an exponent of n-1.
    // However, the significand needed to represent signed two's complement
    // numbers is smaller by one bit because it will only have a non-zero
    // trailing significand field when the exponent is smaller than n-1.
    return from_bits <= primitive_util::SignificandWidth(to_type) &&
           primitive_util::BitWidth(from_type) - 1 <
               primitive_util::OverflowExponent(to_type);
  }
  // S -> U is not safe because it drops negative numbers.
  if (primitive_util::IsSignedIntegralType(from_type) &&
      primitive_util::IsUnsignedIntegralType(to_type)) {
    return false;
  }
  // I -> I is safe if the integer can be represented exactly; we've already
  // ensured that signed to unsigned conversions won't happen here.
  CHECK(primitive_util::IsIntegralType(to_type));
  return from_bits <= to_bits;
}

// Returns the native type (eg, float) corresponding to the given template
// parameter XLA primitive type (eg, F32).
template <PrimitiveType>
struct PrimitiveTypeToNative;

// Declarations of specializations for each native type which correspond to a
// XLA primitive type.
template <>
struct PrimitiveTypeToNative<PRED> {
  using type = bool;
};

// Unsigned integer
template <>
struct PrimitiveTypeToNative<U8> {
  using type = uint8_t;
};

template <>
struct PrimitiveTypeToNative<U16> {
  using type = uint16_t;
};

template <>
struct PrimitiveTypeToNative<U32> {
  using type = uint32_t;
};

template <>
struct PrimitiveTypeToNative<U64> {
  using type = uint64_t;
};

// Signed integer
template <>
struct PrimitiveTypeToNative<S8> {
  using type = int8_t;
};

template <>
struct PrimitiveTypeToNative<S16> {
  using type = int16_t;
};

template <>
struct PrimitiveTypeToNative<S32> {
  using type = int32_t;
};

template <>
struct PrimitiveTypeToNative<S64> {
  using type = int64_t;
};

// Floating point
template <>
struct PrimitiveTypeToNative<F32> {
  using type = float;
};
template <>
struct PrimitiveTypeToNative<F64> {
  using type = double;
};
template <>
struct PrimitiveTypeToNative<F16> {
  using type = half;
};

template <>
struct PrimitiveTypeToNative<BF16> {
  using type = bfloat16;
};

// Complex
template <>
struct PrimitiveTypeToNative<C64> {
  using type = complex64;
};

template <>
struct PrimitiveTypeToNative<C128> {
  using type = complex128;
};

// Returns the lower-case name of the given primitive type.
const std::string& LowercasePrimitiveTypeName(PrimitiveType s);

// Returns the PrimitiveType matching the given name. The given name is expected
// to be lower-case.
StatusOr<PrimitiveType> StringToPrimitiveType(absl::string_view name);

// Returns true if the given name is a primitive type string (lower-case).
bool IsPrimitiveTypeName(absl::string_view name);

// Returns whether `type` can be expressed as an instance of T.
// For example,
//  IsCanonicalRepresentation<float>(F32)          // true
//  IsCanonicalRepresentation<xla::bfloat16>(BF16) // true
//  IsCanonicalRepresentation<int32_t>(S8)         // true, 8 <= 32
//  IsCanonicalRepresentation<uint16_t>(S16)       // false, unsigned.
template <typename T>
bool IsCanonicalRepresentation(PrimitiveType type) {
  switch (type) {
    case F16:
    case F32:
    case BF16:
    case F64:
    case C64:
    case C128:
      return NativeToPrimitiveType<T>() == type;
    case S8:
    case S16:
    case S32:
    case S64:
      return std::is_integral<T>::value && std::is_signed<T>::value &&
             ByteWidth(type) <= sizeof(T);
    case PRED:
    case U8:
    case U16:
    case U32:
    case U64:
      return std::is_integral<T>::value && std::is_unsigned<T>::value &&
             ByteWidth(type) <= sizeof(T);
    case TUPLE:
    case OPAQUE_TYPE:
    case TOKEN:
    case PRIMITIVE_TYPE_INVALID:
    case F8E5M2:
    case F8E4M3FN:
    case PrimitiveType_INT_MAX_SENTINEL_DO_NOT_USE_:
    case PrimitiveType_INT_MIN_SENTINEL_DO_NOT_USE_:
      return false;
  }
}

}  // namespace primitive_util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PRIMITIVE_UTIL_H_
