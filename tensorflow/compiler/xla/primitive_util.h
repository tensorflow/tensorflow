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

#include <type_traits>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/status_macros.h"
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
inline PrimitiveType NativeToPrimitiveType<uint8>() {
  return U8;
}

template <>
inline PrimitiveType NativeToPrimitiveType<uint16>() {
  return U16;
}

template <>
inline PrimitiveType NativeToPrimitiveType<uint32>() {
  return U32;
}

template <>
inline PrimitiveType NativeToPrimitiveType<uint64>() {
  return U64;
}

// Signed integer
template <>
inline PrimitiveType NativeToPrimitiveType<int8>() {
  return S8;
}

template <>
inline PrimitiveType NativeToPrimitiveType<int16>() {
  return S16;
}

template <>
inline PrimitiveType NativeToPrimitiveType<int32>() {
  return S32;
}

template <>
inline PrimitiveType NativeToPrimitiveType<int64>() {
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
bool IsArrayType(PrimitiveType primitive_type);

// Returns the number of bits in the representation for a given type.
int BitWidth(PrimitiveType type);

// Returns the number of bytes in the representation for a given type.
int ByteWidth(PrimitiveType type);

PrimitiveType UnsignedIntegralTypeForBitWidth(int64 src_bitwidth);

PrimitiveType SignedIntegralTypeForBitWidth(int64 src_bitwidth);

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
    return std::make_tuple(
        // Prefer floating point types with more range over other
        // floating-point types or non-floating point types.
        IsFloatingPointType(type) ? OverflowExponent(type) : -1,
        // Prefer floating point types with more precision over less precise
        // types.
        IsFloatingPointType(type) ? SignificandWidth(type) : -1,
        // Prefer wider types over narrower types.
        BitWidth(type),
        // Prefer signed integer types over unsigned integer types.
        IsSignedIntegralType(type));
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

// Returns true if a convert from from_type to to_type looses no precision.
inline bool CastPreservesValues(PrimitiveType from_type,
                                PrimitiveType to_type) {
  if (from_type == to_type) {
    return true;
  }
  switch (to_type) {
    case C128:
      if (from_type == F64) {
        return true;
      }
      ABSL_FALLTHROUGH_INTENDED;
    case F64:
      if (from_type == S32 || from_type == U32 || from_type == F32) {
        return true;
      }
      ABSL_FALLTHROUGH_INTENDED;
    case C64:
      if (from_type == F32) {
        return true;
      }
      ABSL_FALLTHROUGH_INTENDED;
    case F32:
      if (from_type == F16 || from_type == BF16 || from_type == S16 ||
          from_type == U16) {
        return true;
      }
      ABSL_FALLTHROUGH_INTENDED;
    case F16:
    case BF16:
      return from_type == U8 || from_type == S8 || from_type == PRED;
    case S64:
      if (from_type == S32 || from_type == U32) {
        return true;
      }
      ABSL_FALLTHROUGH_INTENDED;
    case S32:
      if (from_type == S16 || from_type == U16) {
        return true;
      }
      ABSL_FALLTHROUGH_INTENDED;
    case S16:
      if (from_type == S8 || from_type == U8) {
        return true;
      }
      ABSL_FALLTHROUGH_INTENDED;
    case S8:
      if (from_type == PRED) {
        return true;
      }
      ABSL_FALLTHROUGH_INTENDED;
    case PRED:
      return false;
    case U64:
      if (from_type == U32) {
        return true;
      }
      ABSL_FALLTHROUGH_INTENDED;
    case U32:
      if (from_type == U16) {
        return true;
      }
      ABSL_FALLTHROUGH_INTENDED;
    case U16:
      if (from_type == U8) {
        return true;
      }
      ABSL_FALLTHROUGH_INTENDED;
    case U8:
      return from_type == PRED;
    default:
      return false;
  }
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
  using type = uint8;
};

template <>
struct PrimitiveTypeToNative<U16> {
  using type = uint16;
};

template <>
struct PrimitiveTypeToNative<U32> {
  using type = uint32;
};

template <>
struct PrimitiveTypeToNative<U64> {
  using type = uint64;
};

// Signed integer
template <>
struct PrimitiveTypeToNative<S8> {
  using type = int8;
};

template <>
struct PrimitiveTypeToNative<S16> {
  using type = int16;
};

template <>
struct PrimitiveTypeToNative<S32> {
  using type = int32;
};

template <>
struct PrimitiveTypeToNative<S64> {
  using type = int64;
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
const string& LowercasePrimitiveTypeName(PrimitiveType s);

// Returns the PrimitiveType matching the given name. The given name is expected
// to be lower-case.
StatusOr<PrimitiveType> StringToPrimitiveType(absl::string_view name);

// Returns true if the given name is a primitive type string (lower-case).
bool IsPrimitiveTypeName(absl::string_view name);

}  // namespace primitive_util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PRIMITIVE_UTIL_H_
