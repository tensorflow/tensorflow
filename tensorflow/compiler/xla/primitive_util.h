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

// The number of exponent bits in a BF16 value.
const int kBFloat16ExponentBits = 8;

// The number of mantissa bits in a BF16 value. There is an implicit leading
// 1, so there is an implicit additional bit of precision.
const int kBFloat16MantissaBits = 7;

// Returns the XLA primitive type (eg, F32) corresponding to the given
// template parameter native type (eg, float).
template <typename NativeT>
PrimitiveType NativeToPrimitiveType() {
  // Make the expression depend on the template parameter NativeT so
  // that this compile-time error only apperas if this function is
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

// Returns the real, imag component type underlying the given complex type.
// LOG(FATAL)'s if complex_type is not complex.
PrimitiveType ComplexComponentType(PrimitiveType complex_type);

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
