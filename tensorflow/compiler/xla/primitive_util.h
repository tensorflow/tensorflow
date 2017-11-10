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

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace primitive_util {

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
// XLA primitive type.
template <>
PrimitiveType NativeToPrimitiveType<bool>();

// Unsigned integer
template <>
PrimitiveType NativeToPrimitiveType<uint8>();

template <>
PrimitiveType NativeToPrimitiveType<uint16>();

template <>
PrimitiveType NativeToPrimitiveType<uint32>();

template <>
PrimitiveType NativeToPrimitiveType<uint64>();

// Signed integer
template <>
PrimitiveType NativeToPrimitiveType<int8>();

template <>
PrimitiveType NativeToPrimitiveType<int16>();

template <>
PrimitiveType NativeToPrimitiveType<int32>();

template <>
PrimitiveType NativeToPrimitiveType<int64>();

// Floating point
template <>
PrimitiveType NativeToPrimitiveType<float>();
template <>
PrimitiveType NativeToPrimitiveType<double>();
template <>
PrimitiveType NativeToPrimitiveType<half>();

// Complex
template <>
PrimitiveType NativeToPrimitiveType<complex64>();

bool IsFloatingPointType(PrimitiveType type);

bool IsComplexType(PrimitiveType type);

bool IsSignedIntegralType(PrimitiveType type);

bool IsUnsignedIntegralType(PrimitiveType type);

bool IsIntegralType(PrimitiveType type);

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

// Complex
template <>
struct PrimitiveTypeToNative<C64> {
  using type = complex64;
};
}  // namespace primitive_util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PRIMITIVE_UTIL_H_
