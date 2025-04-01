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

// Utilities for dealing with XLA primitive types.

#ifndef XLA_PRIMITIVE_UTIL_H_
#define XLA_PRIMITIVE_UTIL_H_

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/ml_dtypes.h"

namespace xla {
namespace primitive_util {

// Returns the count of significand (mantissa) bits for float datatypes.
// This includes the implicit leading mantissa bit. For example, returns 24 for
// F32. For non-float datatypes, results in a LOG(FATAL).
int SignificandWidth(PrimitiveType type);

// Returns the count of exponent bits for float datatypes. For example, returns
// 8 for F32. For non-float datatypes, results in a LOG(FATAL).
int ExponentWidth(PrimitiveType type);

// Returns the smallest integer n such that 2**(n-1) is a normalized number for
// the given float datatype. In other words, returns one plus the exponent of
// the smallest normalized number. For example, returns -125 for F32. For
// non-float datatypes, results in a LOG(FATAL).
int UnderflowExponent(PrimitiveType type);

// Returns the largest integer n such that 2**(n-1) is a finite number for the
// given float datatype. In other words, returns the smallest exponent that
// causes overflow. For example, returns 128 for F32. For non-float datatypes,
// results in a LOG(FATAL).
int OverflowExponent(PrimitiveType type);

// Returns the exponent bias of the given floating point type.
// For non-float datatypes, results in a LOG(FATAL).
int ExponentBias(PrimitiveType type);

// Returns whether the type has a value for infinity.
bool HasInfinity(PrimitiveType type);

// Returns whether the type has a value for NaN.
bool HasNaN(PrimitiveType type);

// Returns whether the type has a value for negative zero.
bool HasNegativeZero(PrimitiveType type);

// Returns the XLA primitive type (eg, F32) corresponding to the given
// template parameter native type (eg, float). Doesn't compile if the native
// type has no corresponding primitive type.
template <typename NativeT>
constexpr PrimitiveType NativeToPrimitiveType() {
  // Make the expression depend on the template parameter NativeT so
  // that this compile-time error only appears if this function is
  // instantiated with some concrete type that is not specialized
  // below.
  static_assert(!std::is_same<NativeT, NativeT>::value,
                "Cannot map native type to primitive type.");
  return PRIMITIVE_TYPE_INVALID;
}

// Declarations of specializations for each native type which correspond to a
// XLA primitive type.
template <>
constexpr PrimitiveType NativeToPrimitiveType<bool>() {
  return PRED;
}

// Unsigned integer
template <>
constexpr PrimitiveType NativeToPrimitiveType<u1>() {
  return U1;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<u2>() {
  return U2;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<u4>() {
  return U4;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<uint8_t>() {
  return U8;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<uint16_t>() {
  return U16;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<uint32_t>() {
  return U32;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<uint64_t>() {
  return U64;
}

// Signed integer
template <>
constexpr PrimitiveType NativeToPrimitiveType<s1>() {
  return S1;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<s2>() {
  return S2;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<s4>() {
  return S4;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<int8_t>() {
  return S8;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<int16_t>() {
  return S16;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<int32_t>() {
  return S32;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<int64_t>() {
  return S64;
}

// Floating point
template <>
constexpr PrimitiveType NativeToPrimitiveType<float>() {
  return F32;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<double>() {
  return F64;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<half>() {
  return F16;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<bfloat16>() {
  return BF16;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<tsl::float4_e2m1fn>() {
  return F4E2M1FN;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<tsl::float8_e5m2>() {
  return F8E5M2;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<tsl::float8_e4m3>() {
  return F8E4M3;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<tsl::float8_e4m3fn>() {
  return F8E4M3FN;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<tsl::float8_e4m3b11fnuz>() {
  return F8E4M3B11FNUZ;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<tsl::float8_e5m2fnuz>() {
  return F8E5M2FNUZ;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<tsl::float8_e4m3fnuz>() {
  return F8E4M3FNUZ;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<tsl::float8_e3m4>() {
  return F8E3M4;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<tsl::float8_e8m0fnu>() {
  return F8E8M0FNU;
}

// Complex
template <>
constexpr PrimitiveType NativeToPrimitiveType<complex64>() {
  return C64;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<complex128>() {
  return C128;
}

// PrimitiveTypeToNative<k>::type is an alias for the native type of the given
// primitive type, and is undefined for primitive types that do not
// have a corresponding native type. E.g. PrimitiveTypeToNative<F32>::type is
// float.
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
struct PrimitiveTypeToNative<U1> {
  using type = u1;
};

template <>
struct PrimitiveTypeToNative<U2> {
  using type = u2;
};

template <>
struct PrimitiveTypeToNative<U4> {
  using type = u4;
};

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
struct PrimitiveTypeToNative<S1> {
  using type = s1;
};

template <>
struct PrimitiveTypeToNative<S2> {
  using type = s2;
};

template <>
struct PrimitiveTypeToNative<S4> {
  using type = s4;
};

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

template <>
struct PrimitiveTypeToNative<F4E2M1FN> {
  using type = tsl::float4_e2m1fn;
};

template <>
struct PrimitiveTypeToNative<F8E5M2> {
  using type = tsl::float8_e5m2;
};

template <>
struct PrimitiveTypeToNative<F8E4M3> {
  using type = tsl::float8_e4m3;
};

template <>
struct PrimitiveTypeToNative<F8E4M3FN> {
  using type = tsl::float8_e4m3fn;
};

template <>
struct PrimitiveTypeToNative<F8E4M3B11FNUZ> {
  using type = tsl::float8_e4m3b11fnuz;
};

template <>
struct PrimitiveTypeToNative<F8E5M2FNUZ> {
  using type = tsl::float8_e5m2fnuz;
};

template <>
struct PrimitiveTypeToNative<F8E4M3FNUZ> {
  using type = tsl::float8_e4m3fnuz;
};

template <>
struct PrimitiveTypeToNative<F8E3M4> {
  using type = tsl::float8_e3m4;
};

template <>
struct PrimitiveTypeToNative<F8E8M0FNU> {
  using type = tsl::float8_e8m0fnu;
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

template <PrimitiveType kType>
using NativeTypeOf =
    typename primitive_util::PrimitiveTypeToNative<kType>::type;

// For each possible value k of type PrimitiveType, PrimitiveTypeConstant<k> is
// a distinct type, and PrimitiveTypeConstant<k>() can be implicitly converted
// to a compile-time constant of value k.
template <PrimitiveType kPrimitiveType>
using PrimitiveTypeConstant =
    std::integral_constant<PrimitiveType, kPrimitiveType>;

// Returns true if the given primitive type is a MX floating-point type.
constexpr bool IsMXType(PrimitiveType type) {
  return type == F4E2M1FN || type == F8E8M0FNU;
}

// Returns true if the given primitive type is an 8-bit floating-point type.
constexpr bool IsF8Type(PrimitiveType type) {
  return type == F8E5M2 || type == F8E4M3 || type == F8E4M3FN ||
         type == F8E4M3B11FNUZ || type == F8E5M2FNUZ || type == F8E4M3FNUZ ||
         type == F8E3M4;
}

// Returns true if the given primitive type is a floating-point type.
constexpr bool IsFloatingPointType(PrimitiveType type) {
  return type == F16 || type == F32 || type == F64 || type == BF16 ||
         IsF8Type(type) || IsMXType(type);
}

// Returns true if the given primitive type is a complex type.
constexpr bool IsComplexType(PrimitiveType type) {
  return type == C64 || type == C128;
}

// Returns true if the given primitive type is a signed integral type.
constexpr bool IsSignedIntegralType(PrimitiveType type) {
  return type == S1 || type == S2 || type == S4 || type == S8 || type == S16 ||
         type == S32 || type == S64;
}

// Returns true if the given primitive type is an unsigned integral type.
constexpr bool IsUnsignedIntegralType(PrimitiveType type) {
  return type == U1 || type == U2 || type == U4 || type == U8 || type == U16 ||
         type == U32 || type == U64;
}

// Returns true if the given primitive type is an integral type.
constexpr bool IsIntegralType(PrimitiveType type) {
  return IsUnsignedIntegralType(type) || IsSignedIntegralType(type);
}

// Returns true if the given primitive type is an 8-bit integral type.
constexpr bool Is8BitIntegralType(PrimitiveType type) {
  return type == S8 || type == U8;
}

// Returns true if values of the given primitive type are held in array shapes.
constexpr bool IsArrayType(PrimitiveType primitive_type) {
  return primitive_type == PRED || IsIntegralType(primitive_type) ||
         IsFloatingPointType(primitive_type) || IsComplexType(primitive_type);
}

// The following *TypeSwitch functions are used to dispatch on the run-time
// value of a PrimitiveType. They each take a polymorphic functor `f` and a
// PrimitiveType value `type` and return the result of applying `f` on a
// PrimitiveTypeConstant<type> value.
//
// They are useful because they allow us to use the run-time value of a
// PrimitiveType in a context expecting a compile-time constant.
//
// For example, consider the following function:
//
//   // Returns the size of the native type of the given primitive type.
//   int GetNativeSizeOf(PrimitiveType type) {
//     ...
//   }
//
// We can use PrimitiveTypeSwitch to implement it as follows:
//
//   int GetNativeSizeOf(PrimitiveType type) {
//     return PrimitiveTypeSwitch<int>(
//         // The functor is polymorphic and can accept any
//         // PrimitiveTypeConstant<type> value.
//         [&](auto primitive_type) -> int {
//           // Use primitive_type as a *compile-time* constant of type
//           // PrimitiveType.
//           return sizeof(NativeTypeOf<primitive_type>());
//         },
//         type);
//   }

// If `type` is an integral type, returns the result of applying polymorphic
// functor f on a PrimitiveTypeConstant<type> value; otherwise crashes.
template <typename R, typename F>
constexpr R IntegralTypeSwitch(F&& f, PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsIntegralType(type))) {
    switch (type) {
      case S1:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::S1>());
      case S2:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::S2>());
      case S4:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::S4>());
      case S8:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::S8>());
      case S16:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::S16>());
      case S32:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::S32>());
      case S64:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::S64>());
      case U1:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::U1>());
      case U2:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::U2>());
      case U4:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::U4>());
      case U8:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::U8>());
      case U16:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::U16>());
      case U32:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::U32>());
      case U64:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::U64>());
      default:
        ABSL_UNREACHABLE();
    }
  }
  LOG(FATAL) << "Not an integral data type " << type;
}

// If `type` is a floating-point type, returns the result of applying
// polymorphic functor f on a PrimitiveTypeConstant<type> value; otherwise
// crashes.
template <typename R, typename F>
constexpr R FloatingPointTypeSwitch(F&& f, PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsFloatingPointType(type))) {
    switch (type) {
      case F4E2M1FN:
        return std::forward<F>(f)(
            PrimitiveTypeConstant<PrimitiveType::F4E2M1FN>());
      case F8E3M4:
        return std::forward<F>(f)(
            PrimitiveTypeConstant<PrimitiveType::F8E3M4>());
      case F8E4M3:
        return std::forward<F>(f)(
            PrimitiveTypeConstant<PrimitiveType::F8E4M3>());
      case F8E4M3FN:
        return std::forward<F>(f)(
            PrimitiveTypeConstant<PrimitiveType::F8E4M3FN>());
      case F8E4M3B11FNUZ:
        return std::forward<F>(f)(
            PrimitiveTypeConstant<PrimitiveType::F8E4M3B11FNUZ>());
      case F8E4M3FNUZ:
        return std::forward<F>(f)(
            PrimitiveTypeConstant<PrimitiveType::F8E4M3FNUZ>());
      case F8E5M2:
        return std::forward<F>(f)(
            PrimitiveTypeConstant<PrimitiveType::F8E5M2>());
      case F8E5M2FNUZ:
        return std::forward<F>(f)(
            PrimitiveTypeConstant<PrimitiveType::F8E5M2FNUZ>());
      case F8E8M0FNU:
        return std::forward<F>(f)(
            PrimitiveTypeConstant<PrimitiveType::F8E8M0FNU>());
      case F16:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::F16>());
      case BF16:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::BF16>());
      case F32:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::F32>());
      case F64:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::F64>());
      default:
        ABSL_UNREACHABLE();
    }
  }
  LOG(FATAL) << "Not a floating point data type " << type;
}

// If `type` is a complex type, returns the result of applying polymorphic
// functor f on a PrimitiveTypeConstant<type> value; otherwise crashes.
template <typename R, typename F>
constexpr R ComplexTypeSwitch(F&& f, PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsComplexType(type))) {
    switch (type) {
      case C64:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::C64>());
      case C128:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::C128>());
      default:
        ABSL_UNREACHABLE();
    }
  }
  LOG(FATAL) << "Not a complex data type " << type;
}

// If `type` is an array type, returns the result of applying polymorphic
// functor f on a PrimitiveTypeConstant<type> value; otherwise crashes.
template <typename R, typename F>
constexpr R ArrayTypeSwitch(F&& f, PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsArrayType(type))) {
    if (IsFloatingPointType(type)) {
      return FloatingPointTypeSwitch<R>(std::forward<F>(f), type);
    }
    if (IsIntegralType(type)) {
      return IntegralTypeSwitch<R>(std::forward<F>(f), type);
    }
    if (IsComplexType(type)) {
      return ComplexTypeSwitch<R>(std::forward<F>(f), type);
    }
    if (type == PRED) {
      return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::PRED>());
    }
  }
  LOG(FATAL) << "Not an array data type " << type;
}

// If `type` is not PRIMITIVE_TYPE_INVALID, returns the result of applying
// polymorphic functor f on a PrimitiveTypeConstant<type> value; otherwise
// crashes.
template <typename R, typename F>
constexpr R PrimitiveTypeSwitch(F&& f, PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsArrayType(type))) {
    return ArrayTypeSwitch<R>(std::forward<F>(f), type);
  }
  if (type == TUPLE) {
    return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::TUPLE>());
  }
  if (type == TOKEN) {
    return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::TOKEN>());
  }
  if (type == OPAQUE_TYPE) {
    return std::forward<F>(f)(
        PrimitiveTypeConstant<PrimitiveType::OPAQUE_TYPE>());
  }
  LOG(FATAL) << "unhandled type " << type;
}

namespace internal {

// Returns the number of bits in the native type for a given primitive type if
// it is an array type. Otherwise, returns 0.
template <PrimitiveType primitive_type>
inline constexpr int PrimitiveTypeBitWidth() {
  if constexpr (IsArrayType(primitive_type)) {
    using NativeT = primitive_util::NativeTypeOf<primitive_type>;
    if constexpr (IsIntegralType(primitive_type)) {
      static_assert(is_specialized_integral_v<NativeT>);
      static_assert(std::numeric_limits<NativeT>::is_signed ==
                    IsSignedIntegralType(primitive_type));
      static_assert(std::numeric_limits<NativeT>::radix == 2);
      return std::numeric_limits<NativeT>::digits +
             (IsSignedIntegralType(primitive_type) ? 1 : 0);
    }
    if constexpr (primitive_type == PRED) {
      return std::numeric_limits<NativeT>::digits;
    }
    if constexpr (IsMXType(primitive_type)) {
      return NativeT::kBits;
    }
    if constexpr (IsFloatingPointType(primitive_type)) {
      return sizeof(NativeT) * std::numeric_limits<uint8_t>::digits;
    }
    if constexpr (IsComplexType(primitive_type)) {
      static_assert(is_complex_v<NativeT>);
      return sizeof(NativeT) * std::numeric_limits<uint8_t>::digits;
    }
  }
  return 0;
}

// BitWidthArrayHelper(<i0, i1, ...>) returns an array of bit widths for the
// given primitive types static_cast<PrimitiveType>(i0),
// static_cast<PrimitiveType>(i1), ...
template <int... Types>
inline constexpr auto BitWidthArrayHelper(
    std::integer_sequence<int, Types...>) {
  return std::array{PrimitiveTypeBitWidth<PrimitiveType{Types}>()...};
}

// An array of bit widths for all primitive types, where kBitWidths[i] is the
// bit width of primitive type static_cast<PrimitiveType>(i).
inline constexpr auto kBitWidths = BitWidthArrayHelper(
    std::make_integer_sequence<int, PrimitiveType_ARRAYSIZE>{});

// ByteWidthArrayHelper(<i0, i1, ...>) returns an array of byte widths for the
// given primitive types static_cast<PrimitiveType>(i0),
// static_cast<PrimitiveType>(i1), ...
template <int... Types>
inline constexpr auto ByteWidthArrayHelper(
    std::integer_sequence<int, Types...>) {
  return std::array{
      // The byte width of a primitive type is the number of bytes needed to
      // store its bits.
      CeilOfRatio(PrimitiveTypeBitWidth<PrimitiveType{Types}>(),
                  // Number of bits in a byte.
                  8)...};
}

// kByteWidths is an array of byte widths for all primitive types, where
// kByteWidths[i] is the byte width of primitive type
// static_cast<PrimitiveType>(i).
inline constexpr auto kByteWidths = ByteWidthArrayHelper(
    std::make_integer_sequence<int, PrimitiveType_ARRAYSIZE>{});

// If type is an array type, returns the width of the array element. Otherwise
// crashes. Depending on the kWidths template parameter, this can return either
// the bit or byte width of the array element.
template <const std::array<int, PrimitiveType_ARRAYSIZE>& kWidths>
inline constexpr int WidthForType(PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsArrayType(type))) {
    return kWidths[type];
  }
  LOG(FATAL) << "Unhandled primitive type " << type;
}

}  // namespace internal

// Returns the number of bits in the representation for a given type. Crashes if
// the type is not an array type.
inline constexpr int BitWidth(PrimitiveType type) {
  return internal::WidthForType<internal::kBitWidths>(type);
}

// Returns the number of bytes in the representation for a given type. Crashes
// if the type is not an array type.
inline constexpr int ByteWidth(PrimitiveType type) {
  return internal::WidthForType<internal::kByteWidths>(type);
}

// Returns the primitive type for the unsigned integral type with the given
// bit width, or PRIMITIVE_TYPE_INVALID if there is no such type.
constexpr PrimitiveType UnsignedIntegralTypeForBitWidth(int64_t src_bitwidth) {
  switch (src_bitwidth) {
    case 1:
      return xla::U1;
    case 2:
      return xla::U2;
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

// Returns the primitive type for the signed integral type with the given
// bit width, or PRIMITIVE_TYPE_INVALID if there is no such type.
PrimitiveType SignedIntegralTypeForBitWidth(int64_t src_bitwidth);

// Returns the real, imag component type for the given complex type.
// Crashes if complex_type is not complex.
constexpr PrimitiveType ComplexComponentType(PrimitiveType complex_type) {
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

// Returns the complex type for the given real, imag component type.
// Crashes if there's no complex type for the given component type.
constexpr PrimitiveType ComplexType(PrimitiveType base_type) {
  if (base_type == F32) {
    return C64;
  }
  if (base_type == F64) {
    return C128;
  }
  return PRIMITIVE_TYPE_INVALID;
}

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

// Returns true if a conversion from from_type to to_type loses no precision.
bool CastPreservesValues(PrimitiveType from_type, PrimitiveType to_type);

// Returns the lower-case name of the given primitive type.
const std::string& LowercasePrimitiveTypeName(PrimitiveType s);

// Returns the PrimitiveType matching the given name. The given name is expected
// to be lower-case.
absl::StatusOr<PrimitiveType> StringToPrimitiveType(
    absl::string_view lower_name);

// Returns true if the given string is a lower-case primitive type name.
bool IsPrimitiveTypeName(absl::string_view name);

// Returns whether `type` can be expressed as an instance of T.
// For example,
//  CanRepresent<float>(F32)          // true
//  CanRepresent<xla::bfloat16>(BF16) // true
//  CanRepresent<int32_t>(S8)         // true, 8 <= 32
//  CanRepresent<uint16_t>(S16)       // false, unsigned.
template <typename T>
constexpr bool CanRepresent(PrimitiveType type) {
  return PrimitiveTypeSwitch<bool>(
      [](auto primitive_type) -> bool {
        if constexpr (primitive_util::IsFloatingPointType(primitive_type) ||
                      primitive_util::IsComplexType(primitive_type)) {
          return NativeToPrimitiveType<T>() == primitive_type;
        }
        if constexpr (primitive_util::IsSignedIntegralType(primitive_type)) {
          return std::numeric_limits<T>::is_integer &&
                 std::numeric_limits<T>::is_signed &&
                 BitWidth(primitive_type) <=
                     (std::numeric_limits<T>::digits + 1);
        }
        if constexpr (primitive_util::IsUnsignedIntegralType(primitive_type) ||
                      primitive_type == PRED) {
          return std::numeric_limits<T>::is_integer &&
                 !std::numeric_limits<T>::is_signed &&
                 BitWidth(primitive_type) <= std::numeric_limits<T>::digits;
        }
        return false;
      },
      type);
}

// Returns true if `x` can be represented by the native type of `ty`.
inline bool FitsInIntegralType(int64_t x, PrimitiveType ty) {
  return primitive_util::IntegralTypeSwitch<bool>(
      [&](auto primitive_type) -> bool {
        using NativeT = primitive_util::NativeTypeOf<primitive_type>;
        return std::numeric_limits<NativeT>::min() <= x &&
               std::numeric_limits<NativeT>::max() >= x;
      },
      ty);
}

// Returns true if `type` is smaller than 8 bits and is not PRED.
constexpr bool IsSubByteNonPredType(PrimitiveType type) {
  return IsArrayType(type) && type != PRED &&
         primitive_util::BitWidth(type) < 8;
}

// Packs the given input of sub-byte values into the given output. The bit width
// of the input type must be 2 or 4, or this function will crash.
inline void PackIntN(PrimitiveType input_type, absl::Span<const char> input,
                     absl::Span<char> output) {
  xla::PackIntN(primitive_util::BitWidth(input_type), input, output);
}

// Unpacks the given input of sub-byte values into the given output. The bit
// width of the input type must be 2 or 4, or this function will crash.
inline void UnpackIntN(PrimitiveType input_type, absl::Span<const char> input,
                       absl::Span<char> output) {
  xla::UnpackIntN(primitive_util::BitWidth(input_type), input, output);
}

}  // namespace primitive_util
}  // namespace xla

#endif  // XLA_PRIMITIVE_UTIL_H_
