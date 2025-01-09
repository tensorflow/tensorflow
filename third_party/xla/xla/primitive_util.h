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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
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

// Returns whether the type has a value for negative zero.
bool HasNegativeZero(PrimitiveType type);

// Returns the XLA primitive type (eg, F32) corresponding to the given
// template parameter native type (eg, float).
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

// Complex
template <>
constexpr PrimitiveType NativeToPrimitiveType<complex64>() {
  return C64;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<complex128>() {
  return C128;
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

// Complex
template <>
struct PrimitiveTypeToNative<C64> {
  using type = complex64;
};

template <>
struct PrimitiveTypeToNative<C128> {
  using type = complex128;
};

// Token
template <>
struct PrimitiveTypeToNative<TOKEN> {
  using type = void;
};

template <PrimitiveType kType>
using NativeTypeOf =
    typename primitive_util::PrimitiveTypeToNative<kType>::type;

template <PrimitiveType kPrimitiveType>
using PrimitiveTypeConstant =
    std::integral_constant<PrimitiveType, kPrimitiveType>;

// Returns true if values of the given primitive type are held in array shapes.
inline constexpr bool IsArrayType(PrimitiveType primitive_type) {
  return primitive_type != TUPLE && primitive_type != OPAQUE_TYPE &&
         primitive_type != TOKEN && primitive_type > PRIMITIVE_TYPE_INVALID &&
         primitive_type < PrimitiveType_ARRAYSIZE;
}

constexpr bool IsF8Type(PrimitiveType type) {
  return type == F8E5M2 || type == F8E4M3 || type == F8E4M3FN ||
         type == F8E4M3B11FNUZ || type == F8E5M2FNUZ || type == F8E4M3FNUZ ||
         type == F8E3M4;
}

constexpr bool IsFloatingPointType(PrimitiveType type) {
  return type == F16 || type == F32 || type == F64 || type == BF16 ||
         IsF8Type(type);
}

constexpr bool IsComplexType(PrimitiveType type) {
  return type == C64 || type == C128;
}

constexpr bool IsSignedIntegralType(PrimitiveType type) {
  return type == S1 || type == S2 || type == S4 || type == S8 || type == S16 ||
         type == S32 || type == S64;
}

constexpr bool IsUnsignedIntegralType(PrimitiveType type) {
  return type == U1 || type == U2 || type == U4 || type == U8 || type == U16 ||
         type == U32 || type == U64;
}

constexpr bool IsIntegralType(PrimitiveType type) {
  return IsUnsignedIntegralType(type) || IsSignedIntegralType(type);
}

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

template <typename R, typename F>
constexpr R FloatingPointTypeSwitch(F&& f, PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsFloatingPointType(type))) {
    switch (type) {
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
template <int... Types>
inline constexpr auto BitWidthArrayHelper(
    std::integer_sequence<int, Types...>) {
  return std::array{PrimitiveTypeBitWidth<PrimitiveType{Types}>()...};
}

inline constexpr auto kBitWidths = BitWidthArrayHelper(
    std::make_integer_sequence<int, PrimitiveType_ARRAYSIZE>{});

template <int... Types>
inline constexpr auto ByteWidthArrayHelper(
    std::integer_sequence<int, Types...>) {
  return std::array{
      CeilOfRatio(PrimitiveTypeBitWidth<PrimitiveType{Types}>(), 8)...};
}
inline constexpr auto kByteWidths = ByteWidthArrayHelper(
    std::make_integer_sequence<int, PrimitiveType_ARRAYSIZE>{});

template <const std::array<int, PrimitiveType_ARRAYSIZE>& kWidths>
inline constexpr int WidthForType(PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsArrayType(type))) {
    return kWidths[type];
  }
  LOG(FATAL) << "Unhandled primitive type " << type;
}
}  // namespace internal

// Returns the number of bits in the representation for a given type.
inline constexpr int BitWidth(PrimitiveType type) {
  return internal::WidthForType<internal::kBitWidths>(type);
}

// Returns the number of bytes in the representation for a given type.
inline constexpr int ByteWidth(PrimitiveType type) {
  return internal::WidthForType<internal::kByteWidths>(type);
}

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

PrimitiveType SignedIntegralTypeForBitWidth(int64_t src_bitwidth);

// Returns the real, imag component type underlying the given complex type.
// LOG(FATAL)'s if complex_type is not complex.
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
  // F -> F is safe if the exponent/significand are preserved and `to_type`
  // preserves infinities in `from_type.
  if (primitive_util::IsFloatingPointType(from_type) &&
      primitive_util::IsFloatingPointType(to_type)) {
    return (!primitive_util::HasInfinity(from_type) ||
            primitive_util::HasInfinity(to_type)) &&
           primitive_util::SignificandWidth(from_type) <=
               primitive_util::SignificandWidth(to_type) &&
           primitive_util::ExponentWidth(from_type) <=
               primitive_util::ExponentWidth(to_type) &&
           (primitive_util::UnderflowExponent(from_type) -
            primitive_util::SignificandWidth(from_type)) >=
               (primitive_util::UnderflowExponent(to_type) -
                primitive_util::SignificandWidth(to_type)) &&
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

// Returns the lower-case name of the given primitive type.
const std::string& LowercasePrimitiveTypeName(PrimitiveType s);

// Returns the PrimitiveType matching the given name. The given name is expected
// to be lower-case.
absl::StatusOr<PrimitiveType> StringToPrimitiveType(absl::string_view name);

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

inline bool FitsInIntegralType(int64_t x, PrimitiveType ty) {
  return primitive_util::IntegralTypeSwitch<bool>(
      [&](auto primitive_type) -> bool {
        using NativeT = primitive_util::NativeTypeOf<primitive_type>;
        return std::numeric_limits<NativeT>::min() <= x &&
               std::numeric_limits<NativeT>::max() >= x;
      },
      ty);
}

constexpr bool IsSubByteNonPredType(PrimitiveType type) {
  return IsArrayType(type) && type != PRED &&
         primitive_util::BitWidth(type) < 8;
}

inline void PackIntN(PrimitiveType input_type, absl::Span<const char> input,
                     absl::Span<char> output) {
  xla::PackIntN(primitive_util::BitWidth(input_type), input, output);
}

inline void UnpackIntN(PrimitiveType input_type, absl::Span<const char> input,
                       absl::Span<char> output) {
  xla::UnpackIntN(primitive_util::BitWidth(input_type), input, output);
}

}  // namespace primitive_util
}  // namespace xla

#endif  // XLA_PRIMITIVE_UTIL_H_
