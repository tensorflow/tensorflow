/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_CODEGEN_INTRINSIC_CPP_VECTOR_OPS_H_
#define XLA_CODEGEN_INTRINSIC_CPP_VECTOR_OPS_H_

#if defined(__FLT16_MANT_DIG__) && defined(__has_attribute) &&    \
    __has_attribute(ext_vector_type) && defined(__has_builtin) && \
    __has_builtin(__builtin_vectorelements)

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "Eigen/Core"

namespace xla {
namespace codegen {
// Half precision (float16)
typedef _Float16 Vec8h __attribute__((vector_size(16)));
typedef _Float16 Vec16h __attribute__((vector_size(32)));

// Single precision (float32)
typedef float Vec4f __attribute__((vector_size(16)));
typedef float Vec8f __attribute__((vector_size(32)));
typedef float Vec16f __attribute__((vector_size(64)));

// Double precision (float64)
typedef double Vec2d __attribute__((vector_size(16)));
typedef double Vec4d __attribute__((vector_size(32)));
typedef double Vec8d __attribute__((vector_size(64)));

// Corresponding unsigned integer types
typedef uint32_t Vec4u __attribute__((vector_size(16)));
typedef uint32_t Vec8u __attribute__((vector_size(32)));
typedef uint32_t Vec16u __attribute__((vector_size(64)));
typedef uint64_t Vec2q __attribute__((vector_size(16)));
typedef uint64_t Vec4q __attribute__((vector_size(32)));
typedef uint64_t Vec8q __attribute__((vector_size(64)));

// Corresponding signed integer types
typedef int32_t Vec4i __attribute__((vector_size(16)));
typedef int32_t Vec8i __attribute__((vector_size(32)));
typedef int32_t Vec16i __attribute__((vector_size(64)));
typedef int64_t Vec2l __attribute__((vector_size(16)));
typedef int64_t Vec4l __attribute__((vector_size(32)));
typedef int64_t Vec8l __attribute__((vector_size(64)));

namespace internal {
// Helper type to select the corresponding unsigned integer vector type.
template <typename ScalarInt, size_t Width>
struct MakeIntVec;

template <>
struct MakeIntVec<uint32_t, 4> {
  using type = Vec4u;
};
template <>
struct MakeIntVec<uint32_t, 8> {
  using type = Vec8u;
};
template <>
struct MakeIntVec<uint32_t, 16> {
  using type = Vec16u;
};
template <>
struct MakeIntVec<uint64_t, 2> {
  using type = Vec2q;
};
template <>
struct MakeIntVec<uint64_t, 4> {
  using type = Vec4q;
};
template <>
struct MakeIntVec<uint64_t, 8> {
  using type = Vec8q;
};

// Helper type to select the corresponding signed integer vector type.
template <typename ScalarInt, size_t Width>
struct MakeSignedIntVec;

template <>
struct MakeSignedIntVec<int32_t, 4> {
  using type = Vec4i;
};
template <>
struct MakeSignedIntVec<int32_t, 8> {
  using type = Vec8i;
};
template <>
struct MakeSignedIntVec<int32_t, 16> {
  using type = Vec16i;
};
template <>
struct MakeSignedIntVec<int64_t, 2> {
  using type = Vec2l;
};
template <>
struct MakeSignedIntVec<int64_t, 4> {
  using type = Vec4l;
};
template <>
struct MakeSignedIntVec<int64_t, 8> {
  using type = Vec8l;
};

// This trait takes a float vector and provides its unsigned integer vector
// partner.
template <typename FloatVec>
struct CorrespondingIntVector {
 private:
  using ScalarFloat = decltype(FloatVec{}[0]);
  static constexpr size_t kWidth = sizeof(FloatVec) / sizeof(ScalarFloat);
  using ScalarInt =
      std::conditional_t<sizeof(ScalarFloat) == 4, uint32_t, uint64_t>;

 public:
  using type = typename MakeIntVec<ScalarInt, kWidth>::type;
};

// This trait takes a float vector and provides its signed integer vector
// partner.
template <typename FloatVec>
struct CorrespondingSignedIntVector {
 private:
  using ScalarFloat = decltype(FloatVec{}[0]);
  static constexpr size_t kWidth = sizeof(FloatVec) / sizeof(ScalarFloat);
  using ScalarInt =
      std::conditional_t<sizeof(ScalarFloat) == 4, int32_t, int64_t>;

 public:
  using type = typename MakeSignedIntVec<ScalarInt, kWidth>::type;
};
}  // namespace internal

// ===--------------------------------------------------------------------===//
// Eigen Array type mapping
// ===--------------------------------------------------------------------===//
template <typename VecType>
struct ArrayMap;

template <>
struct ArrayMap<Vec8h> {
  using type = Eigen::Array<Eigen::half, 8, 1>;
};
template <>
struct ArrayMap<Vec16h> {
  using type = Eigen::Array<Eigen::half, 16, 1>;
};

template <>
struct ArrayMap<Vec4f> {
  using type = Eigen::Array<float, 4, 1>;
};
template <>
struct ArrayMap<Vec8f> {
  using type = Eigen::Array<float, 8, 1>;
};
template <>
struct ArrayMap<Vec16f> {
  using type = Eigen::Array<float, 16, 1>;
};

template <>
struct ArrayMap<Vec2d> {
  using type = Eigen::Array<double, 2, 1>;
};
template <>
struct ArrayMap<Vec4d> {
  using type = Eigen::Array<double, 4, 1>;
};
template <>
struct ArrayMap<Vec8d> {
  using type = Eigen::Array<double, 8, 1>;
};

namespace internal {
template <typename T, typename S, size_t... Is>
constexpr T SplatImpl(S val, std::index_sequence<Is...>) {
  return T{(static_cast<void>(Is), static_cast<decltype(T{}[0])>(val))...};
}
}  // namespace internal

template <typename T, typename S>
constexpr T Splat(S val) {
  if constexpr (std::is_fundamental_v<T>) {
    return static_cast<T>(val);
  } else {
    constexpr size_t kWidth = sizeof(T) / sizeof(decltype(T{}[0]));
    return internal::SplatImpl<T>(val, std::make_index_sequence<kWidth>{});
  }
}

// Computes the absolute value of a vector using bitwise operations.
// FloatVec: The floating-point vector type (e.g., Vec4f).
// x: The input vector.
// Returns a new vector containing the absolute value of each element in x.
template <typename FloatVec>
FloatVec BitwiseAbs(FloatVec x) {
  using IntVec = typename internal::CorrespondingIntVector<FloatVec>::type;
  // Get the underlying scalar integer type (e.g., uint32_t from Vec4u).
  using ScalarInt = decltype(IntVec{}[0]);

  // Create a mask to clear the sign bit (e.g., 0x7FFFFFFF for int).
  // This is a vector where every element is the mask.
  const IntVec abs_mask =
      IntVec{~(static_cast<ScalarInt>(1) << (sizeof(ScalarInt) * 8 - 1))};

  // Reinterpret float as int, apply the mask, and reinterpret back.
  return __builtin_bit_cast(FloatVec, __builtin_bit_cast(IntVec, x) & abs_mask);
}

// Copies the sign of one vector to the value of another.
// FloatVec: The floating-point vector type (e.g., Vec4f).
// value: The vector providing the magnitude.
// sign_source: The vector providing the sign.
// Returns a new vector with the magnitude of `value` and the sign of
// `sign_source`.
template <typename FloatVec>
FloatVec BitwiseCopysign(FloatVec value, FloatVec sign_source) {
  using IntVec = typename internal::CorrespondingIntVector<FloatVec>::type;
  using ScalarInt = decltype(IntVec{}[0]);
  const IntVec sign_mask =
      IntVec{static_cast<ScalarInt>(1) << (sizeof(ScalarInt) * 8 - 1)};
  FloatVec value_abs = BitwiseAbs<FloatVec>(value);
  IntVec sign_bits = __builtin_bit_cast(IntVec, sign_source) & sign_mask;
  return __builtin_bit_cast(FloatVec,
                            __builtin_bit_cast(IntVec, value_abs) | sign_bits);
}

template <typename Vec, typename Scalar>
Vec Clamp(Vec x, Scalar min, Scalar max) {
  return x < min ? min : x > max ? max : x;
}

// Evaluates a polynomial c_0 * x^n + c_1 * x^(n-1) + ... + c_n using Horner's
// method, where `coeffs` are given in descending degree order.
template <typename T, typename S, typename... Coeffs>
constexpr T HornerPoly(T x, S c0, Coeffs... coeffs) {
  T acc = Splat<T>(c0);
  ((acc = acc * x + Splat<T>(coeffs)), ...);
  return acc;
}

// Emits `@llvm.fmuladd` to compute `a * b + c`. When the target CPU has
// hardware FMA, this lowers to a single fused multiply-add instruction;
// without hardware FMA, `@llvm.fmuladd` falls back to unfused multiply/add.
template <typename VecT>
inline VecT Fmuladd(VecT a, VecT b, VecT c) {
#pragma clang fp contract(on)
  return a * b + c;
}

// Computes the error-free transformation `s + e = a + b` assuming `|a| >= |b|`
// (Dekker's FastTwoSum), returning `{s, e}` where `s = RN(a + b)` and `e` is
// the rounding error.
template <typename VecT>
inline std::pair<VecT, VecT> FastTwoSum(VecT a, VecT b) {
  VecT s = a + b;
  VecT e = (a - s) + b;
  return {s, e};
}

template <typename VecT>
inline auto ToSignedInt(VecT x) {
  // Guard NaN lanes to 0 before float-to-int conversion (`fptosi` on NaN is UB
  // and produces `poison`).
  VecT safe_x = x == x ? x : VecT{};
  if constexpr (std::is_arithmetic_v<VecT>) {
    using IntT = std::conditional_t<sizeof(VecT) == 4, int32_t, int64_t>;
    return static_cast<IntT>(safe_x);
  } else {
    using IntVec = typename internal::CorrespondingSignedIntVector<VecT>::type;
    return __builtin_convertvector(safe_x, IntVec);
  }
}
}  // namespace codegen
}  // namespace xla

#endif  // defined(__has_attribute) && __has_attribute(ext_vector_type) &&
        // defined(__has_builtin) && __has_builtin(__builtin_vectorelements)

#endif  // XLA_CODEGEN_INTRINSIC_CPP_VECTOR_OPS_H_
