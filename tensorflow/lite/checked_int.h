/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// This file provides `CheckedInt`, an integer wrapper that tracks whether the
// arithmetic operations used to compute its value have over/underflowed.

#ifndef TENSORFLOW_LITE_CHECKED_INT_H_
#define TENSORFLOW_LITE_CHECKED_INT_H_

#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/macros.h"

namespace tflite {

#if !(defined(TFLITE_HAS_OVERFLOW_BUILTINS)) && \
    (defined(__GNUC__) || defined(__clang__))
#define TFLITE_HAS_OVERFLOW_BUILTINS 1
#endif

#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER >= 1940 && \
    (defined(_M_X64) || defined(_M_IX86))
#define TFLITE_CHECKED_INT_MSVC_INTRINSICS 1
#else
#define TFLITE_CHECKED_INT_MSVC_INTRINSICS 0
#endif

// TFLITE_CHECKED_INT_MERGE_OVERFLOW is used to combine the overflow flag in the
// `CheckedInt` operation implementations either with `||` or with `|`.
//
// We use this macro because different architectures will optimize differently
// the flag update.
#if defined(__aarch64__) || defined(_M_ARM64)
#define TFLITE_CHECKED_INT_MERGE_OVERFLOW(OPERAND_OVERFLOW, OP_OVERFLOW) \
  (static_cast<bool>((OPERAND_OVERFLOW) | (OP_OVERFLOW)))
#else
#define TFLITE_CHECKED_INT_MERGE_OVERFLOW(OPERAND_OVERFLOW, OP_OVERFLOW) \
  (static_cast<bool>(TFLITE_EXPECT_FALSE((OPERAND_OVERFLOW) || (OP_OVERFLOW))))
#endif

namespace checked_int_internal {

template <class R>
using Widen = std::conditional_t<std::is_signed_v<R>, int64_t, uint64_t>;

// Extracts the 32 high bits of `x`.
constexpr uint64_t hi(const uint64_t x) { return x >> 32; }
// Extracts the 32 low bits of `x`.
constexpr uint64_t lo(const uint64_t x) { return x & 0xffffffff; }

// Upper and lower halves of the 128-bit product of `a` and `b`.
inline uint64_t MulWide(uint64_t a, uint64_t b, uint64_t* high) noexcept {
#if defined(_MSC_VER) && !defined(__clang__) && defined(_M_X64)
  return _umul128(a, b, high);
#elif defined(_MSC_VER) && !defined(__clang__) && defined(_M_ARM64)
  *high = __umulh(a, b);
  return a * b;
#else
  const uint64_t hia = hi(a);
  const uint64_t loa = lo(a);
  const uint64_t hib = hi(b);
  const uint64_t lob = lo(b);
  const uint64_t lo_lo = loa * lob;
  const uint64_t hi_lo = hia * lob;
  const uint64_t lo_hi = loa * hib;
  const uint64_t hi_hi = hia * hib;
  const uint64_t cross = hi(lo_lo) + lo(hi_lo) + lo_hi;
  *high = hi_hi + hi(hi_lo) + hi(cross);
  return a * b;
#endif
}

// Representation of operands using a 64 bit magnitude and a sign.
//
// The overloads for operands of different types overflow iff the exact result
// of the operation does not fit in `R`.
struct SignMagnitude {
  bool negative;
  uint64_t magnitude;
};

template <class T>
SignMagnitude ToSignMagnitude(T value) noexcept {
  if constexpr (std::is_signed_v<T>) {
    if (value < 0) {
      return {true, uint64_t{0} - static_cast<uint64_t>(value)};
    }
  }
  return {false, static_cast<uint64_t>(value)};
}

// Stores the exact result `+/-magnitude` modulo 2^N in `*res` and returns
// whether it does not fit in `R`. `wide` means that the magnitude is at least
// 2^64.
template <class R>
bool FromSignMagnitude(bool negative, uint64_t magnitude, bool wide,
                       R* res) noexcept {
  *res = static_cast<R>(negative ? uint64_t{0} - magnitude : magnitude);
  if (wide) {
    return true;
  }
  if (negative) {
    if constexpr (std::is_signed_v<R>) {
      return magnitude >
             static_cast<uint64_t>(std::numeric_limits<R>::max()) + 1;
    } else {
      return magnitude != 0;
    }
  }
  return magnitude > static_cast<uint64_t>(std::numeric_limits<R>::max());
}

template <class R>
bool AddSignMagnitude(SignMagnitude x, SignMagnitude y, R* res) noexcept {
  if (x.negative == y.negative) {
    const uint64_t magnitude = x.magnitude + y.magnitude;
    return FromSignMagnitude(x.negative, magnitude, magnitude < x.magnitude,
                             res);
  }
  if (x.magnitude >= y.magnitude) {
    return FromSignMagnitude(x.negative, x.magnitude - y.magnitude,
                             /*wide=*/false, res);
  }
  return FromSignMagnitude(y.negative, y.magnitude - x.magnitude,
                           /*wide=*/false, res);
}

// Whether every value of `T` is representable in `R`.
template <class T, class R>
inline constexpr bool kFitsIn =
    std::is_signed_v<T> == std::is_signed_v<R>
        ? sizeof(T) <= sizeof(R)
        : std::is_unsigned_v<T> && sizeof(T) < sizeof(R);

// Whether `T` is signed and its conversion to the unsigned `R` only changes
// negative values, by adding 2^N (N being the width of `R`).
template <class T, class R>
inline constexpr bool kWrapsInto =
    std::is_signed_v<T> && std::is_unsigned_v<R> && sizeof(T) <= sizeof(R);

template <class R>
bool AddOverflow(R a, R b, R* res) noexcept {
#if TFLITE_HAS_OVERFLOW_BUILTINS
  return __builtin_add_overflow(a, b, res);
#else
  if constexpr (sizeof(R) < sizeof(int64_t)) {
    using W = Widen<R>;
    const W wide = static_cast<W>(a) + static_cast<W>(b);
    *res = static_cast<R>(wide);
    return wide < static_cast<W>(std::numeric_limits<R>::lowest()) ||
           wide > static_cast<W>(std::numeric_limits<R>::max());
  } else if constexpr (std::is_unsigned_v<R>) {
    *res = a + b;
    return *res < a;
  } else {
    // Signed overflow is undefined behaviour: compute in unsigned and check
    // whether the sign of the result differs from the sign of both operands.
    using UR = std::make_unsigned_t<R>;
    const UR ua = static_cast<UR>(a);
    const UR ub = static_cast<UR>(b);
    const UR ures = ua + ub;
    *res = static_cast<R>(ures);
    return static_cast<R>((ua ^ ures) & (ub ^ ures)) < 0;
  }
#endif
}

#if TFLITE_CHECKED_INT_MSVC_INTRINSICS
template <>
inline bool AddOverflow(int32_t a, int32_t b, int32_t* res) noexcept {
  return _add_overflow_i32(0, a, b, res);
}

template <>
inline bool AddOverflow(uint32_t a, uint32_t b, uint32_t* res) noexcept {
  return _addcarry_u32(0, a, b, res);
}

// `long` and `unsigned long` are 32-bit types distinct from `int32_t` and
// `uint32_t` with MSVC.
template <>
inline bool AddOverflow(long a, long b, long* res) noexcept {
  int32_t r;
  const bool overflow = AddOverflow<int32_t>(a, b, &r);
  *res = r;
  return overflow;
}

template <>
inline bool AddOverflow(unsigned long a, unsigned long b,
                        unsigned long* res) noexcept {
  uint32_t r;
  const bool overflow = AddOverflow<uint32_t>(a, b, &r);
  *res = r;
  return overflow;
}

#if defined(_M_X64)
template <>
inline bool AddOverflow(int64_t a, int64_t b, int64_t* res) noexcept {
  return _add_overflow_i64(0, a, b, res);
}

template <>
inline bool AddOverflow(uint64_t a, uint64_t b, uint64_t* res) noexcept {
  return _addcarry_u64(0, a, b, res);
}
#endif  // defined(_M_X64)
#endif  // TFLITE_CHECKED_INT_MSVC_INTRINSICS

// Defined below; used by the overload for operands of different types.
template <class R>
bool SubOverflow(R a, R b, R* res) noexcept;

template <class A, class B, class R>
bool AddOverflow(A a, B b, R* res) noexcept {
  if constexpr (kFitsIn<A, R> && kFitsIn<B, R>) {
    return AddOverflow<R>(static_cast<R>(a), static_cast<R>(b), res);
  } else if constexpr (kFitsIn<A, R> && kWrapsInto<B, R>) {
    if (b < 0) {
      return SubOverflow<R>(static_cast<R>(a), R{0} - static_cast<R>(b), res);
    }
    return AddOverflow<R>(static_cast<R>(a), static_cast<R>(b), res);
  } else if constexpr (kWrapsInto<A, R> && kFitsIn<B, R>) {
    return AddOverflow(b, a, res);
  } else {
#if TFLITE_HAS_OVERFLOW_BUILTINS
    return __builtin_add_overflow(a, b, res);
#else
    return AddSignMagnitude(ToSignMagnitude(a), ToSignMagnitude(b), res);
#endif
  }
}

template <class R>
bool SubOverflow(R a, R b, R* res) noexcept {
#if TFLITE_HAS_OVERFLOW_BUILTINS
  return __builtin_sub_overflow(a, b, res);
#else
  if constexpr (std::is_unsigned_v<R>) {
    *res = static_cast<R>(a - b);
    return a < b;
  } else if constexpr (sizeof(R) < sizeof(int64_t)) {
    using W = Widen<R>;
    const W wide = static_cast<W>(a) - static_cast<W>(b);
    *res = static_cast<R>(wide);
    return wide < static_cast<W>(std::numeric_limits<R>::lowest()) ||
           wide > static_cast<W>(std::numeric_limits<R>::max());
  } else {
    // Overflow iff the operands have different signs and the sign of the
    // result differs from the sign of `a`.
    using UR = std::make_unsigned_t<R>;
    const UR ua = static_cast<UR>(a);
    const UR ub = static_cast<UR>(b);
    const UR ures = ua - ub;
    *res = static_cast<R>(ures);
    return static_cast<R>((ua ^ ub) & (ua ^ ures)) < 0;
  }
#endif
}

#if TFLITE_CHECKED_INT_MSVC_INTRINSICS
template <>
inline bool SubOverflow(int32_t a, int32_t b, int32_t* res) noexcept {
  return _sub_overflow_i32(0, a, b, res);
}

template <>
inline bool SubOverflow(uint32_t a, uint32_t b, uint32_t* res) noexcept {
  return _subborrow_u32(0, a, b, res);
}

// `long` and `unsigned long` are 32-bit types distinct from `int32_t` and
// `uint32_t` with MSVC.
template <>
inline bool SubOverflow(long a, long b, long* res) noexcept {
  int32_t r;
  const bool overflow = SubOverflow<int32_t>(a, b, &r);
  *res = r;
  return overflow;
}

template <>
inline bool SubOverflow(unsigned long a, unsigned long b,
                        unsigned long* res) noexcept {
  uint32_t r;
  const bool overflow = SubOverflow<uint32_t>(a, b, &r);
  *res = r;
  return overflow;
}

#if defined(_M_X64)
template <>
inline bool SubOverflow(int64_t a, int64_t b, int64_t* res) noexcept {
  return _sub_overflow_i64(0, a, b, res);
}

template <>
inline bool SubOverflow(uint64_t a, uint64_t b, uint64_t* res) noexcept {
  return _subborrow_u64(0, a, b, res);
}
#endif  // defined(_M_X64)
#endif  // TFLITE_CHECKED_INT_MSVC_INTRINSICS

template <class A, class B, class R>
bool SubOverflow(A a, B b, R* res) noexcept {
  if constexpr (kFitsIn<A, R> && kFitsIn<B, R>) {
    return SubOverflow<R>(static_cast<R>(a), static_cast<R>(b), res);
  } else if constexpr (kFitsIn<A, R> && kWrapsInto<B, R>) {
    if (b < 0) {
      return AddOverflow<R>(static_cast<R>(a), R{0} - static_cast<R>(b), res);
    }
    return SubOverflow<R>(static_cast<R>(a), static_cast<R>(b), res);
  } else if constexpr (kWrapsInto<A, R> && kFitsIn<B, R>) {
    if (a < 0) {
      // A negative `a` minus a non-negative `b` is negative.
      *res = static_cast<R>(a) - static_cast<R>(b);
      return true;
    }
    return SubOverflow<R>(static_cast<R>(a), static_cast<R>(b), res);
  } else {
#if TFLITE_HAS_OVERFLOW_BUILTINS
    return __builtin_sub_overflow(a, b, res);
#else
    SignMagnitude y = ToSignMagnitude(b);
    y.negative = !y.negative && y.magnitude != 0;
    return AddSignMagnitude(ToSignMagnitude(a), y, res);
#endif
  }
}

template <class R>
bool MulOverflow(R a, R b, R* res) noexcept {
#if TFLITE_HAS_OVERFLOW_BUILTINS
  return __builtin_mul_overflow(a, b, res);
#else
  if constexpr (sizeof(R) < sizeof(int64_t)) {
    using W = Widen<R>;
    const W wide = static_cast<W>(a) * static_cast<W>(b);
    *res = static_cast<R>(wide);
    return wide < static_cast<W>(std::numeric_limits<R>::lowest()) ||
           wide > static_cast<W>(std::numeric_limits<R>::max());
  } else {
    static_assert(sizeof(R) == sizeof(uint64_t));
    const uint64_t ua = static_cast<uint64_t>(a);
    const uint64_t ub = static_cast<uint64_t>(b);
    uint64_t upper_64;
    const uint64_t lower_64 = MulWide(ua, ub, &upper_64);
    *res = static_cast<R>(lower_64);
    if constexpr (std::is_signed_v<R>) {
      // It took a while to understand this.
      //
      // If a < 0, then ua = a + 2^64.
      // So ua * ub = (a + 2^64) * ub
      //            = a * ub + ub * 2^64
      //                       ~~~~~~~~~
      // This means that the upper_64 that we compute above has an extra ub
      // added to it that we need to remove.
      //
      // The same applies for b.
      if (a < 0) {
        upper_64 -= ub;
      }
      if (b < 0) {
        upper_64 -= ua;
      }
      return upper_64 !=
             static_cast<uint64_t>(static_cast<int64_t>(lower_64) >> 63);
    } else {
      return upper_64 != 0;
    }
  }
#endif
}

// `uint32_t` multiplication keeps the generic 64-bit product, which is faster
// than `_mul_full_overflow_u32` with cl.
#if TFLITE_CHECKED_INT_MSVC_INTRINSICS
template <>
inline bool MulOverflow(int32_t a, int32_t b, int32_t* res) noexcept {
  return _mul_overflow_i32(a, b, res);
}

// `long` and `unsigned long` are 32-bit types distinct from `int32_t` and
// `uint32_t` with MSVC.
template <>
inline bool MulOverflow(long a, long b, long* res) noexcept {
  int32_t r;
  const bool overflow = MulOverflow<int32_t>(a, b, &r);
  *res = r;
  return overflow;
}

template <>
inline bool MulOverflow(unsigned long a, unsigned long b,
                        unsigned long* res) noexcept {
  uint32_t r;
  const bool overflow = MulOverflow<uint32_t>(a, b, &r);
  *res = r;
  return overflow;
}

#if defined(_M_X64)
template <>
inline bool MulOverflow(int64_t a, int64_t b, int64_t* res) noexcept {
  return _mul_overflow_i64(a, b, res);
}

template <>
inline bool MulOverflow(uint64_t a, uint64_t b, uint64_t* res) noexcept {
  uint64_t high;
  return _mul_full_overflow_u64(a, b, res, &high);
}
#endif  // defined(_M_X64)
#endif  // TFLITE_CHECKED_INT_MSVC_INTRINSICS

template <class A, class B, class R>
bool MulOverflow(A a, B b, R* res) noexcept {
  if constexpr (kFitsIn<A, R> && kFitsIn<B, R>) {
    return MulOverflow<R>(static_cast<R>(a), static_cast<R>(b), res);
  } else if constexpr (kFitsIn<A, R> && kWrapsInto<B, R>) {
    if (b < 0) {
      // The product is negative unless `a` is 0.
      *res = static_cast<R>(a) * static_cast<R>(b);
      return a != 0;
    }
    return MulOverflow<R>(static_cast<R>(a), static_cast<R>(b), res);
  } else if constexpr (kWrapsInto<A, R> && kFitsIn<B, R>) {
    return MulOverflow(b, a, res);
  } else {
#if TFLITE_HAS_OVERFLOW_BUILTINS
    return __builtin_mul_overflow(a, b, res);
#else
    const SignMagnitude x = ToSignMagnitude(a);
    const SignMagnitude y = ToSignMagnitude(b);
    uint64_t high;
    const uint64_t magnitude = MulWide(x.magnitude, y.magnitude, &high);
    return FromSignMagnitude(x.negative != y.negative, magnitude, high != 0,
                             res);
#endif
  }
}

template <class R>
bool DivOverflow(R a, R b, R* res) noexcept {
  bool invalid = b == 0;
  if constexpr (std::is_signed_v<R>) {
    invalid = invalid || (a == std::numeric_limits<R>::lowest() && b == -1);
  }
  R val = std::numeric_limits<R>::max();
  if (!TFLITE_EXPECT_FALSE(invalid)) {
    val = static_cast<R>(a / b);
  }
  *res = val;
  return invalid;
}

template <class A, class B, class R>
bool DivOverflow(A a, B b, R* res) noexcept {
  if constexpr (kFitsIn<A, R> && kFitsIn<B, R>) {
    return DivOverflow<R>(static_cast<R>(a), static_cast<R>(b), res);
  } else if constexpr (kFitsIn<A, R> && kWrapsInto<B, R>) {
    if (b < 0) {
      const R q = static_cast<R>(a) / (R{0} - static_cast<R>(b));
      *res = R{0} - q;
      return q != 0;
    }
    return DivOverflow<R>(static_cast<R>(a), static_cast<R>(b), res);
  } else if constexpr (kWrapsInto<A, R> && kFitsIn<B, R>) {
    // Division by zero is left to the same-type primitive.
    if (a < 0 && b != 0) {
      const R q = (R{0} - static_cast<R>(a)) / static_cast<R>(b);
      *res = R{0} - q;
      return q != 0;
    }
    return DivOverflow<R>(static_cast<R>(a), static_cast<R>(b), res);
  } else {
    const SignMagnitude x = ToSignMagnitude(a);
    const SignMagnitude y = ToSignMagnitude(b);
    if (y.magnitude == 0) {
      *res = std::numeric_limits<R>::max();
      return true;
    }
    return FromSignMagnitude(x.negative != y.negative,
                             x.magnitude / y.magnitude,
                             /*wide=*/false, res);
  }
}

}  // namespace checked_int_internal

// Wraps an integer value and allows checking if standard arithmetic operations
// used to generate that value have over/underflowed.
//
// The held value is unspecified if an overflow happens.
template <class T>
class CheckedInt {
 public:
  using type = T;

  static_assert(std::is_integral_v<T>, "T must be an integral value.");

  CheckedInt() = default;
  CheckedInt(const CheckedInt&) = default;
  CheckedInt(CheckedInt&&) = default;
  CheckedInt& operator=(const CheckedInt&) = default;
  CheckedInt& operator=(CheckedInt&&) = default;

  template <class U, class = std::enable_if_t<std::is_convertible_v<U, T>>>
  // NOLINTNEXTLINE(*-explicit-constructor)
  CheckedInt(U val) : value_(static_cast<T>(val)), overflow_(false) {
    if constexpr (std::is_same_v<T, U>) {
      // Don't do anything, the types are the same.
    } else if constexpr (std::is_signed_v<U> == std::is_signed_v<T>) {
      overflow_ = val < std::numeric_limits<T>::lowest() ||
                  val > std::numeric_limits<T>::max();
    } else if constexpr (std::is_unsigned_v<U> && std::is_signed_v<T>) {
      overflow_ = sizeof(T) <= sizeof(U) &&
                  val > static_cast<U>(std::numeric_limits<T>::max());
    } else {  // is_signed_v<U> && is_unsigned_v<T>.
      overflow_ = val < 0 || static_cast<std::make_unsigned_t<U>>(val) >
                                 std::numeric_limits<T>::max();
    }
  }

  template <class U>
  explicit CheckedInt(const CheckedInt<U>& other)
      : CheckedInt<T>(other.value_) {
    overflow_ |= other.overflow_;
  }

  template <class U>
  CheckedInt& operator=(const CheckedInt<U>& other) {
    *this = CheckedInt<T>(other.value_);
    overflow_ |= other.overflow_;
    return *this;
  }

  T Value() const noexcept { return value_; }

  bool Overflow() const noexcept { return overflow_; }

  TfLiteStatus Status() const noexcept {
    return overflow_ ? kTfLiteError : kTfLiteOk;
  }

  template <class U>
  CheckedInt& operator+=(const CheckedInt<U>& b) noexcept {
    return *this = *this + b;
  }

  template <class U>
  CheckedInt& operator-=(const CheckedInt<U>& b) noexcept {
    return *this = *this - b;
  }

  template <class U>
  CheckedInt& operator*=(const CheckedInt<U>& b) noexcept {
    return *this = *this * b;
  }

  template <class U>
  CheckedInt& operator/=(const CheckedInt<U>& b) noexcept {
    return *this = *this / b;
  }

  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>>
  CheckedInt& operator+=(U b) noexcept {
    return *this += CheckedInt<U>(b);
  }

  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>>
  CheckedInt& operator-=(U b) noexcept {
    return *this -= CheckedInt<U>(b);
  }

  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>>
  CheckedInt& operator*=(U b) noexcept {
    return *this *= CheckedInt<U>(b);
  }

  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>>
  CheckedInt& operator/=(U b) noexcept {
    return *this /= CheckedInt<U>(b);
  }

 private:
  // Helper constructor for operators.
  CheckedInt(T val, bool overflow) : value_(val), overflow_(overflow) {}

  template <class V, class U>
  friend CheckedInt<std::common_type_t<V, U>> operator+(
      const CheckedInt<V>& a, const CheckedInt<U>& b) noexcept;

  template <class V, class U>
  friend CheckedInt<std::common_type_t<V, U>> operator-(
      const CheckedInt<V>& a, const CheckedInt<U>& b) noexcept;

  template <class V, class U>
  friend CheckedInt<std::common_type_t<V, U>> operator*(
      const CheckedInt<V>& a, const CheckedInt<U>& b) noexcept;

  template <class V, class U>
  friend CheckedInt<std::common_type_t<V, U>> operator/(
      const CheckedInt<V>& a, const CheckedInt<U>& b) noexcept;

  template <class U>
  friend class CheckedInt;

  template <class U>
  friend bool operator==(const CheckedInt<T>& a,
                         const CheckedInt<U>& b) noexcept {
    return a == b.Value();
  }

  template <class U>
  friend bool operator!=(const CheckedInt<T>& a,
                         const CheckedInt<U>& b) noexcept {
    return !(a == b);
  }

  template <class U>
  friend bool operator<(const CheckedInt<T>& a,
                        const CheckedInt<U>& b) noexcept {
    return a < b.Value();
  }

  template <class U>
  friend bool operator<=(const CheckedInt<T>& a,
                         const CheckedInt<U>& b) noexcept {
    return a <= b.Value();
  }

  template <class U>
  friend bool operator>(const CheckedInt<T>& a,
                        const CheckedInt<U>& b) noexcept {
    return b < a.Value();
  }

  template <class U>
  friend bool operator>=(const CheckedInt<T>& a,
                         const CheckedInt<U>& b) noexcept {
    return b <= a.Value();
  }

#define TFLITE_OVERFLOW_AWARE_INT_MIXED_OP(OP)                           \
  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>> \
  friend auto operator OP(const CheckedInt<T>& a, U b) noexcept {        \
    return a OP CheckedInt<U>(b);                                        \
  }                                                                      \
  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>> \
  friend auto operator OP(U a, const CheckedInt<T>& b) noexcept {        \
    return CheckedInt<U>(a) OP b;                                        \
  }

#define TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(OP)                            \
  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>>      \
  friend bool operator OP(const CheckedInt<T>& a, U b) noexcept {             \
    if constexpr (std::is_signed_v<T> == std::is_signed_v<U>) {               \
      return a.Value() OP b;                                                  \
    } else if constexpr (std::is_signed_v<T>) {                               \
      return a.Value() >= 0 ? static_cast<std::make_unsigned_t<T>>(a.Value()) \
                                  OP b                                        \
                            : (0 OP 1);                                       \
    } else {                                                                  \
      return b >= 0 ? a.Value() OP static_cast<std::make_unsigned_t<U>>(b)    \
                    : (1 OP 0);                                               \
    }                                                                         \
  }                                                                           \
  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>>      \
  friend bool operator OP(U a, const CheckedInt<T>& b) noexcept {             \
    if constexpr (std::is_signed_v<U> == std::is_signed_v<T>) {               \
      return a OP b.Value();                                                  \
    } else if constexpr (std::is_signed_v<U>) {                               \
      return a >= 0 ? static_cast<std::make_unsigned_t<U>>(a) OP b.Value()    \
                    : (0 OP 1);                                               \
    } else {                                                                  \
      return b.Value() >= 0                                                   \
                 ? a OP static_cast<std::make_unsigned_t<T>>(b.Value())       \
                 : (1 OP 0);                                                  \
    }                                                                         \
  }

  // NOLINTBEGIN(whitespace/operators)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_OP(+)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_OP(-)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_OP(*)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_OP(/)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(==)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(!=)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(<)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(<=)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(>)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(>=)
  // NOLINTEND(whitespace/operators)
  //
#undef TFLITE_OVERFLOW_AWARE_INT_MIXED_OP
#undef TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP

 private:
  T value_{};
  // Aligning the flag to the word size keeps `CheckedInt` values of types
  // smaller than a word from being handled as a single packed 64-bit integer,
  // which needs extra instructions to pack and unpack the value and the flag,
  // even after inlining.
  alignas(sizeof(void*)) bool overflow_ = false;
};

template <class T>
CheckedInt(T) -> CheckedInt<T>;

template <class T, class U>
CheckedInt<std::common_type_t<T, U>> operator+(
    const CheckedInt<T>& a, const CheckedInt<U>& b) noexcept {
  CheckedInt<std::common_type_t<T, U>> res;
  res.overflow_ = TFLITE_CHECKED_INT_MERGE_OVERFLOW(
      TFLITE_CHECKED_INT_MERGE_OVERFLOW(a.overflow_, b.overflow_),
      checked_int_internal::AddOverflow(a.value_, b.value_, &res.value_));
  return res;
}

template <class T, class U>
CheckedInt<std::common_type_t<T, U>> operator-(
    const CheckedInt<T>& a, const CheckedInt<U>& b) noexcept {
  CheckedInt<std::common_type_t<T, U>> res;
  res.overflow_ = TFLITE_CHECKED_INT_MERGE_OVERFLOW(
      TFLITE_CHECKED_INT_MERGE_OVERFLOW(a.overflow_, b.overflow_),
      checked_int_internal::SubOverflow(a.value_, b.value_, &res.value_));
  return res;
}

template <class T, class U>
CheckedInt<std::common_type_t<T, U>> operator*(
    const CheckedInt<T>& a, const CheckedInt<U>& b) noexcept {
  CheckedInt<std::common_type_t<T, U>> res;
  res.overflow_ = TFLITE_CHECKED_INT_MERGE_OVERFLOW(
      TFLITE_CHECKED_INT_MERGE_OVERFLOW(a.overflow_, b.overflow_),
      checked_int_internal::MulOverflow(a.value_, b.value_, &res.value_));
  return res;
}

template <class T, class U>
CheckedInt<std::common_type_t<T, U>> operator/(
    const CheckedInt<T>& a, const CheckedInt<U>& b) noexcept {
  CheckedInt<std::common_type_t<T, U>> res;
  if (TFLITE_EXPECT_FALSE(
          checked_int_internal::DivOverflow(a.value_, b.value_, &res.value_))) {
    res.overflow_ = true;
  } else {
    res.overflow_ = TFLITE_CHECKED_INT_MERGE_OVERFLOW(a.overflow_, b.overflow_);
  }
  return res;
}

#undef TFLITE_CHECKED_INT_MERGE_OVERFLOW
#undef TFLITE_CHECKED_INT_MSVC_INTRINSICS

}  // namespace tflite

#endif  // TENSORFLOW_LITE_CHECKED_INT_H_
