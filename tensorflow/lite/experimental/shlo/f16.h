/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_F16_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_F16_H_

#include <cstdint>
#include <type_traits>

#include "fp16.h"  // from @FP16  // IWYU pragma: keep, used with no builtin float16

// Use __FLT16_MAX__ to determine whether _Float16 is builtin
#if defined(__FLT16_MAX__)
#define SHLO_REF_HAS_BUILTIN_FLOAT16 1
#endif

namespace shlo_ref {

class alignas(uint16_t) F16 {
 public:
  F16() = default;

  template <typename T,
            typename = std::enable_if_t<std::is_convertible_v<T, float>>>
  // Allow implicit conversions from types convertible to float.
  // NOLINTNEXTLINE(google-explicit-constructor)
  F16(T x);

  // Tagged constructor to allow construction from bits
  struct bitcast_construct_t {};
  explicit F16(bitcast_construct_t, uint16_t bits) : bits_(bits) {}

  // Allow implicit conversions to float.
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator float() const;

  F16& operator=(float x) { return *this = static_cast<F16>(x); }

  friend F16 operator+(F16 x, F16 y);
  friend F16& operator+=(F16& x, F16 y);
  friend F16 operator-(F16 x, F16 y);
  friend F16& operator-=(F16& x, F16 y);
  friend F16 operator*(F16 x, F16 y);
  friend F16& operator*=(F16& x, F16 y);
  friend F16 operator/(F16 x, F16 y);
  friend F16& operator/=(F16& x, F16 y);

  friend F16 operator-(F16 x);
  friend F16 operator+(F16 x);

 private:
  union {
#ifdef SHLO_REF_HAS_BUILTIN_FLOAT16
    _Float16 native_;
#endif

    uint16_t bits_;
  };
};

static_assert(sizeof(F16) == sizeof(uint16_t));
static_assert(alignof(F16) == alignof(uint16_t));

#ifdef SHLO_REF_HAS_BUILTIN_FLOAT16

template <typename T, typename _>
inline F16::F16(T x) : native_(static_cast<_Float16>(x)) {}

inline F16::operator float() const { return native_; }

#define INTERNAL_F16_ARITHMETIC_OP(OP) \
  inline F16 operator OP(F16 x, F16 y) { return F16(x.native_ OP y.native_); }

#define INTERNAL_F16_ARITHMETIC_ASSIGN_OP(OP)  \
  inline F16& operator OP##=(F16 & x, F16 y) { \
    return x = F16(x.native_ OP y.native_);    \
  }

inline F16 operator-(F16 x) { return F16(-x.native_); }
inline F16 operator+(F16 x) { return F16(x.native_); }

#else  // !SHLO_REF_HAS_BUILTIN_FLOAT16

template <typename T, typename _>
inline F16::F16(T x)
    : bits_(fp16_ieee_from_fp32_value(static_cast<float>(x))) {}

inline F16::operator float() const { return fp16_ieee_to_fp32_value(bits_); }

#define INTERNAL_F16_ARITHMETIC_OP(OP)                          \
  inline F16 operator OP(F16 x, F16 y) {                        \
    return F16(static_cast<float>(x) OP static_cast<float>(y)); \
  }

#define INTERNAL_F16_ARITHMETIC_ASSIGN_OP(OP)                       \
  inline F16& operator OP##=(F16 & x, F16 y) {                      \
    return x = F16(static_cast<float>(x) OP static_cast<float>(y)); \
  }

inline F16 operator-(F16 x) { return F16(-static_cast<float>(x)); }
inline F16 operator+(F16 x) { return F16(static_cast<float>(x)); }

#endif

INTERNAL_F16_ARITHMETIC_OP(+)
INTERNAL_F16_ARITHMETIC_ASSIGN_OP(+)
INTERNAL_F16_ARITHMETIC_OP(-)
INTERNAL_F16_ARITHMETIC_ASSIGN_OP(-)
INTERNAL_F16_ARITHMETIC_OP(*)
INTERNAL_F16_ARITHMETIC_ASSIGN_OP(*)
INTERNAL_F16_ARITHMETIC_OP(/)
INTERNAL_F16_ARITHMETIC_ASSIGN_OP(/)

#undef INTERNAL_F16_ARITHMETIC_OP
#undef INTERNAL_F16_ARITHMETIC_ASSIGN_OP

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_F16_H_
