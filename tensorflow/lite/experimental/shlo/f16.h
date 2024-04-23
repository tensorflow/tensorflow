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
#if defined(__FLT16_MAX__) && !SHLO_REF_EMULATE_F16
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

  explicit operator bool() const;

  F16& operator=(float x) { return *this = static_cast<F16>(x); }

#define SHLO_REF_DECLARE_BINARY_OP(OP)   \
  friend F16 operator OP(F16 x, F16 y);  \
  template <typename T, typename SIFNAE> \
  friend auto operator OP(F16 x, T y);   \
  template <typename T, typename SIFNAE> \
  friend auto operator OP(T x, F16 y);

#define SHLO_REF_DECLARE_BINARY_ASSIGN_OP(OP) \
  SHLO_REF_DECLARE_BINARY_OP(OP);             \
  friend F16& operator OP##=(F16 & x, F16 y); \
  template <typename T, typename SIFNAE>      \
  friend F16& operator OP##=(F16 & x, T y);

  friend F16 operator+(F16 x);
  friend F16 operator-(F16 x);
  friend F16& operator++(F16& x);
  friend F16 operator++(F16& x, int);
  friend F16& operator--(F16& x);
  friend F16 operator--(F16& x, int);
  SHLO_REF_DECLARE_BINARY_ASSIGN_OP(+);
  SHLO_REF_DECLARE_BINARY_ASSIGN_OP(-);
  SHLO_REF_DECLARE_BINARY_ASSIGN_OP(*);
  SHLO_REF_DECLARE_BINARY_ASSIGN_OP(/);
  SHLO_REF_DECLARE_BINARY_OP(<);
  SHLO_REF_DECLARE_BINARY_OP(<=);
  SHLO_REF_DECLARE_BINARY_OP(>);
  SHLO_REF_DECLARE_BINARY_OP(>=);
  SHLO_REF_DECLARE_BINARY_OP(==);
  SHLO_REF_DECLARE_BINARY_OP(!=);
#undef SHLO_REF_DECLARE_BINARY_ASSIGN_OP
#undef SHLO_REF_DECLARE_BINARY_OP

 private:
  union {
#ifdef SHLO_REF_HAS_BUILTIN_FLOAT16
    _Float16 native_;
#endif

    uint16_t bits_;
  };
};
}  // namespace shlo_ref

namespace std {

template <typename T>
struct common_type<shlo_ref::F16, T> {
  static_assert(
      std::is_arithmetic_v<T>,
      "Can only find a common type between F16 and an arithmetic types.");
  using type = shlo_ref::F16;
};

template <typename T>
struct common_type<T, shlo_ref::F16> : common_type<shlo_ref::F16, T> {};

}  // namespace std

namespace shlo_ref {
static_assert(sizeof(F16) == sizeof(uint16_t));
static_assert(alignof(F16) == alignof(uint16_t));

#ifdef SHLO_REF_HAS_BUILTIN_FLOAT16

template <typename T, typename _>
F16::F16(T x) : native_(static_cast<_Float16>(x)) {}

inline F16::operator float() const { return native_; }

inline F16::operator bool() const { return native_; }

inline F16 operator+(F16 x) { return x.native_; }

inline F16 operator-(F16 x) { return -x.native_; }

inline F16& operator++(F16& x) {
  ++x.native_;
  return x;
}

inline F16 operator++(F16& x, int) { return x.native_++; }

inline F16& operator--(F16& x) {
  --x.native_;
  return x;
}

inline F16 operator--(F16& x, int) { return x.native_--; }

#define INTERNAL_F16_ARITHMETIC_OP(OP)                                        \
  inline F16 operator OP(F16 x, F16 y) { return x.native_ OP y.native_; }     \
                                                                              \
  template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>> \
  inline auto operator OP(F16 x, T y) {                                       \
    return x.native_ OP y;                                                    \
  }                                                                           \
                                                                              \
  template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>> \
  inline auto operator OP(T x, F16 y) {                                       \
    return x OP y.native_;                                                    \
  }

#define INTERNAL_F16_ARITHMETIC_ASSIGN_OP(OP)                              \
  INTERNAL_F16_ARITHMETIC_OP(OP);                                          \
  inline F16& operator OP##=(F16 & x, F16 y) {                             \
    x.native_ OP## = y.native_;                                            \
    return x;                                                              \
  }                                                                        \
                                                                           \
  template <class T, typename = std::enable_if_t<std::is_arithmetic_v<T>>> \
  inline F16& operator OP##=(F16 & x, T y) {                               \
    x.native_ OP## = y;                                                    \
    return x;                                                              \
  }

#else  // !SHLO_REF_HAS_BUILTIN_FLOAT16

template <typename T, typename _>
inline F16::F16(T x)
    : bits_(fp16_ieee_from_fp32_value(static_cast<float>(x))) {}

inline F16::operator float() const { return fp16_ieee_to_fp32_value(bits_); }

inline F16::operator bool() const { return bits_; }

#define INTERNAL_F16_ARITHMETIC_OP(OP)                          \
  inline F16 operator OP(F16 x, F16 y) {                        \
    return F16(static_cast<float>(x) OP static_cast<float>(y)); \
  }                                                             \
                                                                \
  template <typename T, typename C = std::common_type<F16, T>>  \
  inline auto operator OP(F16 x, T y) {                         \
    return static_cast<C>(static_cast<float>(x) OP y);          \
  }                                                             \
                                                                \
  template <typename T, typename C = std::common_type<F16, T>>  \
  inline std::common_type<F16, T> operator OP(T x, F16 y) {     \
    return static_cast<C>(x OP static_cast<float>(y));          \
  }

#define INTERNAL_F16_ARITHMETIC_ASSIGN_OP(OP)                              \
  INTERNAL_F16_ARITHMETIC_OP(OP);                                          \
  inline F16& operator OP##=(F16 & x, F16 y) {                             \
    return x = F16(static_cast<float>(x) OP static_cast<float>(y));        \
  }                                                                        \
                                                                           \
  template <class T, typename = std::enable_if_t<std::is_arithmetic_v<T>>> \
  inline F16& operator OP##=(F16 & x, T y) {                               \
    return x = static_cast<float>(x) OP y;                                 \
  }

inline F16 operator-(F16 x) { return F16(-static_cast<float>(x)); }
inline F16 operator+(F16 x) { return F16(static_cast<float>(x)); }

inline F16& operator++(F16& x) { return x += 1; }

inline F16 operator++(F16& x, int) {
  const F16 y = x;
  ++x;
  return y;
}

inline F16& operator--(F16& x) { return x -= 1; }

inline F16 operator--(F16& x, int) {
  const F16 y = x;
  --x;
  return y;
}

#endif

INTERNAL_F16_ARITHMETIC_ASSIGN_OP(+)
INTERNAL_F16_ARITHMETIC_ASSIGN_OP(-)
INTERNAL_F16_ARITHMETIC_ASSIGN_OP(*)
INTERNAL_F16_ARITHMETIC_ASSIGN_OP(/)
INTERNAL_F16_ARITHMETIC_OP(<)
INTERNAL_F16_ARITHMETIC_OP(<=)
INTERNAL_F16_ARITHMETIC_OP(>)
INTERNAL_F16_ARITHMETIC_OP(>=)
INTERNAL_F16_ARITHMETIC_OP(==)
INTERNAL_F16_ARITHMETIC_OP(!=)

#undef INTERNAL_F16_ARITHMETIC_OP
#undef INTERNAL_F16_ARITHMETIC_ASSIGN_OP

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_F16_H_
