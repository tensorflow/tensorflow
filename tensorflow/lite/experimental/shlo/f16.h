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

#ifdef SHLO_REF_HAS_BUILTIN_FLOAT16

#define SHLO_REF_DEFINE_BINARY_OP(OP)                                         \
  friend F16 operator OP(F16 x, F16 y) { return x.native_ OP y.native_; }     \
                                                                              \
  template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>> \
  friend auto operator OP(F16 x, T y) {                                       \
    return x.native_ OP y;                                                    \
  }                                                                           \
                                                                              \
  template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>> \
  friend auto operator OP(T x, F16 y) {                                       \
    return x OP y.native_;                                                    \
  }

#define SHLO_REF_DEFINE_BINARY_ASSIGN_OP(OP)                               \
  SHLO_REF_DEFINE_BINARY_OP(OP);                                           \
  friend F16& operator OP##=(F16 & x, F16 y) {                             \
    x.native_ OP## = y.native_;                                            \
    return x;                                                              \
  }                                                                        \
                                                                           \
  template <class T, typename = std::enable_if_t<std::is_arithmetic_v<T>>> \
  friend F16& operator OP##=(F16 & x, T y) {                               \
    x.native_ OP## = y;                                                    \
    return x;                                                              \
  }

#else  // !SHLO_REF_HAS_BUILTIN_FLOAT16

#define SHLO_REF_DEFINE_BINARY_OP(OP)                              \
  friend F16 operator OP(F16 x, F16 y) {                           \
    return F16(static_cast<float>(x) OP static_cast<float>(y));    \
  }                                                                \
                                                                   \
  template <typename T, typename C = std::common_type_t<F16, T>>   \
  friend C operator OP(F16 x, T y) {                               \
    return static_cast<C>(static_cast<C>(x) OP static_cast<C>(y)); \
  }                                                                \
                                                                   \
  template <typename T, typename C = std::common_type_t<F16, T>>   \
  friend C operator OP(T x, F16 y) {                               \
    return static_cast<C>(static_cast<C>(x) OP static_cast<C>(y)); \
  }

#define SHLO_REF_DEFINE_BINARY_ASSIGN_OP(OP)                               \
  SHLO_REF_DEFINE_BINARY_OP(OP);                                           \
  friend F16& operator OP##=(F16 & x, F16 y) {                             \
    return x = F16(static_cast<float>(x) OP static_cast<float>(y));        \
  }                                                                        \
                                                                           \
  template <class T, typename = std::enable_if_t<std::is_arithmetic_v<T>>> \
  friend F16& operator OP##=(F16 & x, T y) {                               \
    return x = static_cast<float>(x) OP y;                                 \
  }

#endif  // SHLO_REF_HAS_BUILTIN_FLOAT16

  friend F16 operator+(F16 x);
  friend F16 operator-(F16 x);
  friend F16& operator++(F16& x);
  friend F16 operator++(F16& x, int);
  friend F16& operator--(F16& x);
  friend F16 operator--(F16& x, int);
  SHLO_REF_DEFINE_BINARY_ASSIGN_OP(+);
  SHLO_REF_DEFINE_BINARY_ASSIGN_OP(-);
  SHLO_REF_DEFINE_BINARY_ASSIGN_OP(*);
  SHLO_REF_DEFINE_BINARY_ASSIGN_OP(/);
  SHLO_REF_DEFINE_BINARY_OP(<);
  SHLO_REF_DEFINE_BINARY_OP(<=);
  SHLO_REF_DEFINE_BINARY_OP(>);
  SHLO_REF_DEFINE_BINARY_OP(>=);
  SHLO_REF_DEFINE_BINARY_OP(==);
  SHLO_REF_DEFINE_BINARY_OP(!=);
#undef SHLO_REF_DEFINE_BINARY_ASSIGN_OP
#undef SHLO_REF_DEFINE_BINARY_OP

 private:
  union {
#ifdef SHLO_REF_HAS_BUILTIN_FLOAT16
    _Float16 native_;
#endif

    uint16_t bits_;
  };
};

namespace detail {

template <class T, class SFINAE = void>
struct F16CommonType {};

template <class T>
struct F16CommonType<T, std::enable_if_t<std::is_integral_v<T>>> {
  using type = F16;
};

template <class T>
struct F16CommonType<T, std::enable_if_t<std::is_floating_point_v<T>>> {
  using type = T;
};

template <class T>
struct F16CommonType<T, std::enable_if_t<!std::is_arithmetic_v<T> &&
                                         std::is_convertible_v<T, float>>> {
  using type = float;
};

template <class T>
struct F16CommonType<T, std::enable_if_t<!std::is_arithmetic_v<T> &&
                                         !std::is_convertible_v<T, float> &&
                                         std::is_convertible_v<T, double>>> {
  using type = double;
};

}  // namespace detail

}  // namespace shlo_ref

namespace std {

template <>
struct common_type<shlo_ref::F16, shlo_ref::F16> {
  using type = shlo_ref::F16;
};

template <typename T>
struct common_type<shlo_ref::F16, T> : shlo_ref::detail::F16CommonType<T> {};

template <typename T>
struct common_type<T, shlo_ref::F16> : shlo_ref::detail::F16CommonType<T> {};

}  // namespace std

namespace shlo_ref {
static_assert(sizeof(F16) == sizeof(uint16_t));
static_assert(alignof(F16) == alignof(uint16_t));

#ifdef SHLO_REF_HAS_BUILTIN_FLOAT16

template <typename T, typename _>
F16::F16(T x) : native_(static_cast<_Float16>(static_cast<float>(x))) {}

inline F16::operator float() const { return native_; }

inline F16::operator bool() const { return native_; }

inline F16 operator+(F16 x) { return x.native_; }

inline F16 operator-(F16 x) { return -x.native_; }

inline F16& operator++(F16& x) { return x += 1; }

inline F16 operator++(F16& x, int) { return x.native_++; }

inline F16& operator--(F16& x) { return x -= 1; }

inline F16 operator--(F16& x, int) { return x.native_--; }

#else  // !SHLO_REF_HAS_BUILTIN_FLOAT16

template <typename T, typename _>
inline F16::F16(T x)
    : bits_(fp16_ieee_from_fp32_value(static_cast<float>(x))) {}

inline F16::operator float() const { return fp16_ieee_to_fp32_value(bits_); }

inline F16::operator bool() const { return bits_; }

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

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_F16_H_
