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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_I4_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_I4_H_

#include <cstdint>
#include <limits>
#include <ostream>
#include <type_traits>

namespace shlo_ref {

struct I4 {
  int8_t data = 0;

  constexpr I4() = default;
  constexpr I4(const I4&) = default;
  constexpr I4& operator=(const I4&) = default;

  template <class T>
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr I4(T v) : data(v) {}

  template <class T>
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator T() const {
    return static_cast<T>(data);
  }

  // ++a
  friend I4& operator++(I4& lhs) {
    ++lhs.data;
    return lhs;
  }
  // --a
  friend I4& operator--(I4& lhs) {
    --lhs.data;
    return lhs;
  }
  // a++
  friend I4 operator++(I4& lhs, int) {
    I4 ret = lhs;
    ++lhs.data;
    return ret;
  }
  // a--
  friend I4 operator--(I4& lhs, int) {
    I4 ret = lhs;
    --lhs.data;
    return ret;
  }
  // a += b
  friend I4& operator+=(I4& lhs, I4 rhs) {
    lhs.data += rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator+=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data += static_cast<C>(rhs);
    return lhs;
  }
  // a -= b
  friend I4& operator-=(I4& lhs, I4 rhs) {
    lhs.data -= rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator-=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data -= static_cast<C>(rhs);
    return lhs;
  }
  // a *= b
  friend I4& operator*=(I4& lhs, I4 rhs) {
    lhs.data *= rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator*=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data *= static_cast<C>(rhs);
    return lhs;
  }
  // a /= b
  friend I4& operator/=(I4& lhs, I4 rhs) {
    lhs.data /= rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator/=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data /= static_cast<C>(rhs);
    return lhs;
  }
  // a %= b
  friend I4& operator%=(I4& lhs, I4 rhs) {
    lhs.data %= rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator%=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data %= static_cast<C>(rhs);
    return lhs;
  }
  // a &= b
  friend I4& operator&=(I4& lhs, I4 rhs) {
    lhs.data &= rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator&=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data &= static_cast<C>(rhs);
    return lhs;
  }
  // a |= b
  friend I4& operator|=(I4& lhs, I4 rhs) {
    lhs.data |= rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator|=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data |= static_cast<C>(rhs);
    return lhs;
  }
  // a ^= b
  friend I4& operator^=(I4& lhs, I4 rhs) {
    lhs.data ^= rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator^=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data ^= static_cast<C>(rhs);
    return lhs;
  }
  // a <<= b
  friend I4& operator<<=(I4& lhs, I4 rhs) {
    lhs.data <<= rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator<<=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data <<= static_cast<C>(rhs);
    return lhs;
  }
  // a >>= b
  friend I4& operator>>=(I4& lhs, I4 rhs) {
    lhs.data >>= rhs.data;
    return lhs;
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend I4& operator>>=(I4& lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    lhs.data >>= static_cast<C>(rhs);
    return lhs;
  }
  // +a
  friend auto operator+(I4 lhs) { return +lhs.data; }
  // -a
  friend auto operator-(I4 lhs) { return -lhs.data; }
  // a + b
  friend auto operator+(I4 lhs, I4 rhs) { return lhs.data + rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator+(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data + static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator+(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) + rhs.data;
  }
  // a - b
  friend auto operator-(I4 lhs, I4 rhs) { return lhs.data - rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator-(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data - static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator-(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) - rhs.data;
  }
  // a * b
  friend auto operator*(I4 lhs, I4 rhs) { return lhs.data * rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator*(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data * static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator*(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) * rhs.data;
  }
  // a / b
  friend auto operator/(I4 lhs, I4 rhs) { return lhs.data / rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator/(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data / static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator/(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) / rhs.data;
  }
  // a % b
  friend auto operator%(I4 lhs, I4 rhs) { return lhs.data % rhs.data; }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator%(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data % static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator%(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) % rhs.data;
  }
  // ~a
  friend auto operator~(I4 lhs) { return ~lhs.data; }
  // a & b
  friend auto operator&(I4 lhs, I4 rhs) { return lhs.data & rhs.data; }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator&(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data & static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator&(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) & rhs.data;
  }
  // a | b
  friend auto operator|(I4 lhs, I4 rhs) { return lhs.data | rhs.data; }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator|(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data | static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator|(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) | rhs.data;
  }
  // a ^ b
  friend auto operator^(I4 lhs, I4 rhs) { return lhs.data ^ rhs.data; }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator^(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data ^ static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator^(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) ^ rhs.data;
  }
  // a << b
  friend auto operator<<(I4 lhs, I4 rhs) { return lhs.data << rhs.data; }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator<<(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data << static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator<<(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) << rhs.data;
  }
  // a >> b
  friend auto operator>>(I4 lhs, I4 rhs) { return lhs.data >> rhs.data; }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator>>(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data >> static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_integral_v<T>>>
  friend auto operator>>(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) >> rhs.data;
  }
  // !a
  friend bool operator!(I4 v) { return !v.data; }
  // a && b
  friend auto operator&&(I4 lhs, I4 rhs) { return lhs.data && rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator&&(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data && static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator&&(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) && rhs.data;
  }
  // a || b
  friend auto operator||(I4 lhs, I4 rhs) { return lhs.data || rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator||(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data || static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend auto operator||(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) || rhs.data;
  }
  // a == b
  friend bool operator==(I4 lhs, I4 rhs) { return lhs.data == rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator==(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data == static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator==(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) == rhs.data;
  }
  // a != b
  friend bool operator!=(I4 lhs, I4 rhs) { return lhs.data != rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator!=(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data != static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator!=(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) != rhs.data;
  }
  // a < b
  friend bool operator<(I4 lhs, I4 rhs) { return lhs.data < rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator<(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data < static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator<(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) < rhs.data;
  }
  // a > b
  friend bool operator>(I4 lhs, I4 rhs) { return lhs.data > rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator>(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data > static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator>(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) > rhs.data;
  }
  // a <= b
  friend bool operator<=(I4 lhs, I4 rhs) { return lhs.data <= rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator<=(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data <= static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator<=(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) <= rhs.data;
  }
  // a >= b
  friend bool operator>=(I4 lhs, I4 rhs) { return lhs.data >= rhs.data; }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator>=(I4 lhs, T rhs) {
    using C = std::common_type_t<T, int>;
    return lhs.data >= static_cast<C>(rhs);
  }
  template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
  friend bool operator>=(T lhs, I4 rhs) {
    using C = std::common_type_t<T, int>;
    return static_cast<C>(lhs) >= rhs.data;
  }

  friend std::ostream& operator<<(std::ostream& os, I4 v) { return os << +v; }
};

}  // namespace shlo_ref

namespace std {

template <>
struct numeric_limits<shlo_ref::I4> : std::numeric_limits<int8_t> {
  static constexpr shlo_ref::I4 min() noexcept { return shlo_ref::I4(-8); }
  static constexpr shlo_ref::I4 lowest() noexcept { return min(); }
  static constexpr shlo_ref::I4 max() noexcept { return shlo_ref::I4(7); }
};

}  // namespace std

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_I4_H_
