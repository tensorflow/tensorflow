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

#ifndef TENSORFLOW_CORE_PLATFORM_CUS_H_
#define TENSORFLOW_CORE_PLATFORM_CUS_H_

// This type only supports conversion back and forth with float.

#include <complex>

#ifdef __CUDACC__
// All functions callable from CUDA code must be qualified with __device__
#define CUSTOM_DEVICE_FUNC __host__ __device__

#else
#define CUSTOM_DEVICE_FUNC

#endif

namespace tensorflow {

// https://stackoverflow.com/questions/25734477/type-casting-struct-to-integer-c

typedef std::complex<float> complex64;
typedef std::complex<double> complex128;

struct cus {
  uint32_t value;

  constexpr CUSTOM_DEVICE_FUNC cus() : value(0) {}

  constexpr CUSTOM_DEVICE_FUNC cus(const float& f) : value() {
    value = castF32ToValue(f);
  }

  explicit constexpr CUSTOM_DEVICE_FUNC cus(const double& d)
      : cus(static_cast<float>(d)) {}
  explicit constexpr CUSTOM_DEVICE_FUNC cus(const complex64& c64)
      : cus(static_cast<float>(c64.real())) {}
  explicit constexpr CUSTOM_DEVICE_FUNC cus(const complex128& c128)
      : cus(static_cast<float>(c128.real())) {}

  template <class T>
  explicit constexpr CUSTOM_DEVICE_FUNC cus(const T& value)
      : cus(static_cast<float>(value)) {}

  CUSTOM_DEVICE_FUNC operator float() const { return castValueToF32(value); }

  explicit CUSTOM_DEVICE_FUNC operator double() const {
    float f = static_cast<float>(*this);
    return static_cast<double>(f);
  }

  uint32_t castF32ToValue(const float& f) const;
  float castValueToF32(const uint32_t& u) const;

  friend CUSTOM_DEVICE_FUNC cus operator+(const cus& a, const cus& b);
  friend CUSTOM_DEVICE_FUNC cus operator-(const cus& a);
  friend CUSTOM_DEVICE_FUNC cus operator-(const cus& a, const cus& b);
  friend CUSTOM_DEVICE_FUNC cus operator*(const cus& a, const cus& b);
  friend CUSTOM_DEVICE_FUNC cus operator/(const cus& a, const cus& b);
  friend CUSTOM_DEVICE_FUNC cus operator+=(cus& a, const cus& b);
  friend CUSTOM_DEVICE_FUNC cus operator-=(cus& a, const cus& b);
  friend CUSTOM_DEVICE_FUNC cus operator*=(cus& a, const cus& b);
  friend CUSTOM_DEVICE_FUNC cus operator/=(cus& a, const cus& b);
  friend CUSTOM_DEVICE_FUNC bool operator<(const cus& a, const cus& b);
  friend CUSTOM_DEVICE_FUNC bool operator<=(const cus& a, const cus& b);
  friend CUSTOM_DEVICE_FUNC bool operator==(const cus& a, const cus& b);
  friend CUSTOM_DEVICE_FUNC bool operator!=(const cus& a, const cus& b);
  friend CUSTOM_DEVICE_FUNC bool operator>(const cus& a, const cus& b);
  friend CUSTOM_DEVICE_FUNC bool operator>=(const cus& a, const cus& b);
};

}  // namespace tensorflow

namespace std {
template <>
struct hash<tensorflow::cus> {
  std::size_t operator()(tensorflow::cus const& c) const noexcept {
    std::size_t h1 = std::hash<uint32_t>{}(c.value);
    return h1;
  }
};
}  // namespace std

#endif  // TENSORFLOW_CORE_PLATFORM_CUS_H_
