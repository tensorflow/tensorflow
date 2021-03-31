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

using cusWrappedType = uint32_t;
// using cusWrappedType = float;

struct cus {
  cusWrappedType value;

 public:
  void setValue(cusWrappedType v) { value = v; }
  constexpr CUSTOM_DEVICE_FUNC cus() : value(0) {}

  // constexpr CUSTOM_DEVICE_FUNC cus(const cusWrappedType& u) : value(u) {}

  constexpr CUSTOM_DEVICE_FUNC cus(const float& f) : value(static_cast<cusWrappedType>(f)) {}

  explicit constexpr CUSTOM_DEVICE_FUNC cus(const uint32_t& u)
      : value(static_cast<cusWrappedType>(u)) {}
  explicit constexpr CUSTOM_DEVICE_FUNC cus(const double& d)
      : cus(static_cast<cusWrappedType>(d)) {}
  explicit constexpr CUSTOM_DEVICE_FUNC cus(const complex64& c64)
      : cus(static_cast<cusWrappedType>(c64.real())) {}
  explicit constexpr CUSTOM_DEVICE_FUNC cus(const complex128& c128)
      : cus(static_cast<cusWrappedType>(c128.real())) {}

  template <class T>
  explicit constexpr CUSTOM_DEVICE_FUNC cus(const T& value)
      : cus(static_cast<cusWrappedType>(value)) {}

  CUSTOM_DEVICE_FUNC operator float() const {
    return static_cast<float>(value);
  }

  explicit CUSTOM_DEVICE_FUNC operator double() const {
    return static_cast<double>(value);
  }

  explicit CUSTOM_DEVICE_FUNC operator uint32_t() const {
    return static_cast<uint32_t>(value);
  }

  // template <class T>
  // explicit CUSTOM_DEVICE_FUNC operator T() const {
  //   return static_cast<T>(value);
  // }

  // explicit CUSTOM_DEVICE_FUNC operator cusWrappedType() const { return value;
  // }

  // CUSTOM_DEVICE_FUNC inline cus& operator=(cusWrappedType i) {

  //   return *this;
  // }

  // CUSTOM_DEVICE_FUNC inline cus& operator=(const cus& a) {
  //   this->setValue(static_cast<cusWrappedType>(a));
  //   return *this;
  // }
};

CUSTOM_DEVICE_FUNC inline cus operator+(const cus& a, const cus& b) {
  return cus(static_cast<cusWrappedType>(a) + static_cast<cusWrappedType>(b));
}

CUSTOM_DEVICE_FUNC inline cus operator-(const cus& a) {
  return cus(-static_cast<cusWrappedType>(a));
}

CUSTOM_DEVICE_FUNC inline cus operator-(const cus& a, const cus& b) {
  return cus(static_cast<cusWrappedType>(a) - static_cast<cusWrappedType>(b));
}

CUSTOM_DEVICE_FUNC inline cus operator*(const cus& a, const cus& b) {
  return cus(static_cast<cusWrappedType>(a) * static_cast<cusWrappedType>(b));
}

CUSTOM_DEVICE_FUNC inline cus operator/(const cus& a, const cus& b) {
  return cus(static_cast<cusWrappedType>(a) / static_cast<cusWrappedType>(b));
}

CUSTOM_DEVICE_FUNC inline cus operator+=(cus& a, const cus& b) {
  a = a + b;
  return a;
}

CUSTOM_DEVICE_FUNC inline cus operator-=(cus& a, const cus& b) {
  a = a - b;
  return a;
}

CUSTOM_DEVICE_FUNC inline cus operator*=(cus& a, const cus& b) {
  a = a * b;
  return a;
}

CUSTOM_DEVICE_FUNC inline cus operator/=(cus& a, const cus& b) {
  a = a / b;
  return a;
}

CUSTOM_DEVICE_FUNC inline bool operator<(const cus& a, const cus& b) {
  return static_cast<cusWrappedType>(a) < static_cast<cusWrappedType>(b);
}

CUSTOM_DEVICE_FUNC inline bool operator<=(const cus& a, const cus& b) {
  return static_cast<cusWrappedType>(a) <= static_cast<cusWrappedType>(b);
}

CUSTOM_DEVICE_FUNC inline bool operator==(const cus& a, const cus& b) {
  return static_cast<cusWrappedType>(a) == static_cast<cusWrappedType>(b);
}

CUSTOM_DEVICE_FUNC inline bool operator!=(const cus& a, const cus& b) {
  return static_cast<cusWrappedType>(a) != static_cast<cusWrappedType>(b);
}

CUSTOM_DEVICE_FUNC inline bool operator>(const cus& a, const cus& b) {
  return static_cast<cusWrappedType>(a) > static_cast<cusWrappedType>(b);
}

CUSTOM_DEVICE_FUNC inline bool operator>=(const cus& a, const cus& b) {
  return static_cast<cusWrappedType>(a) >= static_cast<cusWrappedType>(b);
}

}  // namespace tensorflow

namespace std {
template <>
struct hash<tensorflow::cus> {
  std::size_t operator()(tensorflow::cus const& c) const noexcept {
    std::size_t h1 = std::hash<tensorflow::cusWrappedType>{}(c.value);
    return h1;
  }
};
}  // namespace std

#endif  // TENSORFLOW_CORE_PLATFORM_CUS_H_
