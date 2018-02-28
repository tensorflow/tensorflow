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

#ifndef TENSORFLOW_CORE_LIB_BFLOAT16_BFLOAT16_H_
#define TENSORFLOW_CORE_LIB_BFLOAT16_BFLOAT16_H_

#include <complex>

#ifdef __CUDACC__
// All functions callable from CUDA code must be qualified with __device__
#define B16_DEVICE_FUNC __host__ __device__

#else
#define B16_DEVICE_FUNC

#endif

namespace Eigen {
struct half;
}

namespace tensorflow {

// Single precision complex.
typedef std::complex<float> complex64;
// Double precision complex.
typedef std::complex<double> complex128;

// see framework/bfloat16.h for description.
struct bfloat16 {
  B16_DEVICE_FUNC bfloat16() {}

  B16_DEVICE_FUNC explicit bfloat16(const float v) {
    if (float_isnan(v)) {
      value = NAN_VALUE;
      return;
    }
    const uint16_t* p = reinterpret_cast<const uint16_t*>(&v);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    value = p[0];
#else
    value = p[1];
#endif
  }

  B16_DEVICE_FUNC explicit bfloat16(const double val)
      : bfloat16(static_cast<float>(val)) {}
  // Following the convention of numpy, converting between complex and
  // float will lead to loss of imag value.
  B16_DEVICE_FUNC explicit bfloat16(const complex64& val)
      : bfloat16(val.real()) {}

  B16_DEVICE_FUNC explicit bfloat16(const complex128& val)
      : bfloat16(static_cast<float>(val.real())) {}

  B16_DEVICE_FUNC explicit bfloat16(const unsigned short val)
      : bfloat16(static_cast<float>(val)) {}

  B16_DEVICE_FUNC explicit bfloat16(const unsigned int val)
      : bfloat16(static_cast<float>(val)) {}

  B16_DEVICE_FUNC explicit bfloat16(const int val)
      : bfloat16(static_cast<float>(val)) {}

  B16_DEVICE_FUNC explicit bfloat16(const long val)
      : bfloat16(static_cast<float>(val)) {}

  B16_DEVICE_FUNC explicit bfloat16(const long long val)
      : bfloat16(static_cast<float>(val)) {}

  template <class T>
  B16_DEVICE_FUNC explicit bfloat16(const T& val)
      : bfloat16(static_cast<float>(val)) {}

  B16_DEVICE_FUNC explicit operator float() const {
    float result;

    uint16_t* q = reinterpret_cast<uint16_t*>(&result);

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    q[0] = value;
    q[1] = 0;
#else
    q[0] = 0;
    q[1] = value;
#endif
    return result;
  }

  B16_DEVICE_FUNC explicit operator bool() const {
    return static_cast<bool>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator Eigen::half() const;

  B16_DEVICE_FUNC explicit operator short() const {
    return static_cast<short>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator int() const {
    return static_cast<int>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator long() const {
    return static_cast<long>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator char() const {
    return static_cast<char>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator signed char() const {
    return static_cast<signed char>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator unsigned char() const {
    return static_cast<unsigned char>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator unsigned short() const {
    return static_cast<unsigned short>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator unsigned int() const {
    return static_cast<unsigned int>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator unsigned long() const {
    return static_cast<unsigned long>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator unsigned long long() const {
    return static_cast<unsigned long long>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator long long() const {
    return static_cast<long long>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator double() const {
    return static_cast<double>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator complex64() const {
    return complex64(float(*this), float(0.0));
  }

  B16_DEVICE_FUNC explicit operator complex128() const {
    return complex128(double(*this), double(0.0));
  }

  static bfloat16 epsilon() {
    bfloat16 x;
    x.value = 0x3c00;  // 0x1.0p-7
    return x;
  }

  uint16_t value;

  // A value that represents "not a number".
  static const uint16_t NAN_VALUE = 0x7FC0;

 private:
  B16_DEVICE_FUNC bool float_isnan(const float& x) {
#ifdef __CUDA_ARCH__
    return ::isnan(x);
#else
    return std::isnan(x);
#endif
  }
};

B16_DEVICE_FUNC inline std::ostream& operator<<(std::ostream& os,
                                                const bfloat16& dt) {
  os << static_cast<float>(dt);
  return os;
}

B16_DEVICE_FUNC inline bfloat16 operator+(bfloat16 a, bfloat16 b) {
  return bfloat16(static_cast<float>(a) + static_cast<float>(b));
}
B16_DEVICE_FUNC inline bfloat16 operator+(bfloat16 a, int b) {
  return bfloat16(static_cast<float>(a) + static_cast<float>(b));
}
B16_DEVICE_FUNC inline bfloat16 operator+(int a, bfloat16 b) {
  return bfloat16(static_cast<float>(a) + static_cast<float>(b));
}
B16_DEVICE_FUNC inline bfloat16 operator-(bfloat16 a, bfloat16 b) {
  return bfloat16(static_cast<float>(a) - static_cast<float>(b));
}
B16_DEVICE_FUNC inline bfloat16 operator*(bfloat16 a, bfloat16 b) {
  return bfloat16(static_cast<float>(a) * static_cast<float>(b));
}
B16_DEVICE_FUNC inline bfloat16 operator/(bfloat16 a, bfloat16 b) {
  return bfloat16(static_cast<float>(a) / static_cast<float>(b));
}
B16_DEVICE_FUNC inline bfloat16 operator-(bfloat16 a) {
  a.value ^= 0x8000;
  return a;
}
B16_DEVICE_FUNC inline bool operator<(bfloat16 a, bfloat16 b) {
  return static_cast<float>(a) < static_cast<float>(b);
}
B16_DEVICE_FUNC inline bool operator<=(bfloat16 a, bfloat16 b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}
B16_DEVICE_FUNC inline bool operator==(bfloat16 a, bfloat16 b) {
  return static_cast<float>(a) == static_cast<float>(b);
}
B16_DEVICE_FUNC inline bool operator!=(bfloat16 a, bfloat16 b) {
  return static_cast<float>(a) != static_cast<float>(b);
}
B16_DEVICE_FUNC inline bool operator>(bfloat16 a, bfloat16 b) {
  return static_cast<float>(a) > static_cast<float>(b);
}
B16_DEVICE_FUNC inline bool operator>=(bfloat16 a, bfloat16 b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}
B16_DEVICE_FUNC inline bfloat16& operator+=(bfloat16& a, bfloat16 b) {
  a = a + b;
  return a;
}
B16_DEVICE_FUNC inline bfloat16& operator-=(bfloat16& a, bfloat16 b) {
  a = a - b;
  return a;
}
B16_DEVICE_FUNC inline bfloat16 operator++(bfloat16& a) {
  a += bfloat16(1);
  return a;
}
B16_DEVICE_FUNC inline bfloat16 operator--(bfloat16& a) {
  a -= bfloat16(1);
  return a;
}
B16_DEVICE_FUNC inline bfloat16 operator++(bfloat16& a, int) {
  bfloat16 original_value = a;
  ++a;
  return original_value;
}
B16_DEVICE_FUNC inline bfloat16 operator--(bfloat16& a, int) {
  bfloat16 original_value = a;
  --a;
  return original_value;
}
B16_DEVICE_FUNC inline bfloat16& operator*=(bfloat16& a, bfloat16 b) {
  a = a * b;
  return a;
}
B16_DEVICE_FUNC inline bfloat16& operator/=(bfloat16& a, bfloat16 b) {
  a = a / b;
  return a;
}
}  // end namespace tensorflow

namespace std {
template <>
struct hash<tensorflow::bfloat16> {
  size_t operator()(const tensorflow::bfloat16& v) const {
    return hash<float>()(static_cast<float>(v));
  }
};
}  // namespace std

#endif  // TENSORFLOW_CORE_LIB_BFLOAT16_BFLOAT16_H_
