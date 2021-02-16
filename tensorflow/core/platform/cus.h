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

#ifndef TENSORFLOW_CORE_PLATFORM_CUS_TYPE_H_
#define TENSORFLOW_CORE_PLATFORM_CUS_TYPE_H_

// This type only supports conversion back and forth with float.

#include<complex>
namespace tensorflow {


// https://stackoverflow.com/questions/25734477/type-casting-struct-to-integer-c


typedef std::complex<float> complex64;
typedef std::complex<double> complex128;

struct cus {
  float value;

  void set(float f) { value = f; }
  cus(){}
  
  explicit cus(const float f) : value(f) {}
  explicit cus(const double d) : cus(static_cast<float>(d)) {}
  explicit cus(const complex64 c64) : cus(c64.real()) {}
  explicit cus(const complex128 c128) : cus(static_cast<float>(c128.real())) {}

  template<class T>
  explicit cus(const T& value) : cus(static_cast<float>(value)) {}


  operator float() const { return value; }

  inline cus& operator=(float i) { 
    this->set(i);
    return *this;
  }

  inline cus& operator=(const cus& a){
    this->set(static_cast<float>(a));
    return *this;
  }
};

inline cus operator+(const cus & a, const cus & b){
  return cus(static_cast<float>(a) + static_cast<float>(b));
}

inline cus operator-(const cus & a, const cus & b){
  return cus(static_cast<float>(a) - static_cast<float>(b));
}

inline cus operator*(const cus & a, const cus & b){
  return cus(static_cast<float>(a) * static_cast<float>(b));
}

inline cus operator/(const cus & a, const cus & b){
  return cus(static_cast<float>(a) / static_cast<float>(b));
}

inline cus operator+=(cus & a, const cus & b){
  a = a + b;
  return a;
}

inline cus operator-=(cus & a, const cus & b){
  a = a - b;
  return a;
}

inline cus operator*=(cus & a, const cus & b){
  a = a * b;
  return a;
}

inline cus operator/=(cus & a, const cus & b){
  a = a / b;
  return a;
}

inline bool operator<(const cus & a, const cus & b){
  return static_cast<float>(a) < static_cast<float>(b);
}

inline bool operator<=(const cus & a, const cus & b){
  return static_cast<float>(a) <= static_cast<float>(b);
}

inline bool operator==(const cus & a, const cus & b){
  return static_cast<float>(a) == static_cast<float>(b);
}

inline bool operator!=(const cus & a, const cus & b){
  return static_cast<float>(a) != static_cast<float>(b);
}

inline bool operator>(const cus & a, const cus & b){
  return static_cast<float>(a) > static_cast<float>(b);
}

inline bool operator>=(const cus & a, const cus & b){
  return static_cast<float>(a) >= static_cast<float>(b);
}




}  // namespace tensorflow



#endif  // TENSORFLOW_CORE_PLATFORM_CUS_TYPE_H_
