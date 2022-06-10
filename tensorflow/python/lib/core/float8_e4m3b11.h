/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PYTHON_LIB_CORE_FLOAT_E4M3B11_H_
#define TENSORFLOW_PYTHON_LIB_CORE_FLOAT_E4M3B11_H_

#include <stdint.h>

#include <cmath>
#include <cstring>
#include <memory>

namespace tensorflow {

uint8_t float_to_float8_e4m3b11(float v);
float float8_e4m3b11_to_float(uint8_t v);

class float8_e4m3b11 {
 public:
  // Exponent: 4, Mantissa: 3, bias: 11
  float8_e4m3b11() {}
  float8_e4m3b11(float v) : rep_(float_to_float8_e4m3b11(v)) {}  // NOLINT

  operator float() const {  // NOLINT: Allow implicit conversion to float,
                            // because it is lossless.
    return float8_e4m3b11_to_float(rep_);
  }

  float8_e4m3b11 operator-() const {
    if ((rep_ & 0x7f) == 0x00) {
      return *this;
    }  // nan or 0.
    float8_e4m3b11 result = *this;
    result.rep_ = result.rep_ ^ 0x80;
    return result;
  }

  uint8_t rep() const { return rep_; }

  static float8_e4m3b11 FromRep(uint8_t rep) {
    float8_e4m3b11 result;
    memcpy(&result, &rep, sizeof(float8_e4m3b11));
    return result;
  }

 private:
  uint8_t rep_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_LIB_CORE_FLOAT_E4M3B11_H_
