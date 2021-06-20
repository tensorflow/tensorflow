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

#include "tensorflow/core/platform/cus.h"

#include <complex>
#include <cstring>

namespace tensorflow {

uint32_t cus::castF32ToValue(const float& f) const {
  uint32_t u;
  static_assert(sizeof f == sizeof u);
  memcpy(&u, &f, sizeof value);
  return u;
}

float cus::castValueToF32(const uint32_t& v) const {
  float f;
  static_assert(sizeof f == sizeof value);
  memcpy(&f, &value, sizeof f);
  return f;
}

CUSTOM_DEVICE_FUNC cus operator+(const cus& a, const cus& b) {
  return cus(static_cast<float>(a) + static_cast<float>(b));
}

CUSTOM_DEVICE_FUNC cus operator-(const cus& a) {
  return cus(-static_cast<float>(a));
}

CUSTOM_DEVICE_FUNC cus operator-(const cus& a, const cus& b) {
  return cus(static_cast<float>(a) - static_cast<float>(b));
}

CUSTOM_DEVICE_FUNC cus operator*(const cus& a, const cus& b) {
  return cus(static_cast<float>(a) * static_cast<float>(b));
}

CUSTOM_DEVICE_FUNC cus operator/(const cus& a, const cus& b) {
  return cus(static_cast<float>(a) / static_cast<float>(b));
}

CUSTOM_DEVICE_FUNC cus operator+=(cus& a, const cus& b) {
  a = a + b;
  return a;
}

CUSTOM_DEVICE_FUNC cus operator-=(cus& a, const cus& b) {
  a = a - b;
  return a;
}

CUSTOM_DEVICE_FUNC cus operator*=(cus& a, const cus& b) {
  a = a * b;
  return a;
}

CUSTOM_DEVICE_FUNC cus operator/=(cus& a, const cus& b) {
  a = a / b;
  return a;
}

CUSTOM_DEVICE_FUNC bool operator<(const cus& a, const cus& b) {
  return static_cast<float>(a) < static_cast<float>(b);
}

CUSTOM_DEVICE_FUNC bool operator<=(const cus& a, const cus& b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}

CUSTOM_DEVICE_FUNC bool operator==(const cus& a, const cus& b) {
  return static_cast<float>(a) == static_cast<float>(b);
}

CUSTOM_DEVICE_FUNC bool operator!=(const cus& a, const cus& b) {
  return static_cast<float>(a) != static_cast<float>(b);
}

CUSTOM_DEVICE_FUNC bool operator>(const cus& a, const cus& b) {
  return static_cast<float>(a) > static_cast<float>(b);
}

CUSTOM_DEVICE_FUNC bool operator>=(const cus& a, const cus& b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

}  // namespace tensorflow
