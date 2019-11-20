/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TYPE_TO_TFLITETYPE_H_
#define TENSORFLOW_LITE_TYPE_TO_TFLITETYPE_H_

// Arduino build defines abs as a macro here. That is invalid C++, and breaks
// libc++'s <complex> header, undefine it.
#ifdef abs
#undef abs
#endif

#include <complex>
#include <string>

#include "tensorflow/lite/c/c_api_internal.h"

namespace tflite {

// Map statically from a c++ type to a TfLiteType. Used in interpreter for safe
// casts.
template <class T>
constexpr TfLiteType typeToTfLiteType() {
  return kTfLiteNoType;
}
template <>
constexpr TfLiteType typeToTfLiteType<int>() {
  return kTfLiteInt32;
}
template <>
constexpr TfLiteType typeToTfLiteType<int16_t>() {
  return kTfLiteInt16;
}
template <>
constexpr TfLiteType typeToTfLiteType<int64_t>() {
  return kTfLiteInt64;
}
template <>
constexpr TfLiteType typeToTfLiteType<float>() {
  return kTfLiteFloat32;
}
template <>
constexpr TfLiteType typeToTfLiteType<unsigned char>() {
  return kTfLiteUInt8;
}
template <>
constexpr TfLiteType typeToTfLiteType<int8_t>() {
  return kTfLiteInt8;
}
template <>
constexpr TfLiteType typeToTfLiteType<bool>() {
  return kTfLiteBool;
}
template <>
constexpr TfLiteType typeToTfLiteType<std::complex<float>>() {
  return kTfLiteComplex64;
}
template <>
constexpr TfLiteType typeToTfLiteType<std::string>() {
  return kTfLiteString;
}
template <>
constexpr TfLiteType typeToTfLiteType<TfLiteFloat16>() {
  return kTfLiteFloat16;
}
}  // namespace tflite
#endif  // TENSORFLOW_LITE_TYPE_TO_TFLITETYPE_H_
