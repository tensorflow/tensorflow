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
#ifndef TENSORFLOW_LITE_PORTABLE_TYPE_TO_TFLITETYPE_H_
#define TENSORFLOW_LITE_PORTABLE_TYPE_TO_TFLITETYPE_H_

// Most of the definitions have been moved to this subheader so that Micro
// can include it without relying on <string> and <complex>, which isn't
// available on all platforms.

// Arduino build defines abs as a macro here. That is invalid C++, and breaks
// libc++'s <complex> header, undefine it.
#ifdef abs
#undef abs
#endif

#include <stdint.h>

#include "tensorflow/lite/core/c/common.h"

namespace tflite {

// Map statically from a C++ type to a TfLiteType. Used in interpreter for
// safe casts.
// Example:
//  typeToTfLiteType<bool>() -> kTfLiteBool
template <typename T>
constexpr TfLiteType typeToTfLiteType() {
  return kTfLiteNoType;
}
// Map from TfLiteType to the corresponding C++ type.
// Example:
//   TfLiteTypeToType<kTfLiteBool>::Type -> bool
template <TfLiteType TFLITE_TYPE_ENUM>
struct TfLiteTypeToType {};  // Specializations below

// Template specialization for both typeToTfLiteType and TfLiteTypeToType.
#define MATCH_TYPE_AND_TFLITE_TYPE(CPP_TYPE, TFLITE_TYPE_ENUM) \
  template <>                                                  \
  constexpr TfLiteType typeToTfLiteType<CPP_TYPE>() {          \
    return TFLITE_TYPE_ENUM;                                   \
  }                                                            \
  template <>                                                  \
  struct TfLiteTypeToType<TFLITE_TYPE_ENUM> {                  \
    using Type = CPP_TYPE;                                     \
  }

// No string mapping is included here, since the TF Lite packed representation
// doesn't correspond to a C++ type well.
MATCH_TYPE_AND_TFLITE_TYPE(int32_t, kTfLiteInt32);
MATCH_TYPE_AND_TFLITE_TYPE(uint32_t, kTfLiteUInt32);
MATCH_TYPE_AND_TFLITE_TYPE(int16_t, kTfLiteInt16);
MATCH_TYPE_AND_TFLITE_TYPE(uint16_t, kTfLiteUInt16);
MATCH_TYPE_AND_TFLITE_TYPE(int64_t, kTfLiteInt64);
MATCH_TYPE_AND_TFLITE_TYPE(float, kTfLiteFloat32);
MATCH_TYPE_AND_TFLITE_TYPE(unsigned char, kTfLiteUInt8);
MATCH_TYPE_AND_TFLITE_TYPE(int8_t, kTfLiteInt8);
MATCH_TYPE_AND_TFLITE_TYPE(bool, kTfLiteBool);
MATCH_TYPE_AND_TFLITE_TYPE(TfLiteFloat16, kTfLiteFloat16);
MATCH_TYPE_AND_TFLITE_TYPE(double, kTfLiteFloat64);
MATCH_TYPE_AND_TFLITE_TYPE(uint64_t, kTfLiteUInt64);

}  // namespace tflite
#endif  // TENSORFLOW_LITE_PORTABLE_TYPE_TO_TFLITETYPE_H_
