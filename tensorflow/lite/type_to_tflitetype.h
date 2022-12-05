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

#include <complex>
#include <string>

#include "tensorflow/lite/core/c/common.h"

// Most of the definitions have been moved to this subheader so that Micro
// can include it without relying on <string> and <complex>, which isn't
// available on all platforms.
#include "tensorflow/lite/portable_type_to_tflitetype.h"

namespace tflite {

// TODO(b/163167649): This string conversion means that only the first entry
// in a string tensor will be returned as a std::string, so it's deprecated.
MATCH_TYPE_AND_TFLITE_TYPE(std::string, kTfLiteString);

MATCH_TYPE_AND_TFLITE_TYPE(std::complex<float>, kTfLiteComplex64);
MATCH_TYPE_AND_TFLITE_TYPE(std::complex<double>, kTfLiteComplex128);

}  // namespace tflite
#endif  // TENSORFLOW_LITE_TYPE_TO_TFLITETYPE_H_
