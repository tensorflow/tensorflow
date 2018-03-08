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

// This file provides general C++ utility functions in TFLite.
// For example: Converting between `TfLiteIntArray`, `std::vector` and
// Flatbuffer vectors. These functions can't live in `context.h` since it's pure
// C.

#ifndef TENSORFLOW_CONTRIB_LITE_UTIL_H_
#define TENSORFLOW_CONTRIB_LITE_UTIL_H_

#include <vector>
#include "tensorflow/contrib/lite/context.h"

namespace tflite {

// Converts a `std::vector` to a `TfLiteIntArray`.
TfLiteIntArray* ConvertVectorToTfLiteIntArray(const std::vector<int>& input);

// Checks whether a `TfLiteIntArray` and `std::vector` have matching elements.
bool EqualVectorAndTfLiteIntArray(const TfLiteIntArray* a,
                                  const std::vector<int>& b);

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_UTIL_H_
