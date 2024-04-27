/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_COMPATIBILITY_CANONICALIZE_VALUE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_COMPATIBILITY_CANONICALIZE_VALUE_H_

#include <string>

#include "absl/strings/string_view.h"

namespace tflite::acceleration {

// Normalises the given ASCII input by converting all alphabets to lower case
// and replacing ' ' and '-' with '_'.
std::string CanonicalizeValue(absl::string_view value);

// Applies the above normalisation plus key specific normalisation.
std::string CanonicalizeValueWithKey(absl::string_view key,
                                     absl::string_view value);
}  // namespace tflite::acceleration

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_COMPATIBILITY_CANONICALIZE_VALUE_H_
