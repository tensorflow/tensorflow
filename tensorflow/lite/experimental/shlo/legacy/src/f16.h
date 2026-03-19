/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_F16_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_F16_H_

#include "tensorflow/lite/experimental/shlo/legacy/src/has_keyword.h"

#if defined(__STDCPP_FLOAT16_T__)
#include <stdfloat>
namespace stablehlo {
using F16 = float16_t;
}  // namespace stablehlo

#elif __has_keyword(_Float16)
namespace stablehlo {
using F16 = _Float16;
}  // namespace stablehlo

#else
#error Type F16 is not available
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_F16_H_
