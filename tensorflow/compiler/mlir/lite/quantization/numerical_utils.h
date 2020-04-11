/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_NUMERICAL_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_NUMERICAL_UTILS_H_

#include <cstdint>
#include <utility>

namespace mlir {
namespace quant {

using QuantizedMultiplier = std::pair<int32_t, int32_t>;

// Decompose double precision multiplier to integer multiplier and exponent.
//    double_multiplier = int_multiplier * 2 ^ (-31 + exponent)
// int_multiplier will be range of (2^31, 2^30].
QuantizedMultiplier QuantizeMultiplier(double double_multiplier);

}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_NUMERICAL_UTILS_H_
