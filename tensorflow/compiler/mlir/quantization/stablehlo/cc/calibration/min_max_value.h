/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_MIN_MAX_VALUE_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_MIN_MAX_VALUE_H_

#include <utility>

namespace stablehlo::quantization {

// Represents the (min, max) value pair, representing the range of values after
// calibrating for quantization.
using MinMaxValue = std::pair<float, float>;

}  // namespace stablehlo::quantization

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_MIN_MAX_VALUE_H_
