/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_QUANTIZATION_UTILS_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_QUANTIZATION_UTILS_H_

#include <cstdint>

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {
namespace utils {

// Returns the number of elements in the given tensor.
TfLiteStatus NumElements(const TensorT& tensor, uint64_t* num_elements);

// Populates the scale and zero point for quantization parameters.
//
// Nudges min and max so that floating point 0 falls exactly on a quantized
// value, returning the nudges scale and zero_point.
void GetAsymmetricQuantizationParams(
    float min, float max, const int quant_min, const int quant_max,
    QuantizationParametersT* quantization_params);

}  // namespace utils
}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_QUANTIZATION_UTILS_H_
