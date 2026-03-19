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
// This file is the MLIR copy of part of
// third_party/tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cc
// as part of the effort to decouple TFLite from MLIR.

#include "tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy/portable_tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace mlir {
namespace lite {
namespace toco_legacy {

// LINT.IfChange(portable_symmetric_quantize_floats)
void PortableSymmetricQuantizeFloats(const float* values, const int size,
                                     int8_t* quantized_values, float* min_value,
                                     float* max_value, float* scaling_factor) {
  auto minmax = std::minmax_element(values, values + size);
  *min_value = *minmax.first;
  *max_value = *minmax.second;

  PortableSymmetricQuantizeFloats(values, size, quantized_values, *min_value,
                                  *max_value, scaling_factor);
}

void PortableSymmetricQuantizeFloats(const float* values, const int size,
                                     int8_t* quantized_values, float min_value,
                                     float max_value, float* scaling_factor) {
  const int32_t kScale = 127;
  const float range = std::max(std::abs(min_value), std::abs(max_value));
  if (range == 0) {
    memset(quantized_values, 0, size * sizeof(int8_t));
    *scaling_factor = 1;
    return;
  }
  *scaling_factor = range / kScale;
  const float scaling_factor_inv = kScale / range;
  for (int i = 0; i < size; ++i) {
    const int32_t quantized_value =
        static_cast<int32_t>(round(values[i] * scaling_factor_inv));
    // Clamp: just in case some odd numeric offset.
    quantized_values[i] = static_cast<int8_t>(
        std::min(kScale, std::max(-kScale, quantized_value)));
  }
}
// LINT.ThenChange(//tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cc:portable_symmetric_quantize_floats)

}  // namespace toco_legacy
}  // namespace lite
}  // namespace mlir
