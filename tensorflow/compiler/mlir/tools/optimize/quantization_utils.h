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
#ifndef TENSORFLOW_COMPILER_MLIR_TOOLS_OPTIMIZE_QUANTIZATION_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TOOLS_OPTIMIZE_QUANTIZATION_UTILS_H_

#include <cstdint>
#include <vector>

namespace tflite_migration {
namespace optimize {
namespace utils {

template <typename BiasType>
std::vector<BiasType> SymmetricBiasQuantize(const float* data,
                                            uint64_t num_elements,
                                            const std::vector<float>& scales);

std::vector<int16_t> SymmetricQuantizeFloatsToInt16(const float* data,
                                                    uint64_t num_elements,
                                                    float scaling_factor);

// Quantize the values given an array of scales.
void SymmetricPerChannelQuantizeValues(const float* input,
                                       const std::vector<float>& scales_inv,
                                       const std::vector<int32_t>& dimension,
                                       int32_t channel_dim_index,
                                       std::vector<int8_t>* output_value);

}  // namespace utils
}  // namespace optimize
}  // namespace tflite_migration

#endif  // TENSORFLOW_COMPILER_MLIR_TOOLS_OPTIMIZE_QUANTIZATION_UTILS_H_
