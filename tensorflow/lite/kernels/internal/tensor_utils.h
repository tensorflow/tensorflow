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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {

// Not all backends support CpuBackendContext usage, so forward declare to avoid
// pulling in its implementation. Use of CpuBackendContext in method
// implementations is purely optional.
class CpuBackendContext;

namespace tensor_utils {

// Apply tanh to elements of a vector
inline void ApplyTanhToVector(const float* __restrict__ vector, int v_size,
                              float* __restrict__ result) {
  using VectorMap = Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>;
  VectorMap input_map(const_cast<float* __restrict__>(vector), v_size);
  VectorMap output_map(result, v_size);
  output_map.array() = input_map.array().tanh();
}

// Apply sigmoid to elements of a vector.
inline void ApplySigmoidToVector(const float* __restrict__ vector, int v_size,
                                 float* __restrict__ result) {
  using VectorMap = Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>;
  VectorMap input_map(const_cast<float* __restrict__>(vector), v_size);
  VectorMap output_map(result, v_size);
  output_map.array() = input_map.array().logistic();
}

// Apply appropriate activation function to elements of a vector.
inline void ApplyActivationToVector(const float* vector, int v_size,
                                    TfLiteFusedActivation activation,
                                    float* result) {
  switch (activation) {
    case kTfLiteActNone:
      return;
    case kTfLiteActRelu:
      return tflite::tensor_utils::ApplyReluToVector(vector, v_size, result);
    case kTfLiteActReluN1To1:
      return tflite::tensor_utils::ApplyRelu1ToVector(vector, v_size, result);
    case kTfLiteActRelu6:
      return tflite::tensor_utils::ApplyRelu6ToVector(vector, v_size, result);
    case kTfLiteActTanh:
      return ApplyTanhToVector(vector, v_size, result);
    case kTfLiteActSignBit:
      return tflite::tensor_utils::ApplySignbitToVector(vector, v_size, result);
    case kTfLiteActSigmoid:
      return ApplySigmoidToVector(vector, v_size, result);
  }
}

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_
