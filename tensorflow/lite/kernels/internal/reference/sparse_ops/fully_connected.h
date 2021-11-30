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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SPARSE_OPS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SPARSE_OPS_FULLY_CONNECTED_H_

#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/utils/sparsity_format_converter.h"

namespace tflite {
namespace reference_ops {

// Convert weights to dense format and run dense fully connected.
inline void FullyConnectedSparseWeight(
    const TfLiteSparsity& sparsity, const FullyConnectedParams& params,
    const RuntimeShape& input_shape, const float* input_data,
    const RuntimeShape& weights_shape, const float* weights_data,
    const RuntimeShape& bias_shape, const float* bias_data,
    const RuntimeShape& output_shape, float* output_data) {
  std::vector<int> weights_shape_vector(weights_shape.DimensionsCount());
  for (int i = 0; i < weights_shape.DimensionsCount(); i++) {
    weights_shape_vector[i] = weights_shape.Dims(i);
  }
  tflite::internal::sparsity::FormatConverter<float> converter(
      weights_shape_vector, sparsity);
  converter.SparseToDense(weights_data);
  const std::vector<float>& dense_weights_data = converter.GetData();
  FullyConnected(params, input_shape, input_data, weights_shape,
                 dense_weights_data.data(), bias_shape, bias_data, output_shape,
                 output_data);
}

}  // namespace reference_ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SPARSE_OPS_FULLY_CONNECTED_H_
