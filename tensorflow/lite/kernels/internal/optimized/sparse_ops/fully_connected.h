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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SPARSE_OPS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SPARSE_OPS_FULLY_CONNECTED_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/round.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

inline void FullyConnectedSparseWeight(
    const TfLiteSparsity& sparsity, const FullyConnectedParams& params,
    const RuntimeShape& input_shape, const float* input_data,
    const RuntimeShape& weights_shape, const float* weights_data,
    const RuntimeShape& bias_shape, const float* bias_data,
    const RuntimeShape& output_shape, float* output_data) {
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;

  const int output_elements = output_shape.FlatSize();
  const int output_dims_count = output_shape.DimensionsCount();
  const int weights_dims_count = weights_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dims_count - 1);
  const int output_depth = MatchingDim(weights_shape, weights_dims_count - 2,
                                       output_shape, output_dims_count - 1);
  const int accum_depth = weights_shape.Dims(weights_dims_count - 1);
  const int w0_size = sparsity.dim_metadata[0].dense_size;
  const int* w1_segments = sparsity.dim_metadata[1].array_segments->data;
  const int* w1_indices = sparsity.dim_metadata[1].array_indices->data;

  for (int i = 0; i < output_elements; ++i) {
    output_data[i] = 0.f;
  }

  for (int b = 0; b < batches; ++b) {
    for (int idx_0 = 0; idx_0 < w0_size; ++idx_0) {
      for (int pw1 = w1_segments[idx_0]; pw1 < w1_segments[idx_0 + 1]; ++pw1) {
        int idx_1 = w1_indices[pw1];
        output_data[b * output_depth + idx_0] +=
            weights_data[pw1] * input_data[b * accum_depth + idx_1];
      }
    }
  }

  for (int b = 0; b < batches; ++b) {
    for (int i = 0; i < output_depth; ++i) {
      float total = output_data[b * output_depth + i];
      float bias_value = bias_data[i];
      output_data[b * output_depth + i] = ActivationFunctionWithMinMax(
          total + bias_value, output_activation_min, output_activation_max);
    }
  }
}

}  // namespace optimized_ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SPARSE_OPS_FULLY_CONNECTED_H_
