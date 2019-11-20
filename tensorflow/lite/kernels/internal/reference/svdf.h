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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SVDF_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SVDF_H_

#include <algorithm>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"

// SVDF op that compresses a fully connected op via low-rank matrix
// factorization. See https://research.google.com/pubs/archive/43813.pdf for
// details.

namespace tflite {
namespace reference_ops {

static inline void ApplyTimeWeightsBiasAndActivation(
    int batch_size, int memory_size, int num_filters, int num_units, int rank,
    const TfLiteTensor* weights_time, const TfLiteTensor* bias,
    TfLiteFusedActivation activation, TfLiteTensor* activation_state,
    TfLiteTensor* scratch, TfLiteTensor* output) {
  // Compute matmul(state, weights_time).
  // The rightmost column is used to save temporary output (with the size of
  // num_filters). This is achieved by starting at
  // GetTensorData<float>(activation_state), and having the stride equal to
  // memory_size.
  for (int b = 0; b < batch_size; ++b) {
    float* state_ptr_batch =
        GetTensorData<float>(activation_state) + b * memory_size * num_filters;
    float* scratch_ptr_batch = GetTensorData<float>(scratch) + b * num_filters;
    tensor_utils::BatchVectorBatchVectorDotProduct(
        GetTensorData<float>(weights_time), state_ptr_batch, memory_size,
        num_filters, scratch_ptr_batch, /*result_stride=*/1);
  }

  // Initialize output with bias if provided.
  if (bias) {
    tensor_utils::VectorBatchVectorAssign(GetTensorData<float>(bias), num_units,
                                          batch_size,
                                          GetTensorData<float>(output));
  } else {
    std::fill_n(GetTensorData<float>(output), batch_size * num_units, 0.0f);
  }

  // Reduction sum.
  for (int b = 0; b < batch_size; ++b) {
    float* output_ptr_batch = GetTensorData<float>(output) + b * num_units;
    float* scratch_ptr_batch = GetTensorData<float>(scratch) + b * num_filters;
    tensor_utils::ReductionSumVector(scratch_ptr_batch, output_ptr_batch,
                                     num_units, rank);
  }

  // Apply activation.
  for (int b = 0; b < batch_size; ++b) {
    float* output_ptr_batch = GetTensorData<float>(output) + b * num_units;
    tensor_utils::ApplyActivationToVector(output_ptr_batch, num_units,
                                          activation, output_ptr_batch);
  }

  // Left shift the activation_state to make room for next cycle's activation.
  // TODO(alanchiao): explore collapsing this into a single loop.
  for (int b = 0; b < batch_size; ++b) {
    float* state_ptr_batch =
        GetTensorData<float>(activation_state) + b * memory_size * num_filters;
    for (int f = 0; f < num_filters; ++f) {
      tensor_utils::VectorShiftLeft(state_ptr_batch, memory_size,
                                    /*shift_value=*/0.0f);
      state_ptr_batch += memory_size;
    }
  }
}

inline void EvalFloatSVDF(TfLiteContext* context, TfLiteNode* node,
                          const TfLiteTensor* input,
                          const TfLiteTensor* weights_feature,
                          const TfLiteTensor* weights_time,
                          const TfLiteTensor* bias,
                          const TfLiteSVDFParams* params, TfLiteTensor* scratch,
                          TfLiteTensor* state, TfLiteTensor* output) {
  const int rank = params->rank;
  const int batch_size = input->dims->data[0];
  const int input_size = input->dims->data[1];
  const int num_filters = weights_feature->dims->data[0];
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];

  // Clear the activation (state's leftmost column).
  // TODO(ghodrat): Add a test which initialize activation_state with invalid
  // values in leftmost column and make sure it passes.
  for (int b = 0; b < batch_size; ++b) {
    float* state_ptr_batch =
        GetTensorData<float>(state) + b * memory_size * num_filters;
    for (int c = 0; c < num_filters; ++c) {
      float* state_ptr = state_ptr_batch + c * memory_size;
      state_ptr[memory_size - 1] = 0.0f;
    }
  }

  // Compute conv1d(inputs, weights_feature).
  // The state's rightmost column is used to save current cycle activation. This
  // is achieved by starting at GetTensorData<float>(state)[memory_size - 1] and
  // having the stride equal to memory_size.
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      GetTensorData<float>(weights_feature), num_filters, input_size,
      GetTensorData<float>(input), batch_size,
      &GetTensorData<float>(state)[memory_size - 1], memory_size);

  ApplyTimeWeightsBiasAndActivation(batch_size, memory_size, num_filters,
                                    num_units, rank, weights_time, bias,
                                    params->activation, state, scratch, output);
}

inline void EvalHybridSVDF(
    TfLiteContext* context, TfLiteNode* node, const TfLiteTensor* input,
    const TfLiteTensor* weights_feature, const TfLiteTensor* weights_time,
    const TfLiteTensor* bias, const TfLiteSVDFParams* params,
    TfLiteTensor* scratch, TfLiteTensor* scaling_factors,
    TfLiteTensor* input_quantized, TfLiteTensor* state, TfLiteTensor* output) {
  const int rank = params->rank;
  const int batch_size = input->dims->data[0];
  const int input_size = input->dims->data[1];
  const int num_filters = weights_feature->dims->data[0];
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];

  // Initialize the pointer to input.
  const float* input_ptr_batch = GetTensorData<float>(input);

  // Initialize the pointer to storage for quantized values and the weights
  // feature.
  int8_t* quantized_input_ptr_batch = GetTensorData<int8_t>(input_quantized);
  const int8_t* weights_feature_ptr = GetTensorData<int8_t>(weights_feature);

  // Initialize the pointer to storage for scaling factors.
  float* scaling_factors_ptr = GetTensorData<float>(scaling_factors);

  // Initialize the weights scale.
  const float weights_feature_scale = weights_feature->params.scale;

  // Clear the activation (state's leftmost column).
  // TODO(ghodrat): Add a test which initialize state with invalid values in
  // the leftmost column and make sure it passes.
  for (int b = 0; b < batch_size; ++b) {
    float* state_ptr_batch =
        GetTensorData<float>(state) + b * memory_size * num_filters;
    for (int c = 0; c < num_filters; ++c) {
      float* state_ptr = state_ptr_batch + c * memory_size;
      state_ptr[memory_size - 1] = 0.0;
    }
  }

  if (!tensor_utils::IsZeroVector(input_ptr_batch, batch_size * input_size)) {
    // Quantize input from float to int8.
    float unused_min, unused_max;
    for (int b = 0; b < batch_size; ++b) {
      const int offset = b * input_size;
      tensor_utils::SymmetricQuantizeFloats(
          input_ptr_batch + offset, input_size,
          quantized_input_ptr_batch + offset, &unused_min, &unused_max,
          &scaling_factors_ptr[b]);
      scaling_factors_ptr[b] *= weights_feature_scale;
    }

    // Compute conv1d(inputs, weights_feature).
    // The rightmost column of state is used to save the current cycle
    // activation.
    // This is achieved by starting at GetTensorData<float>(state)[memory_size -
    // 1] and having the stride equal to memory_size.
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        weights_feature_ptr, num_filters, input_size, quantized_input_ptr_batch,
        scaling_factors_ptr, batch_size,
        &GetTensorData<float>(state)[memory_size - 1], memory_size);
  }

  // TODO(alanchiao): can optimize hybrid case ~5% by unrolling loop in applying
  // time weights so that the inner loop multiplies eight elements at a time.
  ApplyTimeWeightsBiasAndActivation(batch_size, memory_size, num_filters,
                                    num_units, rank, weights_time, bias,
                                    params->activation, state, scratch, output);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SVDF_H_
