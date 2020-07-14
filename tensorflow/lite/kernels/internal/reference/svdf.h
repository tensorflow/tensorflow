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

#include <stdint.h>

#include <algorithm>
#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
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
    const float* const __restrict__ weights_time_ptr,
    const float* const __restrict__ bias_ptr, TfLiteFusedActivation activation,
    float* const __restrict__ state_ptr, float* const __restrict__ scratch_ptr,
    float* const __restrict__ output_ptr) {
  // Compute matmul(state, weights_time).
  for (int b = 0; b < batch_size; ++b) {
    float* state_ptr_batch = state_ptr + b * memory_size * num_filters;
    float* scratch_ptr_batch = scratch_ptr + b * num_filters;
    tensor_utils::BatchVectorBatchVectorDotProduct(
        weights_time_ptr, state_ptr_batch, memory_size, num_filters,
        scratch_ptr_batch);
  }

  // Initialize output with bias if provided.
  if (bias_ptr) {
    tensor_utils::VectorBatchVectorAssign(bias_ptr, num_units, batch_size,
                                          output_ptr);
  } else {
    std::fill_n(output_ptr, batch_size * num_units, 0.0f);
  }

  // Reduction sum.
  for (int b = 0; b < batch_size; ++b) {
    float* output_ptr_batch = output_ptr + b * num_units;
    float* scratch_ptr_batch = scratch_ptr + b * num_filters;
    tensor_utils::ReductionSumVector(scratch_ptr_batch, output_ptr_batch,
                                     num_units, rank);
  }

  // Apply activation.
  for (int b = 0; b < batch_size; ++b) {
    float* output_ptr_batch = output_ptr + b * num_units;
    tensor_utils::ApplyActivationToVector(output_ptr_batch, num_units,
                                          activation, output_ptr_batch);
  }
}

inline void EvalIntegerSVDF(
    TfLiteContext* context, TfLiteNode* node, const TfLiteTensor* input_tensor,
    const TfLiteTensor* weights_feature_tensor,
    const TfLiteTensor* weights_time_tensor, const TfLiteTensor* bias_tensor,
    const TfLiteSVDFParams* params, TfLiteTensor* state_tensor,
    TfLiteTensor* output_tensor, TfLiteTensor* scratch_tensor,
    TfLiteTensor* output_temp_tensor, int32_t scale_1_a, int scale_1_b,
    int32_t scale_2_a, int scale_2_b, int32_t input_zp, int32_t output_zp) {
  const int n_rank = params->rank;
  const int n_batch = input_tensor->dims->data[0];
  const int n_input = input_tensor->dims->data[1];
  const int n_filter = weights_feature_tensor->dims->data[0];
  const int n_unit = n_filter / n_rank;
  const int n_memory = weights_time_tensor->dims->data[1];

  int16_t* const state_ptr = GetTensorData<int16_t>(state_tensor);

  // Left shift the activation_state.
  // std::copy is fine for overlapping ranges if the output is outside of the
  // input range. (This is not true for copy_n.)
  std::copy(state_ptr + 1, state_ptr + n_batch * n_memory * n_filter,
            state_ptr);

  // Feature matmul.
  // Note: no need to clear the latest activation, matmul is not accumulative.
  {
    const int8_t* input = GetTensorData<int8_t>(input_tensor);
    const int8_t* weight_feature =
        GetTensorData<int8_t>(weights_feature_tensor);
    const int32_t output_max = std::numeric_limits<int16_t>::max();
    const int32_t output_min = std::numeric_limits<int16_t>::min();
    int16_t* result_in_batch = state_ptr + (n_memory - 1);
    for (int b = 0; b < n_batch; b++) {
      const int8_t* matrix_ptr = weight_feature;
      for (int r = 0; r < n_filter; r++) {
        int32_t dot_prod = 0;
        const int8_t* vector_in_batch = input + b * n_input;
        for (int c = 0; c < n_input; c++) {
          dot_prod += *matrix_ptr++ * (*vector_in_batch++ - input_zp);
        }
        dot_prod =
            MultiplyByQuantizedMultiplier(dot_prod, scale_1_a, scale_1_b);
        dot_prod = std::min(std::max(output_min, dot_prod), output_max);
        // This assumes state is symmetrically quantized. Otherwise last bit of
        // state should be initialized to its zero point and accumulate the
        // dot_prod.
        // Equivalent as the following:
        //     result_in_batch = zero point, which happens to be zero.
        //     result_in_batch += dot_prod.
        *result_in_batch = dot_prod;
        result_in_batch += n_memory;
      }
    }
  }

  // Time.
  {
    for (int b = 0; b < n_batch; ++b) {
      const int16_t* state_ptr_batch = state_ptr + b * n_memory * n_filter;
      int32_t* scratch_ptr_batch =
          GetTensorData<int32_t>(scratch_tensor) + b * n_filter;
      tensor_utils::BatchVectorBatchVectorDotProduct(
          GetTensorData<int16_t>(weights_time_tensor), state_ptr_batch,
          n_memory, n_filter, scratch_ptr_batch);
    }
  }

  // Reduce, add bias, rescale, activation.
  {
    int32_t* output_temp = GetTensorData<int32_t>(output_temp_tensor);
    // Add bias.
    if (bias_tensor) {
      tensor_utils::VectorBatchVectorAssign(GetTensorData<int32_t>(bias_tensor),
                                            n_unit, n_batch, output_temp);
    } else {
      std::fill_n(output_temp, n_batch * n_unit, 0);
    }
    // Reduce.
    for (int b = 0; b < n_batch; ++b) {
      int32_t* output_temp_ptr = output_temp + b * n_unit;
      int32_t* scratch_ptr_batch =
          GetTensorData<int32_t>(scratch_tensor) + b * n_filter;
      tensor_utils::ReductionSumVector(scratch_ptr_batch, output_temp_ptr,
                                       n_unit, n_rank);
    }
    // Rescale.
    const int32_t output_max = std::numeric_limits<int8_t>::max();
    const int32_t output_min = std::numeric_limits<int8_t>::min();
    for (int i = 0; i < n_batch * n_unit; ++i) {
      int32_t x1 = output_temp[i];
      int32_t x2 = MultiplyByQuantizedMultiplier(x1, scale_2_a, scale_2_b);
      int32_t x3 = x2 + output_zp;
      int32_t x4 = std::min(std::max(output_min, x3), output_max);
      GetTensorData<int8_t>(output_tensor)[i] = static_cast<int8_t>(x4);
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

  // Raw pointers to tensor data.
  const float* input_ptr = GetTensorData<float>(input);
  const float* weights_feature_ptr = GetTensorData<float>(weights_feature);
  const float* weights_time_ptr = GetTensorData<float>(weights_time);
  const float* bias_ptr = GetTensorData<float>(bias);

  float* state_ptr = GetTensorData<float>(state);
  float* scratch_ptr = GetTensorData<float>(scratch);

  float* output_ptr = GetTensorData<float>(output);

  // Left shift the activation_state.
  // std::copy is fine for overlapping ranges if the output is outside of the
  // input range. (This is not true for copy_n.)
  std::copy(state_ptr + 1, state_ptr + batch_size * memory_size * num_filters,
            state_ptr);

  // Clear scratch (the matmul is accumulative).
  std::fill_n(scratch_ptr, batch_size * num_filters, 0.0f);

  // Compute conv1d(inputs, weights_feature).
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      weights_feature_ptr, num_filters, input_size, input_ptr, batch_size,
      scratch_ptr);

  // Copy the latest activation from scratch into activation_state:
  // The last, i.e. (memory_size-1)th entry for each batch, and filter.
  for (int i = 0; i < batch_size * num_filters; ++i) {
    state_ptr[i * memory_size + memory_size - 1] = scratch_ptr[i];
  }

  ApplyTimeWeightsBiasAndActivation(
      batch_size, memory_size, num_filters, num_units, rank, weights_time_ptr,
      bias_ptr, params->activation, state_ptr, scratch_ptr, output_ptr);
}

inline void EvalHybridSVDF(
    TfLiteContext* context, TfLiteNode* node, const TfLiteTensor* input,
    const TfLiteTensor* weights_feature, const TfLiteTensor* weights_time,
    const TfLiteTensor* bias, const TfLiteSVDFParams* params,
    TfLiteTensor* scratch, TfLiteTensor* scaling_factors,
    TfLiteTensor* input_quantized, TfLiteTensor* state, TfLiteTensor* output,
    TfLiteTensor* zero_points, TfLiteTensor* row_sums, bool* compute_row_sums) {
  const int rank = params->rank;
  const int batch_size = input->dims->data[0];
  const int input_size = input->dims->data[1];
  const int num_filters = weights_feature->dims->data[0];
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];

  // Raw pointers to tensor data.
  const float* input_ptr = GetTensorData<float>(input);
  const int8_t* weights_feature_ptr = GetTensorData<int8_t>(weights_feature);
  const float* weights_time_ptr = GetTensorData<float>(weights_time);
  const float* bias_ptr = GetTensorData<float>(bias);

  int8_t* quantized_input_ptr = GetTensorData<int8_t>(input_quantized);
  float* scaling_factors_ptr = GetTensorData<float>(scaling_factors);
  float* state_ptr = GetTensorData<float>(state);
  float* scratch_ptr = GetTensorData<float>(scratch);

  float* output_ptr = GetTensorData<float>(output);

  int32_t* zero_points_ptr = nullptr;
  int32_t* row_sums_ptr = nullptr;
  if (params->asymmetric_quantize_inputs && row_sums != nullptr) {
    zero_points_ptr = GetTensorData<int32_t>(zero_points);
    row_sums_ptr = GetTensorData<int32_t>(row_sums);
  }

  // Initialize the weights scale.
  const float weights_feature_scale = weights_feature->params.scale;

  // Left shift the activation_state.
  // std::copy is fine for overlapping ranges if the output is outside of the
  // input range. (This is not true for copy_n.)
  std::copy(state_ptr + 1, state_ptr + batch_size * memory_size * num_filters,
            state_ptr);

  // Clear scratch (the matmul is accumulative).
  std::fill_n(scratch_ptr, batch_size * num_filters, 0.0f);

  if (!tensor_utils::IsZeroVector(input_ptr, batch_size * input_size)) {
    // Quantize input from float to int8.
    tensor_utils::BatchQuantizeFloats(input_ptr, batch_size, input_size,
                                      quantized_input_ptr, scaling_factors_ptr,
                                      zero_points_ptr,
                                      params->asymmetric_quantize_inputs);
    for (int b = 0; b < batch_size; ++b) {
      scaling_factors_ptr[b] *= weights_feature_scale;
    }

    // Compute conv1d(inputs, weights_feature).
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        weights_feature_ptr, num_filters, input_size, quantized_input_ptr,
        scaling_factors_ptr, batch_size, scratch_ptr,
        /*per_channel_scale=*/nullptr, zero_points_ptr,
        reinterpret_cast<int32_t*>(scratch_ptr), row_sums_ptr, compute_row_sums,
        /*context=*/nullptr);
  }
  // Copy the latest activation from scratch into activation_state:
  // The last, i.e. (memory_size-1)th entry for each batch, and filter.
  for (int i = 0; i < batch_size * num_filters; ++i) {
    state_ptr[i * memory_size + memory_size - 1] = scratch_ptr[i];
  }

  // TODO(alanchiao): can optimize hybrid case ~5% by unrolling loop in applying
  // time weights so that the inner loop multiplies eight elements at a time.
  ApplyTimeWeightsBiasAndActivation(
      batch_size, memory_size, num_filters, num_units, rank, weights_time_ptr,
      bias_ptr, params->activation, state_ptr, scratch_ptr, output_ptr);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SVDF_H_
