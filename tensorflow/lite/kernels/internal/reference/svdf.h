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

  // Rewrite last bit of state.
  // TODO(jianlijianli): move this function into matmul.
  {
    for (int b = 0; b < n_batch; ++b) {
      int16_t* state_ptr_batch =
          GetTensorData<int16_t>(state_tensor) + b * n_memory * n_filter;
      for (int c = 0; c < n_filter; ++c) {
        int16_t* state_ptr = state_ptr_batch + c * n_memory;
        state_ptr[n_memory - 1] = 0;
      }
    }
  }

  // Feature matmul.
  {
    int16_t* state = GetTensorData<int16_t>(state_tensor);
    const int8_t* input = GetTensorData<int8_t>(input_tensor);
    const int8_t* weight_feature =
        GetTensorData<int8_t>(weights_feature_tensor);
    const int32_t output_max = std::numeric_limits<int16_t>::max();
    const int32_t output_min = std::numeric_limits<int16_t>::min();
    int16_t* result_in_batch = state + (n_memory - 1);
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
        *result_in_batch = dot_prod;
        result_in_batch += n_memory;
      }
    }
  }

  // Time.
  {
    for (int b = 0; b < n_batch; ++b) {
      const int16_t* state_ptr_batch =
          GetTensorData<int16_t>(state_tensor) + b * n_memory * n_filter;
      int32_t* scratch_ptr_batch =
          GetTensorData<int32_t>(scratch_tensor) + b * n_filter;
      tensor_utils::BatchVectorBatchVectorDotProduct(
          GetTensorData<int16_t>(weights_time_tensor), state_ptr_batch,
          n_memory, n_filter, scratch_ptr_batch, /*result_stride=*/1);
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

  // Shift state.
  {
    int16_t zero = 0;
    for (int b = 0; b < n_batch; ++b) {
      int16_t* state_ptr_batch =
          GetTensorData<int16_t>(state_tensor) + b * n_memory * n_filter;
      for (int f = 0; f < n_filter; ++f) {
        tensor_utils::VectorShiftLeft(state_ptr_batch, n_memory, zero);
        state_ptr_batch += n_memory;
      }
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
