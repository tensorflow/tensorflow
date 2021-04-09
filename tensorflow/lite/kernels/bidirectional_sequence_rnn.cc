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
#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace bidirectional_sequence_rnn {

namespace {

struct OpData {
  int scratch_tensor_index;
  bool fw_compute_row_sums = false;
  bool bw_compute_row_sums = false;
};

}  // namespace

// LINT.IfChange

constexpr int kInputTensor = 0;
// Forward and backward cell tensors.
constexpr int kFwWeightsTensor = 1;
constexpr int kFwRecurrentWeightsTensor = 2;
constexpr int kFwBiasTensor = 3;
constexpr int kFwHiddenStateTensor = 4;
constexpr int kBwWeightsTensor = 5;
constexpr int kBwRecurrentWeightsTensor = 6;
constexpr int kBwBiasTensor = 7;
constexpr int kBwHiddenStateTensor = 8;
// Used as auxiliary input and weights when stacking for
// tf.contrib.rnn.stack_bidirectional_rnn case (with cross links); Used as input
// to the backward cell when stacking for tf.nn.static_bidirectional_rnn case
// (without cross links).
constexpr int kAuxInputTensor = 9;       // Optional.
constexpr int kFwAuxWeightsTensor = 10;  // Optional.
constexpr int kBwAuxWeightsTensor = 11;  // Optional.
// Output tensors.
constexpr int kFwOutputTensor = 0;
constexpr int kBwOutputTensor = 1;  // Only if merge_outputs is false.

// LINT.ThenChange(//tensorflow/lite/tools/optimize/quantize_weights.cc)

// Temporary tensors.
enum TemporaryTensor {
  kInputQuantized = 0,
  kFwHiddenStateQuantized = 1,
  kBwHiddenStateQuantized = 2,
  kScalingFactors = 3,
  kAccumScratch = 4,
  kZeroPoints = 5,
  kFwRowSums = 6,
  kBwRowSums = 7,
  kAuxInputQuantized = 8,
  kNumTemporaryTensors = 9
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData();
  context->AddTensors(context, kNumTemporaryTensors,
                      &op_data->scratch_tensor_index);
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const auto* params = reinterpret_cast<TfLiteBidirectionalSequenceRNNParams*>(
      node->builtin_data);

  // Check we have all the inputs and outputs we need.
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 12);
  TF_LITE_ENSURE_EQ(context, node->outputs->size,
                    params->merge_outputs ? 1 : 2);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* fw_input_weights;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kFwWeightsTensor,
                                          &fw_input_weights));
  const TfLiteTensor* fw_recurrent_weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kFwRecurrentWeightsTensor,
                                 &fw_recurrent_weights));
  const TfLiteTensor* fw_bias;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kFwBiasTensor, &fw_bias));
  const TfLiteTensor* fw_hidden_state;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kFwHiddenStateTensor,
                                          &fw_hidden_state));
  const TfLiteTensor* bw_input_weights;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kBwWeightsTensor,
                                          &bw_input_weights));
  const TfLiteTensor* bw_recurrent_weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kBwRecurrentWeightsTensor,
                                 &bw_recurrent_weights));
  const TfLiteTensor* bw_bias;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kBwBiasTensor, &bw_bias));
  const TfLiteTensor* bw_hidden_state;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kBwHiddenStateTensor,
                                          &bw_hidden_state));

  const TfLiteTensor* aux_input =
      GetOptionalInputTensor(context, node, kAuxInputTensor);
  const TfLiteTensor* fw_aux_input_weights =
      GetOptionalInputTensor(context, node, kFwAuxWeightsTensor);
  const TfLiteTensor* bw_aux_input_weights =
      GetOptionalInputTensor(context, node, kBwAuxWeightsTensor);

  const bool aux_inputs_weights_or_none =
      ((fw_aux_input_weights != nullptr) &&
       (bw_aux_input_weights != nullptr)) ||
      ((fw_aux_input_weights == nullptr) && (bw_aux_input_weights == nullptr));
  TF_LITE_ENSURE(context, aux_inputs_weights_or_none);
  const bool has_aux_input = (fw_aux_input_weights != nullptr);

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);

  TF_LITE_ENSURE_EQ(context, input->dims->size, 3);
  const bool time_major = params->time_major;
  const int batch_size =
      (time_major) ? input->dims->data[1] : input->dims->data[0];
  const int max_time =
      (time_major) ? input->dims->data[0] : input->dims->data[1];
  const int fw_num_units = fw_input_weights->dims->data[0];
  const int bw_num_units = bw_input_weights->dims->data[0];
  TF_LITE_ENSURE_EQ(context, input->dims->data[2],
                    fw_input_weights->dims->data[1]);
  TF_LITE_ENSURE_EQ(context, input->dims->data[2],
                    bw_input_weights->dims->data[1]);
  TF_LITE_ENSURE_EQ(context, fw_input_weights->dims->data[0],
                    fw_bias->dims->data[0]);
  TF_LITE_ENSURE_EQ(context, bw_input_weights->dims->data[0],
                    bw_bias->dims->data[0]);
  TF_LITE_ENSURE_EQ(context, fw_recurrent_weights->dims->data[0],
                    fw_bias->dims->data[0]);
  TF_LITE_ENSURE_EQ(context, bw_recurrent_weights->dims->data[1],
                    bw_bias->dims->data[0]);
  TF_LITE_ENSURE_EQ(context, NumDimensions(fw_hidden_state), 2);
  TF_LITE_ENSURE_EQ(context, fw_hidden_state->dims->data[0], batch_size);
  TF_LITE_ENSURE_EQ(context, fw_hidden_state->dims->data[1], fw_num_units);
  TF_LITE_ENSURE_EQ(context, NumDimensions(bw_hidden_state), 2);
  TF_LITE_ENSURE_EQ(context, bw_hidden_state->dims->data[0], batch_size);
  TF_LITE_ENSURE_EQ(context, bw_hidden_state->dims->data[1], bw_num_units);

  if (has_aux_input) {
    // Check that aux_input has the same dimensions (except last) as the input.
    TF_LITE_ASSERT_EQ(aux_input->dims->data[0], input->dims->data[0]);
    TF_LITE_ASSERT_EQ(aux_input->dims->data[1], input->dims->data[1]);
    // Check that aux_input_weights has the same dimensions (except last) as
    // the input_weights.
    TF_LITE_ASSERT_EQ(fw_aux_input_weights->dims->data[0], fw_num_units);
    TF_LITE_ASSERT_EQ(bw_aux_input_weights->dims->data[0], bw_num_units);
    TF_LITE_ASSERT_EQ(aux_input->dims->data[2],
                      fw_aux_input_weights->dims->data[1]);
    TF_LITE_ASSERT_EQ(aux_input->dims->data[2],
                      bw_aux_input_weights->dims->data[1]);
  }

  if (IsHybridOp(input, fw_input_weights)) {
    OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
    op_data->fw_compute_row_sums = true;
    op_data->bw_compute_row_sums = true;
    TfLiteIntArrayFree(node->temporaries);
    if (has_aux_input) {
      node->temporaries = TfLiteIntArrayCreate(kNumTemporaryTensors);
    } else {
      // No need to create a temporary tensor for the non-existent aux_input.
      node->temporaries = TfLiteIntArrayCreate(kNumTemporaryTensors - 1);
    }

    node->temporaries->data[kInputQuantized] =
        op_data->scratch_tensor_index + kInputQuantized;
    TfLiteTensor* input_quantized;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, kInputQuantized,
                                                &input_quantized));
    input_quantized->type = fw_input_weights->type;
    input_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
      TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                       input_quantized_size));
    }

    node->temporaries->data[kFwHiddenStateQuantized] =
        op_data->scratch_tensor_index + kFwHiddenStateQuantized;
    TfLiteTensor* fw_hidden_state_quantized;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, kFwHiddenStateQuantized,
                                       &fw_hidden_state_quantized));
    fw_hidden_state_quantized->type = fw_input_weights->type;
    fw_hidden_state_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(fw_hidden_state_quantized->dims,
                             fw_hidden_state->dims)) {
      TfLiteIntArray* fw_hidden_state_quantized_size =
          TfLiteIntArrayCopy(fw_hidden_state->dims);
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, fw_hidden_state_quantized,
                                         fw_hidden_state_quantized_size));
    }

    node->temporaries->data[kBwHiddenStateQuantized] =
        op_data->scratch_tensor_index + kBwHiddenStateQuantized;
    TfLiteTensor* bw_hidden_state_quantized;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, kBwHiddenStateQuantized,
                                       &bw_hidden_state_quantized));
    bw_hidden_state_quantized->type = fw_input_weights->type;
    bw_hidden_state_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(bw_hidden_state_quantized->dims,
                             bw_hidden_state->dims)) {
      TfLiteIntArray* bw_hidden_state_quantized_size =
          TfLiteIntArrayCopy(bw_hidden_state->dims);
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, bw_hidden_state_quantized,
                                         bw_hidden_state_quantized_size));
    }

    // Allocate temporary tensors to store scaling factors of quantization.
    node->temporaries->data[kScalingFactors] =
        op_data->scratch_tensor_index + kScalingFactors;
    TfLiteTensor* scaling_factors;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, kScalingFactors,
                                                &scaling_factors));
    scaling_factors->type = kTfLiteFloat32;
    scaling_factors->allocation_type = kTfLiteArenaRw;
    int scaling_dims[1] = {batch_size};
    if (!TfLiteIntArrayEqualsArray(scaling_factors->dims, 1, scaling_dims)) {
      TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
      scaling_factors_size->data[0] = batch_size;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors,
                                                       scaling_factors_size));
    }
    node->temporaries->data[kAccumScratch] =
        op_data->scratch_tensor_index + kAccumScratch;
    TfLiteTensor* accum_scratch;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, kAccumScratch,
                                                &accum_scratch));
    accum_scratch->type = kTfLiteInt32;
    accum_scratch->allocation_type = kTfLiteArenaRw;
    int accum_scratch_dims[2] = {std::max(fw_num_units, bw_num_units),
                                 batch_size};
    if (!TfLiteIntArrayEqualsArray(accum_scratch->dims, 2,
                                   accum_scratch_dims)) {
      TfLiteIntArray* accum_scratch_size = TfLiteIntArrayCreate(2);
      accum_scratch_size->data[0] = accum_scratch_dims[0];
      accum_scratch_size->data[1] = accum_scratch_dims[1];
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, accum_scratch,
                                                       accum_scratch_size));
    }
    node->temporaries->data[kZeroPoints] =
        op_data->scratch_tensor_index + kZeroPoints;
    TfLiteTensor* zero_points;
    TF_LITE_ENSURE_OK(
        context,
        GetTemporarySafe(context, node, /*index=*/kZeroPoints, &zero_points));
    zero_points->type = kTfLiteInt32;
    zero_points->allocation_type = kTfLiteArenaRw;
    int zero_points_dims[1] = {batch_size};
    if (!TfLiteIntArrayEqualsArray(zero_points->dims, 1, zero_points_dims)) {
      TfLiteIntArray* zero_points_size = TfLiteIntArrayCreate(1);
      zero_points_size->data[0] = batch_size;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, zero_points,
                                                       zero_points_size));
    }
    const int num_row_sums = has_aux_input ? 3 : 2;
    node->temporaries->data[kFwRowSums] =
        op_data->scratch_tensor_index + kFwRowSums;
    TfLiteTensor* fw_row_sums;
    TF_LITE_ENSURE_OK(
        context,
        GetTemporarySafe(context, node, /*index=*/kFwRowSums, &fw_row_sums));
    fw_row_sums->type = kTfLiteInt32;
    fw_row_sums->allocation_type = kTfLiteArenaRwPersistent;
    int fw_row_sums_dims[2] = {num_row_sums, fw_num_units};
    if (!TfLiteIntArrayEqualsArray(fw_row_sums->dims, 2, fw_row_sums_dims)) {
      TfLiteIntArray* fw_row_sums_size = TfLiteIntArrayCreate(2);
      fw_row_sums_size->data[0] = fw_row_sums_dims[0];
      fw_row_sums_size->data[1] = fw_row_sums_dims[1];
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, fw_row_sums,
                                                       fw_row_sums_size));
    }
    node->temporaries->data[kBwRowSums] =
        op_data->scratch_tensor_index + kBwRowSums;
    TfLiteTensor* bw_row_sums;
    TF_LITE_ENSURE_OK(
        context,
        GetTemporarySafe(context, node, /*index=*/kBwRowSums, &bw_row_sums));
    bw_row_sums->type = kTfLiteInt32;
    bw_row_sums->allocation_type = kTfLiteArenaRwPersistent;
    int bw_row_sums_dims[2] = {num_row_sums, bw_num_units};
    if (!TfLiteIntArrayEqualsArray(bw_row_sums->dims, 2, bw_row_sums_dims)) {
      TfLiteIntArray* bw_row_sums_size = TfLiteIntArrayCreate(2);
      bw_row_sums_size->data[0] = bw_row_sums_dims[0];
      bw_row_sums_size->data[1] = bw_row_sums_dims[1];
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, bw_row_sums,
                                                       bw_row_sums_size));
    }
    if (has_aux_input) {
      node->temporaries->data[kAuxInputQuantized] =
          op_data->scratch_tensor_index + kAuxInputQuantized;
      TfLiteTensor* aux_input_quantized;
      TF_LITE_ENSURE_OK(context,
                        GetTemporarySafe(context, node, kAuxInputQuantized,
                                         &aux_input_quantized));
      aux_input_quantized->type = fw_input_weights->type;
      aux_input_quantized->allocation_type = kTfLiteArenaRw;
      if (!TfLiteIntArrayEqual(aux_input_quantized->dims, aux_input->dims)) {
        TfLiteIntArray* aux_input_quantized_size =
            TfLiteIntArrayCopy(aux_input->dims);
        TF_LITE_ENSURE_OK(context,
                          context->ResizeTensor(context, aux_input_quantized,
                                                aux_input_quantized_size));
      }
    }
  }

  // Resize outputs.
  TfLiteTensor* fw_output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kFwOutputTensor, &fw_output));
  TfLiteIntArray* fw_output_size_array = TfLiteIntArrayCreate(3);
  fw_output_size_array->data[0] = (time_major) ? max_time : batch_size;
  fw_output_size_array->data[1] = (time_major) ? batch_size : max_time;
  fw_output_size_array->data[2] =
      params->merge_outputs ? fw_num_units + bw_num_units : fw_num_units;
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, fw_output, fw_output_size_array));
  if (!params->merge_outputs) {
    TfLiteTensor* bw_output;
    TF_LITE_ENSURE_OK(
        context, GetOutputSafe(context, node, kBwOutputTensor, &bw_output));
    TfLiteIntArray* bw_output_size_array = TfLiteIntArrayCreate(3);
    bw_output_size_array->data[0] = batch_size;
    bw_output_size_array->data[1] = max_time;
    bw_output_size_array->data[2] = bw_num_units;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, bw_output,
                                                     bw_output_size_array));
  }

  return kTfLiteOk;
}

TfLiteStatus EvalFloat(const TfLiteTensor* input, const TfLiteTensor* bw_input,
                       const TfLiteTensor* fw_input_weights,
                       const TfLiteTensor* fw_recurrent_weights,
                       const TfLiteTensor* fw_bias,
                       const TfLiteTensor* bw_input_weights,
                       const TfLiteTensor* bw_recurrent_weights,
                       const TfLiteTensor* bw_bias,
                       const TfLiteTensor* aux_input,
                       const TfLiteTensor* fw_aux_input_weights,
                       const TfLiteTensor* bw_aux_input_weights,
                       const TfLiteBidirectionalSequenceRNNParams* params,
                       TfLiteTensor* fw_hidden_state, TfLiteTensor* fw_output,
                       TfLiteTensor* bw_hidden_state, TfLiteTensor* bw_output) {
  const bool time_major = params->time_major;
  const int batch_size =
      (time_major) ? input->dims->data[1] : input->dims->data[0];
  const int max_time =
      (time_major) ? input->dims->data[0] : input->dims->data[1];
  const int input_size = input->dims->data[2];
  const int aux_input_size = (aux_input) ? aux_input->dims->data[2] : 0;

  const int fw_num_units = fw_input_weights->dims->data[0];
  const float* fw_bias_ptr = GetTensorData<float>(fw_bias);
  const float* fw_input_weights_ptr = GetTensorData<float>(fw_input_weights);
  const float* fw_recurrent_weights_ptr =
      GetTensorData<float>(fw_recurrent_weights);

  const int bw_num_units = bw_input_weights->dims->data[0];
  const float* bw_bias_ptr = GetTensorData<float>(bw_bias);
  const float* bw_input_weights_ptr = GetTensorData<float>(bw_input_weights);
  const float* bw_recurrent_weights_ptr =
      GetTensorData<float>(bw_recurrent_weights);

  const float* fw_aux_input_weights_ptr =
      (fw_aux_input_weights != nullptr)
          ? GetTensorData<float>(fw_aux_input_weights)
          : nullptr;
  const float* bw_aux_input_weights_ptr =
      (bw_aux_input_weights != nullptr)
          ? GetTensorData<float>(bw_aux_input_weights)
          : nullptr;

  const int fw_output_step =
      params->merge_outputs ? fw_num_units + bw_num_units : fw_num_units;
  const int bw_output_step =
      params->merge_outputs ? fw_num_units + bw_num_units : bw_num_units;
  if (time_major) {
    // Forward cell.
    float* fw_hidden_state_ptr_batch = GetTensorData<float>(fw_hidden_state);
    for (int s = 0; s < max_time; s++) {
      const float* input_ptr_batch =
          GetTensorData<float>(input) + s * input_size * batch_size;
      const float* aux_input_ptr_batch =
          (aux_input != nullptr)
              ? GetTensorData<float>(aux_input) + s * input_size * batch_size
              : nullptr;
      float* output_ptr_batch =
          GetTensorData<float>(fw_output) + s * fw_output_step * batch_size;

      kernel_utils::RnnBatchStep(
          input_ptr_batch, fw_input_weights_ptr, aux_input_ptr_batch,
          fw_aux_input_weights_ptr, fw_recurrent_weights_ptr, fw_bias_ptr,
          input_size, aux_input_size, fw_num_units, batch_size, fw_output_step,
          params->activation, fw_hidden_state_ptr_batch, output_ptr_batch);
    }
    // Backward cell.
    float* bw_hidden_state_ptr_batch = GetTensorData<float>(bw_hidden_state);
    for (int s = max_time - 1; s >= 0; s--) {
      const float* input_ptr_batch =
          GetTensorData<float>(bw_input) + s * input_size * batch_size;
      const float* aux_input_ptr_batch =
          (aux_input != nullptr)
              ? GetTensorData<float>(aux_input) + s * input_size * batch_size
              : nullptr;
      float* output_ptr_batch =
          (params->merge_outputs
               ? GetTensorData<float>(fw_output) + fw_num_units
               : GetTensorData<float>(bw_output)) +
          s * bw_output_step * batch_size;

      kernel_utils::RnnBatchStep(
          input_ptr_batch, bw_input_weights_ptr, aux_input_ptr_batch,
          bw_aux_input_weights_ptr, bw_recurrent_weights_ptr, bw_bias_ptr,
          input_size, aux_input_size, bw_num_units, batch_size, bw_output_step,
          params->activation, bw_hidden_state_ptr_batch, output_ptr_batch);
    }
  } else {
    for (int b = 0; b < batch_size; b++) {
      // Forward cell.
      float* fw_hidden_state_ptr_batch =
          GetTensorData<float>(fw_hidden_state) + b * fw_num_units;
      float* fw_output_offset =
          GetTensorData<float>(fw_output) + b * fw_output_step * max_time;
      for (int s = 0; s < max_time; s++) {
        const float* input_ptr_batch = GetTensorData<float>(input) +
                                       b * input_size * max_time +
                                       s * input_size;
        const float* aux_input_ptr_batch =
            (aux_input != nullptr)
                ? GetTensorData<float>(aux_input) +
                      b * aux_input_size * max_time + s * aux_input_size
                : nullptr;
        float* output_ptr_batch = fw_output_offset + s * fw_output_step;

        kernel_utils::RnnBatchStep(
            input_ptr_batch, fw_input_weights_ptr, aux_input_ptr_batch,
            fw_aux_input_weights_ptr, fw_recurrent_weights_ptr, fw_bias_ptr,
            input_size, aux_input_size, fw_num_units, /*batch_size=*/1,
            fw_output_step, params->activation, fw_hidden_state_ptr_batch,
            output_ptr_batch);
      }
      // Backward cell.
      float* bw_hidden_state_ptr_batch =
          GetTensorData<float>(bw_hidden_state) + b * bw_num_units;
      float* bw_output_offset =
          params->merge_outputs
              ? GetTensorData<float>(fw_output) +
                    b * bw_output_step * max_time + fw_num_units
              : GetTensorData<float>(bw_output) + b * bw_output_step * max_time;
      for (int s = max_time - 1; s >= 0; s--) {
        const float* input_ptr_batch = GetTensorData<float>(input) +
                                       b * input_size * max_time +
                                       s * input_size;
        const float* aux_input_ptr_batch =
            (aux_input != nullptr)
                ? GetTensorData<float>(aux_input) +
                      b * aux_input_size * max_time + s * aux_input_size
                : nullptr;
        float* output_ptr_batch = bw_output_offset + s * bw_output_step;

        kernel_utils::RnnBatchStep(
            input_ptr_batch, bw_input_weights_ptr, aux_input_ptr_batch,
            bw_aux_input_weights_ptr, bw_recurrent_weights_ptr, bw_bias_ptr,
            input_size, aux_input_size, bw_num_units, /*batch_size=*/1,
            bw_output_step, params->activation, bw_hidden_state_ptr_batch,
            output_ptr_batch);
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalHybrid(
    const TfLiteTensor* input, const TfLiteTensor* bw_input,
    const TfLiteTensor* fw_input_weights,
    const TfLiteTensor* fw_recurrent_weights, const TfLiteTensor* fw_bias,
    const TfLiteTensor* bw_input_weights,
    const TfLiteTensor* bw_recurrent_weights, const TfLiteTensor* bw_bias,
    const TfLiteTensor* aux_input, const TfLiteTensor* aux_fw_input_weights,
    const TfLiteTensor* aux_bw_input_weights,
    const TfLiteBidirectionalSequenceRNNParams* params,
    TfLiteTensor* scaling_factors, TfLiteTensor* input_quantized,
    TfLiteTensor* aux_input_quantized, TfLiteTensor* fw_hidden_state_quantized,
    TfLiteTensor* fw_hidden_state, TfLiteTensor* fw_output,
    TfLiteTensor* bw_hidden_state_quantized, TfLiteTensor* bw_hidden_state,
    TfLiteTensor* bw_output, TfLiteTensor* zero_points,
    TfLiteTensor* accum_scratch, TfLiteTensor* fw_row_sums,
    TfLiteTensor* bw_row_sums, bool* fw_compute_row_sums,
    bool* bw_compute_row_sums) {
  const bool time_major = params->time_major;
  const int batch_size =
      (time_major) ? input->dims->data[1] : input->dims->data[0];
  const int max_time =
      (time_major) ? input->dims->data[0] : input->dims->data[1];
  const int input_size = input->dims->data[2];
  const int aux_input_size = (aux_input) ? aux_input->dims->data[2] : 0;

  const int fw_num_units = fw_input_weights->dims->data[0];
  const float* fw_bias_ptr = GetTensorData<float>(fw_bias);
  const int8_t* fw_input_weights_ptr = GetTensorData<int8_t>(fw_input_weights);
  float fw_input_weights_scale = fw_input_weights->params.scale;
  const int8_t* fw_recurrent_weights_ptr =
      GetTensorData<int8_t>(fw_recurrent_weights);
  float fw_recurrent_weights_scale = fw_recurrent_weights->params.scale;

  const int bw_num_units = bw_input_weights->dims->data[0];
  const float* bw_bias_ptr = GetTensorData<float>(bw_bias);
  const int8_t* bw_input_weights_ptr = GetTensorData<int8_t>(bw_input_weights);
  float bw_input_weights_scale = bw_input_weights->params.scale;
  const int8_t* bw_recurrent_weights_ptr =
      GetTensorData<int8_t>(bw_recurrent_weights);
  float bw_recurrent_weights_scale = bw_recurrent_weights->params.scale;

  // Set the auxiliary pointers and scales if needed.
  const int8_t* aux_fw_input_weights_ptr = nullptr;
  float aux_fw_input_weights_scale = 0.0f;
  const int8_t* aux_bw_input_weights_ptr = nullptr;
  float aux_bw_input_weights_scale = 0.0f;
  int8_t* aux_quantized_input_ptr = nullptr;
  if (aux_input_size > 0) {
    aux_fw_input_weights_ptr = GetTensorData<int8_t>(aux_fw_input_weights);
    aux_fw_input_weights_scale = aux_fw_input_weights->params.scale;
    aux_bw_input_weights_ptr = GetTensorData<int8_t>(aux_bw_input_weights);
    aux_bw_input_weights_scale = aux_bw_input_weights->params.scale;
    aux_quantized_input_ptr = GetTensorData<int8_t>(aux_input_quantized);
  }

  // Initialize temporary storage for quantized values.
  int8_t* quantized_input_ptr = GetTensorData<int8_t>(input_quantized);
  int8_t* fw_quantized_hidden_state_ptr =
      GetTensorData<int8_t>(fw_hidden_state_quantized);
  int8_t* bw_quantized_hidden_state_ptr =
      GetTensorData<int8_t>(bw_hidden_state_quantized);
  float* scaling_factors_ptr = GetTensorData<float>(scaling_factors);
  int32_t* accum_scratch_ptr = GetTensorData<int32_t>(accum_scratch);
  int32_t* zero_points_ptr = nullptr;
  int32_t* fw_row_sums_ptr = nullptr;
  int32_t* bw_row_sums_ptr = nullptr;
  if (params->asymmetric_quantize_inputs) {
    zero_points_ptr = GetTensorData<int32_t>(zero_points);
    fw_row_sums_ptr = GetTensorData<int32_t>(fw_row_sums);
    bw_row_sums_ptr = GetTensorData<int32_t>(bw_row_sums);
  }
  const int fw_output_step =
      params->merge_outputs ? fw_num_units + bw_num_units : fw_num_units;
  const int bw_output_step =
      params->merge_outputs ? fw_num_units + bw_num_units : bw_num_units;

  if (time_major) {
    for (int t = 0; t < max_time; t++) {
      // Forward cell.
      float* fw_hidden_state_ptr_batch = GetTensorData<float>(fw_hidden_state);
      for (int s = 0; s < max_time; s++) {
        const float* input_ptr_batch =
            GetTensorData<float>(input) + s * input_size * batch_size;
        const float* aux_input_ptr_batch =
            (aux_input != nullptr)
                ? GetTensorData<float>(aux_input) + s * input_size * batch_size
                : nullptr;
        float* output_ptr_batch =
            GetTensorData<float>(fw_output) + s * fw_output_step * batch_size;

        kernel_utils::RnnBatchStep(
            input_ptr_batch, fw_input_weights_ptr, fw_input_weights_scale,
            aux_input_ptr_batch, aux_fw_input_weights_ptr,
            aux_fw_input_weights_scale, fw_recurrent_weights_ptr,
            fw_recurrent_weights_scale, fw_bias_ptr, input_size, aux_input_size,
            fw_num_units, batch_size, fw_output_step, params->activation,
            quantized_input_ptr, aux_quantized_input_ptr,
            fw_quantized_hidden_state_ptr, scaling_factors_ptr,
            fw_hidden_state_ptr_batch, output_ptr_batch,
            params->asymmetric_quantize_inputs, zero_points_ptr,
            accum_scratch_ptr, fw_row_sums_ptr, fw_compute_row_sums);
      }
      // Backward cell.
      float* bw_hidden_state_ptr_batch = GetTensorData<float>(bw_hidden_state);
      for (int s = max_time - 1; s >= 0; s--) {
        const float* input_ptr_batch =
            GetTensorData<float>(bw_input) + s * input_size * batch_size;
        const float* aux_input_ptr_batch =
            (aux_input != nullptr)
                ? GetTensorData<float>(aux_input) + s * input_size * batch_size
                : nullptr;
        float* output_ptr_batch =
            (params->merge_outputs
                 ? GetTensorData<float>(fw_output) + fw_num_units
                 : GetTensorData<float>(bw_output)) +
            s * bw_output_step * batch_size;

        kernel_utils::RnnBatchStep(
            input_ptr_batch, bw_input_weights_ptr, bw_input_weights_scale,
            aux_input_ptr_batch, aux_bw_input_weights_ptr,
            aux_bw_input_weights_scale, bw_recurrent_weights_ptr,
            bw_recurrent_weights_scale, bw_bias_ptr, input_size, aux_input_size,
            bw_num_units, batch_size, bw_output_step, params->activation,
            quantized_input_ptr, aux_quantized_input_ptr,
            bw_quantized_hidden_state_ptr, scaling_factors_ptr,
            bw_hidden_state_ptr_batch, output_ptr_batch,
            params->asymmetric_quantize_inputs, zero_points_ptr,
            accum_scratch_ptr, bw_row_sums_ptr, bw_compute_row_sums);
      }
    }
  } else {
    for (int b = 0; b < batch_size; b++) {
      // Forward cell.
      float* fw_hidden_state_ptr_batch =
          GetTensorData<float>(fw_hidden_state) + b * fw_num_units;
      float* fw_output_offset =
          GetTensorData<float>(fw_output) + b * fw_output_step * max_time;
      for (int s = 0; s < max_time; s++) {
        const float* input_ptr_batch = GetTensorData<float>(input) +
                                       b * input_size * max_time +
                                       s * input_size;
        const float* aux_input_ptr_batch =
            (aux_input != nullptr)
                ? GetTensorData<float>(aux_input) + b * input_size * max_time +
                      s * input_size
                : nullptr;
        float* output_ptr_batch = fw_output_offset + s * fw_output_step;

        kernel_utils::RnnBatchStep(
            input_ptr_batch, fw_input_weights_ptr, fw_input_weights_scale,
            aux_input_ptr_batch, aux_fw_input_weights_ptr,
            aux_fw_input_weights_scale, fw_recurrent_weights_ptr,
            fw_recurrent_weights_scale, fw_bias_ptr, input_size, aux_input_size,
            fw_num_units, /*batch_size=*/1, fw_output_step, params->activation,
            quantized_input_ptr, aux_quantized_input_ptr,
            fw_quantized_hidden_state_ptr, scaling_factors_ptr,
            fw_hidden_state_ptr_batch, output_ptr_batch,
            params->asymmetric_quantize_inputs, zero_points_ptr,
            accum_scratch_ptr, fw_row_sums_ptr, fw_compute_row_sums);
      }
      // Backward cell.
      float* bw_hidden_state_ptr_batch =
          GetTensorData<float>(bw_hidden_state) + b * bw_num_units;
      float* bw_output_offset =
          params->merge_outputs
              ? GetTensorData<float>(fw_output) +
                    b * bw_output_step * max_time + fw_num_units
              : GetTensorData<float>(bw_output) + b * bw_output_step * max_time;
      for (int s = max_time - 1; s >= 0; s--) {
        const float* input_ptr_batch = GetTensorData<float>(input) +
                                       b * input_size * max_time +
                                       s * input_size;
        const float* aux_input_ptr_batch =
            (aux_input != nullptr)
                ? GetTensorData<float>(aux_input) + b * input_size * max_time +
                      s * input_size
                : nullptr;
        float* output_ptr_batch = bw_output_offset + s * bw_output_step;

        kernel_utils::RnnBatchStep(
            input_ptr_batch, bw_input_weights_ptr, bw_input_weights_scale,
            aux_input_ptr_batch, aux_bw_input_weights_ptr,
            aux_bw_input_weights_scale, bw_recurrent_weights_ptr,
            bw_recurrent_weights_scale, bw_bias_ptr, input_size, aux_input_size,
            bw_num_units, /*batch_size=*/1, bw_output_step, params->activation,
            quantized_input_ptr, aux_quantized_input_ptr,
            bw_quantized_hidden_state_ptr, scaling_factors_ptr,
            bw_hidden_state_ptr_batch, output_ptr_batch,
            params->asymmetric_quantize_inputs, zero_points_ptr,
            accum_scratch_ptr, bw_row_sums_ptr, bw_compute_row_sums);
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* params = reinterpret_cast<TfLiteBidirectionalSequenceRNNParams*>(
      node->builtin_data);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* fw_input_weights;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kFwWeightsTensor,
                                          &fw_input_weights));
  const TfLiteTensor* fw_recurrent_weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kFwRecurrentWeightsTensor,
                                 &fw_recurrent_weights));
  const TfLiteTensor* fw_bias;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kFwBiasTensor, &fw_bias));
  const TfLiteTensor* bw_input_weights;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kBwWeightsTensor,
                                          &bw_input_weights));
  const TfLiteTensor* bw_recurrent_weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kBwRecurrentWeightsTensor,
                                 &bw_recurrent_weights));
  const TfLiteTensor* bw_bias;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kBwBiasTensor, &bw_bias));

  // Get auxiliary inputs.
  const TfLiteTensor* aux_input =
      GetOptionalInputTensor(context, node, kAuxInputTensor);
  const TfLiteTensor* fw_aux_input_weights =
      GetOptionalInputTensor(context, node, kFwAuxWeightsTensor);
  const TfLiteTensor* bw_aux_input_weights =
      GetOptionalInputTensor(context, node, kBwAuxWeightsTensor);

  TfLiteTensor* fw_hidden_state =
      GetVariableInput(context, node, kFwHiddenStateTensor);
  TFLITE_DCHECK(fw_hidden_state != nullptr);
  TfLiteTensor* bw_hidden_state =
      GetVariableInput(context, node, kBwHiddenStateTensor);
  TFLITE_DCHECK(bw_hidden_state != nullptr);

  TfLiteTensor* fw_output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kFwOutputTensor, &fw_output));
  TfLiteTensor* bw_output = params->merge_outputs
                                ? nullptr
                                : GetOutput(context, node, kBwOutputTensor);

  const bool has_previous_bw_output = (aux_input != nullptr);
  const bool use_aux_input = (fw_aux_input_weights != nullptr);

  // We want to cover the following cases:
  //
  // If not stacking (not connected after other bidi lstms):
  //   both fw & bw will just use `input`; aux_input will be null.
  //
  // If stacking with cross_links, TensorFlow equivalent
  // (tf.contrib.rnn.stack_bidirectional_rnn):
  //   both fw & bw will use `input`, but aux_input will be none null.
  //   Note, this time, whether connected after other bidi lstms both works.
  //
  // If stacking without cross_links, but connected after other bidi lstms,
  // TensorFlow equivalent (tf.nn.static_bidirectional_rnn):
  //   fw will use `input`, bw will use aux_input, and the `real aux_input`
  //   will be null.

  const bool non_stacking_mode = !use_aux_input && has_previous_bw_output;
  const TfLiteTensor* bw_input = non_stacking_mode ? aux_input : input;
  const TfLiteTensor* real_aux_input = non_stacking_mode ? nullptr : aux_input;

  switch (fw_input_weights->type) {
    case kTfLiteFloat32:
      return EvalFloat(input, bw_input, fw_input_weights, fw_recurrent_weights,
                       fw_bias, bw_input_weights, bw_recurrent_weights, bw_bias,
                       real_aux_input, fw_aux_input_weights,
                       bw_aux_input_weights, params, fw_hidden_state, fw_output,
                       bw_hidden_state, bw_output);
    case kTfLiteUInt8:
    case kTfLiteInt8: {
      TfLiteTensor* input_quantized;
      TF_LITE_ENSURE_OK(
          context,
          GetTemporarySafe(context, node, kInputQuantized, &input_quantized));
      TfLiteTensor* fw_hidden_state_quantized;
      TF_LITE_ENSURE_OK(context,
                        GetTemporarySafe(context, node, kFwHiddenStateQuantized,
                                         &fw_hidden_state_quantized));
      TfLiteTensor* bw_hidden_state_quantized;
      TF_LITE_ENSURE_OK(context,
                        GetTemporarySafe(context, node, kBwHiddenStateQuantized,
                                         &bw_hidden_state_quantized));
      TfLiteTensor* scaling_factors;
      TF_LITE_ENSURE_OK(
          context,
          GetTemporarySafe(context, node, kScalingFactors, &scaling_factors));
      TfLiteTensor* zero_points;
      TF_LITE_ENSURE_OK(
          context, GetTemporarySafe(context, node, kZeroPoints, &zero_points));
      TfLiteTensor* accum_scratch;
      TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, kAccumScratch,
                                                  &accum_scratch));
      TfLiteTensor* fw_row_sums;
      TF_LITE_ENSURE_OK(
          context, GetTemporarySafe(context, node, kFwRowSums, &fw_row_sums));
      TfLiteTensor* bw_row_sums;
      TF_LITE_ENSURE_OK(
          context, GetTemporarySafe(context, node, kBwRowSums, &bw_row_sums));
      TfLiteTensor* aux_input_quantized =
          use_aux_input ? GetTemporary(context, node, kAuxInputQuantized)
                        : nullptr;
      auto* op_data = reinterpret_cast<OpData*>(node->user_data);
      return EvalHybrid(
          input, bw_input, fw_input_weights, fw_recurrent_weights, fw_bias,
          bw_input_weights, bw_recurrent_weights, bw_bias, real_aux_input,
          fw_aux_input_weights, bw_aux_input_weights, params, scaling_factors,
          input_quantized, aux_input_quantized, fw_hidden_state_quantized,
          fw_hidden_state, fw_output, bw_hidden_state_quantized,
          bw_hidden_state, bw_output, zero_points, accum_scratch, fw_row_sums,
          bw_row_sums, &op_data->fw_compute_row_sums,
          &op_data->bw_compute_row_sums);
    }
    default:
      context->ReportError(context, "Type not currently supported.");
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace bidirectional_sequence_rnn

TfLiteRegistration* Register_BIDIRECTIONAL_SEQUENCE_RNN() {
  static TfLiteRegistration r = {
      bidirectional_sequence_rnn::Init, bidirectional_sequence_rnn::Free,
      bidirectional_sequence_rnn::Prepare, bidirectional_sequence_rnn::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
