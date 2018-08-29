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
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <limits>

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/activation_functor.h"
#include "tensorflow/contrib/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace bidirectional_sequence_rnn {

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
// Output tensors.
constexpr int kFwOutputTensor = 0;
constexpr int kBwOutputTensor = 1;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* scratch_tensor_index = new int;
  context->AddTensors(context, /*tensors_to_add=*/3, scratch_tensor_index);
  return scratch_tensor_index;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<int*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // Check we have all the inputs and outputs we need.
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 9);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 2);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* fw_input_weights =
      GetInput(context, node, kFwWeightsTensor);
  const TfLiteTensor* fw_recurrent_weights =
      GetInput(context, node, kFwRecurrentWeightsTensor);
  const TfLiteTensor* fw_bias = GetInput(context, node, kFwBiasTensor);
  const TfLiteTensor* fw_hidden_state =
      GetInput(context, node, kFwHiddenStateTensor);
  const TfLiteTensor* bw_input_weights =
      GetInput(context, node, kBwWeightsTensor);
  const TfLiteTensor* bw_recurrent_weights =
      GetInput(context, node, kBwRecurrentWeightsTensor);
  const TfLiteTensor* bw_bias = GetInput(context, node, kBwBiasTensor);
  const TfLiteTensor* bw_hidden_state =
      GetInput(context, node, kBwHiddenStateTensor);

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);

  const int batch_size = input->dims->data[0];
  const int max_time = input->dims->data[1];
  const int fw_num_units = fw_input_weights->dims->data[0];
  const int bw_num_units = bw_input_weights->dims->data[0];
  TF_LITE_ASSERT_EQ(input->dims->data[2], fw_input_weights->dims->data[1]);
  TF_LITE_ASSERT_EQ(input->dims->data[2], bw_input_weights->dims->data[1]);
  TF_LITE_ASSERT_EQ(fw_input_weights->dims->data[0], fw_bias->dims->data[0]);
  TF_LITE_ASSERT_EQ(bw_input_weights->dims->data[0], bw_bias->dims->data[0]);
  TF_LITE_ASSERT_EQ(fw_recurrent_weights->dims->data[0],
                    fw_bias->dims->data[0]);
  TF_LITE_ASSERT_EQ(bw_recurrent_weights->dims->data[1],
                    bw_bias->dims->data[0]);
  TF_LITE_ENSURE_EQ(context, NumDimensions(fw_hidden_state), 2);
  TF_LITE_ENSURE_EQ(context, fw_hidden_state->dims->data[0], batch_size);
  TF_LITE_ENSURE_EQ(context, fw_hidden_state->dims->data[1], fw_num_units);
  TF_LITE_ENSURE_EQ(context, NumDimensions(bw_hidden_state), 2);
  TF_LITE_ENSURE_EQ(context, bw_hidden_state->dims->data[0], batch_size);
  TF_LITE_ENSURE_EQ(context, bw_hidden_state->dims->data[1], bw_num_units);

  TfLiteTensor* fw_output = GetOutput(context, node, kFwOutputTensor);
  TfLiteTensor* bw_output = GetOutput(context, node, kBwOutputTensor);

  const bool is_hybrid_op =
      (fw_input_weights->type == kTfLiteUInt8 && input->type == kTfLiteFloat32);

  if (is_hybrid_op) {
    int* scratch_tensor_index = reinterpret_cast<int*>(node->user_data);
    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(2);
    node->temporaries->data[0] = *scratch_tensor_index;
    TfLiteTensor* input_quantized = GetTemporary(context, node, /*index=*/0);
    input_quantized->type = kTfLiteUInt8;
    input_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
      TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                       input_quantized_size));
    }
    node->temporaries->data[1] = *scratch_tensor_index + 1;
    TfLiteTensor* fw_hidden_state_quantized =
        GetTemporary(context, node, /*index=*/1);
    fw_hidden_state_quantized->type = kTfLiteUInt8;
    fw_hidden_state_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(fw_hidden_state_quantized->dims,
                             fw_hidden_state->dims)) {
      TfLiteIntArray* fw_hidden_state_quantized_size =
          TfLiteIntArrayCopy(fw_hidden_state->dims);
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, fw_hidden_state_quantized,
                                         fw_hidden_state_quantized_size));
    }
    node->temporaries->data[2] = *scratch_tensor_index + 2;
    TfLiteTensor* bw_hidden_state_quantized =
        GetTemporary(context, node, /*index=*/2);
    bw_hidden_state_quantized->type = kTfLiteUInt8;
    bw_hidden_state_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(bw_hidden_state_quantized->dims,
                             bw_hidden_state->dims)) {
      TfLiteIntArray* bw_hidden_state_quantized_size =
          TfLiteIntArrayCopy(bw_hidden_state->dims);
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, bw_hidden_state_quantized,
                                         bw_hidden_state_quantized_size));
    }
  }

  // Resize outputs.
  TfLiteIntArray* fw_output_size_array = TfLiteIntArrayCreate(3);
  fw_output_size_array->data[0] = batch_size;
  fw_output_size_array->data[1] = max_time;
  fw_output_size_array->data[2] = fw_num_units;
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, fw_output, fw_output_size_array));
  TfLiteIntArray* bw_output_size_array = TfLiteIntArrayCreate(3);
  bw_output_size_array->data[0] = batch_size;
  bw_output_size_array->data[1] = max_time;
  bw_output_size_array->data[2] = bw_num_units;
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, bw_output, bw_output_size_array));

  return kTfLiteOk;
}

TfLiteStatus EvalFloat(const TfLiteTensor* input,
                       const TfLiteTensor* fw_input_weights,
                       const TfLiteTensor* fw_recurrent_weights,
                       const TfLiteTensor* fw_bias,
                       const TfLiteTensor* bw_input_weights,
                       const TfLiteTensor* bw_recurrent_weights,
                       const TfLiteTensor* bw_bias,
                       const TfLiteSequenceRNNParams* params,
                       TfLiteTensor* fw_hidden_state, TfLiteTensor* fw_output,
                       TfLiteTensor* bw_hidden_state, TfLiteTensor* bw_output) {
  const int batch_size = input->dims->data[0];
  const int max_time = input->dims->data[1];
  const int input_size = input->dims->data[2];

  const int fw_num_units = fw_input_weights->dims->data[0];
  const float* fw_bias_ptr = fw_bias->data.f;
  const float* fw_input_weights_ptr = fw_input_weights->data.f;
  const float* fw_recurrent_weights_ptr = fw_recurrent_weights->data.f;

  const int bw_num_units = bw_input_weights->dims->data[0];
  const float* bw_bias_ptr = bw_bias->data.f;
  const float* bw_input_weights_ptr = bw_input_weights->data.f;
  const float* bw_recurrent_weights_ptr = bw_recurrent_weights->data.f;

  for (int b = 0; b < batch_size; b++) {
    // Forward cell.
    float* fw_hidden_state_ptr_batch =
        fw_hidden_state->data.f + b * fw_num_units;
    for (int s = 0; s < max_time; s++) {
      const float* input_ptr_batch =
          input->data.f + b * input_size * max_time + s * input_size;
      float* output_ptr_batch =
          fw_output->data.f + b * fw_num_units * max_time + s * fw_num_units;

      kernel_utils::RnnBatchStep(
          input_ptr_batch, fw_input_weights_ptr, fw_recurrent_weights_ptr,
          fw_bias_ptr, input_size, fw_num_units, /*batch_size=*/1,
          params->activation, fw_hidden_state_ptr_batch, output_ptr_batch);
    }
    // Backward cell.
    float* bw_hidden_state_ptr_batch =
        bw_hidden_state->data.f + b * bw_num_units;
    for (int s = max_time - 1; s >= 0; s--) {
      const float* input_ptr_batch =
          input->data.f + b * input_size * max_time + s * input_size;
      float* output_ptr_batch =
          bw_output->data.f + b * bw_num_units * max_time + s * bw_num_units;

      kernel_utils::RnnBatchStep(
          input_ptr_batch, bw_input_weights_ptr, bw_recurrent_weights_ptr,
          bw_bias_ptr, input_size, bw_num_units, /*batch_size=*/1,
          params->activation, bw_hidden_state_ptr_batch, output_ptr_batch);
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalHybrid(
    const TfLiteTensor* input, const TfLiteTensor* fw_input_weights,
    const TfLiteTensor* fw_recurrent_weights, const TfLiteTensor* fw_bias,
    const TfLiteTensor* bw_input_weights,
    const TfLiteTensor* bw_recurrent_weights, const TfLiteTensor* bw_bias,
    const TfLiteSequenceRNNParams* params, TfLiteTensor* input_quantized,
    TfLiteTensor* fw_hidden_state_quantized, TfLiteTensor* fw_scaling_factors,
    TfLiteTensor* fw_hidden_state, TfLiteTensor* fw_output,
    TfLiteTensor* bw_hidden_state_quantized, TfLiteTensor* bw_scaling_factors,
    TfLiteTensor* bw_hidden_state, TfLiteTensor* bw_output) {
  const int batch_size = input->dims->data[0];
  const int max_time = input->dims->data[1];
  const int input_size = input->dims->data[2];

  const int fw_num_units = fw_input_weights->dims->data[0];
  const float* fw_bias_ptr = fw_bias->data.f;
  const int8_t* fw_input_weights_ptr =
      reinterpret_cast<const int8_t*>(fw_input_weights->data.uint8);
  float fw_input_weights_scale = fw_input_weights->params.scale;
  const int8_t* fw_recurrent_weights_ptr =
      reinterpret_cast<const int8_t*>(fw_recurrent_weights->data.uint8);
  float fw_recurrent_weights_scale = fw_recurrent_weights->params.scale;

  const int bw_num_units = bw_input_weights->dims->data[0];
  const float* bw_bias_ptr = bw_bias->data.f;
  const int8_t* bw_input_weights_ptr =
      reinterpret_cast<const int8_t*>(bw_input_weights->data.uint8);
  float bw_input_weights_scale = bw_input_weights->params.scale;
  const int8_t* bw_recurrent_weights_ptr =
      reinterpret_cast<const int8_t*>(bw_recurrent_weights->data.uint8);
  float bw_recurrent_weights_scale = bw_recurrent_weights->params.scale;

  // Initialize temporary storage for quantized values.
  int8_t* quantized_input_ptr =
      reinterpret_cast<int8_t*>(input_quantized->data.uint8);
  int8_t* fw_quantized_hidden_state_ptr =
      reinterpret_cast<int8_t*>(fw_hidden_state_quantized->data.uint8);
  int8_t* bw_quantized_hidden_state_ptr =
      reinterpret_cast<int8_t*>(bw_hidden_state_quantized->data.uint8);
  float* fw_scaling_factors_ptr = fw_scaling_factors->data.f;
  float* bw_scaling_factors_ptr = bw_scaling_factors->data.f;

  for (int b = 0; b < batch_size; b++) {
    // Forward cell.
    float* fw_hidden_state_ptr_batch =
        fw_hidden_state->data.f + b * fw_num_units;
    for (int s = 0; s < max_time; s++) {
      const float* input_ptr_batch =
          input->data.f + b * input_size * max_time + s * input_size;
      float* output_ptr_batch =
          fw_output->data.f + b * fw_num_units * max_time + s * fw_num_units;

      kernel_utils::RnnBatchStep(
          input_ptr_batch, fw_input_weights_ptr, fw_input_weights_scale,
          fw_recurrent_weights_ptr, fw_recurrent_weights_scale, fw_bias_ptr,
          input_size, fw_num_units, /*batch_size=*/1, params->activation,
          quantized_input_ptr, fw_quantized_hidden_state_ptr,
          fw_scaling_factors_ptr, fw_hidden_state_ptr_batch, output_ptr_batch);
    }
    // Backward cell.
    float* bw_hidden_state_ptr_batch =
        bw_hidden_state->data.f + b * bw_num_units;
    for (int s = max_time - 1; s >= 0; s--) {
      const float* input_ptr_batch =
          input->data.f + b * input_size * max_time + s * input_size;
      float* output_ptr_batch =
          bw_output->data.f + b * bw_num_units * max_time + s * bw_num_units;

      kernel_utils::RnnBatchStep(
          input_ptr_batch, bw_input_weights_ptr, bw_input_weights_scale,
          bw_recurrent_weights_ptr, bw_recurrent_weights_scale, bw_bias_ptr,
          input_size, bw_num_units, /*batch_size=*/1, params->activation,
          quantized_input_ptr, bw_quantized_hidden_state_ptr,
          bw_scaling_factors_ptr, bw_hidden_state_ptr_batch, output_ptr_batch);
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* params =
      reinterpret_cast<TfLiteSequenceRNNParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* fw_input_weights =
      GetInput(context, node, kFwWeightsTensor);
  const TfLiteTensor* fw_recurrent_weights =
      GetInput(context, node, kFwRecurrentWeightsTensor);
  const TfLiteTensor* fw_bias = GetInput(context, node, kFwBiasTensor);
  const TfLiteTensor* bw_input_weights =
      GetInput(context, node, kBwWeightsTensor);
  const TfLiteTensor* bw_recurrent_weights =
      GetInput(context, node, kBwRecurrentWeightsTensor);
  const TfLiteTensor* bw_bias = GetInput(context, node, kBwBiasTensor);

  TfLiteTensor* fw_hidden_state =
      const_cast<TfLiteTensor*>(GetInput(context, node, kFwHiddenStateTensor));
  TfLiteTensor* bw_hidden_state =
      const_cast<TfLiteTensor*>(GetInput(context, node, kBwHiddenStateTensor));

  TfLiteTensor* fw_output = GetOutput(context, node, kFwOutputTensor);
  TfLiteTensor* bw_output = GetOutput(context, node, kBwOutputTensor);

  switch (fw_input_weights->type) {
    case kTfLiteFloat32:
      return EvalFloat(input, fw_input_weights, fw_recurrent_weights, fw_bias,
                       bw_input_weights, bw_recurrent_weights, bw_bias, params,
                       fw_hidden_state, fw_output, bw_hidden_state, bw_output);
    case kTfLiteUInt8: {
      TfLiteTensor* input_quantized = GetTemporary(context, node, 0);
      TfLiteTensor* fw_hidden_state_quantized = GetTemporary(context, node, 1);
      TfLiteTensor* bw_hidden_state_quantized = GetTemporary(context, node, 2);
      TfLiteTensor* fw_scaling_factors = GetTemporary(context, node, 3);
      TfLiteTensor* bw_scaling_factors = GetTemporary(context, node, 4);
      return EvalHybrid(input, fw_input_weights, fw_recurrent_weights, fw_bias,
                        bw_input_weights, bw_recurrent_weights, bw_bias, params,
                        input_quantized, fw_hidden_state_quantized,
                        fw_scaling_factors, fw_hidden_state, fw_output,
                        bw_hidden_state_quantized, bw_scaling_factors,
                        bw_hidden_state, bw_output);
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
