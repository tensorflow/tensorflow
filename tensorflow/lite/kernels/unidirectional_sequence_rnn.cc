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
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/activation_functor.h"
#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace unidirectional_sequence_rnn {

// Input tensors.
constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kRecurrentWeightsTensor = 2;
constexpr int kBiasTensor = 3;
constexpr int kHiddenStateTensor = 4;

// Output tensor.
constexpr int kOutputTensor = 0;

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
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 5);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* input_weights = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* recurrent_weights =
      GetInput(context, node, kRecurrentWeightsTensor);
  const TfLiteTensor* bias = GetInput(context, node, kBiasTensor);
  const TfLiteTensor* hidden_state =
      GetInput(context, node, kHiddenStateTensor);

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  auto* params = reinterpret_cast<TfLiteSequenceRNNParams*>(node->builtin_data);
  const bool time_major = params->time_major;
  const int batch_size =
      (time_major) ? input->dims->data[1] : input->dims->data[0];
  const int max_time =
      (time_major) ? input->dims->data[0] : input->dims->data[1];
  const int num_units = input_weights->dims->data[0];
  TF_LITE_ENSURE_EQ(context, input->dims->data[2],
                    input_weights->dims->data[1]);
  TF_LITE_ENSURE_EQ(context, input_weights->dims->data[0], bias->dims->data[0]);
  TF_LITE_ENSURE_EQ(context, recurrent_weights->dims->data[0],
                    bias->dims->data[0]);
  TF_LITE_ENSURE_EQ(context, recurrent_weights->dims->data[1],
                    bias->dims->data[0]);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, input_weights->type, recurrent_weights->type);
  TF_LITE_ENSURE_EQ(context, NumDimensions(hidden_state), 2);
  TF_LITE_ENSURE_EQ(context, hidden_state->dims->data[0], batch_size);
  TF_LITE_ENSURE_EQ(context, hidden_state->dims->data[1], num_units);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // Resize output.
  TfLiteIntArray* output_size_array = TfLiteIntArrayCreate(3);
  output_size_array->data[0] = (time_major) ? max_time : batch_size;
  output_size_array->data[1] = (time_major) ? batch_size : max_time;
  output_size_array->data[2] = num_units;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size_array));

  // Allocate temporary tensors to store quantized values of input and
  // hidden_state tensors.
  if (input->type == kTfLiteFloat32 && input_weights->type == kTfLiteUInt8) {
    int* scratch_tensor_index = reinterpret_cast<int*>(node->user_data);
    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(3);
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
    TfLiteTensor* hidden_state_quantized =
        GetTemporary(context, node, /*index=*/1);
    hidden_state_quantized->type = kTfLiteUInt8;
    hidden_state_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(hidden_state_quantized->dims,
                             hidden_state->dims)) {
      TfLiteIntArray* hidden_state_quantized_size =
          TfLiteIntArrayCopy(hidden_state->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, hidden_state_quantized,
                                              hidden_state_quantized_size));
    }
    node->temporaries->data[2] = *scratch_tensor_index + 2;
    TfLiteTensor* scaling_factors = GetTemporary(context, node, /*index=*/2);
    scaling_factors->type = kTfLiteFloat32;
    scaling_factors->allocation_type = kTfLiteArenaRw;
    int scaling_dims[1] = {batch_size};
    if (!TfLiteIntArrayEqualsArray(scaling_factors->dims, 1, scaling_dims)) {
      TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
      scaling_factors_size->data[0] = batch_size;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors,
                                                       scaling_factors_size));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalFloat(const TfLiteTensor* input,
                       const TfLiteTensor* input_weights,
                       const TfLiteTensor* recurrent_weights,
                       const TfLiteTensor* bias,
                       const TfLiteSequenceRNNParams* params,
                       TfLiteTensor* hidden_state, TfLiteTensor* output) {
  // Initialize the pointer bias.
  const float* bias_ptr = bias->data.f;

  const bool time_major = params->time_major;
  const int batch_size =
      (time_major) ? input->dims->data[1] : input->dims->data[0];
  const int max_time =
      (time_major) ? input->dims->data[0] : input->dims->data[1];
  const int num_units = input_weights->dims->data[0];
  const int input_size = input->dims->data[2];

  // Initialize input_weights and recurrent_weights.
  const float* input_weights_ptr = input_weights->data.f;
  const float* recurrent_weights_ptr = recurrent_weights->data.f;

  if (time_major) {
    // Initialize the pointer to hidden state.
    float* hidden_state_ptr_batch = hidden_state->data.f;
    // Unroll the sequence and use batch operations for efficiency.
    for (int s = 0; s < max_time; s++) {
      // Initialize the pointer to input and output.
      const float* input_ptr_batch =
          input->data.f + s * input_size * batch_size;
      float* output_ptr_batch = output->data.f + s * num_units * batch_size;

      kernel_utils::RnnBatchStep(
          input_ptr_batch, input_weights_ptr, recurrent_weights_ptr, bias_ptr,
          input_size, num_units, batch_size, num_units, params->activation,
          hidden_state_ptr_batch, output_ptr_batch);
    }
  } else {
    // For each batch
    for (int b = 0; b < batch_size; b++) {
      // Initialize the pointer to hidden state.
      float* hidden_state_ptr_batch = hidden_state->data.f + b * num_units;
      for (int s = 0; s < max_time; s++) {
        // Initialize the pointer to input and output.
        const float* input_ptr_batch =
            input->data.f + b * input_size * max_time + s * input_size;
        float* output_ptr_batch =
            output->data.f + b * num_units * max_time + s * num_units;

        kernel_utils::RnnBatchStep(
            input_ptr_batch, input_weights_ptr, recurrent_weights_ptr, bias_ptr,
            input_size, num_units, /*batch_size=*/1, num_units,
            params->activation, hidden_state_ptr_batch, output_ptr_batch);
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalHybrid(
    const TfLiteTensor* input, const TfLiteTensor* input_weights,
    const TfLiteTensor* recurrent_weights, const TfLiteTensor* bias,
    const TfLiteSequenceRNNParams* params, TfLiteTensor* input_scratch,
    TfLiteTensor* hidden_state_scratch, TfLiteTensor* scaling_factors,
    TfLiteTensor* hidden_state, TfLiteTensor* output) {
  const bool time_major = params->time_major;
  const int batch_size =
      (time_major) ? input->dims->data[1] : input->dims->data[0];
  const int max_time =
      (time_major) ? input->dims->data[0] : input->dims->data[1];
  const int num_units = input_weights->dims->data[0];
  const int input_size = input->dims->data[2];

  // Initialize the pointer bias.
  const float* bias_ptr = bias->data.f;
  // Initialize input_weights and recurrent_weights.
  const int8_t* input_weights_ptr =
      reinterpret_cast<const int8_t*>(input_weights->data.uint8);
  const int8_t* recurrent_weights_ptr =
      reinterpret_cast<const int8_t*>(recurrent_weights->data.uint8);
  // Get the scale of the quantized weights.
  float input_weights_scale = input_weights->params.scale;
  float recurrent_weights_scale = recurrent_weights->params.scale;
  // Initialize temporary storage for quantized values.
  int8_t* quantized_input_ptr =
      reinterpret_cast<int8_t*>(input_scratch->data.uint8);
  int8_t* quantized_hidden_state_ptr =
      reinterpret_cast<int8_t*>(hidden_state_scratch->data.uint8);
  float* scaling_factors_ptr = scaling_factors->data.f;

  if (time_major) {
    // Initialize the pointer to hidden state.
    float* hidden_state_ptr_batch = hidden_state->data.f;
    // Unroll the sequence and use batch operations for efficiency.
    for (int s = 0; s < max_time; s++) {
      // Initialize the pointer to input and output.
      const float* input_ptr_batch =
          input->data.f + s * input_size * batch_size;
      float* output_ptr_batch = output->data.f + s * num_units * batch_size;

      kernel_utils::RnnBatchStep(
          input_ptr_batch, input_weights_ptr, input_weights_scale,
          recurrent_weights_ptr, recurrent_weights_scale, bias_ptr, input_size,
          num_units, batch_size, num_units, params->activation,
          quantized_input_ptr, quantized_hidden_state_ptr, scaling_factors_ptr,
          hidden_state_ptr_batch, output_ptr_batch);
    }
  } else {
    // For each batch
    for (int b = 0; b < batch_size; b++) {
      // Initialize the pointer to hidden state.
      float* hidden_state_ptr_batch = hidden_state->data.f + b * num_units;
      for (int s = 0; s < max_time; s++) {
        // Initialize the pointer to input and output.
        const float* input_ptr_batch =
            input->data.f + b * input_size * max_time + s * input_size;
        float* output_ptr_batch =
            output->data.f + b * num_units * max_time + s * num_units;

        kernel_utils::RnnBatchStep(
            input_ptr_batch, input_weights_ptr, input_weights_scale,
            recurrent_weights_ptr, recurrent_weights_scale, bias_ptr,
            input_size, num_units, /*batch_size=*/1, num_units,
            params->activation, quantized_input_ptr, quantized_hidden_state_ptr,
            scaling_factors_ptr, hidden_state_ptr_batch, output_ptr_batch);
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSequenceRNNParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* input_weights = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* recurrent_weights =
      GetInput(context, node, kRecurrentWeightsTensor);
  const TfLiteTensor* bias = GetInput(context, node, kBiasTensor);
  // The hidden_state is a variable input tensor that can be modified.
  TfLiteTensor* hidden_state =
      const_cast<TfLiteTensor*>(GetInput(context, node, kHiddenStateTensor));
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (input_weights->type) {
    case kTfLiteFloat32:
      return EvalFloat(input, input_weights, recurrent_weights, bias, params,
                       hidden_state, output);
    case kTfLiteUInt8: {
      // TODO(mirkov): implement eval with quantized inputs as well.
      TfLiteTensor* input_quantized = GetTemporary(context, node, 0);
      TfLiteTensor* hidden_state_quantized = GetTemporary(context, node, 1);
      TfLiteTensor* scaling_factors = GetTemporary(context, node, 2);
      return EvalHybrid(input, input_weights, recurrent_weights, bias, params,
                        input_quantized, hidden_state_quantized,
                        scaling_factors, hidden_state, output);
    }
    default:
      context->ReportError(context, "Type %d not currently supported.",
                           input_weights->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace unidirectional_sequence_rnn

TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_RNN() {
  static TfLiteRegistration r = {
      unidirectional_sequence_rnn::Init, unidirectional_sequence_rnn::Free,
      unidirectional_sequence_rnn::Prepare, unidirectional_sequence_rnn::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
