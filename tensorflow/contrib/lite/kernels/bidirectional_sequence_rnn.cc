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
#include <unistd.h>
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
constexpr int kBwWeightsTensor = 4;
constexpr int kBwRecurrentWeightsTensor = 5;
constexpr int kBwBiasTensor = 6;
// State and output tensors.
constexpr int kFwHiddenStateTensor = 0;
constexpr int kFwOutputTensor = 1;
constexpr int kBwHiddenStateTensor = 2;
constexpr int kBwOutputTensor = 3;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // Check we have all the inputs and outputs we need.
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 7);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 4);

  TfLiteTensor* input = &context->tensors[node->inputs->data[kInputTensor]];
  TfLiteTensor* fw_input_weights =
      &context->tensors[node->inputs->data[kFwWeightsTensor]];
  TfLiteTensor* fw_recurrent_weights =
      &context->tensors[node->inputs->data[kFwRecurrentWeightsTensor]];
  TfLiteTensor* fw_bias = &context->tensors[node->inputs->data[kFwBiasTensor]];
  TfLiteTensor* bw_input_weights =
      &context->tensors[node->inputs->data[kBwWeightsTensor]];
  TfLiteTensor* bw_recurrent_weights =
      &context->tensors[node->inputs->data[kBwRecurrentWeightsTensor]];
  TfLiteTensor* bw_bias = &context->tensors[node->inputs->data[kBwBiasTensor]];

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
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

  TfLiteTensor* fw_output =
      &context->tensors[node->outputs->data[kFwOutputTensor]];
  TfLiteTensor* bw_output =
      &context->tensors[node->outputs->data[kBwOutputTensor]];

  // Resize hidden states.
  TfLiteIntArray* fw_hidden_state_size_array = TfLiteIntArrayCreate(2);
  fw_hidden_state_size_array->data[0] = batch_size;
  fw_hidden_state_size_array->data[1] = fw_num_units;
  TfLiteTensor* fw_hidden_state =
      &context->tensors[node->outputs->data[kFwHiddenStateTensor]];
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, fw_hidden_state,
                                                   fw_hidden_state_size_array));

  TfLiteIntArray* bw_hidden_state_size_array = TfLiteIntArrayCreate(2);
  bw_hidden_state_size_array->data[0] = batch_size;
  bw_hidden_state_size_array->data[1] = fw_num_units;
  TfLiteTensor* bw_hidden_state =
      &context->tensors[node->outputs->data[kBwHiddenStateTensor]];
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, bw_hidden_state,
                                                   bw_hidden_state_size_array));

  // Mark hidden states as a persistent tensor.
  fw_hidden_state->allocation_type = kTfLiteArenaRwPersistent;
  bw_hidden_state->allocation_type = kTfLiteArenaRwPersistent;

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

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSequenceRNNParams*>(node->builtin_data);

  TfLiteTensor* input = &context->tensors[node->inputs->data[kInputTensor]];
  TfLiteTensor* fw_input_weights =
      &context->tensors[node->inputs->data[kFwWeightsTensor]];
  TfLiteTensor* fw_recurrent_weights =
      &context->tensors[node->inputs->data[kFwRecurrentWeightsTensor]];
  TfLiteTensor* fw_bias = &context->tensors[node->inputs->data[kFwBiasTensor]];
  TfLiteTensor* fw_hidden_state =
      &context->tensors[node->outputs->data[kFwHiddenStateTensor]];
  TfLiteTensor* fw_output =
      &context->tensors[node->outputs->data[kFwOutputTensor]];

  TfLiteTensor* bw_input_weights =
      &context->tensors[node->inputs->data[kBwWeightsTensor]];
  TfLiteTensor* bw_recurrent_weights =
      &context->tensors[node->inputs->data[kBwRecurrentWeightsTensor]];
  TfLiteTensor* bw_bias = &context->tensors[node->inputs->data[kBwBiasTensor]];
  TfLiteTensor* bw_hidden_state =
      &context->tensors[node->outputs->data[kBwHiddenStateTensor]];
  TfLiteTensor* bw_output =
      &context->tensors[node->outputs->data[kBwOutputTensor]];

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

}  // namespace bidirectional_sequence_rnn

TfLiteRegistration* Register_BIDIRECTIONAL_SEQUENCE_RNN() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 bidirectional_sequence_rnn::Prepare,
                                 bidirectional_sequence_rnn::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
