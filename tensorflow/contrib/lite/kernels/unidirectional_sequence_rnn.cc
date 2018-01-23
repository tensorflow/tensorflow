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
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace unidirectional_sequence_rnn {

constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kRecurrentWeightsTensor = 2;
constexpr int kBiasTensor = 3;
constexpr int kHiddenStateTensor = 0;
constexpr int kOutputTensor = 1;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // Check we have all the inputs and outputs we need.
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 4);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 2);

  TfLiteTensor* input = &context->tensors[node->inputs->data[kInputTensor]];
  TfLiteTensor* input_weights =
      &context->tensors[node->inputs->data[kWeightsTensor]];
  TfLiteTensor* recurrent_weights =
      &context->tensors[node->inputs->data[kRecurrentWeightsTensor]];
  TfLiteTensor* bias = &context->tensors[node->inputs->data[kBiasTensor]];

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  auto* params = reinterpret_cast<TfLiteSequenceRNNParams*>(node->builtin_data);
  const bool time_major = params->time_major;
  const int batch_size =
      (time_major) ? input->dims->data[1] : input->dims->data[0];
  const int max_time =
      (time_major) ? input->dims->data[0] : input->dims->data[1];
  const int num_units = input_weights->dims->data[0];
  TF_LITE_ASSERT_EQ(input->dims->data[2], input_weights->dims->data[1]);
  TF_LITE_ASSERT_EQ(input_weights->dims->data[0], bias->dims->data[0]);
  TF_LITE_ASSERT_EQ(recurrent_weights->dims->data[0], bias->dims->data[0]);
  TF_LITE_ASSERT_EQ(recurrent_weights->dims->data[1], bias->dims->data[0]);

  TfLiteTensor* hidden_state =
      &context->tensors[node->outputs->data[kHiddenStateTensor]];
  TfLiteTensor* output = &context->tensors[node->outputs->data[kOutputTensor]];

  // Resize state.
  TfLiteIntArray* hidden_state_size_array = TfLiteIntArrayCreate(2);
  hidden_state_size_array->data[0] = batch_size;
  hidden_state_size_array->data[1] = num_units;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, hidden_state,
                                                   hidden_state_size_array));

  // Mark hidden state as a persistent tensor.
  hidden_state->allocation_type = kTfLiteArenaRwPersistent;

  // Resize output.
  TfLiteIntArray* output_size_array = TfLiteIntArrayCreate(3);
  output_size_array->data[0] = (time_major) ? max_time : batch_size;
  output_size_array->data[1] = (time_major) ? batch_size : max_time;
  output_size_array->data[2] = num_units;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output,
                                                   output_size_array));

  return kTfLiteOk;
}

namespace {
void RnnStep(const float* input_ptr_batch, const float* input_weights_ptr,
             const float* recurrent_weights_ptr, const float* bias_ptr,
             int input_size, int num_units, int input_weights_stride,
             int recurrent_weights_stride, TfLiteFusedActivation activation,
             float* hidden_state_ptr_batch, float* output_ptr_batch) {
  // Output = bias
  for (int o = 0; o < num_units; o++) {
    output_ptr_batch[o] = bias_ptr[o];
  }

  // Output += input * input_weights
  for (int o = 0; o < num_units; o++) {
    for (int i = 0; i < input_size; i++) {
      output_ptr_batch[o] += input_ptr_batch[i] * input_weights_ptr[i];
    }
    input_weights_ptr += input_weights_stride;
  }

  // Output += recurrent_weights * hidden_state
  for (int o = 0; o < num_units; o++) {
    for (int h = 0; h < num_units; h++) {
      output_ptr_batch[o] +=
          hidden_state_ptr_batch[h] * recurrent_weights_ptr[h];
    }
    recurrent_weights_ptr += recurrent_weights_stride;
  }

  // Output = activation(Output) and update hidden_state
  for (int o = 0; o < num_units; o++) {
    output_ptr_batch[o] = (ActivationFunctor(activation))(output_ptr_batch[o]);
    hidden_state_ptr_batch[o] = output_ptr_batch[o];
  }
}
}  // namespace

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSequenceRNNParams*>(node->builtin_data);

  TfLiteTensor* input = &context->tensors[node->inputs->data[kInputTensor]];
  TfLiteTensor* input_weights =
      &context->tensors[node->inputs->data[kWeightsTensor]];
  TfLiteTensor* recurrent_weights =
      &context->tensors[node->inputs->data[kRecurrentWeightsTensor]];
  TfLiteTensor* bias = &context->tensors[node->inputs->data[kBiasTensor]];
  TfLiteTensor* hidden_state =
      &context->tensors[node->outputs->data[kHiddenStateTensor]];
  TfLiteTensor* output = &context->tensors[node->outputs->data[kOutputTensor]];

  // Initialize the pointer bias.
  const float* bias_ptr = bias->data.f;

  const bool time_major = params->time_major;
  const int batch_size =
      (time_major) ? input->dims->data[1] : input->dims->data[0];
  const int max_time =
      (time_major) ? input->dims->data[0] : input->dims->data[1];
  const int num_units = input_weights->dims->data[0];
  const int input_size = input->dims->data[2];
  const int input_weights_stride = input_weights->dims->data[1];
  const int recurrent_weights_stride = recurrent_weights->dims->data[1];

  // Initialize input_weights and recurrent_weights.
  const float* input_weights_ptr = input_weights->data.f;
  const float* recurrent_weights_ptr = recurrent_weights->data.f;

  if (time_major) {
    // Unroll the sequence
    for (int s = 0; s < max_time; s++) {
      for (int b = 0; b < batch_size; b++) {
        // Initialize the pointer to hidden state.
        float* hidden_state_ptr_batch = hidden_state->data.f + b * num_units;
        // Initialize the pointer to input and output.
        const float* input_ptr_batch =
            input->data.f + s * input_size * batch_size + b * input_size;
        float* output_ptr_batch =
            output->data.f + s * num_units * batch_size + b * num_units;

        RnnStep(input_ptr_batch, input_weights_ptr, recurrent_weights_ptr,
                bias_ptr, input_size, num_units, input_weights_stride,
                recurrent_weights_stride, params->activation,
                hidden_state_ptr_batch, output_ptr_batch);
      }
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

        RnnStep(input_ptr_batch, input_weights_ptr, recurrent_weights_ptr,
                bias_ptr, input_size, num_units, input_weights_stride,
                recurrent_weights_stride, params->activation,
                hidden_state_ptr_batch, output_ptr_batch);
      }
    }
  }
  return kTfLiteOk;
}

}  // namespace unidirectional_sequence_rnn

TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_RNN() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 unidirectional_sequence_rnn::Prepare,
                                 unidirectional_sequence_rnn::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
