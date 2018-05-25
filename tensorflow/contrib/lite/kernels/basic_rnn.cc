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
#include <stddef.h>
#include <stdint.h>

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/activation_functor.h"
#include "tensorflow/contrib/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace rnn {

constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kRecurrentWeightsTensor = 2;
constexpr int kBiasTensor = 3;
constexpr int kHiddenStateTensor = 0;
constexpr int kOutputTensor = 1;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* scratch_tensor_index = new int;
  context->AddTensors(context, /*tensors_to_add=*/2, scratch_tensor_index);
  return scratch_tensor_index;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<int*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // Check we have all the inputs and outputs we need.
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 4);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 2);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* input_weights = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* recurrent_weights =
      GetInput(context, node, kRecurrentWeightsTensor);
  const TfLiteTensor* bias = GetInput(context, node, kBiasTensor);

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  const int batch_size = input->dims->data[0];
  const int num_units = input_weights->dims->data[0];
  TF_LITE_ASSERT_EQ(input->dims->data[1], input_weights->dims->data[1]);
  TF_LITE_ASSERT_EQ(input_weights->dims->data[0], bias->dims->data[0]);
  TF_LITE_ASSERT_EQ(recurrent_weights->dims->data[0], bias->dims->data[0]);
  TF_LITE_ASSERT_EQ(recurrent_weights->dims->data[1], bias->dims->data[0]);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, input_weights->type, recurrent_weights->type);

  TfLiteTensor* hidden_state = GetOutput(context, node, kHiddenStateTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // Resize state.
  TfLiteIntArray* hidden_state_size_array = TfLiteIntArrayCreate(2);
  hidden_state_size_array->data[0] = batch_size;
  hidden_state_size_array->data[1] = num_units;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, hidden_state,
                                                   hidden_state_size_array));

  // Mark hidden state as a persistent tensor.
  hidden_state->allocation_type = kTfLiteArenaRwPersistent;

  // Resize output.
  TfLiteIntArray* output_size_array = TfLiteIntArrayCreate(2);
  output_size_array->data[0] = batch_size;
  output_size_array->data[1] = num_units;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size_array));

  // Allocate temporary tensors to store quantized values of input and
  // hidden_state tensors.
  if (input->type == kTfLiteFloat32 && input_weights->type == kTfLiteUInt8) {
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
  }

  return kTfLiteOk;
}

TfLiteStatus EvalFloat(const TfLiteTensor* input,
                       const TfLiteTensor* input_weights,
                       const TfLiteTensor* recurrent_weights,
                       const TfLiteTensor* bias, const TfLiteRNNParams* params,
                       TfLiteTensor* hidden_state, TfLiteTensor* output) {
  const int batch_size = input->dims->data[0];
  const int num_units = input_weights->dims->data[0];
  const int input_size = input->dims->data[1];

  // Initialize the pointer to hidden state.
  float* hidden_state_ptr_batch = hidden_state->data.f;
  // Initialize the pointer to input and output.
  const float* input_ptr_batch = input->data.f;
  float* output_ptr_batch = output->data.f;
  // Initialize input_weights, recurrent_weights and bias.
  const float* input_weights_ptr = input_weights->data.f;
  const float* recurrent_weights_ptr = recurrent_weights->data.f;
  const float* bias_ptr = bias->data.f;

  kernel_utils::RnnBatchStep(input_ptr_batch, input_weights_ptr,
                             recurrent_weights_ptr, bias_ptr, input_size,
                             num_units, batch_size, params->activation,
                             hidden_state_ptr_batch, output_ptr_batch);
  return kTfLiteOk;
}

TfLiteStatus EvalQuantized(const TfLiteTensor* input,
                           const TfLiteTensor* input_weights,
                           const TfLiteTensor* recurrent_weights,
                           const TfLiteTensor* bias,
                           const TfLiteRNNParams* params,
                           TfLiteTensor* input_scratch,
                           TfLiteTensor* hidden_state_scratch,
                           TfLiteTensor* hidden_state, TfLiteTensor* output) {
  const int batch_size = input->dims->data[0];
  const int num_units = input_weights->dims->data[0];
  const int input_size = input->dims->data[1];

  // Initialize the pointer to hidden state.
  float* hidden_state_ptr_batch = hidden_state->data.f;
  // Initialize the pointer to input and output.
  const float* input_ptr_batch = input->data.f;
  float* output_ptr_batch = output->data.f;
  // Initialize input_weights, recurrent_weights and bias.
  const int8_t* input_weights_ptr =
      reinterpret_cast<const int8_t*>(input_weights->data.uint8);
  const int8_t* recurrent_weights_ptr =
      reinterpret_cast<const int8_t*>(recurrent_weights->data.uint8);
  const float* bias_ptr = bias->data.f;
  // Get the scale of the quantized weights.
  float input_weights_scale = input_weights->params.scale;
  float recurrent_weights_scale = recurrent_weights->params.scale;
  // Initialize temporary storage for quantized values.
  int8_t* quantized_input_ptr =
      reinterpret_cast<int8_t*>(input_scratch->data.uint8);
  int8_t* quantized_hidden_state_ptr =
      reinterpret_cast<int8_t*>(hidden_state_scratch->data.uint8);

  kernel_utils::RnnBatchStep(
      input_ptr_batch, input_weights_ptr, input_weights_scale,
      recurrent_weights_ptr, recurrent_weights_scale, bias_ptr, input_size,
      num_units, batch_size, params->activation, quantized_input_ptr,
      quantized_hidden_state_ptr, hidden_state_ptr_batch, output_ptr_batch);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteRNNParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* input_weights = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* recurrent_weights =
      GetInput(context, node, kRecurrentWeightsTensor);
  const TfLiteTensor* bias = GetInput(context, node, kBiasTensor);
  TfLiteTensor* hidden_state = GetOutput(context, node, kHiddenStateTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // We already checked that weight types are consistent, so branch on one.
  switch (input_weights->type) {
    case kTfLiteFloat32:
      return EvalFloat(input, input_weights, recurrent_weights, bias, params,
                       hidden_state, output);
    case kTfLiteUInt8: {
      // TODO(mirkov): implement eval with quantized inputs as well.
      TfLiteTensor* input_quantized = GetTemporary(context, node, 0);
      TfLiteTensor* hidden_state_quantized = GetTemporary(context, node, 1);
      return EvalQuantized(input, input_weights, recurrent_weights, bias,
                           params, input_quantized, hidden_state_quantized,
                           hidden_state, output);
    }
    default:
      context->ReportError(context, "Type %d not currently supported.",
                           input_weights->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace rnn

TfLiteRegistration* Register_RNN() {
  static TfLiteRegistration r = {rnn::Init, rnn::Free, rnn::Prepare, rnn::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
