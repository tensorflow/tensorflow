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
#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace rnn {

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
  const int batch_size = input->dims->data[0];
  const int num_units = input_weights->dims->data[0];
  TF_LITE_ENSURE_EQ(context, input->dims->data[1],
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
  TfLiteIntArray* output_size_array = TfLiteIntArrayCreate(2);
  output_size_array->data[0] = batch_size;
  output_size_array->data[1] = num_units;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size_array));

  const bool is_hybrid = IsHybridOp(input, input_weights);

  // Allocate temporary tensors to store quantized values of input and
  // hidden_state tensors.
  if (is_hybrid) {
    int* scratch_tensor_index = reinterpret_cast<int*>(node->user_data);
    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(3);
    node->temporaries->data[0] = *scratch_tensor_index;
    TfLiteTensor* input_quantized = GetTemporary(context, node, /*index=*/0);
    input_quantized->type = input_weights->type;
    input_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
      TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                       input_quantized_size));
    }
    node->temporaries->data[1] = *scratch_tensor_index + 1;
    TfLiteTensor* hidden_state_quantized =
        GetTemporary(context, node, /*index=*/1);
    hidden_state_quantized->type = input_weights->type;
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
                       const TfLiteTensor* bias, const TfLiteRNNParams* params,
                       TfLiteTensor* hidden_state, TfLiteTensor* output) {
  const int batch_size = input->dims->data[0];
  const int num_units = input_weights->dims->data[0];
  const int input_size = input->dims->data[1];
  const int output_batch_leading_dim =
      output->dims->data[output->dims->size - 1];

  // Initialize the pointer to hidden state.
  float* hidden_state_ptr_batch = GetTensorData<float>(hidden_state);
  // Initialize the pointer to input and output.
  const float* input_ptr_batch = GetTensorData<float>(input);
  float* output_ptr_batch = GetTensorData<float>(output);
  // Initialize input_weights, recurrent_weights and bias.
  const float* input_weights_ptr = GetTensorData<float>(input_weights);
  const float* recurrent_weights_ptr = GetTensorData<float>(recurrent_weights);
  const float* bias_ptr = GetTensorData<float>(bias);

  kernel_utils::RnnBatchStep(
      input_ptr_batch, input_weights_ptr, recurrent_weights_ptr, bias_ptr,
      input_size, num_units, batch_size, output_batch_leading_dim,
      params->activation, hidden_state_ptr_batch, output_ptr_batch);
  return kTfLiteOk;
}

TfLiteStatus EvalHybrid(const TfLiteTensor* input,
                        const TfLiteTensor* input_weights,
                        const TfLiteTensor* recurrent_weights,
                        const TfLiteTensor* bias, const TfLiteRNNParams* params,
                        TfLiteTensor* input_scratch,
                        TfLiteTensor* hidden_state_scratch,
                        TfLiteTensor* scaling_factors,
                        TfLiteTensor* hidden_state, TfLiteTensor* output) {
  const int batch_size = input->dims->data[0];
  const int num_units = input_weights->dims->data[0];
  const int input_size = input->dims->data[1];
  const int output_batch_leading_dim =
      output->dims->data[output->dims->size - 1];

  // Initialize the pointer to hidden state.
  float* hidden_state_ptr_batch = GetTensorData<float>(hidden_state);
  // Initialize the pointer to input and output.
  const float* input_ptr_batch = GetTensorData<float>(input);
  float* output_ptr_batch = GetTensorData<float>(output);
  // Initialize input_weights, recurrent_weights and bias.
  const int8_t* input_weights_ptr = GetTensorData<int8_t>(input_weights);
  const int8_t* recurrent_weights_ptr =
      GetTensorData<int8_t>(recurrent_weights);
  const float* bias_ptr = GetTensorData<float>(bias);
  // Get the scale of the quantized weights.
  float input_weights_scale = input_weights->params.scale;
  float recurrent_weights_scale = recurrent_weights->params.scale;
  // Initialize temporary storage for quantized values.
  int8_t* quantized_input_ptr = GetTensorData<int8_t>(input_scratch);
  int8_t* quantized_hidden_state_ptr =
      GetTensorData<int8_t>(hidden_state_scratch);
  float* scaling_factors_ptr = GetTensorData<float>(scaling_factors);

  kernel_utils::RnnBatchStep(
      input_ptr_batch, input_weights_ptr, input_weights_scale,
      recurrent_weights_ptr, recurrent_weights_scale, bias_ptr, input_size,
      num_units, batch_size, output_batch_leading_dim, params->activation,
      quantized_input_ptr, quantized_hidden_state_ptr, scaling_factors_ptr,
      hidden_state_ptr_batch, output_ptr_batch);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteRNNParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* input_weights = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* recurrent_weights =
      GetInput(context, node, kRecurrentWeightsTensor);
  const TfLiteTensor* bias = GetInput(context, node, kBiasTensor);
  TfLiteTensor* hidden_state =
      &context->tensors[node->inputs->data[kHiddenStateTensor]];
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // We already checked that weight types are consistent, so branch on one.
  switch (input_weights->type) {
    case kTfLiteFloat32:
      return EvalFloat(input, input_weights, recurrent_weights, bias, params,
                       hidden_state, output);
    case kTfLiteUInt8:
    case kTfLiteInt8: {
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

}  // namespace rnn

TfLiteRegistration* Register_RNN() {
  static TfLiteRegistration r = {rnn::Init, rnn::Free, rnn::Prepare, rnn::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
