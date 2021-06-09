/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/add_n.h"

#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

constexpr int kInputTensor0 = 0;
constexpr int kOutputTensor = 0;

constexpr int kAddNIntegerShift = 20;

// only used with INT8 tensors
struct OpData {
  int32_t output_activation_min;
  int32_t output_activation_max;
  int32_t input_offset;
  int32_t output_offset;
  int32_t input_multiplier;
  int32_t output_multiplier;
  int input_shift;
  int output_shift;
  int left_shift;
  int scratch_index;
};

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node) {
  int num_inputs = NumInputs(node);
  TF_LITE_ENSURE(context, num_inputs >= 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input_tensor_first;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputTensor0, &input_tensor_first));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Check that all tensors have the same shape and type.
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input_tensor_first->type);
  for (int i = kInputTensor0 + 1; i < num_inputs; ++i) {
    const TfLiteTensor* input;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &input));
    TF_LITE_ENSURE(context, HaveSameShapes(input_tensor_first, input));
    TF_LITE_ENSURE_TYPES_EQ(context, input_tensor_first->type, input->type);

    // Check that all INT8 input tensors have the same zero-point and scale.
    if (input_tensor_first->type == kTfLiteInt8) {
      TF_LITE_ENSURE(context, input_tensor_first->params.zero_point ==
                                  input->params.zero_point);
      TF_LITE_ENSURE(context,
                     input_tensor_first->params.scale == input->params.scale);
    }
  }

  if (output->type == kTfLiteFloat32) {
    // Allocate scratch buffer space for pointer to each tensor's data
    // and store the scratch buffer index in the node's user_data
    int scratch_index;
    size_t scratch_size = sizeof(float*) * num_inputs;
    TF_LITE_ENSURE_OK(context, context->RequestScratchBufferInArena(
                                   context, scratch_size, &scratch_index));
    node->user_data =
        reinterpret_cast<decltype(node->user_data)>(scratch_index);
  } else if (output->type == kTfLiteInt8) {
    node->user_data =
        context->AllocatePersistentBuffer(context, sizeof(OpData));
    OpData* data = static_cast<OpData*>(node->user_data);

    // Allocate scratch buffer space for pointer to each tensor's data
    // and store the scratch buffer index in OpData
    size_t scratch_size = sizeof(int8_t*) * num_inputs;
    TF_LITE_ENSURE_OK(
        context, context->RequestScratchBufferInArena(context, scratch_size,
                                                      &data->scratch_index));

    // 8bit -> 8bit general quantized path, with general rescalings
    data->input_offset = -input_tensor_first->params.zero_point;
    data->output_offset = output->params.zero_point;
    data->left_shift = kAddNIntegerShift;
    const double twice_max_input_scale =
        2 * static_cast<double>(input_tensor_first->params.scale);
    const double real_input_multiplier =
        static_cast<double>(input_tensor_first->params.scale) /
        twice_max_input_scale;
    const double real_output_multiplier =
        twice_max_input_scale /
        ((1 << data->left_shift) * static_cast<double>(output->params.scale));

    QuantizeMultiplierSmallerThanOneExp(
        real_input_multiplier, &data->input_multiplier, &data->input_shift);

    QuantizeMultiplierSmallerThanOneExp(
        real_output_multiplier, &data->output_multiplier, &data->output_shift);

    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, kTfLiteActNone, output, &data->output_activation_min,
        &data->output_activation_max));
  } else {
    TF_LITE_KERNEL_LOG(context, "ADD_N only supports FLOAT32 and INT8, got %s.",
                       TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return CalculateOpData(context, node);
}

template <typename T>
inline const T** CopyInputsToScratchBuffer(TfLiteContext* context,
                                           TfLiteNode* node,
                                           const int scratch_index) {
  int num_inputs = NumInputs(node);
  void* scratch_buffer = context->GetScratchBuffer(context, scratch_index);
  const T** all_inputs = static_cast<decltype(all_inputs)>(scratch_buffer);
  for (int i = 0; i < num_inputs; i++) {
    const TfLiteEvalTensor* next_input =
        tflite::micro::GetEvalInput(context, node, kInputTensor0 + i);
    all_inputs[i] = tflite::micro::GetTensorData<T>(next_input);
  }

  return all_inputs;
}

template <typename T>
void EvalAddN(TfLiteContext* context, TfLiteNode* node,
              TfLiteEvalTensor* output) {
  int num_inputs = NumInputs(node);

  int scratch_index =
      static_cast<int>(reinterpret_cast<intptr_t>(node->user_data));
  const T** all_inputs =
      CopyInputsToScratchBuffer<T>(context, node, scratch_index);

  reference_ops::AddN<T>(tflite::micro::GetTensorShape(output), num_inputs,
                         all_inputs, tflite::micro::GetTensorData<T>(output));
}

template <typename T>
void EvalAddNQuantized(TfLiteContext* context, TfLiteNode* node,
                       TfLiteEvalTensor* output) {
  int num_inputs = NumInputs(node);

  OpData* data = static_cast<OpData*>(node->user_data);
  const T** all_inputs =
      CopyInputsToScratchBuffer<T>(context, node, data->scratch_index);

  ArithmeticParams params;
  params.left_shift = data->left_shift;
  params.input1_offset = data->input_offset;
  params.input1_multiplier = data->input_multiplier;
  params.input1_shift = data->input_shift;
  params.output_offset = data->output_offset;
  params.output_multiplier = data->output_multiplier;
  params.output_shift = data->output_shift;
  SetActivationParams(data->output_activation_min, data->output_activation_max,
                      &params);

  reference_ops::AddN(params, tflite::micro::GetTensorShape(output), num_inputs,
                      all_inputs, tflite::micro::GetTensorData<T>(output));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  if (output->type == kTfLiteFloat32) {
    EvalAddN<float>(context, node, output);
  } else if (output->type == kTfLiteInt8) {
    EvalAddNQuantized<int8_t>(context, node, output);
  } else {
    TF_LITE_KERNEL_LOG(context, "ADD_N only supports FLOAT32 and INT8, got %s.",
                       TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_ADD_N() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
