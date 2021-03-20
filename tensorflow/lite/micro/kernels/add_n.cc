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
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

constexpr int kInputTensor0 = 0;
constexpr int kOutputTensor = 0;

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
  }

  // Allocate scratch buffer space for pointer to each tensor's data
  // and store the scratch buffer index in the node's user_data
  if (output->type == kTfLiteFloat32) {
    int scratch_index;
    size_t scratch_size = sizeof(float*) * num_inputs;
    TF_LITE_ENSURE_OK(context, context->RequestScratchBufferInArena(
                                   context, scratch_size, &scratch_index));
    node->user_data =
        reinterpret_cast<decltype(node->user_data)>(scratch_index);
  } else {
    TF_LITE_KERNEL_LOG(context, "ADD_N only supports FLOAT32, got %s.",
                       TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return CalculateOpData(context, node);
}

template <typename T>
void EvalAddN(TfLiteContext* context, TfLiteNode* node,
              TfLiteEvalTensor* output) {
  int num_inputs = NumInputs(node);

  int scratch_index =
      static_cast<int>(reinterpret_cast<intptr_t>(node->user_data));
  void* scratch_buffer = context->GetScratchBuffer(context, scratch_index);
  const T** all_inputs = static_cast<decltype(all_inputs)>(scratch_buffer);
  for (int i = 0; i < num_inputs; i++) {
    const TfLiteEvalTensor* next_input =
        tflite::micro::GetEvalInput(context, node, kInputTensor0 + i);
    all_inputs[i] = tflite::micro::GetTensorData<T>(next_input);
  }

  reference_ops::AddN<T>(tflite::micro::GetTensorShape(output), num_inputs,
                         all_inputs, tflite::micro::GetTensorData<T>(output));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  if (output->type == kTfLiteFloat32) {
    EvalAddN<float>(context, node, output);
  } else {
    TF_LITE_KERNEL_LOG(context, "ADD_N only supports FLOAT32, got %s.",
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
