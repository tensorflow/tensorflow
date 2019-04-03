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
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace add_n {

constexpr int kInputTensor1 = 0;
constexpr int kOutputTensor = 0;

struct OpData {
  int num_inputs;
  void* all_inputs;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  return new OpData();
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  int num_inputs = NumInputs(node);
  TF_LITE_ENSURE(context, num_inputs >= 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  output->type = input1->type;

  // Check that all input tensors have the same shape and type.
  for (int i = kInputTensor1 + 1; i < num_inputs; ++i) {
    const TfLiteTensor* input = GetInput(context, node, i);
    TF_LITE_ENSURE(context, HaveSameShapes(input1, input));
    TF_LITE_ENSURE_EQ(context, input1->type, input->type);
  }

  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  data->num_inputs = NumInputs(node);
  if (output->type == kTfLiteFloat32) {
    data->all_inputs = reinterpret_cast<void*>(
        new VectorOfTensors<float>(*context, *node->inputs));
  } else if (output->type == kTfLiteInt32) {
    data->all_inputs = reinterpret_cast<void*>(
        new VectorOfTensors<int32_t>(*context, *node->inputs));
  } else {
    context->ReportError(context,
                         "AddN only supports FLOAT32|INT32 now, got %s.",
                         TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }

  // Use the first input node's dimension to be the dimension of the output
  // node.
  TfLiteIntArray* input1_dims = input1->dims;
  TfLiteIntArray* output_dims = TfLiteIntArrayCopy(input1_dims);
  return context->ResizeTensor(context, output, output_dims);
}

template <typename T>
void EvalAddN(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  VectorOfTensors<T>* all_inputs =
      static_cast<VectorOfTensors<T>*>(data->all_inputs);
  const int num_inputs = data->num_inputs;
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  reference_ops::AddN<T>(GetTensorShape(input1), num_inputs, all_inputs->data(),
                         GetTensorData<T>(output));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  if (output->type == kTfLiteFloat32) {
    EvalAddN<float>(context, node);
  } else if (output->type == kTfLiteInt32) {
    EvalAddN<int32_t>(context, node);
  } else {
    context->ReportError(context,
                         "AddN only supports FLOAT32|INT32 now, got %s.",
                         TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace add_n

TfLiteRegistration* Register_ADD_N() {
  static TfLiteRegistration r = {add_n::Init, add_n::Free, add_n::Prepare,
                                 add_n::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
