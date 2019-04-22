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
#include "tensorflow/lite/kernels/internal/reference/integer_ops/add_n.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
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

typedef AddNParams OpData;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  return data;
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

  // Use the first input node's dimension to be the dimension of the output
  // node.
  TfLiteIntArray* input1_dims = input1->dims;
  TfLiteIntArray* output_dims = TfLiteIntArrayCopy(input1_dims);

  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  data->inputs.reserve(num_inputs);

  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    // 8bit -> 8bit general quantized path, with general rescalings

    const int32 offset = -input1->params.zero_point;
    data->inputs.push_back({0, 0, offset});
    float max_input_scale = input1->params.scale;

    for (int i = kInputTensor1 + 1; i < num_inputs; ++i) {
      const TfLiteTensor* input = GetInput(context, node, i);
      // Check that all input tensors have the same shape and type.
      TF_LITE_ENSURE(context, HaveSameShapes(input1, input));
      TF_LITE_ENSURE_EQ(context, input1->type, input->type);

      const int32 offset = -input->params.zero_point;
      data->inputs.push_back({0, 0, offset});

      max_input_scale = std::max(max_input_scale, input->params.scale);
    }

    const double twice_max_input_scale = 2 * max_input_scale;
    double real_multiplier;
    for (int i = kInputTensor1; i < num_inputs; ++i) {
      const TfLiteTensor* input = GetInput(context, node, i);
      real_multiplier = input->params.scale / twice_max_input_scale;
      QuantizeMultiplierSmallerThanOneExp(
          real_multiplier, &data->inputs[i].multiplier, &data->inputs[i].shift);
    }

    data->output.offset = output->params.zero_point;
    data->left_shift = 20;

    const double real_output_multiplier =
        twice_max_input_scale /
        ((1 << data->left_shift) * output->params.scale);

    QuantizeMultiplierSmallerThanOneExp(
        real_output_multiplier, &data->output.multiplier, &data->output.shift);
  }
  return context->ResizeTensor(context, output, output_dims);
}

template <typename T>
void EvalAddN(TfLiteContext* context, TfLiteNode* node) {
  // TODO(haoliang): Initialize all_inputs only once during init.
  VectorOfTensors<T> all_inputs(*context, *node->inputs);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  int num_inputs = NumInputs(node);
  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  reference_ops::AddN<T>(GetTensorShape(input1), num_inputs, all_inputs.data(),
                         GetTensorData<T>(output));
}

template <typename T>
TfLiteStatus EvalAddNQuantized(TfLiteContext* context, TfLiteNode* node) {
  int num_inputs = NumInputs(node);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  VectorOfTensors<T> all_inputs(*context, *node->inputs);
  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    reference_integer_ops::AddN<T>(data, GetTensorShape(input1), num_inputs,
                                   all_inputs.data(), GetTensorData<T>(output));
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  if (output->type == kTfLiteFloat32) {
    EvalAddN<float>(context, node);
  } else if (output->type == kTfLiteInt32) {
    EvalAddN<int32_t>(context, node);
  } else if (output->type == kTfLiteInt8) {
    TF_LITE_ENSURE_OK(context, EvalAddNQuantized<int8>(context, node));
  } else if (output->type == kTfLiteUInt8) {
    TF_LITE_ENSURE_OK(context, EvalAddNQuantized<uint8>(context, node));
  } else {
    context->ReportError(
        context, "AddN only supports float|int8|uint8|int32 now, got %s.",
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
