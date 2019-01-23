/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace floor_div {
namespace {

// Input/output tensor index.
constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

// Op data for floor_div op.
struct OpData {
  bool requires_broadcast;
};

template <typename T>
T FloorDiv(T input1, T input2) {
  return std::floor(std::divides<double>()(static_cast<double>(input1),
                                           static_cast<double>(input2)));
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  data->requires_broadcast = false;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // Reinterprete the opaque data provided by user.
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, input1->type, input2->type);

  const TfLiteType type = input1->type;
  switch (type) {
    case kTfLiteFloat32:
    case kTfLiteInt32:
    case kTfLiteUInt8:
    case kTfLiteInt64:
      break;
    default:
      context->ReportError(context, "Type '%s' is not supported by floor_div.",
                           TfLiteTypeGetName(type));
      return kTfLiteError;
  }
  output->type = type;

  data->requires_broadcast = !HaveSameShapes(input1, input2);

  TfLiteIntArray* output_size = nullptr;
  if (data->requires_broadcast) {
    TF_LITE_ENSURE_OK(context, CalculateShapeForBroadcast(
                                   context, input1, input2, &output_size));
  } else {
    output_size = TfLiteIntArrayCopy(input1->dims);
  }

  return context->ResizeTensor(context, output, output_size);
}

template <typename T>
TfLiteStatus EvalImpl(TfLiteContext* context, bool requires_broadcast,
                      const TfLiteTensor* input1, const TfLiteTensor* input2,
                      TfLiteTensor* output) {
  const T* denominator_data = GetTensorData<T>(input2);

  // Validate the denominator.
  for (int i = 0; i < NumElements(input2); ++i) {
    if (std::equal_to<T>()(denominator_data[i], 0)) {
      context->ReportError(context, "Division by 0");
      return kTfLiteError;
    }
  }
  if (requires_broadcast) {
    reference_ops::BroadcastBinaryFunction4DSlow<T, T, T>(
        GetTensorShape(input1), GetTensorData<T>(input1),
        GetTensorShape(input2), denominator_data, GetTensorShape(output),
        GetTensorData<T>(output), FloorDiv<T>);
  } else {
    reference_ops::BinaryFunction<T, T, T>(
        GetTensorShape(input1), GetTensorData<T>(input1),
        GetTensorShape(input2), GetTensorData<T>(input2),
        GetTensorShape(output), GetTensorData<T>(output), FloorDiv<T>);
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (input1->type) {
    case kTfLiteInt32: {
      return EvalImpl<int32_t>(context, data->requires_broadcast, input1,
                               input2, output);
    }
    case kTfLiteFloat32: {
      return EvalImpl<float>(context, data->requires_broadcast, input1, input2,
                             output);
    }
    case kTfLiteUInt8: {
      return EvalImpl<uint8_t>(context, data->requires_broadcast, input1,
                               input2, output);
    }
    case kTfLiteInt64: {
      return EvalImpl<int64_t>(context, data->requires_broadcast, input1,
                               input2, output);
    }
    default: {
      context->ReportError(context, "Type '%s' is not supported by floor_div.",
                           TfLiteTypeGetName(input1->type));
      return kTfLiteError;
    }
  }
}

}  // namespace
}  // namespace floor_div

TfLiteRegistration* Register_FLOOR_DIV() {
  // Init, Free, Prepare, Eval are satisfying the Interface required by
  // TfLiteRegistration.
  static TfLiteRegistration r = {floor_div::Init, floor_div::Free,
                                 floor_div::Prepare, floor_div::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
