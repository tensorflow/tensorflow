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
namespace pow {
namespace {

// Input/output tensor index.
constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

// Op data for pow op.
struct OpData {
  bool requires_broadcast;
};

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

  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, input1->type, input2->type);

  const TfLiteType type = input1->type;
  if (type != kTfLiteInt32 && type != kTfLiteFloat32) {
    context->ReportError(context, "Unsupported data type %d.", type);
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
void PowImpl(const TfLiteTensor* input1, const TfLiteTensor* input2,
             TfLiteTensor* output, bool requires_broadcast) {
  if (requires_broadcast) {
    reference_ops::BroadcastPow4DSlow(
        GetTensorShape(input1), GetTensorData<T>(input1),
        GetTensorShape(input2), GetTensorData<T>(input2),
        GetTensorShape(output), GetTensorData<T>(output));
  } else {
    reference_ops::Pow(GetTensorShape(input1), GetTensorData<T>(input1),
                       GetTensorShape(input2), GetTensorData<T>(input2),
                       GetTensorShape(output), GetTensorData<T>(output));
  }
}

TfLiteStatus CheckValue(TfLiteContext* context, const TfLiteTensor* input) {
  const int64_t num_elements = NumElements(input);
  const int32_t* data = GetTensorData<int32_t>(input);
  for (int i = 0; i < num_elements; ++i) {
    if (data[i] < 0) {
      context->ReportError(context,
                           "POW does not support negative value for int32.");
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (output->type) {
    case kTfLiteInt32: {
      // TensorFlow does not support negative for int32.
      TF_LITE_ENSURE_OK(context, CheckValue(context, input2));
      PowImpl<int32_t>(input1, input2, output, data->requires_broadcast);
      break;
    }
    case kTfLiteFloat32: {
      PowImpl<float>(input1, input2, output, data->requires_broadcast);
      break;
    }
    default: {
      context->ReportError(context, "Unsupported data type: %d", output->type);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace
}  // namespace pow

TfLiteRegistration* Register_POW() {
  static TfLiteRegistration r = {pow::Init, pow::Free, pow::Prepare, pow::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
