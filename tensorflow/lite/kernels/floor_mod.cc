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
#include <functional>
#include <type_traits>
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

// TODO(b/117523611): We should factor out a binary_op and put binary ops there.
namespace tflite {
namespace ops {
namespace builtin {
namespace floor_mod {
namespace {

// Input/output tensor index.
constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

// Op data for floor_mod op.
struct OpData {
  bool requires_broadcast;
};

// TODO(b/117912880): Support quantization.

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

  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input2->type);

  const TfLiteType type = input1->type;
  if (type != kTfLiteInt32 && type != kTfLiteFloat32 && type != kTfLiteInt64) {
    context->ReportError(context, "Type '%s' is not supported by floor_mod.",
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

  if (input2->type == kTfLiteInt32 || input2->type == kTfLiteInt64) {
    // Validate the denominator only for integer.
    const int num_elements = NumElements(input2);
    for (int i = 0; i < num_elements; ++i) {
      if (denominator_data[i] == 0) {
        context->ReportError(context, "Division by 0");
        return kTfLiteError;
      }
    }
  }
  if (requires_broadcast) {
    reference_ops::BroadcastBinaryFunction4DSlow<T, T, T>(
        GetTensorShape(input1), GetTensorData<T>(input1),
        GetTensorShape(input2), denominator_data, GetTensorShape(output),
        GetTensorData<T>(output), reference_ops::FloorMod<T>);
  } else {
    reference_ops::BinaryFunction<T, T, T>(
        GetTensorShape(input1), GetTensorData<T>(input1),
        GetTensorShape(input2), GetTensorData<T>(input2),
        GetTensorShape(output), GetTensorData<T>(output),
        reference_ops::FloorMod<T>);
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
    case kTfLiteInt64: {
      return EvalImpl<int64_t>(context, data->requires_broadcast, input1,
                               input2, output);
    }
    case kTfLiteFloat32: {
      return EvalImpl<float>(context, data->requires_broadcast, input1, input2,
                             output);
    }
    default: {
      context->ReportError(context, "Type '%s' is not supported by floor_mod.",
                           TfLiteTypeGetName(input1->type));
      return kTfLiteError;
    }
  }
}

}  // namespace
}  // namespace floor_mod

TfLiteRegistration* Register_FLOOR_MOD() {
  // Init, Free, Prepare, Eval are satisfying the Interface required by
  // TfLiteRegistration.
  static TfLiteRegistration r = {floor_mod::Init, floor_mod::Free,
                                 floor_mod::Prepare, floor_mod::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
