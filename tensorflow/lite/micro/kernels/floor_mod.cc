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

#include "tensorflow/lite/kernels/internal/reference/floor_mod.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/binary_function.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

// OLD-TODO(b/117523611): We should factor out a binary_op and put binary ops
// there.
namespace tflite {
namespace ops {
namespace micro {
namespace floor_mod {
namespace {

// Input/output tensor index.
constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

// OLD-TODO(b/117912880): Support quantization.

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input2->type);

  const TfLiteType type = input1->type;
  if (type != kTfLiteInt32 && type != kTfLiteFloat32 && type != kTfLiteInt64) {
    TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by floor_mod.",
                       TfLiteTypeGetName(type));
    return kTfLiteError;
  }
  output->type = type;

  return kTfLiteError;
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
        TF_LITE_KERNEL_LOG(context, "Division by 0");
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
  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  bool requires_broadcast = false;

  switch (input1->type) {
    case kTfLiteInt32: {
      return EvalImpl<int32_t>(context, requires_broadcast, input1, input2,
                               output);
    }
    case kTfLiteInt64: {
      return EvalImpl<int64_t>(context, requires_broadcast, input1, input2,
                               output);
    }
    case kTfLiteFloat32: {
      return EvalImpl<float>(context, requires_broadcast, input1, input2,
                             output);
    }
    default: {
      TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by floor_mod.",
                         TfLiteTypeGetName(input1->type));
      return kTfLiteError;
    }
  }
}

}  // namespace
}  // namespace floor_mod

TfLiteRegistration* Register_FLOOR_MOD() { return nullptr; }

}  // namespace micro
}  // namespace ops
}  // namespace tflite
