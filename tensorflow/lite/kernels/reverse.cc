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

#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace reverse {
namespace {

constexpr int kInputTensor = 0;
constexpr int kAxisTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* axis;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kAxisTensor, &axis));
  TF_LITE_ENSURE_EQ(context, NumDimensions(axis), 1);
  TF_LITE_ENSURE(context, NumDimensions(input) >= NumElements(axis));

  if (input->type != kTfLiteInt32 && input->type != kTfLiteFloat32 &&
      input->type != kTfLiteUInt8 && input->type != kTfLiteInt8 &&
      input->type != kTfLiteInt16 && input->type != kTfLiteInt64 &&
      input->type != kTfLiteBool) {
    TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by reverse.",
                       TfLiteTypeGetName(input->type));
    return kTfLiteError;
  }

  if (axis->type != kTfLiteInt32) {
    TF_LITE_KERNEL_LOG(context, "Axis Type '%s' is not supported by reverse.",
                       TfLiteTypeGetName(axis->type));
    return kTfLiteError;
  }

  // TODO(b/186320180): support multi-axis case.
  if (NumElements(axis) > 1) {
    TF_LITE_KERNEL_LOG(context, "Current does not support more than 1 axis.");
  }

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input->type);

  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* axis_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kAxisTensor, &axis_tensor));
  int axis = GetTensorData<int32_t>(axis_tensor)[0];
  const int rank = NumDimensions(input);
  if (axis < 0) {
    axis += rank;
  }

  TF_LITE_ENSURE(context, axis >= 0 && axis < rank);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  switch (output->type) {
    case kTfLiteFloat32: {
      reference_ops::Reverse<float>(
          axis, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(output), GetTensorData<float>(output));
      break;
    }
    case kTfLiteUInt8:
    case kTfLiteInt8: {
      reference_ops::Reverse<uint8_t>(
          axis, GetTensorShape(input), GetTensorData<uint8_t>(input),
          GetTensorShape(output), GetTensorData<uint8_t>(output));
      break;
    }
    case kTfLiteInt16: {
      reference_ops::Reverse<int16_t>(
          axis, GetTensorShape(input), GetTensorData<int16_t>(input),
          GetTensorShape(output), GetTensorData<int16_t>(output));
      break;
    }
    case kTfLiteInt32: {
      reference_ops::Reverse<int32_t>(
          axis, GetTensorShape(input), GetTensorData<int32_t>(input),
          GetTensorShape(output), GetTensorData<int32_t>(output));
      break;
    }
    case kTfLiteInt64: {
      reference_ops::Reverse<int64_t>(
          axis, GetTensorShape(input), GetTensorData<int64_t>(input),
          GetTensorShape(output), GetTensorData<int64_t>(output));
      break;
    }
    case kTfLiteBool: {
      reference_ops::Reverse<bool>(
          axis, GetTensorShape(input), GetTensorData<bool>(input),
          GetTensorShape(output), GetTensorData<bool>(output));
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by reverse.",
                         TfLiteTypeGetName(output->type));
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

}  // namespace
}  // namespace reverse

TfLiteRegistration* Register_REVERSE_V2() {
  static TfLiteRegistration r = {nullptr, nullptr, reverse::Prepare,
                                 reverse::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
