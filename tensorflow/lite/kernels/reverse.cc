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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace reverse {
namespace {

constexpr int kInputTensor = 0;
constexpr int kAxisTensor = 1;
constexpr int kOutputTensor = 0;

// NOTE: Currently Tensorflow limits input of Reverse to be max 8-D, if changes
// then please update accordingly here
constexpr int kMaxInputDim = 8;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* axis = GetInput(context, node, kAxisTensor);
  TF_LITE_ENSURE(context, NumDimensions(input) >= NumElements(axis));
  TF_LITE_ENSURE(context, NumDimensions(input) <= kMaxInputDim);

  if (input->type != kTfLiteInt32 && input->type != kTfLiteFloat32 &&
      input->type != kTfLiteUInt8 && input->type != kTfLiteInt16 &&
      input->type != kTfLiteInt64) {
    context->ReportError(context, "Type '%s' is not supported by reverse.",
                         TfLiteTypeGetName(input->type));
    return kTfLiteError;
  }

  if (axis->type != kTfLiteInt32) {
    context->ReportError(context, "Axis Type '%s' is not supported by reverse.",
                         TfLiteTypeGetName(axis->type));
    return kTfLiteError;
  }

  // Allocate temporary tensor which is required in kernel operation
  int temp_tensor_id = 0;
  context->AddTensors(context, 1, &temp_tensor_id);

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(1);
  node->temporaries->data[0] = temp_tensor_id;

  TfLiteTensor* input_copy = GetTemporary(context, node, /*index=*/0);
  input_copy->type = input->type;
  input_copy->allocation_type = kTfLiteArenaRw;

  TfLiteIntArray* input_copy_size = TfLiteIntArrayCopy(input->dims);
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, input_copy, input_copy_size));

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  TF_LITE_ENSURE_EQ(context, output->type, input->type);

  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* axis_tensor = GetInput(context, node, kAxisTensor);
  const int dimension_size = NumDimensions(input);
  int axes_size = NumElements(axis_tensor);
  bool axes[kMaxInputDim] = {false};
  const int32_t* axis_data = GetTensorData<int32_t>(axis_tensor);
  int axis = 0;
  for (int index = 0; index < axes_size; ++index) {
    axis = axis_data[index];
    if (axis < 0) {
      axis += dimension_size;
    }
    TF_LITE_ENSURE(context, 0 <= axis && axis < dimension_size);
    // To ensure no repeating axis
    TF_LITE_ENSURE(context, axes[axis] == false);
    axes[axis] = true;
  }

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TfLiteTensor* input_copy = GetTemporary(context, node, /*index=*/0);

  switch (output->type) {
    case kTfLiteFloat32: {
      reference_ops::Reverse<float>(
          axes, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(output), GetTensorData<float>(output),
          GetTensorData<float>(input_copy));
      break;
    }
    case kTfLiteUInt8: {
      reference_ops::Reverse<uint8_t>(
          axes, GetTensorShape(input), GetTensorData<uint8_t>(input),
          GetTensorShape(output), GetTensorData<uint8_t>(output),
          GetTensorData<uint8_t>(input_copy));
      break;
    }
    case kTfLiteInt16: {
      reference_ops::Reverse<int16_t>(
          axes, GetTensorShape(input), GetTensorData<int16_t>(input),
          GetTensorShape(output), GetTensorData<int16_t>(output),
          GetTensorData<int16_t>(input_copy));
      break;
    }
    case kTfLiteInt32: {
      reference_ops::Reverse<int32_t>(
          axes, GetTensorShape(input), GetTensorData<int32_t>(input),
          GetTensorShape(output), GetTensorData<int32_t>(output),
          GetTensorData<int32_t>(input_copy));
      break;
    }
    case kTfLiteInt64: {
      reference_ops::Reverse<int64_t>(
          axes, GetTensorShape(input), GetTensorData<int64_t>(input),
          GetTensorShape(output), GetTensorData<int64_t>(output),
          GetTensorData<int64_t>(input_copy));
      break;
    }
    default: {
      context->ReportError(context, "Type '%s' is not supported by reverse.",
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
