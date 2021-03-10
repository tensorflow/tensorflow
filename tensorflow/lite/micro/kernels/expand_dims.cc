/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kAxisTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus ExpandTensorDim(TfLiteContext* context,
                             const TfLiteEvalTensor* input, int32_t axis,
                             TfLiteEvalTensor* output) {
  const TfLiteIntArray* input_dims = input->dims;
  TfLiteIntArray* output_dims = output->dims;
  if (axis < 0) {
    axis = input_dims->size + 1 + axis;
  }
  TF_LITE_ENSURE(context, (axis <= input_dims->size));

  output_dims->size = input_dims->size + 1;
  for (int i = 0; i < output_dims->size; ++i) {
    if (i < axis) {
      output_dims->data[i] = input_dims->data[i];
    } else if (i == axis) {
      output_dims->data[i] = 1;
    } else {
      output_dims->data[i] = input_dims->data[i - 1];
    }
  }
  return kTfLiteOk;
}

TfLiteStatus GetAxisValueFromTensor(TfLiteContext* context,
                                    const TfLiteEvalTensor* axis,
                                    int32_t* axis_value) {
  const int axis_dims = (tflite::micro::GetTensorShape(axis)).DimensionsCount();
  if (axis_dims > 1) {
    TF_LITE_KERNEL_LOG(context, "Axis has only one element for Expand_Dims.",
                       axis_dims);
    return kTfLiteError;
  }

  if (kTfLiteInt32 == (axis->type)) {
    const int32_t* axis_ptr = tflite::micro::GetTensorData<int32_t>(axis);
    *axis_value = axis_ptr[0];
    return kTfLiteOk;
  } else {
    TF_LITE_KERNEL_LOG(context,
                       "Axis type %s (%d) not supported by Expand_Dims.",
                       TfLiteTypeGetName(axis->type), axis->type);
    return kTfLiteError;
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* axis;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kAxisTensor, &axis));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  output->type = input->type;
  if (IsDynamicTensor(axis)) {
    TF_LITE_KERNEL_LOG(context,
                       "DynamicTensor is not yet supported by Expand_Dims.");
    return kTfLiteError;
  }
  return kTfLiteOk;
}

template <typename T>
void memCopyN(T* out, const T* in, const int num_elements) {
  for (int i = 0; i < num_elements; ++i) {
    out[i] = in[i];
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* axis =
      tflite::micro::GetEvalInput(context, node, kAxisTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  const int flat_size = ElementCount(*input->dims);
  const int input_dims = input->dims->size;

  int32_t axis_value;
  TF_LITE_ENSURE_OK(context,
                    GetAxisValueFromTensor(context, axis, &axis_value));
  if ((axis_value > static_cast<int32_t>(input_dims)) ||
      (axis_value < static_cast<int32_t>(-(input_dims + 1)))) {
    TF_LITE_KERNEL_LOG(context, "Invalid Expand_Dims axis value (%d).",
                       axis_value);
    return kTfLiteError;
  }
  ExpandTensorDim(context, input, axis_value, output);

  switch (input->type) {
    case kTfLiteFloat32: {
      memCopyN(tflite::micro::GetTensorData<float>(output),
               tflite::micro::GetTensorData<float>(input), flat_size);
    } break;
    case kTfLiteInt8: {
      memCopyN(tflite::micro::GetTensorData<int8_t>(output),
               tflite::micro::GetTensorData<int8_t>(input), flat_size);
    } break;
    default:
      TF_LITE_KERNEL_LOG(
          context,
          "Expand_Dims only currently supports int8 and float32, got %d.",
          input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}
}  // namespace

TfLiteRegistration Register_EXPAND_DIMS() {
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
