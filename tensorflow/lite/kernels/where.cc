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
#include <stdint.h>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace where {

constexpr int kInputConditionTensor = 0;
constexpr int kOutputTensor = 0;

template <typename T>
TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const TfLiteTensor* cond_tensor,
                                TfLiteTensor* output_tensor) {
  // Output tensor should have shape:
  // (num_true, cond_rank), where num_true denotes the number of true values
  // in condition.
  const RuntimeShape& cond_shape = GetTensorShape(cond_tensor);
  const int size = cond_shape.FlatSize();
  const int cond_rank = cond_shape.DimensionsCount();
  const T* cond_data = GetTensorData<T>(cond_tensor);

  int true_count = 0;
  for (int i = 0; i < size; ++i) {
    if (cond_data[i] != T(0)) {
      true_count++;
    }
  }
  TfLiteIntArray* output_dims = TfLiteIntArrayCreate(2);
  output_dims->data[0] = true_count;
  output_dims->data[1] = cond_rank;
  return context->ResizeTensor(context, output_tensor, output_dims);
}

template <typename T>
TfLiteStatus PrepareOutput(TfLiteContext* context,
                           const TfLiteTensor* cond_tensor,
                           TfLiteTensor* output) {
  // As output will be a 2D tensor of indices, use int64 to be consistent with
  // tensorflow.
  output->type = kTfLiteInt64;

  // Exit early if cond is a non-const tensor. Set output tensor to dynamic so
  // output size can be determined in Eval.
  if (!IsConstantOrPersistentTensor(cond_tensor)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }
  return ResizeOutputTensor<T>(context, cond_tensor, output);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* cond_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputConditionTensor,
                                          &cond_tensor));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  switch (cond_tensor->type) {
    case kTfLiteBool:
      return PrepareOutput<bool>(context, cond_tensor, output);
    case kTfLiteFloat32:
      return PrepareOutput<float>(context, cond_tensor, output);
    case kTfLiteInt64:
      return PrepareOutput<int64_t>(context, cond_tensor, output);
    case kTfLiteInt32:
      return PrepareOutput<int32_t>(context, cond_tensor, output);
    case kTfLiteInt8:
      return PrepareOutput<int8_t>(context, cond_tensor, output);
    case kTfLiteUInt8:
      return PrepareOutput<uint8_t>(context, cond_tensor, output);
    case kTfLiteUInt32:
      return PrepareOutput<uint32_t>(context, cond_tensor, output);
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Condition tensor has unsupported type: '%s'.",
                         TfLiteTypeGetName(cond_tensor->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* cond_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputConditionTensor,
                                          &cond_tensor));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  if (IsDynamicTensor(output)) {
    switch (cond_tensor->type) {
      case kTfLiteBool:
        TF_LITE_ENSURE_OK(
            context, ResizeOutputTensor<bool>(context, cond_tensor, output));
        break;
      case kTfLiteFloat32:
        TF_LITE_ENSURE_OK(
            context, ResizeOutputTensor<float>(context, cond_tensor, output));
        break;
      case kTfLiteInt64:
        TF_LITE_ENSURE_OK(
            context, ResizeOutputTensor<int64_t>(context, cond_tensor, output));
        break;
      case kTfLiteInt32:
        TF_LITE_ENSURE_OK(
            context, ResizeOutputTensor<int32_t>(context, cond_tensor, output));
        break;
      case kTfLiteInt8:
        TF_LITE_ENSURE_OK(
            context, ResizeOutputTensor<int8_t>(context, cond_tensor, output));
        break;
      case kTfLiteUInt8:
        TF_LITE_ENSURE_OK(
            context, ResizeOutputTensor<uint8_t>(context, cond_tensor, output));
        break;
      case kTfLiteUInt32:
        TF_LITE_ENSURE_OK(context, ResizeOutputTensor<uint32_t>(
                                       context, cond_tensor, output));
        break;
      default:
        TF_LITE_KERNEL_LOG(context,
                           "Condition tensor has unsupported type: '%s'.",
                           TfLiteTypeGetName(cond_tensor->type));
        return kTfLiteError;
    }
  }

  TfLiteIntArray* dims = cond_tensor->dims;
  if (dims->size == 0) {
    // Scalar tensors are not supported.
    TF_LITE_KERNEL_LOG(context, "Where op requires condition w/ rank > 0");
    return kTfLiteError;
  }

  switch (cond_tensor->type) {
    case kTfLiteBool:
      reference_ops::SelectTrueCoords(GetTensorShape(cond_tensor),
                                      GetTensorData<bool>(cond_tensor),
                                      GetTensorData<int64_t>(output));
      break;
    case kTfLiteFloat32:
      reference_ops::SelectTrueCoords(GetTensorShape(cond_tensor),
                                      GetTensorData<float>(cond_tensor),
                                      GetTensorData<int64_t>(output));
      break;
    case kTfLiteInt64:
      reference_ops::SelectTrueCoords(GetTensorShape(cond_tensor),
                                      GetTensorData<int64_t>(cond_tensor),
                                      GetTensorData<int64_t>(output));
      break;
    case kTfLiteInt32:
      reference_ops::SelectTrueCoords(GetTensorShape(cond_tensor),
                                      GetTensorData<int32_t>(cond_tensor),
                                      GetTensorData<int64_t>(output));
      break;
    case kTfLiteInt8:
      reference_ops::SelectTrueCoords(GetTensorShape(cond_tensor),
                                      GetTensorData<int8_t>(cond_tensor),
                                      GetTensorData<int64_t>(output));
      break;
    case kTfLiteUInt8:
      reference_ops::SelectTrueCoords(GetTensorShape(cond_tensor),
                                      GetTensorData<uint8_t>(cond_tensor),
                                      GetTensorData<int64_t>(output));
      break;
    case kTfLiteUInt32:
      reference_ops::SelectTrueCoords(GetTensorShape(cond_tensor),
                                      GetTensorData<uint32_t>(cond_tensor),
                                      GetTensorData<int64_t>(output));
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Condition tensor has unsupported type: '%s'.",
                         TfLiteTypeGetName(cond_tensor->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}
}  // namespace where

TfLiteRegistration* Register_WHERE() {
  static TfLiteRegistration r = {/*init*/ nullptr, /*free*/ nullptr,
                                 where::Prepare, where::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
