/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/resize_nearest_neighbor.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace resize_nearest_neighbor {

constexpr int kInputTensor = 0;
constexpr int kSizeTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* size = GetInput(context, node, kSizeTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // Our current implementations rely on the input being 4D,
  // and the size being 1D tensor with exactly 2 elements.
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(size), 1);
  TF_LITE_ENSURE_EQ(context, size->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, size->dims->data[0], 2);

  output->type = input->type;

  if (!IsConstantTensor(size)) {
    TF_LITE_KERNEL_LOG(context, "Dynamic tensors are unsupported in tfmicro.");
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteResizeNearestNeighborParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* size =
      tflite::micro::GetEvalInput(context, node, kSizeTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  tflite::ResizeNearestNeighborParams op_params;
  op_params.align_corners = params->align_corners;
  op_params.half_pixel_centers = false;

  if (output->type == kTfLiteFloat32) {
    reference_ops::ResizeNearestNeighbor(
        op_params, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int32_t>(input),
        tflite::micro::GetTensorShape(size),
        tflite::micro::GetTensorData<int32_t>(size),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int32_t>(output));
  } else if (output->type == kTfLiteUInt8) {
    reference_ops::ResizeNearestNeighbor(
        op_params, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<uint8_t>(input),
        tflite::micro::GetTensorShape(size),
        tflite::micro::GetTensorData<int32_t>(size),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<uint8_t>(output));
  } else if (output->type == kTfLiteInt8) {
    reference_ops::ResizeNearestNeighbor(
        op_params, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(size),
        tflite::micro::GetTensorData<int32_t>(size),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
  } else {
    TF_LITE_KERNEL_LOG(context,
                       "Output type is %d, requires float, uint8_t or int8_t.",
                       output->type);
    return kTfLiteError;
  }

  return kTfLiteOk;
}
}  // namespace resize_nearest_neighbor

TfLiteRegistration Register_RESIZE_NEAREST_NEIGHBOR() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/resize_nearest_neighbor::Prepare,
          /*invoke=*/resize_nearest_neighbor::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
