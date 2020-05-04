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

namespace tflite {
namespace ops {
namespace micro {
namespace resize_nearest_neighbor {

constexpr int kInputTensor = 0;
constexpr int kSizeTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
#if defined(DEBUG)
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
    TF_LITE_KERNEL_LOG(context,
                         "Dynamic tensors are unsupported in tfmicro.");
    return kTfLiteError;
  }
#endif
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteResizeNearestNeighborParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* size = GetInput(context, node, kSizeTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  tflite::ResizeNearestNeighborParams op_params;
  op_params.align_corners = params->align_corners;
  op_params.half_pixel_centers = false;

  if (output->type == kTfLiteFloat32) {
    reference_ops::ResizeNearestNeighbor(
        op_params, GetTensorShape(input), GetTensorData<int32>(input),
        GetTensorShape(size), GetTensorData<int32>(size),
        GetTensorShape(output), GetTensorData<int32>(output));
  } else if (output->type == kTfLiteUInt8) {
    reference_ops::ResizeNearestNeighbor(
        op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
        GetTensorShape(size), GetTensorData<int32>(size),
        GetTensorShape(output), GetTensorData<uint8_t>(output));
  } else if (output->type == kTfLiteInt8) {
    reference_ops::ResizeNearestNeighbor(
        op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
        GetTensorShape(size), GetTensorData<int32>(size),
        GetTensorShape(output), GetTensorData<int8_t>(output));
  } else {
    TF_LITE_KERNEL_LOG(context,
                       "Output type is %d, requires float, uint8 or int8.",
                       output->type);
    return kTfLiteError;
  }

  return kTfLiteOk;
}
}  // namespace resize_nearest_neighbor

TfLiteRegistration* Register_RESIZE_NEAREST_NEIGHBOR() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 /*prepare=*/resize_nearest_neighbor::Prepare,
                                 /*invoke=*/resize_nearest_neighbor::Eval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
