/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace unsorted_segment_prod {

static const int kInputDataTensor = 0;
static const int kInputSegmentIdsTensor = 1;
static const int kOutputTensor = 0;

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const TfLiteTensor* data,
                                const int num_segments, TfLiteTensor* output) {
  const int data_rank = NumDimensions(data);
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(NumDimensions(data));
  output_shape->data[0] = num_segments;
  for (int i = 1; i < data_rank; ++i) {
    output_shape->data[i] = data->dims->data[i];
  }
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* data;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputDataTensor, &data));
  const TfLiteTensor* segment_ids;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputSegmentIdsTensor,
                                          &segment_ids));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE(context,
                 data->type == kTfLiteInt32 || data->type == kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, segment_ids->type, kTfLiteInt32);

  if (!IsConstantTensor(data) || !IsConstantTensor(segment_ids)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }

  const auto no_segments =
      reinterpret_cast<const TfLiteUnsortedSegmentProdParams*>(
          node->builtin_data)
          ->num_segments;
  return ResizeOutputTensor(context, data, no_segments, output);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* params = reinterpret_cast<const TfLiteUnsortedSegmentProdParams*>(
      node->builtin_data);
  const TfLiteTensor* data;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputDataTensor, &data));
  const TfLiteTensor* segment_ids;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputSegmentIdsTensor,
                                          &segment_ids));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(
        context,
        ResizeOutputTensor(context, data, params->num_segments, output));
  }
  TF_LITE_ENSURE_EQ(context, GetTensorShape(data).Dims(0),
                    GetTensorShape(segment_ids).Dims(0));

#define TF_LITE_UNSORTED_SEGMENT_PROD(dtype)                            \
  reference_ops::UnsortedSegmentProd<dtype>(                            \
      GetTensorShape(data), GetTensorData<dtype>(data),                 \
      GetTensorShape(segment_ids), GetTensorData<int32_t>(segment_ids), \
      params->num_segments, GetTensorShape(output),                     \
      GetTensorData<dtype>(output));
  switch (data->type) {
    case kTfLiteInt32:
      TF_LITE_UNSORTED_SEGMENT_PROD(int32_t);
      break;
    case kTfLiteFloat32:
      TF_LITE_UNSORTED_SEGMENT_PROD(float);
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context, "Currently UnsortedSegmentProd doesn't support type: %s",
          TfLiteTypeGetName(data->type));
      return kTfLiteError;
  }
#undef TF_LITE_UNSORTED_SEGMENT_PROD
  return kTfLiteOk;
}

}  // namespace unsorted_segment_prod

TfLiteRegistration* Register_UNSORTED_SEGMENT_PROD() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 unsorted_segment_prod::Prepare,
                                 unsorted_segment_prod::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
