/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace segment_sum {

static const int kInputDataTensor = 0;
static const int kInputSegmentIdsTensor = 1;
static const int kOutputTensor = 0;

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const TfLiteTensor* data,
                                const TfLiteTensor* segment_ids,
                                TfLiteTensor* output) {
  int max_index = -1;
  const int segment_id_size = segment_ids->dims->data[0];
  if (segment_id_size > 0) {
    max_index = segment_ids->data.i32[segment_id_size - 1];
  }
  const int data_rank = NumDimensions(data);
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(NumDimensions(data));
  output_shape->data[0] = max_index + 1;
  for (int i = 1; i < data_rank; ++i) {
    output_shape->data[i] = data->dims->data[i];
  }
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* data = GetInput(context, node, kInputDataTensor);
  const TfLiteTensor* segment_ids =
      GetInput(context, node, kInputSegmentIdsTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE(context,
                 data->type == kTfLiteInt32 || data->type == kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, segment_ids->type, kTfLiteInt32);

  if (!IsConstantTensor(data) || !IsConstantTensor(segment_ids)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }

  return ResizeOutputTensor(context, data, segment_ids, output);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* data = GetInput(context, node, kInputDataTensor);
  const TfLiteTensor* segment_ids =
      GetInput(context, node, kInputSegmentIdsTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeOutputTensor(context, data, segment_ids, output));
  }

#define TF_LITE_SEGMENT_SUM(dtype)                                      \
  reference_ops::SegmentSum<dtype>(                                     \
      GetTensorShape(data), GetTensorData<dtype>(data),                 \
      GetTensorShape(segment_ids), GetTensorData<int32_t>(segment_ids), \
      GetTensorShape(output), GetTensorData<dtype>(output));
  switch (data->type) {
    case kTfLiteInt32:
      TF_LITE_SEGMENT_SUM(int32_t);
      break;
    case kTfLiteFloat32:
      TF_LITE_SEGMENT_SUM(float);
      break;
    default:
      context->ReportError(context,
                           "Currently SegmentSum doesn't support type: %s",
                           TfLiteTypeGetName(data->type));
      return kTfLiteError;
  }
#undef TF_LITE_SEGMENT_SUM
  return kTfLiteOk;
}

}  // namespace segment_sum

TfLiteRegistration* Register_SEGMENT_SUM() {
  static TfLiteRegistration r = {nullptr, nullptr, segment_sum::Prepare,
                                 segment_sum::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
