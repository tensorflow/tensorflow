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

#include <stdint.h>

#include <memory>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace segment_sum {

static const int kInputDataTensor = 0;
static const int kInputSegmentIdsTensor = 1;
static const int kOutputTensor = 0;

TfLiteStatus CheckedDimensionProduct(TfLiteContext* context,
                                     const TfLiteTensor* tensor, int begin,
                                     int end, int* product) {
  TF_LITE_ENSURE(context, tensor != nullptr);
  TF_LITE_ENSURE(context, product != nullptr);
  TF_LITE_ENSURE(context, begin >= 0);
  TF_LITE_ENSURE(context, end >= begin);
  TF_LITE_ENSURE(context, end <= NumDimensions(tensor));

  CheckedInt<int> checked_product = 1;
  for (int i = begin; i < end; ++i) {
    const int dim = SizeOfDimension(tensor, i);
    TF_LITE_ENSURE(context, dim >= 0);
    checked_product *= dim;
    TF_LITE_ENSURE_MSG(context, !checked_product.Overflow(),
                       "Dimension product overflows int.");
  }
  *product = checked_product.Value();
  return kTfLiteOk;
}

TfLiteStatus ValidateSegmentIds(TfLiteContext* context,
                                const TfLiteTensor* data,
                                const TfLiteTensor* segment_ids,
                                int* max_segment_id) {
  TF_LITE_ENSURE(context, max_segment_id != nullptr);
  TF_LITE_ENSURE(context, NumDimensions(data) >= 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(segment_ids), 1);
  int segment_id_size = 0;
  TF_LITE_ENSURE_MSG(
      context, CheckedNumElements(segment_ids, segment_id_size) == kTfLiteOk,
      "segment_ids element count overflows int.");
  TF_LITE_ENSURE_EQ(context, segment_id_size, data->dims->data[0]);

  int previous_segment_id = -1;
  for (int i = 0; i < segment_id_size; i++) {
    const int current_segment_id = GetTensorData<int32_t>(segment_ids)[i];
    if (i == 0) {
      TF_LITE_ENSURE_EQ(context, current_segment_id, 0);
    } else {
      const int64_t delta = static_cast<int64_t>(current_segment_id) -
                            static_cast<int64_t>(previous_segment_id);
      TF_LITE_ENSURE(context, delta == 0 || delta == 1);
    }
    previous_segment_id = current_segment_id;
  }

  *max_segment_id = previous_segment_id;
  return kTfLiteOk;
}

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const TfLiteTensor* data,
                                const TfLiteTensor* segment_ids,
                                TfLiteTensor* output) {
  // Segment ids should be of same cardinality as first input dimension and they
  // should be increasing by at most 1, from 0 (e.g., [0, 0, 1, 2, 3] is valid)
  int max_index = -1;
  TF_LITE_ENSURE_OK(context,
                    ValidateSegmentIds(context, data, segment_ids, &max_index));
  const CheckedInt<int> output_first_dim = CheckedInt<int>(max_index) + 1;
  TF_LITE_ENSURE_MSG(context, !output_first_dim.Overflow(),
                     "Output dimension overflows int.");

  const int data_rank = NumDimensions(data);
  std::unique_ptr<TfLiteIntArray, decltype(&TfLiteIntArrayFree)> output_shape(
      TfLiteIntArrayCreate(NumDimensions(data)), TfLiteIntArrayFree);
  output_shape->data[0] = output_first_dim.Value();
  for (int i = 1; i < data_rank; ++i) {
    output_shape->data[i] = data->dims->data[i];
  }
  int output_elements = 0;
  TF_LITE_ENSURE_MSG(
      context,
      CheckedNumElements(output_shape.get(), output_elements) == kTfLiteOk,
      "Output element count overflows int.");
  return context->ResizeTensor(context, output, output_shape.release());
}

TfLiteStatus ValidateSegmentSumEval(TfLiteContext* context,
                                    const TfLiteTensor* data,
                                    const TfLiteTensor* segment_ids,
                                    const TfLiteTensor* output) {
  int max_segment_id = -1;
  TF_LITE_ENSURE_OK(
      context, ValidateSegmentIds(context, data, segment_ids, &max_segment_id));
  TF_LITE_ENSURE(context, output->dims->data[0] > max_segment_id);

  int segment_flat_size = 0;
  TF_LITE_ENSURE_OK(context, CheckedDimensionProduct(context, output, 1,
                                                     NumDimensions(output),
                                                     &segment_flat_size));
  int output_elements = 0;
  TF_LITE_ENSURE_MSG(context,
                     CheckedNumElements(output, output_elements) == kTfLiteOk,
                     "Output element count overflows int.");
  return kTfLiteOk;
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

  if (!IsConstantOrPersistentTensor(data) ||
      !IsConstantOrPersistentTensor(segment_ids)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }

  return ResizeOutputTensor(context, data, segment_ids, output);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
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
    TF_LITE_ENSURE_OK(context,
                      ResizeOutputTensor(context, data, segment_ids, output));
  }
  TF_LITE_ENSURE_OK(context,
                    ValidateSegmentSumEval(context, data, segment_ids, output));

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
      TF_LITE_KERNEL_LOG(context,
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
