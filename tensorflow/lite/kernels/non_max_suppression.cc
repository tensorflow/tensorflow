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
#include "tensorflow/lite/kernels/internal/reference/non_max_suppression.h"

#include <initializer_list>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace non_max_suppression {

// Boxes in format [y1, x1, y2, x2]. Shape: [num_boxes, 4]
// Type: Float.
constexpr int kInputTensorBoxes = 0;
// Shape: [num_boxes]
// Type: Float.
constexpr int kInputTensorScores = 1;
// Max number of boxes to output. Actual output can be smaller.
// The output tensors (indices/scores) are of this length.
// Type: Int32.
constexpr int kInputTensorMaxOutputSize = 2;
// Type: Float.
constexpr int kInputTensorIouThreshold = 3;
// Type: Float.
constexpr int kInputTensorScoreThreshold = 4;
// Only applies to NON_MAX_SUPPRESSION_V5.
// Type: Float.
constexpr int kInputTensorSigma = 5;

// Indices of selected boxes. Shape: [num_selected_indices]
// Type: Int32.
constexpr int kNMSOutputTensorSelectedIndices = 0;
// Type: Int32.
constexpr int kNMSOutputTensorNumSelectedIndices = 1;

// Indices of selected boxes. Shape: [num_selected_indices]
// Type: Int32.
constexpr int kSoftNMSOutputTensorSelectedIndices = 0;
// Scores of selected boxes. Shape: [num_selected_indices]
// Type: Float.
constexpr int kSoftNMSOutputTensorSelectedScores = 1;
// Type: Int32.
constexpr int kSoftNMSOutputTensorNumSelectedIndices = 2;

TfLiteStatus SetTensorSizes(TfLiteContext* context, TfLiteTensor* tensor,
                            std::initializer_list<int> values) {
  TfLiteIntArray* size = TfLiteIntArrayCreate(values.size());
  int index = 0;
  for (const auto& v : values) {
    size->data[index++] = v;
  }
  return context->ResizeTensor(context, tensor, size);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const int num_inputs = NumInputs(node);
  const bool is_soft_nms = num_inputs == 6;
  if (num_inputs != 5 && num_inputs != 6) {
    context->ReportError(context, "Found NMS op with invalid num inputs: %d",
                         NumInputs(node));
    return kTfLiteError;
  }

  // Boxes & Scores.
  const TfLiteTensor* input_boxes;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputTensorBoxes, &input_boxes));
  TF_LITE_ENSURE_EQ(context, input_boxes->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_boxes), 2);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(input_boxes, 1), 4);
  const int num_boxes = SizeOfDimension(input_boxes, 0);
  const TfLiteTensor* input_scores;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputTensorScores, &input_scores));
  TF_LITE_ENSURE_EQ(context, input_scores->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_scores), 1);
  TF_LITE_ENSURE_EQ(context, num_boxes, SizeOfDimension(input_scores, 0));

  // Max output size.
  const TfLiteTensor* input_max_output_size;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorMaxOutputSize,
                                 &input_max_output_size));
  TF_LITE_ENSURE_EQ(context, input_max_output_size->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_max_output_size), 0);
  const bool is_max_output_size_const = IsConstantTensor(input_max_output_size);
  int max_output_size_value = 0;
  if (is_max_output_size_const) {
    max_output_size_value = *GetTensorData<int>(input_max_output_size);
    TF_LITE_ENSURE(context, (max_output_size_value >= 0));
  }

  // IoU & Score thresholds.
  const TfLiteTensor* input_iou_threshold;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorIouThreshold,
                                 &input_iou_threshold));
  TF_LITE_ENSURE_EQ(context, input_iou_threshold->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_iou_threshold), 0);
  const TfLiteTensor* input_score_threshold;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorScoreThreshold,
                                 &input_score_threshold));
  TF_LITE_ENSURE_EQ(context, input_iou_threshold->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_score_threshold), 0);

  if (is_soft_nms) {
    const TfLiteTensor* input_sigma;
    TF_LITE_ENSURE_OK(
        context, GetInputSafe(context, node, kInputTensorSigma, &input_sigma));
    TF_LITE_ENSURE_EQ(context, input_sigma->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, NumDimensions(input_sigma), 0);

    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 3);
    TfLiteTensor* output_selected_indices;
    TF_LITE_ENSURE_OK(
        context,
        GetOutputSafe(context, node, kSoftNMSOutputTensorSelectedIndices,
                      &output_selected_indices));
    output_selected_indices->type = kTfLiteInt32;
    TfLiteTensor* output_selected_scores;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node,
                                             kSoftNMSOutputTensorSelectedScores,
                                             &output_selected_scores));
    output_selected_scores->type = kTfLiteFloat32;
    TfLiteTensor* output_num_selected_indices;
    TF_LITE_ENSURE_OK(
        context,
        GetOutputSafe(context, node, kSoftNMSOutputTensorNumSelectedIndices,
                      &output_num_selected_indices));
    output_num_selected_indices->type = kTfLiteInt32;
    SetTensorSizes(context, output_num_selected_indices, {});

    if (is_max_output_size_const) {
      SetTensorSizes(context, output_selected_indices, {max_output_size_value});
      SetTensorSizes(context, output_selected_scores, {max_output_size_value});
    } else {
      SetTensorToDynamic(output_selected_indices);
      SetTensorToDynamic(output_selected_scores);
    }
  } else {
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);
    TfLiteTensor* output_selected_indices;
    TF_LITE_ENSURE_OK(
        context, GetOutputSafe(context, node, kNMSOutputTensorSelectedIndices,
                               &output_selected_indices));
    output_selected_indices->type = kTfLiteInt32;
    TfLiteTensor* output_num_selected_indices;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node,
                                             kNMSOutputTensorNumSelectedIndices,
                                             &output_num_selected_indices));
    output_num_selected_indices->type = kTfLiteInt32;
    SetTensorSizes(context, output_num_selected_indices, {});

    if (is_max_output_size_const) {
      SetTensorSizes(context, output_selected_indices, {max_output_size_value});
    } else {
      SetTensorToDynamic(output_selected_indices);
    }
  }

  return kTfLiteOk;
}

// If num_selected_indices < max_output_size, the output tensor can contain
// garbage values initially present in memory. This causes segfault in
// downstream ops such as GATHER, since one of the outputs denotes indices and
// int garbage values can be pretty large. This method zeroes-out the remaining
// values.
// NOTE: We ensure memory being reset is valid, by setting pertinent output
// tensors to max_output_size length in Prepare.
void ResetUnusedElementsToZeroes(const int max_output_size,
                                 const int num_selected_indices,
                                 int* selected_indices,
                                 float* selected_scores) {
  for (int i = num_selected_indices; i < max_output_size; ++i) {
    selected_indices[i] = 0;
    if (selected_scores) {
      selected_scores[i] = 0.0;
    }
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const bool is_soft_nms = NumInputs(node) == 6;

  const TfLiteTensor* input_boxes;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputTensorBoxes, &input_boxes));
  const int num_boxes = SizeOfDimension(input_boxes, 0);
  const TfLiteTensor* input_scores;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputTensorScores, &input_scores));
  const TfLiteTensor* input_max_output_size;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorMaxOutputSize,
                                 &input_max_output_size));
  const int max_output_size_value = *GetTensorData<int>(input_max_output_size);
  TF_LITE_ENSURE(context, (max_output_size_value >= 0));
  const bool is_max_output_size_const = IsConstantTensor(input_max_output_size);
  const TfLiteTensor* input_iou_threshold;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorIouThreshold,
                                 &input_iou_threshold));
  const float iou_threshold = *GetTensorData<float>(input_iou_threshold);
  const TfLiteTensor* input_score_threshold;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorScoreThreshold,
                                 &input_score_threshold));
  const float score_threshold = *GetTensorData<float>(input_score_threshold);

  TfLiteTensor* output_selected_indices = nullptr;
  TfLiteTensor* output_selected_scores = nullptr;
  TfLiteTensor* output_num_selected_indices = nullptr;

  if (is_soft_nms) {
    const TfLiteTensor* input_sigma;
    TF_LITE_ENSURE_OK(
        context, GetInputSafe(context, node, kInputTensorSigma, &input_sigma));
    const float soft_nms_sigma = *GetTensorData<float>(input_sigma);
    if (soft_nms_sigma < 0) {
      context->ReportError(context, "Invalid sigma value for soft NMS: %f",
                           soft_nms_sigma);
      return kTfLiteError;
    }

    TF_LITE_ENSURE_OK(
        context,
        GetOutputSafe(context, node, kSoftNMSOutputTensorSelectedIndices,
                      &output_selected_indices));
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node,
                                             kSoftNMSOutputTensorSelectedScores,
                                             &output_selected_scores));
    TF_LITE_ENSURE_OK(
        context,
        GetOutputSafe(context, node, kSoftNMSOutputTensorNumSelectedIndices,
                      &output_num_selected_indices));
    if (!is_max_output_size_const) {
      SetTensorSizes(context, output_selected_indices, {max_output_size_value});
      SetTensorSizes(context, output_selected_scores, {max_output_size_value});
    }
    reference_ops::NonMaxSuppression(
        input_boxes->data.f, num_boxes, input_scores->data.f,
        max_output_size_value, iou_threshold, score_threshold, soft_nms_sigma,
        output_selected_indices->data.i32, output_selected_scores->data.f,
        output_num_selected_indices->data.i32);
    ResetUnusedElementsToZeroes(
        max_output_size_value, *output_num_selected_indices->data.i32,
        output_selected_indices->data.i32, output_selected_scores->data.f);
  } else {
    TF_LITE_ENSURE_OK(
        context, GetOutputSafe(context, node, kNMSOutputTensorSelectedIndices,
                               &output_selected_indices));
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node,
                                             kNMSOutputTensorNumSelectedIndices,
                                             &output_num_selected_indices));
    if (!is_max_output_size_const) {
      SetTensorSizes(context, output_selected_indices, {max_output_size_value});
    }
    reference_ops::NonMaxSuppression(
        input_boxes->data.f, num_boxes, input_scores->data.f,
        max_output_size_value, iou_threshold, score_threshold, /**sigma=**/ 0.0,
        output_selected_indices->data.i32, /**selected_scores=**/ nullptr,
        output_num_selected_indices->data.i32);
    ResetUnusedElementsToZeroes(max_output_size_value,
                                *output_num_selected_indices->data.i32,
                                output_selected_indices->data.i32, nullptr);
  }

  return kTfLiteOk;
}
}  // namespace non_max_suppression

TfLiteRegistration* Register_NON_MAX_SUPPRESSION_V4() {
  static TfLiteRegistration r = {nullptr, nullptr, non_max_suppression::Prepare,
                                 non_max_suppression::Eval};
  return &r;
}

TfLiteRegistration* Register_NON_MAX_SUPPRESSION_V5() {
  static TfLiteRegistration r = {nullptr, nullptr, non_max_suppression::Prepare,
                                 non_max_suppression::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
