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
#include <limits>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace non_max_suppression {

// Boxes in format [y1, x1, y2, x2]. Shape: [num_boxes, 4]
// Type: Float, Int8 or Int16.
constexpr int kInputTensorBoxes = 0;
// Shape: [num_boxes]
// Type: Float, Int8 or Int16.
constexpr int kInputTensorScores = 1;
// Max number of boxes to output. Actual output can be smaller.
// The output tensors (indices/scores) are of this length.
// Type: Int32.
constexpr int kInputTensorMaxOutputSize = 2;
// Type: Float, Int8 or Int16.
constexpr int kInputTensorIouThreshold = 3;
// Type: Float, Int8 or Int16.
constexpr int kInputTensorScoreThreshold = 4;
// Only applies to NON_MAX_SUPPRESSION_V5.
// Type: Float, Int8 or Int16.
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
// Type: Float, Int8 or Int16.
constexpr int kSoftNMSOutputTensorSelectedScores = 1;
// Type: Int32.
constexpr int kSoftNMSOutputTensorNumSelectedIndices = 2;

// Restricted inverse scale and zero point values for IOUs and scores.
// Type: Int8
constexpr int32_t kIouAndScoreInverseScaleInt8 = 256;
constexpr int32_t kIouAndScoreZeroPointInt8 = -128;
// Type: Int16
constexpr int32_t kIouAndScoreInverseScaleInt16 = 32768;
constexpr int32_t kIouAndScoreZeroPointInt16 = 0;

struct OpData {
  // INT8 or INT16 LUT populated with quantized exponents of possible similarity
  // scores. LUT size depends on input type.
  void* soft_nms_lut{nullptr};

  int32_t iou_and_score_inverse_scale;
  int32_t iou_and_score_zero_point;

  int32_t scores_rescale_multiplier;
  int32_t scores_rescale_shift;

  int32_t score_threshold_rescale_multiplier;
  int32_t score_threshold_rescale_shift;

  int32_t iou_threshold_rescale_multiplier;
  int32_t iou_threshold_rescale_shift;

  int32_t selected_scores_rescale_multiplier;
  int32_t selected_scores_rescale_shift;
};

TfLiteStatus SetTensorSizes(TfLiteContext* context, TfLiteTensor* tensor,
                            std::initializer_list<int> values) {
  TfLiteIntArray* size = TfLiteIntArrayCreate(values.size());
  int index = 0;
  for (const auto& v : values) {
    size->data[index++] = v;
  }
  return context->ResizeTensor(context, tensor, size);
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData;
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  OpData* op_data = reinterpret_cast<OpData*>(buffer);
  free(op_data->soft_nms_lut);
  delete op_data;
}

bool IsSoftNms(TfLiteNode* node) { return NumInputs(node) == 6; }

template <typename T>
void GenSoftNmsLut(float sigma_scale, T* soft_nms_lut) {
  TFLITE_DCHECK_LT(sigma_scale, 0.f);

  const void* lut_func_params = static_cast<const void*>(&sigma_scale);
  const auto lut_func = [](float iou, const void* lut_func_params) {
    const float sigma_scale = *static_cast<const float*>(lut_func_params);
    return std::exp(sigma_scale * iou * iou);
  };

  // As the IOU is [0; 1] the LUT inputs in the [0; 1] range. The LUT output
  // will also be in the same range as the parameter of std::exp will always be
  // <= 0 due to the scale parameter being < 0.
  const float lut_scale =
      1.0f / (std::numeric_limits<T>::max() - std::numeric_limits<T>::min());
  LUTPopulate<T>(lut_scale, std::numeric_limits<T>::min(), lut_scale,
                 std::numeric_limits<T>::min(), lut_func, lut_func_params,
                 reinterpret_cast<T*>(soft_nms_lut));
}

template <typename T>
TfLiteStatus PrepareQuantized(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = static_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, op_data != nullptr);

  op_data->iou_and_score_inverse_scale = std::is_same<T, int16_t>::value
                                             ? kIouAndScoreInverseScaleInt16
                                             : kIouAndScoreInverseScaleInt8;
  op_data->iou_and_score_zero_point = std::is_same<T, int16_t>::value
                                          ? kIouAndScoreZeroPointInt16
                                          : kIouAndScoreZeroPointInt8;

  const TfLiteTensor* input_scores;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputTensorScores, &input_scores));
  QuantizeMultiplier(
      input_scores->params.scale * op_data->iou_and_score_inverse_scale,
      &op_data->scores_rescale_multiplier, &op_data->scores_rescale_shift);

  const TfLiteTensor* input_iou_threshold;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorIouThreshold,
                                 &input_iou_threshold));
  QuantizeMultiplier(
      input_iou_threshold->params.scale * op_data->iou_and_score_inverse_scale,
      &op_data->iou_threshold_rescale_multiplier,
      &op_data->iou_threshold_rescale_shift);

  const TfLiteTensor* input_score_threshold;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorScoreThreshold,
                                 &input_score_threshold));
  QuantizeMultiplier(input_score_threshold->params.scale *
                         op_data->iou_and_score_inverse_scale,
                     &op_data->score_threshold_rescale_multiplier,
                     &op_data->score_threshold_rescale_shift);

  // Generate LUT (only for Soft-NMS).
  if (IsSoftNms(node)) {
    TfLiteTensor* output_selected_scores;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node,
                                             kSoftNMSOutputTensorSelectedScores,
                                             &output_selected_scores));
    QuantizeMultiplier(output_selected_scores->params.scale,
                       &op_data->selected_scores_rescale_multiplier,
                       &op_data->selected_scores_rescale_shift);

    // Get the Soft-NMS sigma.
    const TfLiteTensor* input_sigma;
    TF_LITE_ENSURE_OK(
        context, GetInputSafe(context, node, kInputTensorSigma, &input_sigma));

    if (!IsConstantTensor(input_sigma)) {
      TF_LITE_KERNEL_LOG(context, "Sigma needs to be constant for type %s.",
                         TfLiteTypeGetName(input_sigma->type));
      return kTfLiteError;
    }

    const T soft_nms_sigma = *GetTensorData<T>(input_sigma);

    // Sigma is expected to be positive and non-zero, otherwise no LUT will be
    // generated.
    if (soft_nms_sigma > input_sigma->params.zero_point) {
      // Calculate scale from sigma.
      const float sigma_scale =
          -0.5f / ((soft_nms_sigma - input_sigma->params.zero_point) *
                   input_sigma->params.scale);

      // Generate LUT.
      void* soft_nms_lut = malloc(sizeof(T) * LUTSize<T>());
      if (!soft_nms_lut) {
        TF_LITE_KERNEL_LOG(context, "Failed to allocate Soft-NMS LUT.");
        return kTfLiteError;
      }

      GenSoftNmsLut(sigma_scale, static_cast<T*>(soft_nms_lut));
      op_data->soft_nms_lut = soft_nms_lut;
    }
  }

  return kTfLiteOk;
}

bool IsSupportedType(TfLiteType type) {
  return type == kTfLiteFloat32 || type == kTfLiteInt8 || type == kTfLiteInt16;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const int num_inputs = NumInputs(node);
  if (num_inputs != 5 && num_inputs != 6) {
    TF_LITE_KERNEL_LOG(context, "Found NMS op with invalid num inputs: %d",
                       NumInputs(node));
    return kTfLiteError;
  }

  // Boxes & Scores.
  const TfLiteTensor* input_boxes;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputTensorBoxes, &input_boxes));
  TF_LITE_ENSURE(context, IsSupportedType(input_boxes->type));
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_boxes), 2);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(input_boxes, 1), 4);
  const int num_boxes = SizeOfDimension(input_boxes, 0);

  const TfLiteTensor* input_scores;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputTensorScores, &input_scores));
  TF_LITE_ENSURE(context, IsSupportedType(input_scores->type));
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_scores), 1);
  TF_LITE_ENSURE_EQ(context, num_boxes, SizeOfDimension(input_scores, 0));

  // Max output size.
  const TfLiteTensor* input_max_output_size;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorMaxOutputSize,
                                 &input_max_output_size));
  TF_LITE_ENSURE_EQ(context, input_max_output_size->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_max_output_size), 0);
  const bool is_max_output_size_const =
      IsConstantOrPersistentTensor(input_max_output_size);
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
  TF_LITE_ENSURE(context, IsSupportedType(input_iou_threshold->type));
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_iou_threshold), 0);
  const TfLiteTensor* input_score_threshold;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorScoreThreshold,
                                 &input_score_threshold));
  TF_LITE_ENSURE(context, IsSupportedType(input_score_threshold->type));
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_score_threshold), 0);

  if (IsSoftNms(node)) {
    const TfLiteTensor* input_sigma;
    TF_LITE_ENSURE_OK(
        context, GetInputSafe(context, node, kInputTensorSigma, &input_sigma));
    TF_LITE_ENSURE(context, IsSupportedType(input_sigma->type));
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
    output_selected_scores->type = input_scores->type;
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

  // Prepare auxiliary structures used only for quantized types.
  if (input_boxes->type == kTfLiteInt8) {
    TF_LITE_ENSURE_OK(context, PrepareQuantized<int8_t>(context, node));
  } else if (input_boxes->type == kTfLiteInt16) {
    TF_LITE_ENSURE_OK(context, PrepareQuantized<int16_t>(context, node));
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
template <typename T>
void ResetUnusedElementsToZeroes(const int max_output_size,
                                 const int num_selected_indices,
                                 int* selected_indices, T* selected_scores,
                                 T selected_scores_zero_point) {
  for (int i = num_selected_indices; i < max_output_size; ++i) {
    selected_indices[i] = 0;
    if (selected_scores) {
      selected_scores[i] = selected_scores_zero_point;
    }
  }
}

template <typename T>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  NonMaxSuppressionParams nms_params;

  OpData* op_data = static_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, op_data != nullptr);

  nms_params.iou_zero_point = op_data->iou_and_score_zero_point;
  nms_params.iou_inverse_scale = op_data->iou_and_score_inverse_scale;

  const TfLiteTensor* input_boxes;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputTensorBoxes, &input_boxes));
  const int num_boxes = SizeOfDimension(input_boxes, 0);

  const TfLiteTensor* input_scores;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputTensorScores, &input_scores));

  nms_params.scores_zero_point = input_scores->params.zero_point;
  nms_params.scores_rescale_zero_point = op_data->iou_and_score_zero_point;
  nms_params.scores_rescale_multiplier = op_data->scores_rescale_multiplier;
  nms_params.scores_rescale_shift = op_data->scores_rescale_shift;

  const TfLiteTensor* input_max_output_size;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorMaxOutputSize,
                                 &input_max_output_size));
  const int max_output_size_value = *GetTensorData<int>(input_max_output_size);
  TF_LITE_ENSURE(context, (max_output_size_value >= 0));
  const bool is_max_output_size_const =
      IsConstantOrPersistentTensor(input_max_output_size);

  const TfLiteTensor* input_iou_threshold;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorIouThreshold,
                                 &input_iou_threshold));
  const T iou_threshold = *GetTensorData<T>(input_iou_threshold);

  nms_params.iou_threshold_zero_point = input_iou_threshold->params.zero_point;
  nms_params.iou_threshold_rescale_zero_point =
      op_data->iou_and_score_zero_point;
  nms_params.iou_threshold_rescale_multiplier =
      op_data->iou_threshold_rescale_multiplier;
  nms_params.iou_threshold_rescale_shift = op_data->iou_threshold_rescale_shift;

  const TfLiteTensor* input_score_threshold;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorScoreThreshold,
                                 &input_score_threshold));
  const T score_threshold = *GetTensorData<T>(input_score_threshold);

  nms_params.score_threshold_zero_point =
      input_score_threshold->params.zero_point;
  nms_params.score_threshold_rescale_zero_point =
      op_data->iou_and_score_zero_point;
  nms_params.score_threshold_rescale_multiplier =
      op_data->score_threshold_rescale_multiplier;
  nms_params.score_threshold_rescale_shift =
      op_data->score_threshold_rescale_shift;

  TfLiteTensor* output_selected_indices = nullptr;
  TfLiteTensor* output_selected_scores = nullptr;
  TfLiteTensor* output_num_selected_indices = nullptr;

  if (IsSoftNms(node)) {
    const TfLiteTensor* input_sigma;
    TF_LITE_ENSURE_OK(
        context, GetInputSafe(context, node, kInputTensorSigma, &input_sigma));
    const T soft_nms_sigma = *GetTensorData<T>(input_sigma);

    // Ensure sigma is not negative.
    if (soft_nms_sigma < input_sigma->params.zero_point) {
      TF_LITE_KERNEL_LOG(context, "Invalid sigma value for Soft-NMS: %f",
                         soft_nms_sigma);
      return kTfLiteError;
    }

    nms_params.soft_nms_lut = const_cast<const void*>(op_data->soft_nms_lut);

    TF_LITE_ENSURE_OK(
        context,
        GetOutputSafe(context, node, kSoftNMSOutputTensorSelectedIndices,
                      &output_selected_indices));

    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node,
                                             kSoftNMSOutputTensorSelectedScores,
                                             &output_selected_scores));
    nms_params.selected_scores_rescale_zero_point =
        output_selected_scores->params.zero_point;
    nms_params.selected_scores_rescale_multiplier =
        op_data->selected_scores_rescale_multiplier;
    nms_params.selected_scores_rescale_shift =
        op_data->selected_scores_rescale_shift;

    TF_LITE_ENSURE_OK(
        context,
        GetOutputSafe(context, node, kSoftNMSOutputTensorNumSelectedIndices,
                      &output_num_selected_indices));
    if (!is_max_output_size_const) {
      SetTensorSizes(context, output_selected_indices, {max_output_size_value});
      SetTensorSizes(context, output_selected_scores, {max_output_size_value});
    }
    reference_ops::NonMaxSuppression(
        nms_params, GetTensorData<T>(input_boxes), num_boxes,
        GetTensorData<T>(input_scores), max_output_size_value, iou_threshold,
        score_threshold, soft_nms_sigma,
        GetTensorData<int32_t>(output_selected_indices),
        GetTensorData<T>(output_selected_scores),
        GetTensorData<int32_t>(output_num_selected_indices));
    ResetUnusedElementsToZeroes<T>(
        max_output_size_value,
        *GetTensorData<int32_t>(output_num_selected_indices),
        GetTensorData<int32_t>(output_selected_indices),
        GetTensorData<T>(output_selected_scores),
        output_selected_scores->params.zero_point);
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

    const T soft_nms_sigma{0};
    reference_ops::NonMaxSuppression(
        nms_params, GetTensorData<T>(input_boxes), num_boxes,
        GetTensorData<T>(input_scores), max_output_size_value, iou_threshold,
        score_threshold, soft_nms_sigma,
        GetTensorData<int32_t>(output_selected_indices),
        /**selected_scores=**/ static_cast<T*>(nullptr),
        GetTensorData<int32_t>(output_num_selected_indices));
    ResetUnusedElementsToZeroes<T>(
        max_output_size_value,
        *GetTensorData<int32_t>(output_num_selected_indices),
        GetTensorData<int32_t>(output_selected_indices), nullptr, 0);
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_boxes;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputTensorBoxes, &input_boxes));

  const TfLiteType nms_type = input_boxes->type;
  switch (nms_type) {
    case kTfLiteFloat32:
      return Eval<float>(context, node);
    case kTfLiteInt8:
      return Eval<int8_t>(context, node);
    case kTfLiteInt16:
      return Eval<int16_t>(context, node);
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Type '%s' is not supported by non-max supression.",
                         TfLiteTypeGetName(nms_type));
  }

  return kTfLiteError;
}

}  // namespace non_max_suppression

TfLiteRegistration* Register_NON_MAX_SUPPRESSION_V4() {
  static TfLiteRegistration r = {
      non_max_suppression::Init, non_max_suppression::Free,
      non_max_suppression::Prepare, non_max_suppression::Eval};
  return &r;
}

TfLiteRegistration* Register_NON_MAX_SUPPRESSION_V5() {
  static TfLiteRegistration r = {
      non_max_suppression::Init, non_max_suppression::Free,
      non_max_suppression::Prepare, non_max_suppression::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite