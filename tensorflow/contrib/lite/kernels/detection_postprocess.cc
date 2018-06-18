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
#include <string.h>
#include <numeric>
#include <vector>
#include "flatbuffers/flexbuffers.h"
#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace custom {
namespace detection_postprocess {

// Input tensors
constexpr int kInputTensorBoxEncodings = 0;
constexpr int kInputTensorClassPredictions = 1;
constexpr int kInputTensorAnchors = 2;

// Output tensors
constexpr int kOutputTensorDetectionBoxes = 0;
constexpr int kOutputTensorDetectionClasses = 1;
constexpr int kOutputTensorDetectionScores = 2;
constexpr int kOutputTensorNumDetections = 3;

constexpr size_t kNumCoordBox = 4;
constexpr size_t kBatchSize = 1;

// Object Detection model produces axis-aligned boxes in two formats:
// BoxCorner represents the upper right (xmin, ymin) and
// lower left corner (xmax, ymax).
// CenterSize represents the center (xcenter, ycenter), height and width.
// BoxCornerEncoding and CenterSizeEncoding are related as follows:
// ycenter = y / y_scale * anchor.h + anchor.y;
// xcenter = x / x_scale * anchor.w + anchor.x;
// half_h = 0.5*exp(h/ h_scale)) * anchor.h;
// half_w = 0.5*exp(w / w_scale)) * anchor.w;
// ymin = ycenter - half_h
// ymax = ycenter + half_h
// xmin = xcenter - half_w
// xmax = xcenter + half_w
struct BoxCornerEncoding {
  float ymin;
  float xmin;
  float ymax;
  float xmax;
};

struct CenterSizeEncoding {
  float y;
  float x;
  float h;
  float w;
};
// We make sure that the memory allocations are contiguous with static assert.
static_assert(sizeof(BoxCornerEncoding) == sizeof(float) * kNumCoordBox,
              "Size of BoxCornerEncoding is 4 float values");
static_assert(sizeof(CenterSizeEncoding) == sizeof(float) * kNumCoordBox,
              "Size of CenterSizeEncoding is 4 float values");

struct OpData {
  int max_detections;
  int max_classes_per_detection;
  float non_max_suppression_score_threshold;
  float intersection_over_union_threshold;
  int num_classes;
  CenterSizeEncoding scale_values;
  // Indices of Temporary tensors
  int decoded_boxes_index;
  int scores_index;
  int active_candidate_index;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData;
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  op_data->max_detections = m["max_detections"].AsInt32();
  op_data->max_classes_per_detection = m["max_classes_per_detection"].AsInt32();
  op_data->non_max_suppression_score_threshold =
      m["nms_score_threshold"].AsFloat();
  op_data->intersection_over_union_threshold = m["nms_iou_threshold"].AsFloat();
  op_data->num_classes = m["num_classes"].AsInt32();
  op_data->scale_values.y = m["y_scale"].AsFloat();
  op_data->scale_values.x = m["x_scale"].AsFloat();
  op_data->scale_values.h = m["h_scale"].AsFloat();
  op_data->scale_values.w = m["w_scale"].AsFloat();
  context->AddTensors(context, 1, &op_data->decoded_boxes_index);
  context->AddTensors(context, 1, &op_data->scores_index);
  context->AddTensors(context, 1, &op_data->active_candidate_index);
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

// TODO(chowdhery): Add to kernel_util.h
TfLiteStatus SetTensorSizes(TfLiteContext* context, TfLiteTensor* tensor,
                            std::initializer_list<int> values) {
  TfLiteIntArray* size = TfLiteIntArrayCreate(values.size());
  int index = 0;
  for (int v : values) {
    size->data[index] = v;
    ++index;
  }
  return context->ResizeTensor(context, tensor, size);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = reinterpret_cast<OpData*>(node->user_data);
  // Inputs: box_encodings, scores, anchors
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  const TfLiteTensor* input_box_encodings =
      GetInput(context, node, kInputTensorBoxEncodings);
  const TfLiteTensor* input_class_predictions =
      GetInput(context, node, kInputTensorClassPredictions);
  const TfLiteTensor* input_anchors =
      GetInput(context, node, kInputTensorAnchors);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_box_encodings), 3);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_class_predictions), 3);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_anchors), 2);
  // number of detected boxes
  const int num_detected_boxes =
      op_data->max_detections * op_data->max_classes_per_detection;

  // Outputs: detection_boxes, detection_scores, detection_classes,
  // num_detections
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 4);
  // Output Tensor detection_boxes: size is set to (1, num_detected_boxes, 4)
  TfLiteTensor* detection_boxes =
      GetOutput(context, node, kOutputTensorDetectionBoxes);
  detection_boxes->type = kTfLiteFloat32;
  SetTensorSizes(context, detection_boxes,
                 {kBatchSize, num_detected_boxes, kNumCoordBox});

  // Output Tensor detection_classes: size is set to (1, num_detected_boxes)
  TfLiteTensor* detection_classes =
      GetOutput(context, node, kOutputTensorDetectionClasses);
  detection_classes->type = kTfLiteFloat32;
  SetTensorSizes(context, detection_classes, {kBatchSize, num_detected_boxes});

  // Output Tensor detection_scores: size is set to (1, num_detected_boxes)
  TfLiteTensor* detection_scores =
      GetOutput(context, node, kOutputTensorDetectionScores);
  detection_scores->type = kTfLiteFloat32;
  SetTensorSizes(context, detection_scores, {kBatchSize, num_detected_boxes});

  // Output Tensor num_detections: size is set to 1
  TfLiteTensor* num_detections =
      GetOutput(context, node, kOutputTensorNumDetections);
  num_detections->type = kTfLiteFloat32;
  // TODO (chowdhery): Make it a scalar when available
  SetTensorSizes(context, num_detections, {1});

  // Temporary tensors
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(3);
  node->temporaries->data[0] = op_data->decoded_boxes_index;
  node->temporaries->data[1] = op_data->scores_index;
  node->temporaries->data[2] = op_data->active_candidate_index;

  // decoded_boxes
  TfLiteTensor* decoded_boxes = &context->tensors[op_data->decoded_boxes_index];
  decoded_boxes->type = kTfLiteFloat32;
  decoded_boxes->allocation_type = kTfLiteArenaRw;
  SetTensorSizes(context, decoded_boxes,
                 {input_box_encodings->dims->data[1], kNumCoordBox});

  // scores
  TfLiteTensor* scores = &context->tensors[op_data->scores_index];
  scores->type = kTfLiteFloat32;
  scores->allocation_type = kTfLiteArenaRw;
  SetTensorSizes(context, scores,
                 {input_class_predictions->dims->data[1],
                  input_class_predictions->dims->data[2]});

  // active_candidate
  TfLiteTensor* active_candidate =
      &context->tensors[op_data->active_candidate_index];
  active_candidate->type = kTfLiteUInt8;
  active_candidate->allocation_type = kTfLiteArenaRw;
  SetTensorSizes(context, active_candidate,
                 {input_box_encodings->dims->data[1]});

  return kTfLiteOk;
}

class Dequantizer {
 public:
  Dequantizer(int zero_point, float scale)
      : zero_point_(zero_point), scale_(scale) {}
  float operator()(uint8 x) {
    return (static_cast<float>(x) - zero_point_) * scale_;
  }

 private:
  int zero_point_;
  float scale_;
};

void DequantizeBoxEncodings(const TfLiteTensor* input_box_encodings, int idx,
                            float quant_zero_point, float quant_scale,
                            CenterSizeEncoding* box_centersize) {
  const uint8* boxes =
      GetTensorData<uint8>(input_box_encodings) + kNumCoordBox * idx;
  Dequantizer dequantize(quant_zero_point, quant_scale);
  box_centersize->y = dequantize(boxes[0]);
  box_centersize->x = dequantize(boxes[1]);
  box_centersize->h = dequantize(boxes[2]);
  box_centersize->w = dequantize(boxes[3]);
}

template <class T>
T ReInterpretTensor(const TfLiteTensor* tensor) {
  // TODO (chowdhery): check float
  const float* tensor_base = tensor->data.f;
  return reinterpret_cast<T>(tensor_base);
}

template <class T>
T ReInterpretTensor(TfLiteTensor* tensor) {
  // TODO (chowdhery): check float
  float* tensor_base = tensor->data.f;
  return reinterpret_cast<T>(tensor_base);
}

TfLiteStatus DecodeCenterSizeBoxes(TfLiteContext* context, TfLiteNode* node,
                                   OpData* op_data) {
  // Parse input tensor boxencodings
  const TfLiteTensor* input_box_encodings =
      GetInput(context, node, kInputTensorBoxEncodings);
  TF_LITE_ENSURE_EQ(context, input_box_encodings->dims->data[0], kBatchSize);
  const int num_boxes = input_box_encodings->dims->data[1];
  TF_LITE_ENSURE_EQ(context, input_box_encodings->dims->data[2], kNumCoordBox);

  // Decode the boxes to get (ymin, xmin, ymax, xmax) based on the anchors
  CenterSizeEncoding box_centersize;
  CenterSizeEncoding scale_values = op_data->scale_values;
  const float quant_zero_point =
      static_cast<float>(input_box_encodings->params.zero_point);
  const float quant_scale =
      static_cast<float>(input_box_encodings->params.scale);
  for (int idx = 0; idx < num_boxes; ++idx) {
    switch (input_box_encodings->type) {
        // Quantized
      case kTfLiteUInt8:
        DequantizeBoxEncodings(input_box_encodings, idx, quant_zero_point,
                               quant_scale, &box_centersize);
        break;
        // Float
      case kTfLiteFloat32:
        box_centersize = ReInterpretTensor<const CenterSizeEncoding*>(
            input_box_encodings)[idx];
        break;
      default:
        // Unsupported type.
        return kTfLiteError;
    }

    const TfLiteTensor* input_anchors =
        GetInput(context, node, kInputTensorAnchors);

    const auto& anchor =
        ReInterpretTensor<const CenterSizeEncoding*>(input_anchors)[idx];

    float ycenter = box_centersize.y / scale_values.y * anchor.h + anchor.y;
    float xcenter = box_centersize.x / scale_values.x * anchor.w + anchor.x;
    float half_h =
        0.5f * static_cast<float>(std::exp(box_centersize.h / scale_values.h)) *
        anchor.h;
    float half_w =
        0.5f * static_cast<float>(std::exp(box_centersize.w / scale_values.w)) *
        anchor.w;
    TfLiteTensor* decoded_boxes =
        &context->tensors[op_data->decoded_boxes_index];
    auto& box = ReInterpretTensor<BoxCornerEncoding*>(decoded_boxes)[idx];
    box.ymin = ycenter - half_h;
    box.xmin = xcenter - half_w;
    box.ymax = ycenter + half_h;
    box.xmax = xcenter + half_w;
  }
  return kTfLiteOk;
}

void DecreasingPartialArgSort(const float* values, int num_values,
                              int num_to_sort, int* indices) {
  std::iota(indices, indices + num_values, 0);
  std::partial_sort(
      indices, indices + num_to_sort, indices + num_values,
      [&values](const int i, const int j) { return values[i] > values[j]; });
}

void SelectDetectionsAboveScoreThreshold(const std::vector<float>& values,
                                         const float threshold,
                                         std::vector<float>* keep_values,
                                         std::vector<int>* keep_indices) {
  for (int i = 0; i < values.size(); i++) {
    if (values[i] >= threshold) {
      keep_values->emplace_back(values[i]);
      keep_indices->emplace_back(i);
    }
  }
}

bool ValidateBoxes(const TfLiteTensor* decoded_boxes, const int num_boxes) {
  for (int i = 0; i < num_boxes; ++i) {
    // ymax>=ymin, xmax>=xmin
    auto& box = ReInterpretTensor<const BoxCornerEncoding*>(decoded_boxes)[i];
    if (box.ymin >= box.ymax || box.xmin >= box.xmax) {
      return false;
    }
  }
  return true;
}

float ComputeIntersectionOverUnion(const TfLiteTensor* decoded_boxes,
                                   const int i, const int j) {
  auto& box_i = ReInterpretTensor<const BoxCornerEncoding*>(decoded_boxes)[i];
  auto& box_j = ReInterpretTensor<const BoxCornerEncoding*>(decoded_boxes)[j];
  const float area_i = (box_i.ymax - box_i.ymin) * (box_i.xmax - box_i.xmin);
  const float area_j = (box_j.ymax - box_j.ymin) * (box_j.xmax - box_j.xmin);
  if (area_i <= 0 || area_j <= 0) return 0.0;
  const float intersection_ymin = std::max<float>(box_i.ymin, box_j.ymin);
  const float intersection_xmin = std::max<float>(box_i.xmin, box_j.xmin);
  const float intersection_ymax = std::min<float>(box_i.ymax, box_j.ymax);
  const float intersection_xmax = std::min<float>(box_i.xmax, box_j.xmax);
  const float intersection_area =
      std::max<float>(intersection_ymax - intersection_ymin, 0.0) *
      std::max<float>(intersection_xmax - intersection_xmin, 0.0);
  return intersection_area / (area_i + area_j - intersection_area);
}

// NonMaxSuppressionSingleClass() is O(n^2) pairwise comparison between boxes
// It assumes all boxes are good in beginning and sorts based on the scores.
// If lower-scoring box has too much overlap with a higher-scoring box,
// we get rid of the lower-scoring box.
TfLiteStatus NonMaxSuppressionSingleClassHelper(
    TfLiteContext* context, TfLiteNode* node, OpData* op_data,
    const std::vector<float>& scores, std::vector<int>* selected) {
  const TfLiteTensor* input_box_encodings =
      GetInput(context, node, kInputTensorBoxEncodings);
  const TfLiteTensor* decoded_boxes =
      &context->tensors[op_data->decoded_boxes_index];
  const int num_boxes = input_box_encodings->dims->data[1];
  const int max_detections = op_data->max_detections;
  const float non_max_suppression_score_threshold =
      op_data->non_max_suppression_score_threshold;
  const float intersection_over_union_threshold =
      op_data->intersection_over_union_threshold;
  // Maximum detections should be positive.
  TF_LITE_ENSURE(context, (max_detections >= 0));
  // intersection_over_union_threshold should be positive
  // and should be less than 1.
  TF_LITE_ENSURE(context, (intersection_over_union_threshold > 0.0f) &&
                              (intersection_over_union_threshold <= 1.0f));
  // Validate boxes
  TF_LITE_ENSURE(context, ValidateBoxes(decoded_boxes, num_boxes));

  // threshold scores
  std::vector<int> keep_indices;
  // TODO (chowdhery): Remove the dynamic allocation and replace it
  // with temporaries, esp for std::vector<float>
  std::vector<float> keep_scores;
  SelectDetectionsAboveScoreThreshold(
      scores, non_max_suppression_score_threshold, &keep_scores, &keep_indices);

  int num_scores_kept = keep_scores.size();
  std::vector<int> sorted_indices;
  sorted_indices.resize(num_scores_kept);
  DecreasingPartialArgSort(keep_scores.data(), num_scores_kept, num_scores_kept,
                           sorted_indices.data());

  const int num_boxes_kept = keep_scores.size();
  const int output_size = std::min(num_boxes_kept, max_detections);
  selected->clear();
  TfLiteTensor* active_candidate =
      &context->tensors[op_data->active_candidate_index];
  TF_LITE_ENSURE(context, (active_candidate->dims->data[0]) == num_boxes);
  int num_active_candidate = num_boxes;
  uint8_t* active_box_candidate = (active_candidate->data.uint8);
  for (int row = 0; row < num_boxes; row++) {
    active_box_candidate[row] = 1;
  }

  for (int i = 0; i < num_boxes; ++i) {
    if (num_active_candidate == 0 || selected->size() >= output_size) break;
    if (active_box_candidate[i] == 1) {
      selected->push_back(keep_indices[sorted_indices[i]]);
      active_box_candidate[i] = 0;
      num_active_candidate--;
    } else {
      continue;
    }
    for (int j = i + 1; j < num_boxes; ++j) {
      if (active_box_candidate[j] == 1) {
        float intersection_over_union = ComputeIntersectionOverUnion(
            decoded_boxes, keep_indices[sorted_indices[i]],
            keep_indices[sorted_indices[j]]);

        if (intersection_over_union > intersection_over_union_threshold) {
          active_box_candidate[j] = 0;
          num_active_candidate--;
        }
      }
    }
  }
  return kTfLiteOk;
}

// This function implements a fast version of Non Maximal Suppression for
// multiple classes where
// 1) we keep the top-k scores for each anchor and
// 2) during NMS, each anchor only uses the highest class score for sorting.
// 3) Compared to standard NMS, the worst runtime of this version is O(N^2)
// instead of O(KN^2) where N is the number of anchors and K the number of
// classes.
TfLiteStatus NonMaxSuppressionMultiClassFastHelper(TfLiteContext* context,
                                                   TfLiteNode* node,
                                                   OpData* op_data,
                                                   const float* scores) {
  const TfLiteTensor* input_box_encodings =
      GetInput(context, node, kInputTensorBoxEncodings);
  const TfLiteTensor* decoded_boxes =
      &context->tensors[op_data->decoded_boxes_index];

  TfLiteTensor* detection_boxes =
      GetOutput(context, node, kOutputTensorDetectionBoxes);
  TfLiteTensor* detection_classes =
      GetOutput(context, node, kOutputTensorDetectionClasses);
  TfLiteTensor* detection_scores =
      GetOutput(context, node, kOutputTensorDetectionScores);
  TfLiteTensor* num_detections =
      GetOutput(context, node, kOutputTensorNumDetections);

  const int num_boxes = input_box_encodings->dims->data[1];
  const int num_classes = op_data->num_classes;
  const int max_categories_per_anchor = op_data->max_classes_per_detection;
  // The row index offset is 1 if background class is included and 0 otherwise.
  const int label_offset = 1;
  TF_LITE_ENSURE(context, (label_offset != -1));
  TF_LITE_ENSURE(context, (max_categories_per_anchor > 0));
  const int num_classes_with_background = num_classes + label_offset;
  const int num_categories_per_anchor =
      std::min(max_categories_per_anchor, num_classes);
  std::vector<float> max_scores;
  max_scores.resize(num_boxes);
  std::vector<int> sorted_class_indices;
  sorted_class_indices.resize(num_boxes * num_classes);
  for (int row = 0; row < num_boxes; row++) {
    const float* box_scores =
        scores + row * num_classes_with_background + label_offset;
    int* class_indices = sorted_class_indices.data() + row * num_classes;
    DecreasingPartialArgSort(box_scores, num_classes, num_categories_per_anchor,
                             class_indices);
    max_scores[row] = box_scores[class_indices[0]];
  }
  // Perform non-maximal suppression on max scores
  std::vector<int> selected;
  NonMaxSuppressionSingleClassHelper(context, node, op_data, max_scores,
                                     &selected);
  // Allocate output tensors
  int output_box_index = 0;
  for (const auto& selected_index : selected) {
    const float* box_scores =
        scores + selected_index * num_classes_with_background + label_offset;
    const int* class_indices =
        sorted_class_indices.data() + selected_index * num_classes;

    for (int col = 0; col < num_categories_per_anchor; ++col) {
      int box_offset = num_categories_per_anchor * output_box_index + col;
      // detection_boxes
      ReInterpretTensor<BoxCornerEncoding*>(detection_boxes)[box_offset] =
          ReInterpretTensor<const BoxCornerEncoding*>(
              decoded_boxes)[selected_index];
      // detection_classes
      detection_classes->data.f[box_offset] = class_indices[col];
      // detection_scores
      detection_scores->data.f[box_offset] = box_scores[class_indices[col]];
      output_box_index++;
    }
  }
  num_detections->data.f[0] = output_box_index;
  return kTfLiteOk;
}

void DequantizeClassPredictions(const TfLiteTensor* input_class_predictions,
                                const int num_boxes,
                                const int num_classes_with_background,
                                const TfLiteTensor* scores) {
  float quant_zero_point =
      static_cast<float>(input_class_predictions->params.zero_point);
  float quant_scale = static_cast<float>(input_class_predictions->params.scale);
  Dequantizer dequantize(quant_zero_point, quant_scale);
  const uint8* scores_quant = GetTensorData<uint8>(input_class_predictions);
  for (int idx = 0; idx < num_boxes * num_classes_with_background; ++idx) {
    scores->data.f[idx] = dequantize(scores_quant[idx]);
  }
}

TfLiteStatus NonMaxSuppressionMultiClass(TfLiteContext* context,
                                         TfLiteNode* node, OpData* op_data) {
  // Get the input tensors
  const TfLiteTensor* input_box_encodings =
      GetInput(context, node, kInputTensorBoxEncodings);
  const TfLiteTensor* input_class_predictions =
      GetInput(context, node, kInputTensorClassPredictions);
  const int num_boxes = input_box_encodings->dims->data[1];
  const int num_classes = op_data->num_classes;
  TF_LITE_ENSURE_EQ(context, input_class_predictions->dims->data[0],
                    kBatchSize);
  TF_LITE_ENSURE_EQ(context, input_class_predictions->dims->data[1], num_boxes);
  const int num_classes_with_background =
      input_class_predictions->dims->data[2];

  TF_LITE_ENSURE(context, (num_classes_with_background == num_classes + 1));

  const TfLiteTensor* scores;
  switch (input_class_predictions->type) {
    case kTfLiteUInt8: {
      TfLiteTensor* temporary_scores = &context->tensors[op_data->scores_index];
      DequantizeClassPredictions(input_class_predictions, num_boxes,
                                 num_classes_with_background, temporary_scores);
      scores = temporary_scores;
    } break;
    case kTfLiteFloat32:
      scores = input_class_predictions;
      break;
    default:
      // Unsupported type.
      return kTfLiteError;
  }
  NonMaxSuppressionMultiClassFastHelper(context, node, op_data,
                                        GetTensorData<float>(scores));
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  // TODO(chowdhery): Generalize for any batch size
  TF_LITE_ENSURE(context, (kBatchSize == 1));
  auto* op_data = reinterpret_cast<OpData*>(node->user_data);
  // These two functions correspond to two blocks in the Object Detection model.
  // In future, we would like to break the custom op in two blocks, which is
  // currently not feasible because we would like to input quantized inputs
  // and do all calculations in float. Mixed quantized/float calculations are
  // currently not supported in TFLite.

  // This fills in temporary decoded_boxes
  // by transforming input_box_encodings and input_anchors from
  // CenterSizeEncodings to BoxCornerEncoding
  DecodeCenterSizeBoxes(context, node, op_data);
  // This fills in the output tensors
  // by choosing effective set of decoded boxes
  // based on Non Maximal Suppression, i.e. selecting
  // highest scoring non-overlapping boxes.
  NonMaxSuppressionMultiClass(context, node, op_data);

  return kTfLiteOk;
}
}  // namespace detection_postprocess

TfLiteRegistration* Register_DETECTION_POSTPROCESS() {
  static TfLiteRegistration r = {detection_postprocess::Init,
                                 detection_postprocess::Free,
                                 detection_postprocess::Prepare,
                                 detection_postprocess::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
