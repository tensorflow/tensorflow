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
#include "tensorflow/lite/tools/evaluation/stages/object_detection_average_precision_stage.h"

#include <stdint.h>

#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/stages/utils/image_metrics.h"

namespace tflite {
namespace evaluation {
namespace {

image::Detection ConvertProtoToDetection(
    const ObjectDetectionResult::ObjectInstance& input, int image_id) {
  image::Detection detection;
  detection.box.x.min = input.bounding_box().normalized_left();
  detection.box.x.max = input.bounding_box().normalized_right();
  detection.box.y.min = input.bounding_box().normalized_top();
  detection.box.y.max = input.bounding_box().normalized_bottom();
  detection.imgid = image_id;
  detection.score = input.score();
  return detection;
}

}  // namespace

TfLiteStatus ObjectDetectionAveragePrecisionStage::Init() {
  num_classes_ = config_.specification()
                     .object_detection_average_precision_params()
                     .num_classes();
  if (num_classes_ <= 0) {
    LOG(ERROR) << "num_classes cannot be <= 0";
    return kTfLiteError;
  }

  // Initialize per-class data structures.
  for (int i = 0; i < num_classes_; ++i) {
    ground_truth_object_vectors_.emplace_back();
    predicted_object_vectors_.emplace_back();
  }
  return kTfLiteOk;
}

TfLiteStatus ObjectDetectionAveragePrecisionStage::Run() {
  for (int i = 0; i < ground_truth_objects_.objects_size(); ++i) {
    const int class_id = ground_truth_objects_.objects(i).class_id();
    if (class_id >= num_classes_) {
      LOG(ERROR) << "Encountered invalid class ID: " << class_id;
      return kTfLiteError;
    }

    ground_truth_object_vectors_[class_id].push_back(ConvertProtoToDetection(
        ground_truth_objects_.objects(i), current_image_index_));
  }

  for (int i = 0; i < predicted_objects_.objects_size(); ++i) {
    const int class_id = predicted_objects_.objects(i).class_id();
    if (class_id >= num_classes_) {
      LOG(ERROR) << "Encountered invalid class ID: " << class_id;
      return kTfLiteError;
    }

    predicted_object_vectors_[class_id].push_back(ConvertProtoToDetection(
        predicted_objects_.objects(i), current_image_index_));
  }

  current_image_index_++;
  return kTfLiteOk;
}

EvaluationStageMetrics ObjectDetectionAveragePrecisionStage::LatestMetrics() {
  EvaluationStageMetrics metrics;
  if (current_image_index_ == 0) return metrics;

  metrics.set_num_runs(current_image_index_);
  auto* ap_metrics = metrics.mutable_process_metrics()
                         ->mutable_object_detection_average_precision_metrics();
  auto& ap_params =
      config_.specification().object_detection_average_precision_params();

  std::vector<float> iou_thresholds;
  if (ap_params.iou_thresholds_size() == 0) {
    // Default IoU thresholds as defined by COCO evaluation.
    // Refer: http://cocodataset.org/#detection-eval
    float threshold = 0.5;
    for (int i = 0; i < 10; ++i) {
      iou_thresholds.push_back(threshold + i * 0.05);
    }
  } else {
    for (auto& threshold : ap_params.iou_thresholds()) {
      iou_thresholds.push_back(threshold);
    }
  }

  image::AveragePrecision::Options opts;
  opts.num_recall_points = ap_params.num_recall_points();

  float ap_sum = 0;
  int num_total_aps = 0;
  for (float threshold : iou_thresholds) {
    float threshold_ap_sum = 0;
    int num_counted_classes = 0;

    for (int i = 0; i < num_classes_; ++i) {
      // Skip if this class wasn't encountered at all.
      // TODO(b/133772912): Investigate the validity of this snippet when a
      // subset of the classes is encountered in datasets.
      if (ground_truth_object_vectors_[i].empty() &&
          predicted_object_vectors_[i].empty())
        continue;

      // Output is NaN if there are no ground truth objects.
      // So we assume 0.
      float ap_value = 0.0;
      if (!ground_truth_object_vectors_[i].empty()) {
        opts.iou_threshold = threshold;
        ap_value = image::AveragePrecision(opts).FromBoxes(
            ground_truth_object_vectors_[i], predicted_object_vectors_[i]);
      }

      ap_sum += ap_value;
      num_total_aps += 1;
      threshold_ap_sum += ap_value;
      num_counted_classes += 1;
    }

    if (num_counted_classes == 0) continue;
    auto* threshold_ap = ap_metrics->add_individual_average_precisions();
    threshold_ap->set_average_precision(threshold_ap_sum / num_counted_classes);
    threshold_ap->set_iou_threshold(threshold);
  }

  if (num_total_aps == 0) return metrics;
  ap_metrics->set_overall_mean_average_precision(ap_sum / num_total_aps);
  return metrics;
}

}  // namespace evaluation
}  // namespace tflite
