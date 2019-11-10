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

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tflite {
namespace evaluation {
namespace {

constexpr char kAveragePrecisionStageName[] =
    "object_detection_average_precision";

EvaluationStageConfig GetAveragePrecisionStageConfig(int num_classes) {
  EvaluationStageConfig config;
  config.set_name(kAveragePrecisionStageName);
  auto* params = config.mutable_specification()
                     ->mutable_object_detection_average_precision_params();
  params->add_iou_thresholds(0.5);
  params->add_iou_thresholds(0.999);
  params->set_num_classes(num_classes);
  return config;
}

ObjectDetectionResult GetGroundTruthDetectionResult() {
  ObjectDetectionResult ground_truth;
  ground_truth.set_image_name("some_image.jpg");

  auto* object_1 = ground_truth.add_objects();
  object_1->set_class_id(1);
  auto* object_1_bbox = object_1->mutable_bounding_box();
  object_1_bbox->set_normalized_top(0.5);
  object_1_bbox->set_normalized_bottom(1.0);
  object_1_bbox->set_normalized_left(0.5);
  object_1_bbox->set_normalized_right(1.0);

  auto* object_2 = ground_truth.add_objects();
  object_2->set_class_id(1);
  auto* object_2_bbox = object_2->mutable_bounding_box();
  object_2_bbox->set_normalized_top(0);
  object_2_bbox->set_normalized_bottom(1.0);
  object_2_bbox->set_normalized_left(0);
  object_2_bbox->set_normalized_right(1.0);

  auto* object_3 = ground_truth.add_objects();
  object_3->set_class_id(2);
  auto* object_3_bbox = object_3->mutable_bounding_box();
  object_3_bbox->set_normalized_top(0.5);
  object_3_bbox->set_normalized_bottom(1.0);
  object_3_bbox->set_normalized_left(0.5);
  object_3_bbox->set_normalized_right(1.0);

  return ground_truth;
}

ObjectDetectionResult GetPredictedDetectionResult() {
  ObjectDetectionResult predicted;

  auto* object_1 = predicted.add_objects();
  object_1->set_class_id(1);
  object_1->set_score(0.8);
  auto* object_1_bbox = object_1->mutable_bounding_box();
  object_1_bbox->set_normalized_top(0.091);
  object_1_bbox->set_normalized_bottom(1.0);
  object_1_bbox->set_normalized_left(0.091);
  object_1_bbox->set_normalized_right(1.0);

  auto* object_2 = predicted.add_objects();
  object_2->set_class_id(1);
  object_2->set_score(0.9);
  auto* object_2_bbox = object_2->mutable_bounding_box();
  object_2_bbox->set_normalized_top(0.474);
  object_2_bbox->set_normalized_bottom(1.0);
  object_2_bbox->set_normalized_left(0.474);
  object_2_bbox->set_normalized_right(1.0);

  auto* object_3 = predicted.add_objects();
  object_3->set_class_id(1);
  object_3->set_score(0.95);
  auto* object_3_bbox = object_3->mutable_bounding_box();
  object_3_bbox->set_normalized_top(0.474);
  object_3_bbox->set_normalized_bottom(1.0);
  object_3_bbox->set_normalized_left(0.474);
  object_3_bbox->set_normalized_right(1.0);
  return predicted;
}

TEST(ObjectDetectionAveragePrecisionStage, ZeroClasses) {
  // Create stage.
  EvaluationStageConfig config = GetAveragePrecisionStageConfig(0);
  ObjectDetectionAveragePrecisionStage stage =
      ObjectDetectionAveragePrecisionStage(config);

  EXPECT_EQ(stage.Init(), kTfLiteError);
}

// Tests ObjectDetectionAveragePrecisionStage with sample inputs & outputs.
// The underlying library is tested extensively in utils/image_metrics_test.
TEST(ObjectDetectionAveragePrecisionStage, SampleInputs) {
  // Create & initialize stage.
  EvaluationStageConfig config = GetAveragePrecisionStageConfig(3);
  ObjectDetectionAveragePrecisionStage stage =
      ObjectDetectionAveragePrecisionStage(config);
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  const ObjectDetectionResult ground_truth = GetGroundTruthDetectionResult();
  const ObjectDetectionResult predicted = GetPredictedDetectionResult();

  // Run with no predictions.
  stage.SetEvalInputs(ObjectDetectionResult(), ground_truth);
  EXPECT_EQ(stage.Run(), kTfLiteOk);

  EvaluationStageMetrics metrics = stage.LatestMetrics();
  ObjectDetectionAveragePrecisionMetrics detection_metrics =
      metrics.process_metrics().object_detection_average_precision_metrics();
  EXPECT_FLOAT_EQ(detection_metrics.overall_mean_average_precision(), 0.0);
  EXPECT_EQ(detection_metrics.individual_average_precisions_size(), 2);

  // Run with matching predictions.
  stage.SetEvalInputs(ground_truth, ground_truth);
  EXPECT_EQ(stage.Run(), kTfLiteOk);

  metrics = stage.LatestMetrics();
  detection_metrics =
      metrics.process_metrics().object_detection_average_precision_metrics();
  EXPECT_FLOAT_EQ(detection_metrics.overall_mean_average_precision(),
                  0.50495052);
  EXPECT_EQ(metrics.num_runs(), 2);

  // Run.
  stage.SetEvalInputs(predicted, ground_truth);
  EXPECT_EQ(stage.Run(), kTfLiteOk);

  metrics = stage.LatestMetrics();
  detection_metrics =
      metrics.process_metrics().object_detection_average_precision_metrics();
  EXPECT_FLOAT_EQ(
      detection_metrics.individual_average_precisions(0).iou_threshold(), 0.5);
  EXPECT_FLOAT_EQ(
      detection_metrics.individual_average_precisions(0).average_precision(),
      0.4841584);
  EXPECT_FLOAT_EQ(
      detection_metrics.individual_average_precisions(1).iou_threshold(),
      0.999);
  EXPECT_FLOAT_EQ(
      detection_metrics.individual_average_precisions(1).average_precision(),
      0.33663365);
  // Should be average of above two values.
  EXPECT_FLOAT_EQ(detection_metrics.overall_mean_average_precision(),
                  0.41039604);
}

TEST(ObjectDetectionAveragePrecisionStage, DefaultIoUThresholds) {
  // Create & initialize stage.
  EvaluationStageConfig config = GetAveragePrecisionStageConfig(3);
  auto* params = config.mutable_specification()
                     ->mutable_object_detection_average_precision_params();
  params->clear_iou_thresholds();
  ObjectDetectionAveragePrecisionStage stage =
      ObjectDetectionAveragePrecisionStage(config);
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  const ObjectDetectionResult ground_truth = GetGroundTruthDetectionResult();
  const ObjectDetectionResult predicted = GetPredictedDetectionResult();

  // Run with matching predictions.
  stage.SetEvalInputs(ground_truth, ground_truth);
  EXPECT_EQ(stage.Run(), kTfLiteOk);

  EvaluationStageMetrics metrics = stage.LatestMetrics();
  ObjectDetectionAveragePrecisionMetrics detection_metrics =
      metrics.process_metrics().object_detection_average_precision_metrics();
  // Full AP, since ground-truth & predictions match.
  EXPECT_FLOAT_EQ(detection_metrics.overall_mean_average_precision(), 1.0);
  // Should be 10 IoU thresholds.
  EXPECT_EQ(detection_metrics.individual_average_precisions_size(), 10);
  EXPECT_FLOAT_EQ(
      detection_metrics.individual_average_precisions(0).iou_threshold(), 0.5);
  EXPECT_FLOAT_EQ(
      detection_metrics.individual_average_precisions(9).iou_threshold(), 0.95);
}

}  // namespace
}  // namespace evaluation
}  // namespace tflite
