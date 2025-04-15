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
#include "tensorflow/lite/tools/evaluation/stages/topk_accuracy_eval_stage.h"

#include <stdint.h>

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tflite {
namespace evaluation {
namespace {

constexpr char kTopkAccuracyEvalStageName[] = "topk_accuracy_eval_stage";
constexpr int kNumCategories = 1001;

EvaluationStageConfig GetTopkAccuracyEvalStageConfig() {
  EvaluationStageConfig config;
  config.set_name(kTopkAccuracyEvalStageName);
  auto* params =
      config.mutable_specification()->mutable_topk_accuracy_eval_params();
  params->set_k(5);
  return config;
}

template <typename T>
T* ResetOutputArray(T array[]) {
  for (int i = 0; i < kNumCategories; i++) {
    array[i] = 0;
  }
  return array;
}

std::vector<std::string> CreateGroundTruthLabels() {
  std::vector<std::string> ground_truth_labels;
  ground_truth_labels.reserve(kNumCategories);
  for (int i = 0; i < kNumCategories; i++) {
    ground_truth_labels.push_back(std::to_string(i));
  }
  return ground_truth_labels;
}

TEST(TopkAccuracyEvalStage, NoInitializers) {
  // Create stage.
  EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
  TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);

  EXPECT_EQ(stage.Init(), kTfLiteError);
}

TEST(TopkAccuracyEvalStage, NoK) {
  // Create stage.
  EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
  config.mutable_specification()
      ->mutable_topk_accuracy_eval_params()
      ->clear_k();
  TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);

  // Initialize.
  std::vector<std::string> ground_truth_labels = CreateGroundTruthLabels();
  TfLiteIntArray* model_output_shape = TfLiteIntArrayCreate(2);
  model_output_shape->data[0] = 1;
  model_output_shape->data[1] = kNumCategories;
  TfLiteType model_output_type = kTfLiteFloat32;
  stage.SetTaskInfo(ground_truth_labels, model_output_type, model_output_shape);
  EXPECT_EQ(stage.Init(), kTfLiteError);
  TfLiteIntArrayFree(model_output_shape);
}

TEST(TopkAccuracyEvalStage, NoGroundTruthLabels) {
  // Create stage.
  EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
  TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);

  // Initialize.
  std::vector<std::string> ground_truth_labels = {};
  TfLiteIntArray* model_output_shape = TfLiteIntArrayCreate(2);
  model_output_shape->data[0] = 1;
  model_output_shape->data[1] = kNumCategories;
  TfLiteType model_output_type = kTfLiteFloat32;
  stage.SetTaskInfo(ground_truth_labels, model_output_type, model_output_shape);
  EXPECT_EQ(stage.Init(), kTfLiteError);
  TfLiteIntArrayFree(model_output_shape);
}

TEST(TopkAccuracyEvalStage, KTooLarge) {
  // Create stage.
  EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
  config.mutable_specification()->mutable_topk_accuracy_eval_params()->set_k(
      10000);
  TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);

  // Initialize.
  std::vector<std::string> ground_truth_labels = CreateGroundTruthLabels();
  TfLiteIntArray* model_output_shape = TfLiteIntArrayCreate(2);
  model_output_shape->data[0] = 1;
  model_output_shape->data[1] = kNumCategories;
  TfLiteType model_output_type = kTfLiteFloat32;
  stage.SetTaskInfo(ground_truth_labels, model_output_type, model_output_shape);
  EXPECT_EQ(stage.Init(), kTfLiteError);
  TfLiteIntArrayFree(model_output_shape);
}

TEST(TopkAccuracyEvalStage, WeirdModelOutputShape) {
  // Create stage.
  EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
  TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);

  // Initialize.
  std::vector<std::string> ground_truth_labels = CreateGroundTruthLabels();
  TfLiteIntArray* model_output_shape = TfLiteIntArrayCreate(2);
  model_output_shape->data[0] = 1;
  model_output_shape->data[1] = kNumCategories + 1;
  TfLiteType model_output_type = kTfLiteFloat32;
  stage.SetTaskInfo(ground_truth_labels, model_output_type, model_output_shape);
  EXPECT_EQ(stage.Init(), kTfLiteError);
  TfLiteIntArrayFree(model_output_shape);
}

TEST(TopkAccuracyEvalStage, UnsupportedModelOutputType) {
  // Create stage.
  EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
  TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);

  // Initialize.
  std::vector<std::string> ground_truth_labels = CreateGroundTruthLabels();
  TfLiteIntArray* model_output_shape = TfLiteIntArrayCreate(2);
  model_output_shape->data[0] = 1;
  model_output_shape->data[1] = kNumCategories + 1;
  TfLiteType model_output_type = kTfLiteComplex64;
  stage.SetTaskInfo(ground_truth_labels, model_output_type, model_output_shape);
  EXPECT_EQ(stage.Init(), kTfLiteError);
  TfLiteIntArrayFree(model_output_shape);
}

TEST(TopkAccuracyEvalStage, NoInputs) {
  // Create stage.
  EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
  TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);
  // Initialize.
  std::vector<std::string> ground_truth_labels = CreateGroundTruthLabels();
  TfLiteIntArray* model_output_shape = TfLiteIntArrayCreate(2);
  model_output_shape->data[0] = 1;
  model_output_shape->data[1] = kNumCategories;
  TfLiteType model_output_type = kTfLiteFloat32;
  stage.SetTaskInfo(ground_truth_labels, model_output_type, model_output_shape);
  EXPECT_EQ(stage.Init(), kTfLiteOk);
  TfLiteIntArrayFree(model_output_shape);

  EXPECT_EQ(stage.Run(), kTfLiteError);
}

TEST(TopkAccuracyEvalStage, InvalidGroundTruth) {
  // Create stage.
  EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
  TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);
  // Initialize.
  std::vector<std::string> ground_truth_labels = CreateGroundTruthLabels();
  TfLiteIntArray* model_output_shape = TfLiteIntArrayCreate(2);
  model_output_shape->data[0] = 1;
  model_output_shape->data[1] = kNumCategories;
  TfLiteType model_output_type = kTfLiteFloat32;
  stage.SetTaskInfo(ground_truth_labels, model_output_type, model_output_shape);
  EXPECT_EQ(stage.Init(), kTfLiteOk);
  TfLiteIntArrayFree(model_output_shape);

  float array[kNumCategories];
  float* tensor = ResetOutputArray(array);
  tensor[0] = 0.8;
  stage.SetEvalInputs(tensor, /*ground_truth_label=*/nullptr);
  EXPECT_EQ(stage.Run(), kTfLiteError);
}

TEST(TopkAccuracyEvalStage, FloatTest_CorrectLabelsAtLastIndices) {
  // Create stage.
  EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
  TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);
  // Initialize.
  std::vector<std::string> ground_truth_labels = CreateGroundTruthLabels();
  TfLiteIntArray* model_output_shape = TfLiteIntArrayCreate(2);
  model_output_shape->data[0] = 1;
  model_output_shape->data[1] = kNumCategories;
  TfLiteType model_output_type = kTfLiteFloat32;
  stage.SetTaskInfo(ground_truth_labels, model_output_type, model_output_shape);
  EXPECT_EQ(stage.Init(), kTfLiteOk);
  TfLiteIntArrayFree(model_output_shape);

  float array[kNumCategories];

  // The ground truth is index 0, but it is 5th most likely based on model's
  // output.
  float* tensor = ResetOutputArray(array);
  tensor[4] = 0.9;
  tensor[3] = 0.8;
  tensor[2] = 0.7;
  tensor[1] = 0.6;
  tensor[0] = 0.5;
  std::string ground_truth = "0";
  stage.SetEvalInputs(tensor, &ground_truth);
  EXPECT_EQ(stage.Run(), kTfLiteOk);
  EvaluationStageMetrics metrics = stage.LatestMetrics();
  EXPECT_EQ(1, metrics.num_runs());
  auto accuracy_metrics = metrics.process_metrics().topk_accuracy_metrics();
  // Only top-5 count is 1.0, rest are 0.0
  EXPECT_FLOAT_EQ(1.0, accuracy_metrics.topk_accuracies(4));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(0.0, accuracy_metrics.topk_accuracies(i));
  }

  // The ground truth is index 1, but it is 4th highest based on model's output.
  ground_truth = "1";
  stage.SetEvalInputs(tensor, &ground_truth);
  EXPECT_EQ(stage.Run(), kTfLiteOk);
  metrics = stage.LatestMetrics();
  EXPECT_EQ(2, metrics.num_runs());
  accuracy_metrics = metrics.process_metrics().topk_accuracy_metrics();
  // 1/2 images had the currect output in top-4, 2/2 has currect output in
  // top-5.
  EXPECT_FLOAT_EQ(1.0, accuracy_metrics.topk_accuracies(4));
  EXPECT_FLOAT_EQ(0.5, accuracy_metrics.topk_accuracies(3));
  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(0.0, accuracy_metrics.topk_accuracies(i));
  }
}

class CorrectTopkAccuracyEvalTest : public ::testing::Test {
 protected:
  template <typename T>
  void VerifyCorrectBehaviorForType(T ground_truth_0_value,
                                    T ground_truth_1_value,
                                    TfLiteType model_output_type) {
    // Create stage.
    EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
    TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);
    // Initialize.
    std::vector<std::string> ground_truth_labels = CreateGroundTruthLabels();
    TfLiteIntArray* model_output_shape = TfLiteIntArrayCreate(2);
    model_output_shape->data[0] = 1;
    model_output_shape->data[1] = kNumCategories;
    stage.SetTaskInfo(ground_truth_labels, model_output_type,
                      model_output_shape);
    EXPECT_EQ(stage.Init(), kTfLiteOk);
    TfLiteIntArrayFree(model_output_shape);

    // Pre-run state.
    EvaluationStageMetrics metrics = stage.LatestMetrics();
    EXPECT_EQ(0, metrics.num_runs());
    auto accuracy_metrics = metrics.process_metrics().topk_accuracy_metrics();
    EXPECT_EQ(0, accuracy_metrics.topk_accuracies_size());

    T array[kNumCategories];

    // First image was correctly identified as "0".
    T* tensor = ResetOutputArray(array);
    tensor[0] = ground_truth_0_value;
    std::string ground_truth = "0";
    stage.SetEvalInputs(tensor, &ground_truth);
    EXPECT_EQ(stage.Run(), kTfLiteOk);
    metrics = stage.LatestMetrics();
    EXPECT_EQ(1, metrics.num_runs());
    accuracy_metrics = metrics.process_metrics().topk_accuracy_metrics();
    for (int i = 0; i < accuracy_metrics.topk_accuracies_size(); ++i) {
      EXPECT_FLOAT_EQ(1.0, accuracy_metrics.topk_accuracies(i));
    }

    // Second image was also correctly identified as "1".
    // Hence, for the second image as well, the top output ("1") was correct.
    tensor[1] = ground_truth_1_value;
    ground_truth = "1";
    stage.SetEvalInputs(tensor, &ground_truth);
    EXPECT_EQ(stage.Run(), kTfLiteOk);
    metrics = stage.LatestMetrics();
    EXPECT_EQ(2, metrics.num_runs());
    accuracy_metrics = metrics.process_metrics().topk_accuracy_metrics();
    for (int i = 0; i < accuracy_metrics.topk_accuracies_size(); ++i) {
      EXPECT_FLOAT_EQ(1.0, accuracy_metrics.topk_accuracies(i));
    }
  }
};

TEST_F(CorrectTopkAccuracyEvalTest, FloatTest) {
  VerifyCorrectBehaviorForType(static_cast<float>(0.8), static_cast<float>(0.9),
                               kTfLiteFloat32);
}

TEST_F(CorrectTopkAccuracyEvalTest, Int8Test) {
  VerifyCorrectBehaviorForType(static_cast<int8_t>(1), static_cast<int8_t>(2),
                               kTfLiteInt8);
}

TEST_F(CorrectTopkAccuracyEvalTest, UInt8Test) {
  VerifyCorrectBehaviorForType(static_cast<uint8_t>(1), static_cast<uint8_t>(2),
                               kTfLiteUInt8);
}

}  // namespace
}  // namespace evaluation
}  // namespace tflite
