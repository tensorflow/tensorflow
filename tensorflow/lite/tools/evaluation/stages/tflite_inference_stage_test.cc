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
#include "tensorflow/lite/tools/evaluation/stages/tflite_inference_stage.h"

#include <stdint.h>

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace tflite {
namespace evaluation {
namespace {

constexpr char kTfliteInferenceStageName[] = "tflite_inference_stage";
constexpr char kModelPath[] =
    "tensorflow/lite/testdata/add_quantized.bin";
constexpr int kTotalElements = 1 * 8 * 8 * 3;

template <typename T>
T* SetValues(T array[], T value) {
  for (int i = 0; i < kTotalElements; i++) {
    array[i] = value;
  }
  return array;
}

EvaluationStageConfig GetTfliteInferenceStageConfig() {
  EvaluationStageConfig config;
  config.set_name(kTfliteInferenceStageName);
  auto* params =
      config.mutable_specification()->mutable_tflite_inference_params();
  params->set_model_file_path(kModelPath);
  params->set_invocations_per_run(2);
  return config;
}

TEST(TfliteInferenceStage, NoParams) {
  EvaluationStageConfig config = GetTfliteInferenceStageConfig();
  config.mutable_specification()->clear_tflite_inference_params();
  TfliteInferenceStage stage(config);
  EXPECT_EQ(stage.Init(), kTfLiteError);
}

TEST(TfliteInferenceStage, NoModelPath) {
  EvaluationStageConfig config = GetTfliteInferenceStageConfig();
  config.mutable_specification()
      ->mutable_tflite_inference_params()
      ->clear_model_file_path();
  TfliteInferenceStage stage(config);
  EXPECT_EQ(stage.Init(), kTfLiteError);
}

TEST(TfliteInferenceStage, IncorrectModelPath) {
  EvaluationStageConfig config = GetTfliteInferenceStageConfig();
  config.mutable_specification()
      ->mutable_tflite_inference_params()
      ->set_model_file_path("xyz.tflite");
  TfliteInferenceStage stage(config);
  EXPECT_EQ(stage.Init(), kTfLiteError);
}

TEST(TfliteInferenceStage, NoInputData) {
  // Create stage.
  EvaluationStageConfig config = GetTfliteInferenceStageConfig();
  TfliteInferenceStage stage(config);

  // Initialize.
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  // Run.
  EXPECT_EQ(stage.Run(), kTfLiteError);
}

TEST(TfliteInferenceStage, CorrectModelInfo) {
  // Create stage.
  EvaluationStageConfig config = GetTfliteInferenceStageConfig();
  TfliteInferenceStage stage(config);

  // Initialize.
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  const TfLiteModelInfo* model_info = stage.GetModelInfo();
  // Verify Input
  EXPECT_EQ(model_info->inputs.size(), 1);
  const TfLiteTensor* tensor = model_info->inputs[0];
  EXPECT_EQ(tensor->type, kTfLiteUInt8);
  EXPECT_EQ(tensor->bytes, kTotalElements);
  const TfLiteIntArray* input_shape = tensor->dims;
  EXPECT_EQ(input_shape->data[0], 1);
  EXPECT_EQ(input_shape->data[1], 8);
  EXPECT_EQ(input_shape->data[2], 8);
  EXPECT_EQ(input_shape->data[3], 3);
  // Verify Output
  EXPECT_EQ(model_info->outputs.size(), 1);
  tensor = model_info->outputs[0];
  EXPECT_EQ(tensor->type, kTfLiteUInt8);
  EXPECT_EQ(tensor->bytes, kTotalElements);
  const TfLiteIntArray* output_shape = tensor->dims;
  EXPECT_EQ(output_shape->data[0], 1);
  EXPECT_EQ(output_shape->data[1], 8);
  EXPECT_EQ(output_shape->data[2], 8);
  EXPECT_EQ(output_shape->data[3], 3);
}

TEST(TfliteInferenceStage, TestResizeModel) {
  // Create stage.
  EvaluationStageConfig config = GetTfliteInferenceStageConfig();
  TfliteInferenceStage stage(config);

  // Initialize.
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  // Resize.
  EXPECT_EQ(stage.ResizeInputs({{3, 8, 8, 3}}), kTfLiteOk);

  const TfLiteModelInfo* model_info = stage.GetModelInfo();
  // Verify Input
  EXPECT_EQ(model_info->inputs.size(), 1);
  const TfLiteTensor* tensor = model_info->inputs[0];
  EXPECT_EQ(tensor->type, kTfLiteUInt8);
  EXPECT_EQ(tensor->bytes, 3 * kTotalElements);
  const TfLiteIntArray* input_shape = tensor->dims;
  EXPECT_EQ(input_shape->data[0], 3);
  EXPECT_EQ(input_shape->data[1], 8);
  EXPECT_EQ(input_shape->data[2], 8);
  EXPECT_EQ(input_shape->data[3], 3);
  // Verify Output
  EXPECT_EQ(model_info->outputs.size(), 1);
  tensor = model_info->outputs[0];
  EXPECT_EQ(tensor->type, kTfLiteUInt8);
  EXPECT_EQ(tensor->bytes, 3 * kTotalElements);
  const TfLiteIntArray* output_shape = tensor->dims;
  EXPECT_EQ(output_shape->data[0], 3);
  EXPECT_EQ(output_shape->data[1], 8);
  EXPECT_EQ(output_shape->data[2], 8);
  EXPECT_EQ(output_shape->data[3], 3);
}

TEST(TfliteInferenceStage, CorrectOutput) {
  // Create stage.
  EvaluationStageConfig config = GetTfliteInferenceStageConfig();
  TfliteInferenceStage stage(config);

  // Initialize.
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  // Set input data.
  uint8_t input_tensor[kTotalElements];
  SetValues(input_tensor, static_cast<uint8_t>(2));
  std::vector<void*> inputs;
  inputs.push_back(input_tensor);
  stage.SetInputs(inputs);

  // Run.
  EXPECT_EQ(stage.Run(), kTfLiteOk);

  // Verify outputs.
  uint8_t* output_tensor = static_cast<uint8_t*>(stage.GetOutputs()->at(0));
  for (int i = 0; i < kTotalElements; i++) {
    EXPECT_EQ(output_tensor[i], static_cast<uint8_t>(6));
  }

  // Verify metrics.
  EvaluationStageMetrics metrics = stage.LatestMetrics();
  EXPECT_EQ(metrics.num_runs(), 1);
  const auto& latency = metrics.process_metrics().total_latency();
  const auto max_latency = latency.max_us();
  EXPECT_GT(max_latency, 0);
  EXPECT_LT(max_latency, 1e7);
  EXPECT_LE(latency.last_us(), max_latency);
  EXPECT_LE(latency.min_us(), max_latency);
  EXPECT_GE(latency.sum_us(), max_latency);
  EXPECT_LE(latency.avg_us(), max_latency);
  EXPECT_TRUE(latency.has_std_deviation_us());
  EXPECT_EQ(
      metrics.process_metrics().tflite_inference_metrics().num_inferences(), 2);
}

TEST(TfliteInferenceStage, CustomDelegate) {
  // Create stage.
  EvaluationStageConfig config = GetTfliteInferenceStageConfig();
  TfliteInferenceStage stage(config);

  Interpreter::TfLiteDelegatePtr test_delegate = CreateNNAPIDelegate();

  // Delegate application should only work after initialization of stage.
  EXPECT_NE(stage.ApplyCustomDelegate(std::move(test_delegate)), kTfLiteOk);
  EXPECT_EQ(stage.Init(), kTfLiteOk);
  EXPECT_EQ(stage.ApplyCustomDelegate(std::move(test_delegate)), kTfLiteOk);
}

}  // namespace
}  // namespace evaluation
}  // namespace tflite
