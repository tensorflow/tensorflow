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
#include "tensorflow/lite/tools/evaluation/stages/inference_profiler_stage.h"

#include <stdint.h>

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tflite {
namespace evaluation {
namespace {

constexpr char kInferenceProfilerStageName[] = "inference_profiler_stage";
constexpr char kModelPath[] =
    "tensorflow/lite/testdata/add_quantized.bin";

EvaluationStageConfig GetInferenceProfilerStageConfig(int num_threads = 1) {
  EvaluationStageConfig config;
  config.set_name(kInferenceProfilerStageName);
  auto* params =
      config.mutable_specification()->mutable_tflite_inference_params();
  params->set_model_file_path(kModelPath);
  params->set_invocations_per_run(2);
  params->set_num_threads(num_threads);
  return config;
}

TEST(InferenceProfilerStage, NoParams) {
  EvaluationStageConfig config = GetInferenceProfilerStageConfig();
  config.mutable_specification()->clear_tflite_inference_params();
  InferenceProfilerStage stage(config);
  EXPECT_EQ(stage.Init(), kTfLiteError);
}

TEST(InferenceProfilerStage, NoModelPath) {
  EvaluationStageConfig config = GetInferenceProfilerStageConfig();
  config.mutable_specification()
      ->mutable_tflite_inference_params()
      ->clear_model_file_path();
  InferenceProfilerStage stage(config);
  EXPECT_EQ(stage.Init(), kTfLiteError);
}

TEST(InferenceProfilerStage, NoOutputDiffForDefaultConfig) {
  // Create stage.
  EvaluationStageConfig config = GetInferenceProfilerStageConfig();
  InferenceProfilerStage stage(config);

  // Initialize.
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(stage.Run(), kTfLiteOk);
  }
  EvaluationStageMetrics metrics = stage.LatestMetrics();
  EXPECT_TRUE(metrics.process_metrics().has_inference_profiler_metrics());
  auto profiler_metrics =
      metrics.process_metrics().inference_profiler_metrics();
  EXPECT_TRUE(profiler_metrics.has_reference_latency());
  EXPECT_TRUE(profiler_metrics.has_test_latency());
  EXPECT_EQ(profiler_metrics.output_errors_size(), 1);
  EXPECT_EQ(profiler_metrics.output_errors(0).avg_value(), 0);
}

}  // namespace
}  // namespace evaluation
}  // namespace tflite
