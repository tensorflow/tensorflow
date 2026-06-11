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
#include "tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h"

#include <cstdint>
#include <fstream>
#include <ios>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tflite {
namespace evaluation {
namespace {

constexpr absl::string_view kImagePreprocessingStageName =
    "inception_preprocessing_stage";
constexpr absl::string_view kTestImage =
    "tensorflow/lite/tools/evaluation/stages/testdata/"
    "grace_hopper.jpg";
constexpr int kImageDim = 224;

void VerifyLatencyMetrics(const EvaluationStageMetrics& metrics) {
  EXPECT_EQ(metrics.num_runs(), 1);
  int64_t last_latency = metrics.process_metrics().total_latency().last_us();
  EXPECT_GT(last_latency, 0);
  EXPECT_LT(last_latency, 1e7);
  EXPECT_EQ(metrics.process_metrics().total_latency().max_us(), last_latency);
  EXPECT_EQ(metrics.process_metrics().total_latency().min_us(), last_latency);
  EXPECT_EQ(metrics.process_metrics().total_latency().sum_us(), last_latency);
  EXPECT_EQ(metrics.process_metrics().total_latency().avg_us(), last_latency);
}

TEST(ImagePreprocessingStage, NoParams) {
  ImagePreprocessingConfigBuilder builder(
      std::string(kImagePreprocessingStageName), kTfLiteFloat32);
  EvaluationStageConfig config = builder.build();
  config.mutable_specification()->clear_image_preprocessing_params();
  ImagePreprocessingStage stage(config);
  EXPECT_EQ(stage.Init(), kTfLiteError);
}

TEST(ImagePreprocessingStage, InvalidCroppingFraction) {
  ImagePreprocessingConfigBuilder builder(
      std::string(kImagePreprocessingStageName), kTfLiteFloat32);
  builder.AddCroppingStep(-0.8);
  ImagePreprocessingStage stage(builder.build());
  EXPECT_EQ(stage.Init(), kTfLiteError);
}

TEST(ImagePreprocessingStage, ImagePathNotSet) {
  ImagePreprocessingConfigBuilder builder(
      std::string(kImagePreprocessingStageName), kTfLiteFloat32);
  ImagePreprocessingStage stage(builder.build());
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  EXPECT_EQ(stage.Run(), kTfLiteError);
  EXPECT_EQ(stage.GetPreprocessedImageData(), nullptr);
}

TEST(ImagePreprocessingStage, TestImagePreprocessingFloat) {
  std::string image_path(kTestImage);

  ImagePreprocessingConfigBuilder builder(
      std::string(kImagePreprocessingStageName), kTfLiteFloat32);
  builder.AddCroppingStep(0.875);
  builder.AddResizingStep(224, 224, false);
  builder.AddNormalizationStep(127.5, 1.0 / 127.5);
  ImagePreprocessingStage stage(builder.build());
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  // Pre-run.
  EXPECT_EQ(stage.GetPreprocessedImageData(), nullptr);

  stage.SetImagePath(&image_path);
  EXPECT_EQ(stage.Run(), kTfLiteOk);
  EvaluationStageMetrics metrics = stage.LatestMetrics();

  float* preprocessed_image_ptr =
      static_cast<float*>(stage.GetPreprocessedImageData());
  EXPECT_NE(preprocessed_image_ptr, nullptr);
  EXPECT_FLOAT_EQ(preprocessed_image_ptr[0], -0.74901962);
  EXPECT_FLOAT_EQ(preprocessed_image_ptr[1], -0.74901962);
  EXPECT_FLOAT_EQ(preprocessed_image_ptr[2], -0.68627453);
  VerifyLatencyMetrics(metrics);
}

TEST(ImagePreprocessingStage, TestImagePreprocessingFloat_NoCrop) {
  std::string image_path(kTestImage);

  ImagePreprocessingConfigBuilder builder(
      std::string(kImagePreprocessingStageName), kTfLiteFloat32);
  builder.AddResizingStep(224, 224, false);
  builder.AddNormalizationStep(127.5, 1.0 / 127.5);
  ImagePreprocessingStage stage(builder.build());
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  // Pre-run.
  EXPECT_EQ(stage.GetPreprocessedImageData(), nullptr);

  stage.SetImagePath(&image_path);
  EXPECT_EQ(stage.Run(), kTfLiteOk);
  EvaluationStageMetrics metrics = stage.LatestMetrics();

  float* preprocessed_image_ptr =
      static_cast<float*>(stage.GetPreprocessedImageData());
  EXPECT_NE(preprocessed_image_ptr, nullptr);
  EXPECT_FLOAT_EQ(preprocessed_image_ptr[0], -0.83529419);
  EXPECT_FLOAT_EQ(preprocessed_image_ptr[1], -0.7960785);
  EXPECT_FLOAT_EQ(preprocessed_image_ptr[2], -0.35686275);
  VerifyLatencyMetrics(metrics);
}

TEST(ImagePreprocessingStage, TestImagePreprocessingUInt8Quantized) {
  std::string image_path(kTestImage);

  ImagePreprocessingConfigBuilder builder(
      std::string(kImagePreprocessingStageName), kTfLiteUInt8);
  builder.AddCroppingStep(0.875);
  builder.AddResizingStep(224, 224, false);
  ImagePreprocessingStage stage(builder.build());
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  // Pre-run.
  EXPECT_EQ(stage.GetPreprocessedImageData(), nullptr);

  stage.SetImagePath(&image_path);
  EXPECT_EQ(stage.Run(), kTfLiteOk);
  EvaluationStageMetrics metrics = stage.LatestMetrics();

  uint8_t* preprocessed_image_ptr =
      static_cast<uint8_t*>(stage.GetPreprocessedImageData());
  EXPECT_NE(preprocessed_image_ptr, nullptr);
  EXPECT_EQ(preprocessed_image_ptr[0], 32);
  EXPECT_EQ(preprocessed_image_ptr[1], 32);
  EXPECT_EQ(preprocessed_image_ptr[2], 40);
  VerifyLatencyMetrics(metrics);
}

TEST(ImagePreprocessingStage, TestImagePreprocessingInt8Quantized) {
  std::string image_path(kTestImage);

  ImagePreprocessingConfigBuilder builder(
      std::string(kImagePreprocessingStageName), kTfLiteInt8);
  builder.AddCroppingStep(0.875);
  builder.AddResizingStep(224, 224, false);
  builder.AddNormalizationStep(128.0, 1.0);
  ImagePreprocessingStage stage(builder.build());
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  // Pre-run.
  EXPECT_EQ(stage.GetPreprocessedImageData(), nullptr);

  stage.SetImagePath(&image_path);
  EXPECT_EQ(stage.Run(), kTfLiteOk);
  EvaluationStageMetrics metrics = stage.LatestMetrics();

  int8_t* preprocessed_image_ptr =
      static_cast<int8_t*>(stage.GetPreprocessedImageData());
  EXPECT_NE(preprocessed_image_ptr, nullptr);
  EXPECT_EQ(preprocessed_image_ptr[0], -96);
  EXPECT_EQ(preprocessed_image_ptr[1], -96);
  EXPECT_EQ(preprocessed_image_ptr[2], -88);
  VerifyLatencyMetrics(metrics);
}

TEST(ImagePreprocessingStage, TestImagePreprocessingPadding) {
  std::string image_path(kTestImage);

  ImagePreprocessingConfigBuilder builder(
      std::string(kImagePreprocessingStageName), kTfLiteInt8);
  builder.AddCroppingStep(0.875);
  builder.AddResizingStep(224, 224, false);
  builder.AddPaddingStep(225, 225, 0);
  builder.AddNormalizationStep(128.0, 1.0);
  ImagePreprocessingStage stage(builder.build());
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  // Pre-run.
  EXPECT_EQ(stage.GetPreprocessedImageData(), nullptr);

  stage.SetImagePath(&image_path);
  EXPECT_EQ(stage.Run(), kTfLiteOk);
  EvaluationStageMetrics metrics = stage.LatestMetrics();

  int8_t* preprocessed_image_ptr =
      static_cast<int8_t*>(stage.GetPreprocessedImageData());
  EXPECT_NE(preprocessed_image_ptr, nullptr);
  EXPECT_EQ(preprocessed_image_ptr[0], -128);
  EXPECT_EQ(preprocessed_image_ptr[224], -128);
  EXPECT_EQ(preprocessed_image_ptr[225 * 3], -128);
  EXPECT_EQ(preprocessed_image_ptr[225 * 3 + 3], -96);
  EXPECT_EQ(preprocessed_image_ptr[225 * 3 + 4], -96);
  EXPECT_EQ(preprocessed_image_ptr[225 * 3 + 5], -88);
  VerifyLatencyMetrics(metrics);
}

TEST(ImagePreprocessingStage, TestImagePreprocessingSubtractMean) {
  std::string image_path(kTestImage);

  ImagePreprocessingConfigBuilder builder(
      std::string(kImagePreprocessingStageName), kTfLiteFloat32);
  builder.AddCroppingStep(0.875);
  builder.AddResizingStep(224, 224, false);
  builder.AddPerChannelNormalizationStep(110.0, 120.0, 123.0, 1.0);
  ImagePreprocessingStage stage(builder.build());
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  // Pre-run.
  EXPECT_EQ(stage.GetPreprocessedImageData(), nullptr);

  stage.SetImagePath(&image_path);
  EXPECT_EQ(stage.Run(), kTfLiteOk);
  EvaluationStageMetrics metrics = stage.LatestMetrics();

  float* preprocessed_image_ptr =
      static_cast<float*>(stage.GetPreprocessedImageData());
  EXPECT_NE(preprocessed_image_ptr, nullptr);
  EXPECT_EQ(preprocessed_image_ptr[0], -78);
  EXPECT_EQ(preprocessed_image_ptr[1], -88);
  EXPECT_EQ(preprocessed_image_ptr[2], -83);
  VerifyLatencyMetrics(metrics);
}

TEST(ImagePreprocessingStage, TestInvalidRawImageSize) {
  std::string image_path = testing::TempDir() + "/invalid.rgb8";
  std::ofstream stream(image_path, std::ios::out | std::ios::binary);
  stream.write("1234", 4);
  stream.close();

  ImagePreprocessingConfigBuilder builder(
      std::string(kImagePreprocessingStageName), kTfLiteFloat32);
  builder.AddNormalizationStep(127.5, 1.0 / 127.5);
  ImagePreprocessingStage stage(builder.build());
  EXPECT_EQ(stage.Init(), kTfLiteOk);
  stage.SetImagePath(&image_path);
  EXPECT_EQ(stage.Run(), kTfLiteError);
}

}  // namespace
}  // namespace evaluation
}  // namespace tflite
