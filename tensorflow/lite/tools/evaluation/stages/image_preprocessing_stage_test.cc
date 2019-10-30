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

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tflite {
namespace evaluation {
namespace {

constexpr char kImagePreprocessingStageName[] = "inception_preprocessing_stage";
constexpr char kTestImage[] =
    "tensorflow/lite/tools/evaluation/stages/testdata/"
    "grace_hopper.jpg";
constexpr int kImageDim = 224;

EvaluationStageConfig GetImagePreprocessingStageConfig(TfLiteType output_type) {
  EvaluationStageConfig config;
  config.set_name(kImagePreprocessingStageName);
  auto* params =
      config.mutable_specification()->mutable_image_preprocessing_params();
  params->set_image_height(kImageDim);
  params->set_image_width(kImageDim);
  params->set_output_type(static_cast<int>(output_type));
  return config;
}

TEST(ImagePreprocessingStage, NoParams) {
  EvaluationStageConfig config =
      GetImagePreprocessingStageConfig(kTfLiteFloat32);
  config.mutable_specification()->clear_image_preprocessing_params();
  ImagePreprocessingStage stage = ImagePreprocessingStage(config);
  EXPECT_EQ(stage.Init(), kTfLiteError);
}

TEST(ImagePreprocessingStage, NoImageHeight) {
  EvaluationStageConfig config =
      GetImagePreprocessingStageConfig(kTfLiteFloat32);
  config.mutable_specification()
      ->mutable_image_preprocessing_params()
      ->clear_image_height();
  ImagePreprocessingStage stage = ImagePreprocessingStage(config);
  EXPECT_EQ(stage.Init(), kTfLiteError);
}

TEST(ImagePreprocessingStage, NoImageWidth) {
  EvaluationStageConfig config =
      GetImagePreprocessingStageConfig(kTfLiteFloat32);
  config.mutable_specification()
      ->mutable_image_preprocessing_params()
      ->clear_image_width();
  ImagePreprocessingStage stage = ImagePreprocessingStage(config);
  EXPECT_EQ(stage.Init(), kTfLiteError);
}

TEST(ImagePreprocessingStage, InvalidCroppingFraction) {
  EvaluationStageConfig config =
      GetImagePreprocessingStageConfig(kTfLiteFloat32);
  config.mutable_specification()
      ->mutable_image_preprocessing_params()
      ->set_cropping_fraction(-0.8);
  ImagePreprocessingStage stage = ImagePreprocessingStage(config);
  EXPECT_EQ(stage.Init(), kTfLiteError);
}

TEST(ImagePreprocessingStage, ImagePathNotSet) {
  EvaluationStageConfig config =
      GetImagePreprocessingStageConfig(kTfLiteFloat32);
  ImagePreprocessingStage stage = ImagePreprocessingStage(config);
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  EXPECT_EQ(stage.Run(), kTfLiteError);
  EXPECT_EQ(stage.GetPreprocessedImageData(), nullptr);
}

TEST(ImagePreprocessingStage, TestImagePreprocessingFloat) {
  std::string image_path = kTestImage;

  EvaluationStageConfig config =
      GetImagePreprocessingStageConfig(kTfLiteFloat32);
  ImagePreprocessingStage stage = ImagePreprocessingStage(config);
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  // Pre-run.
  EXPECT_EQ(stage.GetPreprocessedImageData(), nullptr);

  stage.SetImagePath(&image_path);
  EXPECT_EQ(stage.Run(), kTfLiteOk);
  EvaluationStageMetrics metrics = stage.LatestMetrics();

  float* preprocessed_image_ptr =
      static_cast<float*>(stage.GetPreprocessedImageData());
  EXPECT_NE(preprocessed_image_ptr, nullptr);
  // We check raw values computed from central-cropping & bilinear interpolation
  // on the test image. The interpolation math is similar to Unit Square formula
  // here: https://en.wikipedia.org/wiki/Bilinear_interpolation#Unit_square
  // These values were verified by running entire image classification pipeline
  // & ensuring output is accurate. We test 3 values, one for each of R/G/B
  // channels.
  EXPECT_FLOAT_EQ(preprocessed_image_ptr[0], -0.74901962);
  EXPECT_FLOAT_EQ(preprocessed_image_ptr[1], -0.74901962);
  EXPECT_FLOAT_EQ(preprocessed_image_ptr[2], -0.68627453);
  EXPECT_EQ(metrics.num_runs(), 1);
  const auto& last_latency =
      metrics.process_metrics().total_latency().last_us();
  EXPECT_GT(last_latency, 0);
  EXPECT_LT(last_latency, 1e7);
  EXPECT_EQ(metrics.process_metrics().total_latency().max_us(), last_latency);
  EXPECT_EQ(metrics.process_metrics().total_latency().min_us(), last_latency);
  EXPECT_EQ(metrics.process_metrics().total_latency().sum_us(), last_latency);
  EXPECT_EQ(metrics.process_metrics().total_latency().avg_us(), last_latency);
}

TEST(ImagePreprocessingStage, TestImagePreprocessingFloat_NoCrop) {
  std::string image_path = kTestImage;

  EvaluationStageConfig config =
      GetImagePreprocessingStageConfig(kTfLiteFloat32);
  config.mutable_specification()
      ->mutable_image_preprocessing_params()
      ->set_cropping_fraction(0);
  ImagePreprocessingStage stage = ImagePreprocessingStage(config);
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  // Pre-run.
  EXPECT_EQ(stage.GetPreprocessedImageData(), nullptr);

  stage.SetImagePath(&image_path);
  EXPECT_EQ(stage.Run(), kTfLiteOk);
  EvaluationStageMetrics metrics = stage.LatestMetrics();

  float* preprocessed_image_ptr =
      static_cast<float*>(stage.GetPreprocessedImageData());
  EXPECT_NE(preprocessed_image_ptr, nullptr);
  // We check raw values computed from central-cropping & bilinear interpolation
  // on the test image. The interpolation math is similar to Unit Square formula
  // here: https://en.wikipedia.org/wiki/Bilinear_interpolation#Unit_square
  // These values were verified by running entire image classification pipeline
  // & ensuring output is accurate. We test 3 values, one for each of R/G/B
  // channels.
  EXPECT_FLOAT_EQ(preprocessed_image_ptr[0], -0.83529419);
  EXPECT_FLOAT_EQ(preprocessed_image_ptr[1], -0.7960785);
  EXPECT_FLOAT_EQ(preprocessed_image_ptr[2], -0.35686275);
  EXPECT_EQ(metrics.num_runs(), 1);
  const auto& last_latency =
      metrics.process_metrics().total_latency().last_us();
  EXPECT_GT(last_latency, 0);
  EXPECT_LT(last_latency, 1e7);
  EXPECT_EQ(metrics.process_metrics().total_latency().max_us(), last_latency);
  EXPECT_EQ(metrics.process_metrics().total_latency().min_us(), last_latency);
  EXPECT_EQ(metrics.process_metrics().total_latency().sum_us(), last_latency);
  EXPECT_EQ(metrics.process_metrics().total_latency().avg_us(), last_latency);
}

TEST(ImagePreprocessingStage, TestImagePreprocessingUInt8Quantized) {
  std::string image_path = kTestImage;

  EvaluationStageConfig config = GetImagePreprocessingStageConfig(kTfLiteUInt8);
  ImagePreprocessingStage stage = ImagePreprocessingStage(config);
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  // Pre-run.
  EXPECT_EQ(stage.GetPreprocessedImageData(), nullptr);

  stage.SetImagePath(&image_path);
  EXPECT_EQ(stage.Run(), kTfLiteOk);
  EvaluationStageMetrics metrics = stage.LatestMetrics();

  uint8_t* preprocessed_image_ptr =
      static_cast<uint8_t*>(stage.GetPreprocessedImageData());
  EXPECT_NE(preprocessed_image_ptr, nullptr);
  // We check raw values computed from central-cropping & bilinear interpolation
  // on the test image. The interpolation math is similar to Unit Square formula
  // here: https://en.wikipedia.org/wiki/Bilinear_interpolation#Unit_square
  // These values were verified by running entire image classification pipeline
  // & ensuring output is accurate. We test 3 values, one for each of R/G/B
  // channels.
  EXPECT_EQ(preprocessed_image_ptr[0], 32);
  EXPECT_EQ(preprocessed_image_ptr[1], 32);
  EXPECT_EQ(preprocessed_image_ptr[2], 40);
  EXPECT_EQ(metrics.num_runs(), 1);
  const auto& last_latency =
      metrics.process_metrics().total_latency().last_us();
  EXPECT_GT(last_latency, 0);
  EXPECT_LT(last_latency, 1e7);
  EXPECT_EQ(metrics.process_metrics().total_latency().max_us(), last_latency);
  EXPECT_EQ(metrics.process_metrics().total_latency().min_us(), last_latency);
  EXPECT_EQ(metrics.process_metrics().total_latency().sum_us(), last_latency);
  EXPECT_EQ(metrics.process_metrics().total_latency().avg_us(), last_latency);
}

TEST(ImagePreprocessingStage, TestImagePreprocessingInt8Quantized) {
  std::string image_path = kTestImage;

  EvaluationStageConfig config = GetImagePreprocessingStageConfig(kTfLiteInt8);
  ImagePreprocessingStage stage = ImagePreprocessingStage(config);
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  // Pre-run.
  EXPECT_EQ(stage.GetPreprocessedImageData(), nullptr);

  stage.SetImagePath(&image_path);
  EXPECT_EQ(stage.Run(), kTfLiteOk);
  EvaluationStageMetrics metrics = stage.LatestMetrics();

  int8_t* preprocessed_image_ptr =
      static_cast<int8_t*>(stage.GetPreprocessedImageData());
  EXPECT_NE(preprocessed_image_ptr, nullptr);
  // We check raw values computed from central-cropping & bilinear interpolation
  // on the test image. The interpolation math is similar to Unit Square formula
  // here: https://en.wikipedia.org/wiki/Bilinear_interpolation#Unit_square
  // These values were verified by running entire image classification pipeline
  // & ensuring output is accurate. We test 3 values, one for each of R/G/B
  // channels.
  EXPECT_EQ(preprocessed_image_ptr[0], -96);
  EXPECT_EQ(preprocessed_image_ptr[1], -96);
  EXPECT_EQ(preprocessed_image_ptr[2], -88);
  EXPECT_EQ(metrics.num_runs(), 1);
  const auto& last_latency =
      metrics.process_metrics().total_latency().last_us();
  EXPECT_GT(last_latency, 0);
  EXPECT_LT(last_latency, 1e7);
  EXPECT_EQ(metrics.process_metrics().total_latency().max_us(), last_latency);
  EXPECT_EQ(metrics.process_metrics().total_latency().min_us(), last_latency);
  EXPECT_EQ(metrics.process_metrics().total_latency().sum_us(), last_latency);
  EXPECT_EQ(metrics.process_metrics().total_latency().avg_us(), last_latency);
}

}  // namespace
}  // namespace evaluation
}  // namespace tflite
