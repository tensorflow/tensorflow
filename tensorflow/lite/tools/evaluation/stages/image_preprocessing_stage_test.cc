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

#include <gmock/gmock.h>
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

MATCHER(HasValidLatencyMetrics, "") {
  if (arg.num_runs() != 1) {
    *result_listener << "num_runs is " << arg.num_runs() << " (expected 1)";
    return false;
  }
  const auto& total_latency = arg.process_metrics().total_latency();
  int64_t last_latency = total_latency.last_us();
  if (last_latency <= 0 || last_latency >= 1e7) {
    *result_listener << "last_latency is " << last_latency
                     << " (expected to be in range (0, 1e7))";
    return false;
  }
  if (total_latency.max_us() != last_latency) {
    *result_listener << "max_us (" << total_latency.max_us()
                     << ") != last_latency (" << last_latency << ")";
    return false;
  }
  if (total_latency.min_us() != last_latency) {
    *result_listener << "min_us (" << total_latency.min_us()
                     << ") != last_latency (" << last_latency << ")";
    return false;
  }
  if (total_latency.sum_us() != last_latency) {
    *result_listener << "sum_us (" << total_latency.sum_us()
                     << ") != last_latency (" << last_latency << ")";
    return false;
  }
  if (total_latency.avg_us() != last_latency) {
    *result_listener << "avg_us (" << total_latency.avg_us()
                     << ") != last_latency (" << last_latency << ")";
    return false;
  }
  return true;
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
  // We check raw values computed from central-cropping & bilinear interpolation
  // on the test image. The interpolation math is similar to Unit Square formula
  // here: https://en.wikipedia.org/wiki/Bilinear_interpolation#Unit_square
  // These values were verified by running entire image classification pipeline
  // & ensuring output is accurate. We test 3 values, one for each of R/G/B
  // channels.
  EXPECT_FLOAT_EQ(preprocessed_image_ptr[0], -0.74901962);
  EXPECT_FLOAT_EQ(preprocessed_image_ptr[1], -0.74901962);
  EXPECT_FLOAT_EQ(preprocessed_image_ptr[2], -0.68627453);
  EXPECT_THAT(metrics, HasValidLatencyMetrics());
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
  // We check raw values computed from central-cropping & bilinear interpolation
  // on the test image. The interpolation math is similar to Unit Square formula
  // here: https://en.wikipedia.org/wiki/Bilinear_interpolation#Unit_square
  // These values were verified by running entire image classification pipeline
  // & ensuring output is accurate. We test 3 values, one for each of R/G/B
  // channels.
  EXPECT_FLOAT_EQ(preprocessed_image_ptr[0], -0.83529419);
  EXPECT_FLOAT_EQ(preprocessed_image_ptr[1], -0.7960785);
  EXPECT_FLOAT_EQ(preprocessed_image_ptr[2], -0.35686275);
  EXPECT_THAT(metrics, HasValidLatencyMetrics());
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
  // We check raw values computed from central-cropping & bilinear interpolation
  // on the test image. The interpolation math is similar to Unit Square formula
  // here: https://en.wikipedia.org/wiki/Bilinear_interpolation#Unit_square
  // These values were verified by running entire image classification pipeline
  // & ensuring output is accurate. We test 3 values, one for each of R/G/B
  // channels.
  EXPECT_EQ(preprocessed_image_ptr[0], 32);
  EXPECT_EQ(preprocessed_image_ptr[1], 32);
  EXPECT_EQ(preprocessed_image_ptr[2], 40);
  EXPECT_THAT(metrics, HasValidLatencyMetrics());
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
  // We check raw values computed from central-cropping & bilinear interpolation
  // on the test image. The interpolation math is similar to Unit Square formula
  // here: https://en.wikipedia.org/wiki/Bilinear_interpolation#Unit_square
  // These values were verified by running entire image classification pipeline
  // & ensuring output is accurate. We test 3 values, one for each of R/G/B
  // channels.
  EXPECT_EQ(preprocessed_image_ptr[0], -96);
  EXPECT_EQ(preprocessed_image_ptr[1], -96);
  EXPECT_EQ(preprocessed_image_ptr[2], -88);
  EXPECT_THAT(metrics, HasValidLatencyMetrics());
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
  // We check raw values computed from central-cropping & bilinear interpolation
  // on the test image. The interpolation math is similar to Unit Square formula
  // here: https://en.wikipedia.org/wiki/Bilinear_interpolation#Unit_square
  // These values were verified by running entire image classification pipeline
  // & ensuring output is accurate. We test 3 values, one for each of R/G/B
  // channels.
  EXPECT_EQ(preprocessed_image_ptr[0], -128);
  EXPECT_EQ(preprocessed_image_ptr[224], -128);
  EXPECT_EQ(preprocessed_image_ptr[225 * 3], -128);
  EXPECT_EQ(preprocessed_image_ptr[225 * 3 + 3], -96);
  EXPECT_EQ(preprocessed_image_ptr[225 * 3 + 4], -96);
  EXPECT_EQ(preprocessed_image_ptr[225 * 3 + 5], -88);
  EXPECT_THAT(metrics, HasValidLatencyMetrics());
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
  // We check raw values computed from central-cropping & bilinear interpolation
  // on the test image. The interpolation math is similar to Unit Square formula
  // here: https://en.wikipedia.org/wiki/Bilinear_interpolation#Unit_square
  // These values were verified by running entire image classification pipeline
  // & ensuring output is accurate. We test 3 values, one for each of R/G/B
  // channels.
  EXPECT_EQ(preprocessed_image_ptr[0], -78);
  EXPECT_EQ(preprocessed_image_ptr[1], -88);
  EXPECT_EQ(preprocessed_image_ptr[2], -83);
  EXPECT_THAT(metrics, HasValidLatencyMetrics());
}

TEST(ImagePreprocessingStage, TestCropTargetLargerThanImage) {
  std::string image_path(kTestImage);

  ImagePreprocessingConfigBuilder builder(
      std::string(kImagePreprocessingStageName), kTfLiteFloat32);
  // A crop target larger than the image would produce negative start offsets
  // and read out of bounds. It must be rejected at run time.
  builder.AddCroppingStep(/*width=*/100000U, /*height=*/100000U);
  builder.AddNormalizationStep(127.5, 1.0 / 127.5);
  ImagePreprocessingStage stage(builder.build());
  EXPECT_EQ(stage.Init(), kTfLiteOk);
  stage.SetImagePath(&image_path);
  EXPECT_EQ(stage.Run(), kTfLiteError);
}

TEST(ImagePreprocessingStage, TestPaddingTargetSmallerThanImage) {
  std::string image_path(kTestImage);

  ImagePreprocessingConfigBuilder builder(
      std::string(kImagePreprocessingStageName), kTfLiteFloat32);
  // A padding target smaller than the image would produce negative padding
  // counts and write out of bounds. It must be rejected at run time.
  builder.AddPaddingStep(/*width=*/1, /*height=*/1, 0);
  builder.AddNormalizationStep(127.5, 1.0 / 127.5);
  ImagePreprocessingStage stage(builder.build());
  EXPECT_EQ(stage.Init(), kTfLiteOk);
  stage.SetImagePath(&image_path);
  EXPECT_EQ(stage.Run(), kTfLiteError);
}

TEST(ImagePreprocessingStage, TestPaddingTargetTooLarge) {
  std::string image_path(kTestImage);

  ImagePreprocessingConfigBuilder builder(
      std::string(kImagePreprocessingStageName), kTfLiteFloat32);
  // A padding target large enough to overflow the output buffer size
  // computation must be rejected at run time.
  builder.AddPaddingStep(/*width=*/100000, /*height=*/100000, 0);
  builder.AddNormalizationStep(127.5, 1.0 / 127.5);
  ImagePreprocessingStage stage(builder.build());
  EXPECT_EQ(stage.Init(), kTfLiteOk);
  stage.SetImagePath(&image_path);
  EXPECT_EQ(stage.Run(), kTfLiteError);
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

TEST(ImagePreprocessingStage, TestRawImageSizeMismatchWithCroppingTarget) {
  std::string image_path = testing::TempDir() + "/mismatch.rgb8";
  std::ofstream stream(image_path, std::ios::out | std::ios::binary);
  // 6 bytes = 2 pixels (if 3 channels)
  stream.write("123456", 6);
  stream.close();

  ImagePreprocessingConfigBuilder builder(
      std::string(kImagePreprocessingStageName), kTfLiteFloat32);
  // Target size 2x2 = 4 pixels = 12 bytes.
  builder.AddCroppingStep(/*width=*/2U, /*height=*/2U);
  ImagePreprocessingStage stage(builder.build());
  EXPECT_EQ(stage.Init(), kTfLiteOk);
  stage.SetImagePath(&image_path);
  EXPECT_EQ(stage.Run(), kTfLiteError);
}

}  // namespace
}  // namespace evaluation
}  // namespace tflite
