/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/jni/accuracy_benchmark.h"

#include <fcntl.h>
#include <stdio.h>

#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_validation_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.h"

namespace tflite {
namespace benchmark {
namespace accuracy {
namespace {

class AccuracyBenchmarkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    acceleration::MiniBenchmarkTestHelper helper;
    should_perform_test_ = helper.should_perform_test();

    if (!should_perform_test_) {
      return;
    }
    std::string embedded_model_path = helper.DumpToTempFile(
        "mobilenet_quant_with_validation.tflite",
        g_tflite_acceleration_embedded_mobilenet_validation_model,
        g_tflite_acceleration_embedded_mobilenet_validation_model_len);
    ASSERT_TRUE(!embedded_model_path.empty());

    model_fp_ = fopen(embedded_model_path.c_str(), "rb");
    ASSERT_TRUE(model_fp_ != nullptr);
    ASSERT_EQ(fseek(model_fp_, 0, SEEK_END), 0);
    model_size_ = ftell(model_fp_);
    ASSERT_NE(model_size_, -1);
    ASSERT_EQ(fseek(model_fp_, 0, SEEK_SET), 0);

    result_path_ = ::testing::TempDir();
  }

  void TearDown() override { fclose(model_fp_); }

  std::string result_path_;
  size_t model_size_;
  FILE* model_fp_;
  bool should_perform_test_ = true;
};

TEST_F(AccuracyBenchmarkTest, FailedWithInvalidModelFileDescriptor) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }
  std::vector<std::string> args;

  EXPECT_EQ(Benchmark(args, 0, 0, 0, result_path_.c_str()),
            kAccuracyBenchmarkRunnerInitializationFailed);
}

TEST_F(AccuracyBenchmarkTest, FailedWithMissingArguments) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }
  std::vector<std::string> args = {"--other_arguments=other"};

  EXPECT_EQ(
      Benchmark(args, fileno(model_fp_), 0, model_size_, result_path_.c_str()),
      kAccuracyBenchmarkArgumentParsingFailed);
}

TEST_F(AccuracyBenchmarkTest, FailedWithInvalidTFliteSettings) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }
  std::vector<std::string> args = {
      "--stable_delegate_settings_file=third_party/tensorflow/lite/tools/"
      "delegates/experimental/stable_delegate/"
      "test_invalid_settings.json"};

  EXPECT_EQ(
      Benchmark(args, fileno(model_fp_), 0, model_size_, result_path_.c_str()),
      kAccuracyBenchmarkTfLiteSettingsParsingFailed);
}

TEST_F(AccuracyBenchmarkTest, SucceedWithSampleStableDelegate) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }
  std::vector<std::string> args = {
      "--stable_delegate_settings_file=third_party/tensorflow/lite/tools/"
      "delegates/experimental/stable_delegate/"
      "test_sample_stable_delegate_settings.json"};

  AccuracyBenchmarkStatus status =
      Benchmark(args, fileno(model_fp_), 0, model_size_, result_path_.c_str());

  // TODO(b/253442685): verify that stable delegate was used.
  EXPECT_EQ(status, kAccuracyBenchmarkPass);
}

TEST_F(AccuracyBenchmarkTest, SucceedWithEmbeddedValidationAndXNNPack) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }
  std::vector<std::string> args = {
      "--stable_delegate_settings_file=third_party/tensorflow/lite/delegates/"
      "utils/experimental/stable_delegate/test_xnnpack_settings.json"};

  AccuracyBenchmarkStatus status =
      Benchmark(args, fileno(model_fp_), 0, model_size_, result_path_.c_str());

  // TODO(b/253442685): verify that XNNPack was used.
  EXPECT_EQ(status, kAccuracyBenchmarkPass);
}

}  // namespace
}  // namespace accuracy
}  // namespace benchmark
}  // namespace tflite
