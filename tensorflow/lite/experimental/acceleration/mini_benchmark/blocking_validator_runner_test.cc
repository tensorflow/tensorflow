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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/blocking_validator_runner.h"

#include <fcntl.h>

#include <iostream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/benchmark_result_evaluator.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_validation_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_options.h"

namespace tflite {
namespace acceleration {
namespace {

using ::flatbuffers::FlatBufferBuilder;

class CustomResultEvaluator : public AbstractBenchmarkResultEvaluator {
 public:
  bool HasPassedAccuracyCheck(const BenchmarkResult& result) override {
    return true;
  }
};

class BlockingValidatorRunnerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    MiniBenchmarkTestHelper helper;
    should_perform_test_ = helper.should_perform_test();

    if (!should_perform_test_) {
      return;
    }
    options_.model_path = helper.DumpToTempFile(
        "mobilenet_quant_with_validation.tflite",
        g_tflite_acceleration_embedded_mobilenet_validation_model,
        g_tflite_acceleration_embedded_mobilenet_validation_model_len);
    ASSERT_TRUE(!options_.model_path.empty());

    options_.data_directory_path = ::testing::TempDir();
    options_.storage_path = ::testing::TempDir() + "/storage_path.fb";
    (void)unlink(options_.storage_path.c_str());
    options_.per_test_timeout_ms = 5000;

    plain_model_path_ = MiniBenchmarkTestHelper::DumpToTempFile(
        "mobilenet_quant.tflite",
        g_tflite_acceleration_embedded_mobilenet_model,
        g_tflite_acceleration_embedded_mobilenet_model_len);
  }

  std::string plain_model_path_;
  ValidatorRunnerOptions options_;
  bool should_perform_test_ = true;
};

TEST_F(BlockingValidatorRunnerTest, SucceedWithEmbeddedValidation) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }

  BlockingValidatorRunner runner(options_);
  ASSERT_EQ(runner.Init(), kMinibenchmarkSuccess);
  FlatBufferBuilder fbb;
#ifdef __ANDROID__
  fbb.Finish(CreateTFLiteSettings(fbb, Delegate_GPU));
#else
  fbb.Finish(CreateTFLiteSettings(fbb));
#endif  // __ANDROID__

  std::vector<const BenchmarkEvent*> results = runner.TriggerValidation(
      {flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer())});
  EXPECT_THAT(results, testing::Not(testing::IsEmpty()));
  for (auto& result : results) {
    EXPECT_EQ(result->event_type(), BenchmarkEventType_END);
    EXPECT_TRUE(result->result()->ok());
  }
}

TEST_F(BlockingValidatorRunnerTest, SucceedWithFdModelCustomValidation) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }

  options_.model_path.clear();
  options_.model_fd = open(plain_model_path_.c_str(), O_RDONLY);
  ASSERT_GE(options_.model_fd, 0);
  struct stat stat_buf = {0};
  ASSERT_EQ(fstat(options_.model_fd, &stat_buf), 0);
  options_.model_size = stat_buf.st_size;
  options_.model_offset = 0;
  options_.custom_input_batch_size = 3;
  options_.custom_input_data = {std::vector<uint8_t>(3 * 224 * 224 * 3, 1)};
  CustomResultEvaluator evaluator;
  options_.benchmark_result_evaluator = &evaluator;

  BlockingValidatorRunner runner(options_);
  ASSERT_EQ(runner.Init(), kMinibenchmarkSuccess);
  FlatBufferBuilder fbb;
#ifdef __ANDROID__
  fbb.Finish(CreateTFLiteSettings(fbb, Delegate_XNNPACK));
#else
  fbb.Finish(CreateTFLiteSettings(fbb));
#endif  // __ANDROID__

  std::vector<const BenchmarkEvent*> results = runner.TriggerValidation(
      {flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer())});
  EXPECT_THAT(results, testing::Not(testing::IsEmpty()));
  for (auto& result : results) {
    EXPECT_EQ(result->event_type(), BenchmarkEventType_END);
  }
}
#ifndef __ANDROID__
TEST_F(BlockingValidatorRunnerTest, SucceedWhenRunningMultipleTimes) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }

  BlockingValidatorRunner runner(options_);
  ASSERT_EQ(runner.Init(), kMinibenchmarkSuccess);
  FlatBufferBuilder fbb;
  fbb.Finish(CreateTFLiteSettings(fbb));

  int num_runs = 3;
  for (int i = 0; i < num_runs; i++) {
    std::vector<const BenchmarkEvent*> results = runner.TriggerValidation(
        {flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer())});
    EXPECT_THAT(results, testing::Not(testing::IsEmpty()));
    for (auto& result : results) {
      EXPECT_EQ(result->event_type(), BenchmarkEventType_END);
      EXPECT_TRUE(result->result()->ok());
    }
  }
}
#endif  // !__ANDROID__

TEST_F(BlockingValidatorRunnerTest, ReturnEmptyWhenTimedOut) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }

  options_.per_test_timeout_ms = 100;
  BlockingValidatorRunner runner(options_);
  ASSERT_EQ(runner.Init(), kMinibenchmarkSuccess);
  FlatBufferBuilder fbb;
  fbb.Finish(CreateTFLiteSettings(fbb));

  std::vector<const BenchmarkEvent*> results = runner.TriggerValidation(
      {flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer())});
  EXPECT_THAT(results, testing::IsEmpty());
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite
