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
#include "absl/strings/str_cat.h"
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
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
using ::flatbuffers::GetRoot;

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
    options_.model_path = helper.DumpToTempFile(
        "mobilenet_quant_with_validation.tflite",
        g_tflite_acceleration_embedded_mobilenet_validation_model,
        g_tflite_acceleration_embedded_mobilenet_validation_model_len);
    ASSERT_TRUE(!options_.model_path.empty());

    options_.data_directory_path = ::testing::TempDir();
    options_.storage_path =
        absl::StrCat(::testing::TempDir(), "storage_path.fb.1");
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

  std::vector<FlatBufferBuilder> results = runner.TriggerValidation(
      {flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer())});
  EXPECT_THAT(results, testing::Not(testing::IsEmpty()));
  for (auto& result : results) {
    const BenchmarkEvent* event =
        GetRoot<BenchmarkEvent>(result.GetBufferPointer());
    EXPECT_EQ(event->event_type(), BenchmarkEventType_END);
    EXPECT_TRUE(event->result()->ok());
  }
}

TEST_F(BlockingValidatorRunnerTest, SucceedWithFdCloexecEmbeddedValidation) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }

  options_.model_fd = open(options_.model_path.c_str(), O_RDONLY | O_CLOEXEC);
  ASSERT_GE(options_.model_fd, 0);
  struct stat stat_buf = {0};
  ASSERT_EQ(fstat(options_.model_fd, &stat_buf), 0);
  options_.model_size = stat_buf.st_size;
  options_.model_offset = 0;
  options_.model_path.clear();

  BlockingValidatorRunner runner(options_);
  ASSERT_EQ(runner.Init(), kMinibenchmarkSuccess);
  FlatBufferBuilder fbb;
#ifdef __ANDROID__
  fbb.Finish(CreateTFLiteSettings(fbb, Delegate_GPU));
#else
  fbb.Finish(CreateTFLiteSettings(fbb));
#endif  // __ANDROID__

  std::vector<FlatBufferBuilder> results = runner.TriggerValidation(
      {flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer())});
  EXPECT_THAT(results, testing::Not(testing::IsEmpty()));
  for (auto& result : results) {
    const BenchmarkEvent* event =
        GetRoot<BenchmarkEvent>(result.GetBufferPointer());
    EXPECT_EQ(event->event_type(), BenchmarkEventType_END);
    EXPECT_TRUE(event->result()->ok());
  }
}

TEST_F(BlockingValidatorRunnerTest, SucceedWithBufferModel) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }

  options_.model_buffer =
      g_tflite_acceleration_embedded_mobilenet_validation_model;
  options_.model_size =
      g_tflite_acceleration_embedded_mobilenet_validation_model_len;
  options_.model_path.clear();

  BlockingValidatorRunner runner(options_);
  ASSERT_EQ(runner.Init(), kMinibenchmarkSuccess);
  FlatBufferBuilder fbb;
  fbb.Finish(CreateTFLiteSettings(fbb));

  std::vector<FlatBufferBuilder> results = runner.TriggerValidation(
      {flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer())});
  EXPECT_THAT(results, testing::Not(testing::IsEmpty()));
  for (auto& result : results) {
    const BenchmarkEvent* event =
        GetRoot<BenchmarkEvent>(result.GetBufferPointer());
    EXPECT_EQ(event->event_type(), BenchmarkEventType_END);
    EXPECT_TRUE(event->result()->ok());
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

  std::vector<FlatBufferBuilder> results = runner.TriggerValidation(
      {flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer())});
  EXPECT_THAT(results, testing::Not(testing::IsEmpty()));
  for (auto& result : results) {
    const BenchmarkEvent* event =
        GetRoot<BenchmarkEvent>(result.GetBufferPointer());
    EXPECT_EQ(event->event_type(), BenchmarkEventType_END);
  }
}

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
    std::vector<FlatBufferBuilder> results = runner.TriggerValidation(
        {flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer()),
         flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer())});
    EXPECT_THAT(results, testing::Not(testing::IsEmpty()));
    for (auto& result : results) {
      const BenchmarkEvent* event =
          GetRoot<BenchmarkEvent>(result.GetBufferPointer());
      EXPECT_EQ(event->event_type(), BenchmarkEventType_END);
      EXPECT_TRUE(event->result()->ok());
    }
  }
}

TEST_F(BlockingValidatorRunnerTest, ReturnErrorWhenTimedOut) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }
  options_.per_test_timeout_ms = 50;
  BlockingValidatorRunner runner(options_);
  ASSERT_EQ(runner.Init(), kMinibenchmarkSuccess);
  FlatBufferBuilder fbb;
  fbb.Finish(CreateTFLiteSettings(fbb));

  std::vector<FlatBufferBuilder> results = runner.TriggerValidation(
      {flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer())});
  EXPECT_THAT(results, testing::SizeIs(1));
  for (auto& result : results) {
    const BenchmarkEvent* event =
        GetRoot<BenchmarkEvent>(result.GetBufferPointer());
    EXPECT_EQ(event->event_type(), BenchmarkEventType_ERROR);
    ASSERT_NE(nullptr, event->error());
    // The timeout can result in two different behaviors:
    // 1. The popen() subprocess got killed by the detached thread because the
    // timeout has reached, and the thread wrote error code
    // kMinibenchmarkCommandTimedOut, or
    // 2. The thread didn't respond the main process in time, and the main
    // process returned after the timeout, with error code
    // kMinibenchmarkCompletionEventMissing.
    EXPECT_THAT(event->error()->mini_benchmark_error_code(),
                testing::AnyOf(kMinibenchmarkCommandTimedOut,
                               kMinibenchmarkCompletionEventMissing));
  }
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite
