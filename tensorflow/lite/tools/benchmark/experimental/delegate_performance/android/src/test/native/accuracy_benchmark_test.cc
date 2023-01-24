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
#include "tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/src/main/native/accuracy_benchmark.h"

#include <fcntl.h>
#include <stdio.h>

#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/tflite_settings_json_parser.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_validation_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.h"
#include "tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/src/main/native/status_codes.h"

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
    ASSERT_FALSE(embedded_model_path.empty());

    model_fp_ = fopen(embedded_model_path.c_str(), "rb");
    ASSERT_NE(model_fp_, nullptr);
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
  delegates::utils::TfLiteSettingsJsonParser parser;
  flatbuffers::FlatBufferBuilder builder;
  std::vector<std::string> args;
  const TFLiteSettings* tflite_settings = parser.Parse(
      "third_party/tensorflow/lite/tools/delegates/experimental/"
      "stable_delegate/test_sample_stable_delegate_settings.json");

  flatbuffers::Offset<BenchmarkEvent> offset =
      Benchmark(builder, *tflite_settings, /*model_fd=*/0,
                /*model_offset=*/0, /*model_size=*/0, result_path_.c_str());
  builder.Finish(offset);
  const BenchmarkEvent* event =
      flatbuffers::GetRoot<BenchmarkEvent>(builder.GetBufferPointer());

  ASSERT_NE(event, nullptr);
  EXPECT_EQ(event->event_type(), BenchmarkEventType_ERROR);
  ASSERT_NE(event->error(), nullptr);
  EXPECT_EQ(event->error()->stage(), BenchmarkStage_INITIALIZATION);
  EXPECT_EQ(
      event->error()->exit_code(),
      DelegatePerformanceBenchmarkStatus::kBenchmarkRunnerInitializationFailed);
}

TEST_F(AccuracyBenchmarkTest, SucceedWithSampleStableDelegate) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }
  delegates::utils::TfLiteSettingsJsonParser parser;
  flatbuffers::FlatBufferBuilder builder;
  const TFLiteSettings* tflite_settings = parser.Parse(
      "third_party/tensorflow/lite/tools/delegates/experimental/"
      "stable_delegate/test_sample_stable_delegate_settings.json");

  flatbuffers::Offset<BenchmarkEvent> offset = Benchmark(
      builder, *tflite_settings, /*model_fd=*/fileno(model_fp_),
      /*model_offset=*/0, /*model_size=*/model_size_, result_path_.c_str());
  builder.Finish(offset);
  const BenchmarkEvent* event =
      flatbuffers::GetRoot<BenchmarkEvent>(builder.GetBufferPointer());

  // TODO(b/253442685): verify that the stable delegate was used.
  ASSERT_NE(event, nullptr);
  EXPECT_EQ(event->event_type(), BenchmarkEventType_END);
  EXPECT_EQ(event->error(), nullptr);
}

TEST_F(AccuracyBenchmarkTest, SucceedWithEmbeddedValidationAndXNNPack) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }
  delegates::utils::TfLiteSettingsJsonParser parser;
  flatbuffers::FlatBufferBuilder builder;
  const TFLiteSettings* tflite_settings = parser.Parse(
      "third_party/tensorflow/lite/delegates/utils/experimental/"
      "stable_delegate/test_xnnpack_settings.json");

  flatbuffers::Offset<BenchmarkEvent> offset = Benchmark(
      builder, *tflite_settings, /*model_fd=*/fileno(model_fp_),
      /*model_offset=*/0, /*model_size=*/model_size_, result_path_.c_str());
  builder.Finish(offset);
  const BenchmarkEvent* event =
      flatbuffers::GetRoot<BenchmarkEvent>(builder.GetBufferPointer());

  // TODO(b/253442685): verify that the XNNPack delegate was used.
  ASSERT_NE(event, nullptr);
  EXPECT_EQ(event->event_type(), BenchmarkEventType_END);
  EXPECT_EQ(event->error(), nullptr);
}

}  // namespace
}  // namespace accuracy
}  // namespace benchmark
}  // namespace tflite
