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
#include "tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/src/main/native/latency_benchmark.h"

#include <fcntl.h>

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/tflite_settings_json_parser.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/proto/delegate_performance.pb.h"

namespace tflite {
namespace benchmark {
namespace latency {
namespace {

static constexpr char kModelPath[] =
    "third_party/tensorflow/lite/java/demo/app/src/main/assets/"
    "mobilenet_v1_1.0_224.tflite";
static constexpr char kSettingsFilePath[] =
    "third_party/tensorflow/lite/tools/delegates/experimental/stable_delegate/"
    "test_sample_stable_delegate_settings.json";

class LatencyBenchmarkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_fp_ = fopen(kModelPath, "rb");
    ASSERT_TRUE(model_fp_ != nullptr);
    ASSERT_EQ(fseek(model_fp_, 0, SEEK_END), 0);
    model_size_ = ftell(model_fp_);
    ASSERT_NE(model_size_, -1);
    ASSERT_EQ(fseek(model_fp_, 0, SEEK_SET), 0);
    settings_ = parser_.Parse(kSettingsFilePath);
  }

  delegates::utils::TfLiteSettingsJsonParser parser_;
  const TFLiteSettings* settings_;
  size_t model_size_;
  FILE* model_fp_;
  std::vector<std::string> args_;
};

TEST_F(LatencyBenchmarkTest, FailedWithNullFileDescriptor) {
  EXPECT_TRUE(Benchmark(*settings_, kSettingsFilePath,
                        /*model_fd=*/0, /*model_offset=*/0,
                        /*model_size=*/0, args_)
                  .has_error());
}

TEST_F(LatencyBenchmarkTest, FailedWithInvalidNumThreadsSettings) {
  flatbuffers::FlatBufferBuilder fbb;
  flatbuffers::Offset<tflite::XNNPackSettings> xnnpack_settings =
      CreateXNNPackSettings(fbb, /*num_threads=*/-3);
  TFLiteSettingsBuilder tflite_settings_builder(fbb);
  tflite_settings_builder.add_delegate(Delegate_XNNPACK);
  tflite_settings_builder.add_xnnpack_settings(xnnpack_settings);
  fbb.Finish(tflite_settings_builder.Finish());
  const TFLiteSettings* settings =
      flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer());

  EXPECT_TRUE(Benchmark(*settings,
                        /*tflite_settings_path=*/"example_path",
                        fileno(model_fp_),
                        /*model_offset=*/0, model_size_, args_)
                  .has_error());
}

TEST_F(LatencyBenchmarkTest, SucceedWithEmptyTfLiteSettings) {
  flatbuffers::FlatBufferBuilder fbb;
  TFLiteSettingsBuilder tflite_settings_builder(fbb);
  fbb.Finish(tflite_settings_builder.Finish());
  const TFLiteSettings* settings =
      flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer());

  // TODO(b/253442685): verify that the default delegate was used.
  EXPECT_EQ(Benchmark(*settings, /*tflite_settings_path=*/"example_path",
                      fileno(model_fp_), /*model_offset=*/0, model_size_, args_)
                .event_type(),
            proto::benchmark::BENCHMARK_EVENT_TYPE_END);
}

TEST_F(LatencyBenchmarkTest, SucceedWithCpuTfLiteSettings) {
  flatbuffers::FlatBufferBuilder fbb;
  TFLiteSettingsBuilder tflite_settings_builder(fbb);
  tflite_settings_builder.add_disable_default_delegates(true);
  fbb.Finish(tflite_settings_builder.Finish());
  const TFLiteSettings* settings =
      flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer());

  // TODO(b/253442685): verify that no model delegation has occurred.
  EXPECT_EQ(Benchmark(*settings, /*tflite_settings_path=*/"example_path",
                      fileno(model_fp_), /*model_offset=*/0, model_size_, args_)
                .event_type(),
            proto::benchmark::BENCHMARK_EVENT_TYPE_END);
}

#ifdef __ANDROID__
TEST_F(LatencyBenchmarkTest, SucceedWithGpuTfLiteSettings) {
  flatbuffers::FlatBufferBuilder fbb;
  TFLiteSettingsBuilder tflite_settings_builder(fbb);
  tflite_settings_builder.add_delegate(Delegate_GPU);
  fbb.Finish(tflite_settings_builder.Finish());
  const TFLiteSettings* settings =
      flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer());

  // TODO(b/253442685): verify that the GPU delegate was used.
  EXPECT_EQ(Benchmark(*settings, /*tflite_settings_path=*/"example_path",
                      fileno(model_fp_), /*model_offset=*/0, model_size_, args_)
                .event_type(),
            proto::benchmark::BENCHMARK_EVENT_TYPE_END);
}
#endif  // __ANDROID__

TEST_F(LatencyBenchmarkTest, SucceedWithSampleStableDelegate) {
  // TODO(b/253442685): verify that the stable delegate was used.
  EXPECT_EQ(Benchmark(*settings_, kSettingsFilePath, fileno(model_fp_),
                      /*model_offset=*/0, model_size_, args_)
                .event_type(),
            proto::benchmark::BENCHMARK_EVENT_TYPE_END);
}

TEST_F(LatencyBenchmarkTest,
       SucceedWithSampleStableDelegateAndBenchmarkToolArguments) {
  std::vector<std::string> args = {"--warmup_runs=10"};

  // TODO(b/253442685): verify that the stable delegate was used.
  EXPECT_EQ(Benchmark(*settings_, kSettingsFilePath, fileno(model_fp_),
                      /*model_offset=*/0, model_size_, args)
                .event_type(),
            proto::benchmark::BENCHMARK_EVENT_TYPE_END);
}

}  // namespace
}  // namespace latency
}  // namespace benchmark
}  // namespace tflite
