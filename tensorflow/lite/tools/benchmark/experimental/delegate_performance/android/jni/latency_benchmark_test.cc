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
#include "tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/jni/latency_benchmark.h"

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

static constexpr char kGraphArgument[] =
    "--graph=third_party/tensorflow/lite/java/demo/app/src/main/assets/"
    "mobilenet_v1_1.0_224.tflite";
static constexpr char kSettingsFilePath[] =
    "third_party/tensorflow/lite/tools/delegates/experimental/stable_delegate/"
    "test_sample_stable_delegate_settings.json";

TEST(LatencyBenchmarkTest, FailedWithMissingModel) {
  delegates::utils::TfLiteSettingsJsonParser parser;
  std::vector<std::string> args = {"--other_arguments=other"};

  EXPECT_TRUE(
      Benchmark(args, *parser.Parse(kSettingsFilePath), kSettingsFilePath)
          .has_error());
}

TEST(LatencyBenchmarkTest, FailedWithInvalidNumThreadsSettings) {
  std::vector<std::string> args = {kGraphArgument};
  flatbuffers::FlatBufferBuilder fbb;
  flatbuffers::Offset<tflite::XNNPackSettings> xnnpack_settings =
      CreateXNNPackSettings(fbb, /*num_threads=*/-3);
  TFLiteSettingsBuilder tflite_settings_builder(fbb);
  tflite_settings_builder.add_delegate(Delegate_XNNPACK);
  tflite_settings_builder.add_xnnpack_settings(xnnpack_settings);
  fbb.Finish(tflite_settings_builder.Finish());
  const TFLiteSettings* settings =
      flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer());

  EXPECT_TRUE(Benchmark(args, *settings,
                        /*tflite_settings_path=*/"example_path")
                  .has_error());
}

TEST(LatencyBenchmarkTest, SucceedWithSampleStableDelegate) {
  std::vector<std::string> args = {kGraphArgument};
  delegates::utils::TfLiteSettingsJsonParser parser;

  // TODO(b/253442685): verify that stable delegate was used.
  EXPECT_EQ(Benchmark(args, *parser.Parse(kSettingsFilePath), kSettingsFilePath)
                .event_type(),
            proto::benchmark::BENCHMARK_EVENT_TYPE_END);
}

}  // namespace
}  // namespace latency
}  // namespace benchmark
}  // namespace tflite
