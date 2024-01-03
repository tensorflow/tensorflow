/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"

#include <fcntl.h>
#include <sys/stat.h>

#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/tools/benchmark/benchmark_model.h"
#include "tensorflow/lite/tools/benchmark/benchmark_params.h"
#include "tensorflow/lite/tools/tool_params.h"

namespace tflite {
namespace benchmark {
namespace {

static constexpr char kModelPath[] =
    "../tflite_mobilenet_float/"
    "mobilenet_v1_1.0_224.tflite";

class TestBenchmarkListener : public BenchmarkListener {
 public:
  void OnBenchmarkEnd(const BenchmarkResults& results) override {
    results_ = results;
  }

  BenchmarkResults results_;
};

TEST(BenchmarkTfLiteModelTest, GetModelSizeFromPathSucceeded) {
  BenchmarkParams params = BenchmarkTfLiteModel::DefaultParams();
  params.Set<std::string>("graph", kModelPath);
  params.Set<int>("num_runs", 1);
  params.Set<int>("warmup_runs", 0);
  BenchmarkTfLiteModel benchmark = BenchmarkTfLiteModel(std::move(params));
  TestBenchmarkListener listener;
  benchmark.AddListener(&listener);

  benchmark.Run();

  EXPECT_GE(listener.results_.model_size_mb(), 0);
}

TEST(BenchmarkTfLiteModelTest, GetModelSizeFromFileDescriptorSucceeded) {
  BenchmarkParams params = BenchmarkTfLiteModel::DefaultParams();
  int fd = open(kModelPath, O_RDONLY);
  ASSERT_GE(fd, 0);
  int model_offset = 0;
  struct stat stat_buf = {0};
  ASSERT_EQ(fstat(fd, &stat_buf), 0);
  params.Set<std::string>("graph", absl::StrCat("fd:", fd, ":", model_offset,
                                                ":", stat_buf.st_size));
  params.Set<int>("num_runs", 1);
  params.Set<int>("warmup_runs", 0);
  BenchmarkTfLiteModel benchmark = BenchmarkTfLiteModel(std::move(params));
  TestBenchmarkListener listener;
  benchmark.AddListener(&listener);

  benchmark.Run();

  EXPECT_EQ(listener.results_.model_size_mb(), stat_buf.st_size / 1e6);
}

TEST(BenchmarkTfLiteModelTest, ResizeInputWithDelegate) {
  BenchmarkParams params = BenchmarkTfLiteModel::DefaultParams();
  params.Set<std::string>("graph", kModelPath);
  params.Set<bool>("use_xnnpack", true);
  params.Set<std::string>("input_layer", "input_87");
  params.Set<std::string>("input_layer_shape", "2,224,224,3");

  BenchmarkTfLiteModel benchmark = BenchmarkTfLiteModel(std::move(params));

  EXPECT_EQ(benchmark.Run(), kTfLiteOk);
}

}  // namespace
}  // namespace benchmark
}  // namespace tflite
