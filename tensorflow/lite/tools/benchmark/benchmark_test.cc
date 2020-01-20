/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <iostream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/algorithm.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/tools/benchmark/benchmark_performance_options.h"
#include "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"
#include "tensorflow/lite/tools/command_line_flags.h"

namespace {
const std::string* g_fp32_model_path = nullptr;
const std::string* g_int8_model_path = nullptr;
}  // namespace

namespace tflite {
namespace benchmark {
namespace {

enum class ModelGraphType { FP32, INT8 };

BenchmarkParams CreateParams(int32_t num_runs, float min_secs, float max_secs,
                             ModelGraphType graph_type = ModelGraphType::FP32) {
  BenchmarkParams params;
  params.AddParam("num_runs", BenchmarkParam::Create<int32_t>(num_runs));
  params.AddParam("min_secs", BenchmarkParam::Create<float>(min_secs));
  params.AddParam("max_secs", BenchmarkParam::Create<float>(max_secs));
  params.AddParam("run_delay", BenchmarkParam::Create<float>(-1.0f));
  params.AddParam("num_threads", BenchmarkParam::Create<int32_t>(1));
  params.AddParam("benchmark_name", BenchmarkParam::Create<std::string>(""));
  params.AddParam("output_prefix", BenchmarkParam::Create<std::string>(""));
  params.AddParam("warmup_runs", BenchmarkParam::Create<int32_t>(1));

  if (graph_type == ModelGraphType::INT8) {
    params.AddParam("graph",
                    BenchmarkParam::Create<std::string>(*g_int8_model_path));
  } else {
    // by default, simply use the fp32 one.
    params.AddParam("graph",
                    BenchmarkParam::Create<std::string>(*g_fp32_model_path));
  }

  params.AddParam("input_layer", BenchmarkParam::Create<std::string>(""));
  params.AddParam("input_layer_shape", BenchmarkParam::Create<std::string>(""));
  params.AddParam("input_layer_value_range",
                  BenchmarkParam::Create<std::string>(""));
  params.AddParam("use_hexagon", BenchmarkParam::Create<bool>(false));
  params.AddParam("use_nnapi", BenchmarkParam::Create<bool>(false));
  params.AddParam("allow_fp16", BenchmarkParam::Create<bool>(false));
  params.AddParam("require_full_delegation",
                  BenchmarkParam::Create<bool>(false));
  params.AddParam("warmup_min_secs", BenchmarkParam::Create<float>(0.5f));
  params.AddParam("use_legacy_nnapi", BenchmarkParam::Create<bool>(false));
  params.AddParam("use_gpu", BenchmarkParam::Create<bool>(false));
  params.AddParam("enable_op_profiling", BenchmarkParam::Create<bool>(false));
  params.AddParam("max_profiling_buffer_entries",
                  BenchmarkParam::Create<int32_t>(1024));
  params.AddParam("nnapi_accelerator_name",
                  BenchmarkParam::Create<std::string>(""));
  params.AddParam("nnapi_execution_preference",
                  BenchmarkParam::Create<std::string>(""));
  return params;
}

BenchmarkParams CreateParams() { return CreateParams(2, 1.0f, 150.0f); }
BenchmarkParams CreateFp32Params() {
  return CreateParams(2, 1.0f, 150.0f, ModelGraphType::FP32);
}
BenchmarkParams CreateInt8Params() {
  return CreateParams(2, 1.0f, 150.0f, ModelGraphType::INT8);
}

class TestBenchmark : public BenchmarkTfLiteModel {
 public:
  explicit TestBenchmark(BenchmarkParams params)
      : BenchmarkTfLiteModel(std::move(params)) {}
  const tflite::Interpreter* GetInterpreter() { return interpreter_.get(); }

  void Prepare() {
    PrepareInputData();
    ResetInputsAndOutputs();
  }
};

TEST(BenchmarkTest, DoesntCrashFp32Model) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());

  BenchmarkTfLiteModel benchmark(CreateFp32Params());
  benchmark.Run();
}

TEST(BenchmarkTest, DoesntCrashInt8Model) {
  ASSERT_THAT(g_int8_model_path, testing::NotNull());

  BenchmarkTfLiteModel benchmark(CreateInt8Params());
  benchmark.Run();
}

TEST(BenchmarkTest, DoesntCrashMultiPerfOptions) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());

  BenchmarkTfLiteModel benchmark(CreateFp32Params());
  BenchmarkPerformanceOptions all_options_benchmark(&benchmark);
  all_options_benchmark.Run();
}

TEST(BenchmarkTest, DoesntCrashMultiPerfOptionsWithProfiling) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());

  BenchmarkParams params = CreateFp32Params();
  params.Set<bool>("enable_op_profiling", true);
  BenchmarkTfLiteModel benchmark(std::move(params));
  BenchmarkPerformanceOptions all_options_benchmark(&benchmark);
  all_options_benchmark.Run();
}

TEST(BenchmarkTest, DoesntCrashWithExplicitInputFp32Model) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());

  // Note: the following input-related params are *specific* to model
  // 'g_fp32_model_path' which is specified as 'lite:testdata/multi_add.bin for
  // the test.
  BenchmarkParams params = CreateFp32Params();
  params.Set<std::string>("input_layer", "a,b,c,d");
  params.Set<std::string>("input_layer_shape",
                          "1,8,8,3:1,8,8,3:1,8,8,3:1,8,8,3");
  params.Set<std::string>("input_layer_value_range", "d,1,10:b,0,100");
  BenchmarkTfLiteModel benchmark(std::move(params));
  benchmark.Run();
}

TEST(BenchmarkTest, DoesntCrashWithExplicitInputInt8Model) {
  ASSERT_THAT(g_int8_model_path, testing::NotNull());

  // Note: the following input-related params are *specific* to model
  // 'g_int8_model_path' which is specified as
  // 'lite:testdata/add_quantized_int8.bin for the test.
  BenchmarkParams params = CreateInt8Params();
  params.Set<std::string>("input_layer", "a");
  params.Set<std::string>("input_layer_shape", "1,8,8,3");
  params.Set<std::string>("input_layer_value_range", "a,1,10");
  BenchmarkTfLiteModel benchmark(std::move(params));
  benchmark.Run();
}

class MaxDurationWorksTestListener : public BenchmarkListener {
  void OnBenchmarkEnd(const BenchmarkResults& results) override {
    const int64_t num_actul_runs = results.inference_time_us().count();
    TFLITE_LOG(INFO) << "number of actual runs: " << num_actul_runs;
    EXPECT_GE(num_actul_runs, 1);
    EXPECT_LT(num_actul_runs, 100000000);
  }
};

TEST(BenchmarkTest, MaxDurationWorks) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());
  BenchmarkTfLiteModel benchmark(CreateParams(100000000 /* num_runs */,
                                              1000000.0f /* min_secs */,
                                              0.001f /* max_secs */));
  MaxDurationWorksTestListener listener;
  benchmark.AddListener(&listener);
  benchmark.Run();
}

TEST(BenchmarkTest, ParametersArePopulatedWhenInputShapeIsNotSpecified) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());

  TestBenchmark benchmark(CreateParams());
  benchmark.Init();
  benchmark.Prepare();

  auto interpreter = benchmark.GetInterpreter();
  auto inputs = interpreter->inputs();
  ASSERT_GE(inputs.size(), 1);
  auto input_tensor = interpreter->tensor(inputs[0]);

  // Copy input tensor to a vector
  std::vector<char> input_bytes(input_tensor->data.raw,
                                input_tensor->data.raw + input_tensor->bytes);

  benchmark.Prepare();

  // Expect data is not the same.
  EXPECT_EQ(input_bytes.size(), input_tensor->bytes);
  EXPECT_FALSE(absl::equal(input_bytes.begin(), input_bytes.end(),
                           input_tensor->data.raw,
                           input_tensor->data.raw + input_tensor->bytes));
}

}  // namespace
}  // namespace benchmark
}  // namespace tflite

int main(int argc, char** argv) {
  std::string fp32_model_path, int8_model_path;
  std::vector<tflite::Flag> flags = {
      tflite::Flag::CreateFlag("fp32_graph", &fp32_model_path,
                               "Path to a fp32 model file."),
      tflite::Flag::CreateFlag("int8_graph", &int8_model_path,
                               "Path to a int8 model file."),
  };

  g_fp32_model_path = &fp32_model_path;
  g_int8_model_path = &int8_model_path;

  const bool parse_result =
      tflite::Flags::Parse(&argc, const_cast<const char**>(argv), flags);
  if (!parse_result) {
    std::cerr << tflite::Flags::Usage(argv[0], flags);
    return 1;
  }

  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
