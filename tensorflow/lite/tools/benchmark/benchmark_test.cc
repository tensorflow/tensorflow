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
#ifndef _WIN32
#include <fcntl.h>
#endif  // !defined(_WIN32)

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/algorithm.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/tools/benchmark/benchmark_performance_options.h"
#include "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/logging.h"

namespace {
const std::string* g_fp32_model_path = nullptr;
const std::string* g_int8_model_path = nullptr;
const std::string* g_string_model_path = nullptr;
const std::string* g_string_model_path_no_signature = nullptr;
const std::string* g_multi_signature_model_path = nullptr;
}  // namespace

namespace tflite {
namespace benchmark {
namespace {

enum class ModelGraphType { FP32, INT8, STRING };
enum class ModelReadOption { FROM_PATH, FROM_FD };

void InitializeParams(
    BenchmarkParams& params, int32_t num_runs, float min_secs, float max_secs,
    ModelReadOption model_read_option = ModelReadOption::FROM_PATH,
    ModelGraphType graph_type = ModelGraphType::FP32,
    absl::string_view signature_key = "",
    bool use_legacy_string_model = false) {
  params.Set<int32_t>("num_runs", num_runs);
  params.Set<float>("min_secs", min_secs);
  params.Set<float>("max_secs", max_secs);

  // by default, simply use the fp32 one.
  std::string graph_path = *g_fp32_model_path;
  if (graph_type == ModelGraphType::INT8) {
    graph_path = *g_int8_model_path;
  } else if (graph_type == ModelGraphType::STRING) {
    graph_path = use_legacy_string_model ? *g_string_model_path_no_signature
                                         : *g_string_model_path;
  } else if (!signature_key.empty()) {
    graph_path = *g_multi_signature_model_path;
  }
  std::string fd_or_graph_path = graph_path;
#ifndef _WIN32
  if (model_read_option == ModelReadOption::FROM_FD) {
    int fd = open(graph_path.c_str(), O_RDONLY);
    ASSERT_GE(fd, 0);
    struct stat stat_buf = {0};
    ASSERT_EQ(fstat(fd, &stat_buf), 0);
    size_t model_size = stat_buf.st_size;
    size_t model_offset = 0;
    fd_or_graph_path =
        absl::StrFormat("fd:%d:%zu:%zu", fd, model_offset, model_size);
  }
#endif  // !defined(_WIN32)
  params.Set<std::string>("graph", fd_or_graph_path);
  if (!signature_key.empty()) {
    params.Set<std::string>("signature_to_run_for", std::string(signature_key));
  }
}

BenchmarkParams InitializeParams() {
  BenchmarkParams params = BenchmarkTfLiteModel::DefaultParams();
  InitializeParams(params, /*num_runs=*/2, /*min_secs=*/1.0f,
                   /*max_secs=*/150.0f);
  return params;
}
BenchmarkParams CreateFp32Params() {
  BenchmarkParams params = BenchmarkTfLiteModel::DefaultParams();
  InitializeParams(
      params, /*num_runs=*/2, /*min_secs=*/1.0f, /*max_secs=*/150.0f,
      /*model_read_option=*/ModelReadOption::FROM_PATH, ModelGraphType::FP32);
  return params;
}
BenchmarkParams CreateInt8Params() {
  BenchmarkParams params = BenchmarkTfLiteModel::DefaultParams();
  InitializeParams(
      params, /*num_runs=*/2, /*min_secs=*/1.0f, /*max_secs=*/150.0f,
      /*model_read_option=*/ModelReadOption::FROM_PATH, ModelGraphType::INT8);
  return params;
}
BenchmarkParams CreateStringParams() {
  BenchmarkParams params = BenchmarkTfLiteModel::DefaultParams();
  InitializeParams(
      params, /*num_runs=*/2, /*min_secs=*/1.0f, /*max_secs=*/150.0f,
      /*model_read_option=*/ModelReadOption::FROM_PATH, ModelGraphType::STRING);
  return params;
}
BenchmarkParams CreateLegacyStringParams() {
  BenchmarkParams params = BenchmarkTfLiteModel::DefaultParams();
  InitializeParams(
      params, /*num_runs=*/2, /*min_secs=*/1.0f, /*max_secs=*/150.0f,
      /*model_read_option=*/ModelReadOption::FROM_PATH, ModelGraphType::STRING,
      /*signature_key=*/"", /*use_legacy_string_model=*/true);
  return params;
}
BenchmarkParams CreateStringFdParams() {
  BenchmarkParams params = BenchmarkTfLiteModel::DefaultParams();
  InitializeParams(
      params, /*num_runs=*/2, /*min_secs=*/1.0f, /*max_secs=*/150.0f,
      /*model_read_option=*/ModelReadOption::FROM_FD, ModelGraphType::STRING);
  return params;
}
BenchmarkParams CreateMultiSignatureParams(std::string signature_key) {
  BenchmarkParams params = BenchmarkTfLiteModel::DefaultParams();
  InitializeParams(
      params, /*num_runs=*/2, /*min_secs=*/1.0f, /*max_secs=*/150.0f,
      /*model_read_option=*/ModelReadOption::FROM_PATH, ModelGraphType::FP32,
      /*signature_key=*/signature_key);
  return params;
}

std::string CreateFilePath(const std::string& file_name) {
  const char* tmp_dir = getenv("TEST_TMPDIR");
  return std::string(tmp_dir ? tmp_dir : "./") + file_name;
}

void WriteInputLayerValueFile(const std::string& file_path,
                              ModelGraphType graph_type, int num_elements,
                              char file_value = 'a') {
  std::ofstream file(file_path);
  int bytes = 0;
  switch (graph_type) {
    case ModelGraphType::FP32:
      bytes = 4 * num_elements;
      break;
    case ModelGraphType::INT8:
      bytes = num_elements;
      break;
    default:
      LOG(WARNING) << absl::StrFormat(
          "ModelGraphType(enum_value:%d) is not known.", graph_type);
      LOG(WARNING) << "The size of the ModelGraphType will be 1 byte in tests.";
      bytes = num_elements;
      break;
  }
  std::vector<char> buffer(bytes, file_value);
  file.write(buffer.data(), bytes);
}

void CheckInputTensorValue(const TfLiteTensor* input_tensor,
                           char expected_value) {
  ASSERT_THAT(input_tensor, testing::NotNull());
  EXPECT_TRUE(std::all_of(
      input_tensor->data.raw, input_tensor->data.raw + input_tensor->bytes,
      [expected_value](char c) { return c == expected_value; }));
}

void CheckInputTensorValue(const TfLiteTensor* input_tensor,
                           int tensor_dim_index,
                           const std::string& expected_value) {
  StringRef tensor_value = GetString(input_tensor, tensor_dim_index);
  EXPECT_TRUE(absl::equal(tensor_value.str, tensor_value.str + tensor_value.len,
                          expected_value.c_str(),
                          expected_value.c_str() + expected_value.length()));
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

  const TfLiteTensor* GetInputTensor(int index) {
    return index >= interpreter_runner_->inputs().size()
               ? nullptr
               : interpreter_runner_->tensor(
                     interpreter_runner_->inputs()[index]);
  }
};

TEST(BenchmarkTest, DoesntCrashFp32Model) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());

  TestBenchmark benchmark(CreateFp32Params());
  benchmark.Run();
}

TEST(BenchmarkTest, DoesntCrashInt8Model) {
  ASSERT_THAT(g_int8_model_path, testing::NotNull());

  TestBenchmark benchmark(CreateInt8Params());
  benchmark.Run();
}

TEST(BenchmarkTest, DoesntCrashStringModel) {
  ASSERT_THAT(g_string_model_path, testing::NotNull());

  TestBenchmark benchmark(CreateStringParams());
  benchmark.Run();
}

TEST(BenchmarkTest, DoesntCrashStringLegacyModel) {
  ASSERT_THAT(g_string_model_path_no_signature, testing::NotNull());

  TestBenchmark benchmark(CreateLegacyStringParams());
  benchmark.Run();
}

#ifndef _WIN32
TEST(BenchmarkTest, DoesntCrashStringModelWithFd) {
  ASSERT_THAT(g_string_model_path, testing::NotNull());

  TestBenchmark benchmark(CreateStringFdParams());
  benchmark.Run();
}
#endif  // !defined(_WIN32)

TEST(BenchmarkTest, DoesntCrashMultiSignatureModel) {
  ASSERT_THAT(g_multi_signature_model_path, testing::NotNull());

  TestBenchmark benchmark(CreateMultiSignatureParams("add"));
  auto status = benchmark.Run();
  EXPECT_EQ(kTfLiteOk, status);

  TestBenchmark benchmark_sub(CreateMultiSignatureParams("sub"));
  auto status_sub = benchmark_sub.Run();
  EXPECT_EQ(kTfLiteOk, status_sub);
}

TEST(BenchmarkTest, MultiSignatureModelWithInvalidSignatureKeyFails) {
  ASSERT_THAT(g_multi_signature_model_path, testing::NotNull());

  TestBenchmark benchmark(CreateMultiSignatureParams("addisabbaba"));
  auto status = benchmark.Run();
  EXPECT_EQ(kTfLiteError, status);
}

TEST(BenchmarkTest, SplitInputLayerNameAndValueFile) {
  std::vector<std::string> input_layer_value_files = {
      "input:/tmp/input",
      "input::0:/tmp/input",
      "input::0::0:/tmp/input",
      "input::::0:/tmp::input",
  };
  std::vector<std::pair<std::string, std::string>> expected = {
      {"input", "/tmp/input"},
      {"input:0", "/tmp/input"},
      {"input:0:0", "/tmp/input"},
      {"input::0", "/tmp:input"},
  };
  std::pair<std::string, std::string> name_file_pair;
  for (int i = 0; i < input_layer_value_files.size(); ++i) {
    SplitInputLayerNameAndValueFile(input_layer_value_files[i], name_file_pair);
    EXPECT_EQ(name_file_pair.first, expected[i].first);
    EXPECT_EQ(name_file_pair.second, expected[i].second);
  }

  EXPECT_EQ(SplitInputLayerNameAndValueFile("a:b:c", name_file_pair),
            kTfLiteError);
  EXPECT_EQ(SplitInputLayerNameAndValueFile("abc", name_file_pair),
            kTfLiteError);
}

class TestMultiRunStatsRecorder : public MultiRunStatsRecorder {
 public:
  void OutputStats() override {
    MultiRunStatsRecorder::OutputStats();

    // Check results have been sorted according to avg. latency in increasing
    // order, and the incomplete runs are at the back of the results.
    double pre_avg_latency = -1e6;
    bool has_incomplete = false;  // ensure complete/incomplete are not mixed.
    for (const auto& result : results_) {
      const auto current_avg_latency = result.metrics.inference_time_us().avg();
      if (result.completed) {
        EXPECT_GE(current_avg_latency, pre_avg_latency);
        EXPECT_FALSE(has_incomplete);
      } else {
        EXPECT_EQ(0, result.metrics.inference_time_us().count());
        has_incomplete = true;
      }
      pre_avg_latency = current_avg_latency;
    }
  }
};

TEST(BenchmarkTest, DoesntCrashMultiPerfOptions) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());

  TestBenchmark benchmark(CreateFp32Params());
  BenchmarkPerformanceOptions all_options_benchmark(
      &benchmark, std::make_unique<TestMultiRunStatsRecorder>());
  all_options_benchmark.Run();
}

TEST(BenchmarkTest, DoesntCrashMultiPerfOptionsWithProfiling) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());

  BenchmarkParams params = CreateFp32Params();
  params.Set<bool>("enable_op_profiling", true);
  TestBenchmark benchmark(std::move(params));
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
  TestBenchmark benchmark(std::move(params));
  benchmark.Run();
}

TEST(BenchmarkTest, DoesntCrashWithExplicitInputInt8Model) {
  ASSERT_THAT(g_int8_model_path, testing::NotNull());

  // Note: the following input-related params are *specific* to model
  // 'g_int8_model_path' which is specified as
  // 'lite:testdata/add_quantized_int8.bin for the test.
  int a_min = 1;
  int a_max = 10;
  BenchmarkParams params = CreateInt8Params();
  params.Set<std::string>("input_layer", "a");
  params.Set<std::string>("input_layer_shape", "1,8,8,3");
  params.Set<std::string>("input_layer_value_range",
                          absl::StrFormat("a,%d,%d", a_min, a_max));
  TestBenchmark benchmark(std::move(params));
  benchmark.Run();

  auto input_tensor = benchmark.GetInputTensor(0);
  ASSERT_THAT(input_tensor, testing::NotNull());
  EXPECT_TRUE(std::all_of(
      input_tensor->data.raw, input_tensor->data.raw + input_tensor->bytes,
      [a_min, a_max](int i) { return a_min <= i && i <= a_max; }));
}

TEST(BenchmarkTest, DoesntCrashWithExplicitInputValueFilesFp32Model) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());
  char file_value_b = 'b';
  const std::string file_path_b = CreateFilePath("fp32_binary_b");
  WriteInputLayerValueFile(file_path_b, ModelGraphType::FP32, 192,
                           file_value_b);
  char file_value_d = 'd';
  const std::string file_path_d = CreateFilePath("fp32_binary_d");
  WriteInputLayerValueFile(file_path_d, ModelGraphType::FP32, 192,
                           file_value_d);

  // Note: the following input-related params are *specific* to model
  // 'g_fp32_model_path' which is specified as 'lite:testdata/multi_add.bin for
  // the test.
  BenchmarkParams params = CreateFp32Params();
  params.Set<std::string>("input_layer", "a,b,c,d");
  params.Set<std::string>("input_layer_shape",
                          "1,8,8,3:1,8,8,3:1,8,8,3:1,8,8,3");
  params.Set<std::string>("input_layer_value_files",
                          "d:" + file_path_d + ",b:" + file_path_b);
  TestBenchmark benchmark(std::move(params));
  benchmark.Run();

  CheckInputTensorValue(benchmark.GetInputTensor(1), file_value_b);
  CheckInputTensorValue(benchmark.GetInputTensor(3), file_value_d);
}

TEST(BenchmarkTest, DoesntCrashWithExplicitInputValueFilesInt8Model) {
  ASSERT_THAT(g_int8_model_path, testing::NotNull());
  const std::string file_path = CreateFilePath("int8_binary");
  char file_value = 'a';
  WriteInputLayerValueFile(file_path, ModelGraphType::INT8, 192, file_value);

  // Note: the following input-related params are *specific* to model
  // 'g_int8_model_path' which is specified as
  // 'lite:testdata/add_quantized_int8.bin for the test.
  BenchmarkParams params = CreateInt8Params();
  params.Set<std::string>("input_layer", "a");
  params.Set<std::string>("input_layer_shape", "1,8,8,3");
  params.Set<std::string>("input_layer_value_files", "a:" + file_path);
  TestBenchmark benchmark(std::move(params));
  benchmark.Run();

  CheckInputTensorValue(benchmark.GetInputTensor(0), file_value);
}

TEST(BenchmarkTest, DoesntCrashWithExplicitInputValueFilesMultiSignatureModel) {
  ASSERT_THAT(g_multi_signature_model_path, testing::NotNull());
  const std::string file_path_add =
      CreateFilePath("multi_signature_binary_add");
  char file_value_add = 'a';
  WriteInputLayerValueFile(file_path_add, ModelGraphType::FP32, 192,
                           file_value_add);

  // Note: the following input-related params are *specific* to model
  // 'g_multi_signature_model_path' which is specified as
  // 'lite:testdata/add_quantized_int8.bin for the test.
  BenchmarkParams params = CreateMultiSignatureParams("add");
  params.Set<std::string>("input_layer", "x");
  params.Set<std::string>("input_layer_shape", "192");
  params.Set<std::string>("input_layer_value_files", "x:" + file_path_add);
  TestBenchmark benchmark(std::move(params));
  benchmark.Run();

  CheckInputTensorValue(benchmark.GetInputTensor(0), file_value_add);

  const std::string file_path_sub =
      CreateFilePath("multi_signature_binary_sub");
  char file_value_sub = 'z';
  WriteInputLayerValueFile(file_path_sub, ModelGraphType::FP32, 192,
                           file_value_sub);
  // Note: the following input-related params are *specific* to model
  // 'g_multi_signature_model_path' which is specified as
  // 'lite:testdata/add_quantized_int8.bin for the test.
  BenchmarkParams params_2 = CreateMultiSignatureParams("sub");
  params_2.Set<std::string>("input_layer", "x");
  params_2.Set<std::string>("input_layer_shape", "192");
  params_2.Set<std::string>("input_layer_value_files", "x:" + file_path_sub);
  TestBenchmark benchmark_2(std::move(params_2));
  benchmark_2.Run();

  CheckInputTensorValue(benchmark_2.GetInputTensor(0), file_value_sub);
}

TEST(BenchmarkTest, DoesntCrashWithExplicitInputValueFilesStringModel) {
  ASSERT_THAT(g_string_model_path, testing::NotNull());
  const std::string file_path = CreateFilePath("string_binary");
  const std::string string_value_0 = "abcd";
  const std::string string_value_1 = "12345";
  const std::string string_value_2 = "a1b2c3d4e5";
  std::ofstream file(file_path);
  // Store the terminating null-character ('\0') at the end of the returned
  // value by std::string::c_str().
  file.write(string_value_0.c_str(), string_value_0.length() + 1);
  file.write(string_value_1.c_str(), string_value_1.length() + 1);
  file.write(string_value_2.c_str(), string_value_2.length() + 1);
  file.close();

  // Note: the following input-related params are *specific* to model
  // 'g_string_model_path' which is specified as
  // 'lite:testdata/string_input_model.bin for the test.
  BenchmarkParams params = CreateStringParams();
  params.Set<std::string>("input_layer", "a");
  params.Set<std::string>("input_layer_shape", "1,3");
  params.Set<std::string>("input_layer_value_files", "a:" + file_path);
  TestBenchmark benchmark(std::move(params));
  benchmark.Run();

  auto input_tensor = benchmark.GetInputTensor(0);
  ASSERT_THAT(input_tensor, testing::NotNull());
  EXPECT_EQ(GetStringCount(input_tensor), 3);
  CheckInputTensorValue(input_tensor, 0, string_value_0);
  CheckInputTensorValue(input_tensor, 1, string_value_1);
  CheckInputTensorValue(input_tensor, 2, string_value_2);
}

class ScopedCommandlineArgs {
 public:
  explicit ScopedCommandlineArgs(const std::vector<std::string>& actual_args) {
    argc_ = actual_args.size() + 1;
    argv_ = new char*[argc_];
    const std::string program_name = "benchmark_model";
    int buffer_size = program_name.length() + 1;
    for (const auto& arg : actual_args) buffer_size += arg.length() + 1;
    buffer_ = new char[buffer_size];
    auto next_start = program_name.copy(buffer_, program_name.length());
    buffer_[next_start++] = '\0';
    argv_[0] = buffer_;
    for (int i = 0; i < actual_args.size(); ++i) {
      const auto& arg = actual_args[i];
      argv_[i + 1] = buffer_ + next_start;
      next_start += arg.copy(argv_[i + 1], arg.length());
      buffer_[next_start++] = '\0';
    }
  }
  ~ScopedCommandlineArgs() {
    delete[] argv_;
    delete[] buffer_;
  }

  int argc() const { return argc_; }

  char** argv() const { return argv_; }

 private:
  char* buffer_;  // the buffer for all arguments.
  int argc_;
  char** argv_;  // Each char* element points to each argument.
};

TEST(BenchmarkTest, RunWithCorrectFlags) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());
  TestBenchmark benchmark(CreateFp32Params());
  ScopedCommandlineArgs scoped_argv({"--num_threads=4"});
  auto status = benchmark.Run(scoped_argv.argc(), scoped_argv.argv());
  EXPECT_EQ(kTfLiteOk, status);
}

TEST(BenchmarkTest, RunWithWrongFlags) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());
  TestBenchmark benchmark(CreateFp32Params());
  ScopedCommandlineArgs scoped_argv({"--num_threads=str"});
  auto status = benchmark.Run(scoped_argv.argc(), scoped_argv.argv());
  EXPECT_EQ(kTfLiteError, status);
}

TEST(BenchmarkTest, RunWithUseCaching) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());
  TestBenchmark benchmark(CreateFp32Params());
  ScopedCommandlineArgs scoped_argv({"--use_caching=false"});
  auto status = benchmark.Run(scoped_argv.argc(), scoped_argv.argv());
  EXPECT_EQ(kTfLiteOk, status);
}

class MaxDurationWorksTestListener : public BenchmarkListener {
  void OnBenchmarkEnd(const BenchmarkResults& results) override {
    const int64_t num_actual_runs = results.inference_time_us().count();
    TFLITE_LOG(INFO) << "number of actual runs: " << num_actual_runs;
    EXPECT_GE(num_actual_runs, 1);
    EXPECT_LT(num_actual_runs, 100000000);
  }
};

TEST(BenchmarkTest, MaxDurationWorks) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());
  BenchmarkParams params = BenchmarkTfLiteModel::DefaultParams();
  InitializeParams(params, 100000000 /* num_runs */, 1000000.0f /* min_secs */,
                   0.001f /* max_secs */);
  TestBenchmark benchmark(std::move(params));
  MaxDurationWorksTestListener listener;
  benchmark.AddListener(&listener);
  benchmark.Run();
}

TEST(BenchmarkTest, ParametersArePopulatedWhenInputShapeIsNotSpecified) {
  ASSERT_THAT(g_fp32_model_path, testing::NotNull());

  TestBenchmark benchmark(InitializeParams());
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

TEST(BenchmarkTest, InitializationFailedWhenInvalidGraphPathIsProvided) {
  BenchmarkParams params = BenchmarkTfLiteModel::DefaultParams();
  params.Set<std::string>("graph", "invalid/path");

  TestBenchmark benchmark(std::move(params));

  EXPECT_EQ(benchmark.Init(), kTfLiteError);
}

TEST(BenchmarkTest, InitializationFailedWhenInvalidGraphFdIsProvided) {
  BenchmarkParams params = BenchmarkTfLiteModel::DefaultParams();
  params.Set<std::string>("graph", "fd:file:descriptor");

  TestBenchmark benchmark(std::move(params));

  EXPECT_EQ(benchmark.Init(), kTfLiteError);
}

}  // namespace
}  // namespace benchmark
}  // namespace tflite

int main(int argc, char** argv) {
  std::string fp32_model_path, int8_model_path, string_model_path,
      string_model_path_with_no_signature, multi_signature_model_path;
  std::vector<tflite::Flag> flags = {
      tflite::Flag::CreateFlag("fp32_graph", &fp32_model_path,
                               "Path to a fp32 model file."),
      tflite::Flag::CreateFlag("int8_graph", &int8_model_path,
                               "Path to a int8 model file."),
      tflite::Flag::CreateFlag("string_graph_with_signature",
                               &string_model_path,
                               "Path to a string model file with a signature."),
      tflite::Flag::CreateFlag(
          "string_graph_without_signature",
          &string_model_path_with_no_signature,
          "Path to a string model file without signatures."),
      tflite::Flag::CreateFlag("multi_signature_graph",
                               &multi_signature_model_path,
                               "Path to a multi-signature model file."),
  };

  g_fp32_model_path = &fp32_model_path;
  g_int8_model_path = &int8_model_path;
  g_string_model_path = &string_model_path;
  g_multi_signature_model_path = &multi_signature_model_path;
  g_string_model_path_no_signature = &string_model_path_with_no_signature;

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
