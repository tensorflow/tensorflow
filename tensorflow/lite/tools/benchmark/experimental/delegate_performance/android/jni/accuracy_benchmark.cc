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

#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>

#include <cstddef>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/blocking_validator_runner.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_options.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/tool_params.h"

namespace tflite {
namespace benchmark {
namespace accuracy {
namespace {

template <typename T>
Flag CreateFlag(const char* name, tools::ToolParams* params,
                const std::string& usage) {
  return Flag(
      name,
      [params, name](const T& val, int argv_position) {
        params->Set<T>(name, val, argv_position);
      },
      params->Get<T>(name), usage, Flag::kOptional);
}

AccuracyBenchmarkStatus ConfigureTFLiteSettingsFromArgs(
    const std::vector<std::string>& args, flatbuffers::FlatBufferBuilder& fbb) {
  // TODO(b/241781387): Improve argument parsing for TFLite settings parameters
  // and validator runner options.
  tools::ToolParams params;
  // Apply XNNPack delegate by default.
  params.AddParam("use_xnnpack", tools::ToolParam::Create<bool>(true));
  params.AddParam("use_nnapi", tools::ToolParam::Create<bool>(false));
  params.AddParam("use_gpu", tools::ToolParam::Create<bool>(false));

  std::vector<const char*> argv;
  std::string arg0 = "(MiniBenchmarkAndroid)";
  argv.push_back(const_cast<char*>(arg0.data()));
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }
  int argc = argv.size();
  if (!Flags::Parse(
          &argc, argv.data(),
          {
              CreateFlag<bool>("use_gpu", &params,
                               "Apply GPU delegate for benchmarking."),
              CreateFlag<bool>("use_nnapi", &params,
                               "Apply NNAPI delegate for benchmarking."),
              CreateFlag<bool>("use_xnnpack", &params,
                               "Apply XNNPack delegate for benchmarking."),
          })) {
    return kAccuracyBenchmarkArgumentParsingFailed;
  }
  bool use_xnnpack = params.Get<bool>("use_xnnpack");
  bool use_gpu = params.Get<bool>("use_gpu");
  bool use_nnapi = params.Get<bool>("use_nnapi");
  // Use Delegate_NONE as the default value here for delegate because XNNPack
  // delegate will still be applied as the default delegate unless it is
  // specified as disabled.
  Delegate delegate = Delegate_NONE;
  if (use_gpu && use_nnapi) {
    return kAccuracyBenchmarkMoreThanOneDelegateProvided;
  } else if (use_gpu) {
    delegate = Delegate_GPU;
  } else if (use_nnapi) {
    delegate = Delegate_NNAPI;
  }

  TFLiteSettingsBuilder tflite_settings_builder(fbb);
  tflite_settings_builder.add_delegate(delegate);
  tflite_settings_builder.add_disable_default_delegates(!use_xnnpack);
  flatbuffers::Offset<TFLiteSettings> tflite_settings =
      tflite_settings_builder.Finish();
  fbb.Finish(tflite_settings);
  return kAccuracyBenchmarkSuccess;
}

}  // namespace

AccuracyBenchmarkStatus Benchmark(const std::vector<std::string>& args,
                                  int model_fd, size_t model_offset,
                                  size_t model_size,
                                  const char* result_path_chars) {
  std::string result_path(result_path_chars);
  acceleration::ValidatorRunnerOptions options;
  options.model_fd = model_fd;
  options.model_offset = model_offset;
  options.model_size = model_size;
  options.data_directory_path = result_path;
  options.storage_path = result_path + "/storage_path.fb";
  int return_code = std::remove(options.storage_path.c_str());
  if (return_code) {
    TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                    "Failed to remove storage file (%s): %s.",
                    options.storage_path.c_str(), strerror(errno));
  }
  options.per_test_timeout_ms = 5000;

  acceleration::BlockingValidatorRunner runner(options);
  acceleration::MinibenchmarkStatus status = runner.Init();
  if (status != acceleration::kMinibenchmarkSuccess) {
    TFLITE_LOG_PROD(
        TFLITE_LOG_ERROR,
        "MiniBenchmark BlockingValidatorRunner initialization failed with "
        "error code %d",
        status);
    return kAccuracyBenchmarkRunnerInitializationFailed;
  }

  flatbuffers::FlatBufferBuilder fbb;
  AccuracyBenchmarkStatus parse_status =
      ConfigureTFLiteSettingsFromArgs(args, fbb);
  if (parse_status != kAccuracyBenchmarkSuccess) {
    TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                    "Failed to parse arguments with error code %d",
                    parse_status);
    return parse_status;
  }
  std::vector<const TFLiteSettings*> settings = {
      flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer())};
  std::vector<const BenchmarkEvent*> results =
      runner.TriggerValidation(settings);
  if (results.size() != settings.size()) {
    TFLITE_LOG_PROD(
        TFLITE_LOG_ERROR,
        "Number of result events (%zu) doesn't match the expectation (%zu).",
        results.size(), settings.size());
    return kAccuracyBenchmarkResultCountMismatch;
  }
  // The settings contains one test only. Therefore, the benchmark checks for
  // the first result only.
  if (!results[0]->result()->ok()) {
    return kAccuracyBenchmarkFail;
  }
  return kAccuracyBenchmarkPass;
}

}  // namespace accuracy
}  // namespace benchmark
}  // namespace tflite
