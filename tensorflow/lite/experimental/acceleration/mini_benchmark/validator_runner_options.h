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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_RUNNER_OPTIONS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_RUNNER_OPTIONS_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/experimental/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/benchmark_result_evaluator.h"
#include "tensorflow/lite/nnapi/sl/include/SupportLibrary.h"
#include "tensorflow/lite/stderr_reporter.h"

namespace tflite {
namespace acceleration {

inline const char* TfLiteValidationEntrypointName() {
  static constexpr char kEntrypointName[] =
      "Java_org_tensorflow_lite_acceleration_validation_entrypoint";
  return kEntrypointName;
}

// Option class for constructing ValidatorRunner and BlockingValidatorRunner.
struct ValidatorRunnerOptions {
  // Required: Where to read the model.
  // Option 1: Read model from model_path.
  std::string model_path;
  // Option 2: Read model from file descriptor.
  int model_fd = -1;
  size_t model_offset = 0;
  size_t model_size = 0;

  // Optional: Custom validation info.
  // Number of sample input.
  int custom_input_batch_size = 1;
  // The sample input data.
  // Suppose the model has N input tensors, and each tensor is of size M, then
  // custom_input_data.size() == N, and each custom_input_data[i] .size() ==
  // M*N. The input data from different batches are concatenated so that the
  // j-th input data maps to custom_input_data[i][j * M to(j + 1) * M].
  std::vector<std::vector<uint8_t>> custom_input_data;
  // The custom validation rule that decides whether the output is considered
  // passing accuracy checks. The lifetime of this evaluator should last longer
  // than validator runner.
  AbstractBenchmarkResultEvaluator* benchmark_result_evaluator =
      EmbeddedResultEvaluator::GetInstance();

  // Required: The 'storage_path' must be model-specific.
  std::string storage_path;
  // Required: 'data_directory_path' must be suitable for extracting an
  // executable file to.
  std::string data_directory_path;
  // Optional: The timeout for each acceleration config test. By default
  // timeout is not enabled.
  int per_test_timeout_ms = 0;

  // Optional: The nnapi_sl pointer can be used to configure the runner to use
  // the NNAPI implementation coming from the Support Library instead of
  // the NNAPI platform drivers.
  // If nnapi_sl is not null we expect the functions referenced by the
  // structure lifetime to be enclosing the one of the mini-benchmark. In
  // particular we expect that if the NnApiSupportLibrary was loaded by a
  // shared library, dlclose is called only after all this mini-benchmark
  // object has been deleted.
  const NnApiSLDriverImplFL5* nnapi_sl = nullptr;
  // Optional: A handle to a gpu_plugin provided by TFLite-in-PlayServices GPU
  // Module. It will be used to lookup the shared object that provides GPU
  // Delegate Plugin.
  const TfLiteDelegatePlugin* gpu_plugin_handle = nullptr;

  std::string validation_entrypoint_name = TfLiteValidationEntrypointName();
  ErrorReporter* error_reporter = DefaultErrorReporter();
};

// Create a ValidatorRunnerOptions based on the given settings.
ValidatorRunnerOptions CreateValidatorRunnerOptionsFrom(
    const MinibenchmarkSettings& settings);

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_RUNNER_OPTIONS_H_
