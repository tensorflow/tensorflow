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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_BLOCKING_VALIDATOR_RUNNER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_BLOCKING_VALIDATOR_RUNNER_H_

#include <memory>
#include <string>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_impl.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_options.h"

namespace tflite {
namespace acceleration {

// Class that runs mini-benchmark validation in a separate process and gives
// access to the results. This class provides a synchronous API for the callers
// to wait until the all the tests have finished.
//
// This class is thread-safe when using different storage_path_. When
// storage_path_ is shared between multiple runners, they will interfere with
// each other.
class BlockingValidatorRunner {
 public:
  explicit BlockingValidatorRunner(const ValidatorRunnerOptions& options);

  MinibenchmarkStatus Init();

  // Trigger the validation tests with for_settings, and return the test result.
  // Each for_settings will have a corresponding result. The result is of schema
  // BenchmarkEvent.
  std::vector<flatbuffers::FlatBufferBuilder> TriggerValidation(
      const std::vector<const TFLiteSettings*>& for_settings);

 private:
  int per_test_timeout_ms_ = 0;
  std::string storage_path_;
  std::unique_ptr<ValidatorRunnerImpl> validator_runner_impl_;
};

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_BLOCKING_VALIDATOR_RUNNER_H_
