/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_RUNNER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_RUNNER_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_impl.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_options.h"

namespace tflite {
namespace acceleration {
// Class that runs mini-benchmark validation in a separate process and gives
// access to the results.
//
// It is safe to construct more than one instance of the ValidatorRunner in one
// or more processes. File locks are used to ensure the storage is mutated
// safely and that we run at most one validation at a time for a given
// data_directory_path.
//
// A single instance of ValidatorRunner is thread-compatible (access from
// multiple threads must be guarded with a mutex).
class ValidatorRunner {
 public:
  static constexpr int64_t kDefaultEventTimeoutUs = 30 * 1000 * 1000;

  explicit ValidatorRunner(const ValidatorRunnerOptions& options);

  MinibenchmarkStatus Init();

  // The following methods invalidate previously returned pointers.

  // Run validation for those settings in 'for_settings' where validation has
  // not yet been run. Incomplete validation may be retried a small number of
  // times (e.g., 2).
  // Returns number of runs triggered (this may include runs triggered through a
  // different instance, and is meant for debugging).
  int TriggerMissingValidation(
      const std::vector<const TFLiteSettings*>& for_settings);

  // Get results for successfully completed validation runs. The caller can then
  // pick the best configuration based on timings.
  std::vector<const BenchmarkEvent*> GetSuccessfulResults() {
    return validator_runner_impl_->GetSuccessfulResultsFromStorage();
  }

  // Get results for completed validation runs regardless whether it is
  // successful or not.
  int GetNumCompletedResults() {
    return validator_runner_impl_->GetNumCompletedResults();
  }

  // Get all relevant results for telemetry. Will contain:
  // - Start events if an incomplete test is found. Tests are considered
  // incomplete, if they started more than timeout_us ago and do not have
  // results/errors.
  // - Error events where the test ended with an error
  // - End events where the test was completed (even if results were incorrect).
  // The returned events will be marked as logged and not returned again on
  // subsequent calls.
  std::vector<const BenchmarkEvent*> GetAndFlushEventsToLog(
      int64_t timeout_us = kDefaultEventTimeoutUs);

 private:
  FlatbufferStorage<BenchmarkEvent> storage_;
  ErrorReporter* error_reporter_;
  bool triggered_ = false;
  std::unique_ptr<ValidatorRunnerImpl> validator_runner_impl_;
};

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_RUNNER_H_
