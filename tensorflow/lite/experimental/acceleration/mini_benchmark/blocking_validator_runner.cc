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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/blocking_validator_runner.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_options.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace acceleration {
namespace {

using ::flatbuffers::FlatBufferBuilder;
using ::flatbuffers::GetRoot;

// Wait time between each query to the test result file, defined in
// microseconds.
constexpr absl::Duration kWaitBetweenRefresh = absl::Milliseconds(20);

// Generate a string of 10 chars.
std::string GenerateRandomString() {
  static const char charset[] =
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";
  const int size = 10;
  std::string result;
  result.reserve(size);
  for (int i = 0; i < size; ++i) {
    result.data()[i] = charset[rand() % (sizeof(charset) - 1)];
  }

  return result;
}

}  // namespace

BlockingValidatorRunner::BlockingValidatorRunner(
    const ValidatorRunnerOptions& options)
    : per_test_timeout_ms_(options.per_test_timeout_ms),
      storage_path_base_(options.storage_path) {
  validator_runner_impl_ = std::make_unique<ValidatorRunnerImpl>(
      CreateModelLoaderPath(options), options.storage_path,
      options.data_directory_path, options.per_test_timeout_ms,
      options.custom_input_data.empty()
          ? nullptr
          : std::make_unique<CustomValidationEmbedder>(
                options.custom_input_batch_size, options.custom_input_data,
                options.error_reporter),
      options.error_reporter, options.nnapi_sl, options.gpu_plugin_handle,
      options.validation_entrypoint_name, options.benchmark_result_evaluator);
}

MinibenchmarkStatus BlockingValidatorRunner::Init() {
  return validator_runner_impl_->Init();
}

std::vector<FlatBufferBuilder> BlockingValidatorRunner::TriggerValidation(
    const std::vector<const TFLiteSettings*>& for_settings) {
  if (for_settings.empty()) {
    return {};
  }

  // Create a unique storage_path.
  std::string storage_path =
      absl::StrCat(storage_path_base_, ".", GenerateRandomString());
  auto to_be_run =
      std::make_unique<std::vector<flatbuffers::FlatBufferBuilder>>();
  std::vector<TFLiteSettingsT> for_settings_obj;
  for_settings_obj.reserve(for_settings.size());
  for (auto settings : for_settings) {
    TFLiteSettingsT tflite_settings;
    settings->UnPackTo(&tflite_settings);
    flatbuffers::FlatBufferBuilder copy;
    copy.Finish(CreateTFLiteSettings(copy, &tflite_settings));
    to_be_run->emplace_back(std::move(copy));
    for_settings_obj.emplace_back(tflite_settings);
  }
  validator_runner_impl_->TriggerValidationAsync(std::move(to_be_run),
                                                 storage_path);

  // The underlying process runner should ensure each test finishes on time or
  // timed out. deadline_us is added here as an extra safety guard.
  int64_t total_timeout_ms = per_test_timeout_ms_ * (1 + for_settings.size());
  int64_t deadline_us = Validator::BootTimeMicros() + total_timeout_ms * 1000;

  bool within_timeout = true;
  // TODO(b/249274787): GetNumCompletedResults() loads the file from disk each
  // time when called. We should find a way to optimize the FlatbufferStorage to
  // reduce the I/O and remove the sleep().
  while ((validator_runner_impl_->GetNumCompletedResults()) <
             for_settings.size() &&
         (within_timeout = Validator::BootTimeMicros() < deadline_us)) {
    usleep(absl::ToInt64Microseconds(kWaitBetweenRefresh));
  }

  std::vector<FlatBufferBuilder> results =
      validator_runner_impl_->GetCompletedResults();
  if (!within_timeout) {
    TFLITE_LOG_PROD(
        TFLITE_LOG_WARNING,
        "Validation timed out after %ld ms. Return before all tests finished.",
        total_timeout_ms);
  } else if (for_settings.size() != results.size()) {
    TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                    "Validation completed.Started benchmarking for %d "
                    "TFLiteSettings, received %d results.",
                    for_settings.size(), results.size());
  }

  // If there are any for_settings missing from results, add an error event.
  std::vector<TFLiteSettingsT> result_settings;
  result_settings.reserve(results.size());
  for (auto& result : results) {
    const BenchmarkEvent* event =
        GetRoot<BenchmarkEvent>(result.GetBufferPointer());
    TFLiteSettingsT event_settings;
    event->tflite_settings()->UnPackTo(&event_settings);
    result_settings.emplace_back(std::move(event_settings));
  }
  for (auto& settings_obj : for_settings_obj) {
    auto result_it =
        std::find(result_settings.begin(), result_settings.end(), settings_obj);
    if (result_it == result_settings.end()) {
      FlatBufferBuilder fbb;
      fbb.Finish(CreateBenchmarkEvent(
          fbb, CreateTFLiteSettings(fbb, &settings_obj),
          BenchmarkEventType_ERROR, /* result */ 0,
          CreateBenchmarkError(fbb, BenchmarkStage_UNKNOWN,
                               /* exit_code */ 0, /* signal */ 0,
                               /* error_code */ 0,
                               /* mini_benchmark_error_code */
                               kMinibenchmarkCompletionEventMissing),
          Validator::BootTimeMicros(), Validator::WallTimeMicros()));
      results.emplace_back(std::move(fbb));
    }
  }
  // Delete storage_file before returning. In case of test timeout, the child
  // thread or process may create and continue to write to the storage_path. In
  // this case we cannot delete the file.
  (void)unlink(storage_path.c_str());
  return results;
}

}  // namespace acceleration
}  // namespace tflite
