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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_options.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"

namespace tflite {
namespace acceleration {

ValidatorRunnerOptions CreateValidatorRunnerOptionsFrom(
    const MinibenchmarkSettings& minibenchmark_settings) {
  ValidatorRunnerOptions options;
  if (minibenchmark_settings.model_file()) {
    if (minibenchmark_settings.model_file()->filename()) {
      options.model_path =
          minibenchmark_settings.model_file()->filename()->str();
    } else if (minibenchmark_settings.model_file()->buffer_handle() > 0) {
      options.model_buffer = reinterpret_cast<const uint8_t*>(
          minibenchmark_settings.model_file()->buffer_handle());
      options.model_size = minibenchmark_settings.model_file()->length();
    } else {
      options.model_fd = minibenchmark_settings.model_file()->fd();
      options.model_offset = minibenchmark_settings.model_file()->offset();
      options.model_size = minibenchmark_settings.model_file()->length();
    }
  }
  if (minibenchmark_settings.storage_paths()) {
    options.data_directory_path =
        minibenchmark_settings.storage_paths()->data_directory_path()->str();
    options.storage_path =
        minibenchmark_settings.storage_paths()->storage_file_path()->str();
  }
  if (minibenchmark_settings.validation_settings()) {
    options.per_test_timeout_ms =
        minibenchmark_settings.validation_settings()->per_test_timeout_ms();
  }
  return options;
}

std::string CreateModelLoaderPath(const ValidatorRunnerOptions& options) {
  std::string model_path;
  if (!options.model_path.empty()) {
    model_path = options.model_path;
  } else if (options.model_fd >= 0) {
    model_path = absl::StrCat("fd:", options.model_fd, ":",
                              options.model_offset, ":", options.model_size);
  } else if (options.model_buffer) {
    model_path =
        absl::StrCat("buffer:", reinterpret_cast<int64_t>(options.model_buffer),
                     ":", options.model_size);
  }
  return model_path;
}

}  // namespace acceleration
}  // namespace tflite
