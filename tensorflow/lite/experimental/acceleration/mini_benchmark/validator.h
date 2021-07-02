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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_H_

#include <cstdint>
#include <map>
#include <string>

#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/configuration/delegate_registry.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace acceleration {

// Class to run the validation subgraph of a tflite model with embedded
// validation.
//
// The API is split into multiple steps so that callers can construct detailed
// telemetry from it.
//
// TODO(b/172541832): add wrapper for running in separate process and using a
// file to communicate results.
class Validator {
 public:
  // Construct Validator for given model path and compute settings. The
  // compute_settings must be valid for the lifetime of the Validator instance.
  Validator(const std::string& model_path,
            const ComputeSettings* compute_settings);
  // Construct Validator for given model file descriptor and compute settings.
  // The model_fd only has to be valid for the duration of the constructor (it's
  // dup'ed inside). The compute_settings must be valid for the lifetime of the
  // Validator instance.
  Validator(int model_fd, size_t model_offset, size_t model_size,
            const ComputeSettings* compute_settings);
  // Check that the model is valid for validation.
  MinibenchmarkStatus CheckModel(bool load_only = false);

  // Results from validation.
  struct Results {
    // Are the results correct (metrics below threshold).
    bool ok = false;
    // What are the metrics results, for telemetry.
    std::map<std::string, std::vector<float>> metrics;
    // How long did compilation (ModifyGraphWithDelegate) take. -1 if running on
    // CPU (or in rare cases when reading the system clock fails).
    int64_t compilation_time_us;
    // How long did execution (Invoke) take. (Empty in rare cases when reading
    // the system clock fails).
    std::vector<int64_t> execution_time_us;
    // Any possible error from the delegate.
    int delegate_error = 0;
  };
  // Run the validation graph and return validation results.
  MinibenchmarkStatus RunValidation(Results* results_out);

  // Get timestamps.
  static int64_t BootTimeMicros();
  static int64_t WallTimeMicros();

  ~Validator();

  Validator(Validator&) = delete;
  Validator& operator=(Validator&) = delete;
  Validator(Validator&&) = delete;
  Validator& operator=(Validator&&) = delete;

 private:
  // Load delegate plugin and create delegate.
  MinibenchmarkStatus LoadDelegate();
  // Apply the compute settings (typically applying a delegate to the
  // interpreter).
  MinibenchmarkStatus ApplyComputeSettings(int* delegate_error_out);

  std::string model_path_;
  int model_fd_ = -1;
  size_t model_offset_, model_size_;
  const ComputeSettings* compute_settings_;
  std::unique_ptr<Interpreter> interpreter_;
  ::tflite::ops::builtin::BuiltinOpResolver resolver_;
  std::unique_ptr<FlatBufferModel> model_;
  ::tflite::delegates::TfLiteDelegatePtr delegate_;
  std::unique_ptr<tflite::delegates::DelegatePluginInterface> delegate_plugin_;
  Subgraph* validation_entrypoint_ = nullptr;
  Subgraph* main_model_ = nullptr;
  int64_t compilation_time_us_ = -1;
};

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_H_
