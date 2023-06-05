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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/core/acceleration/configuration/delegate_registry.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/tools/model_loader.h"

namespace tflite {
namespace acceleration {

// Class to run the validation subgraph of a tflite model with embedded
// validation.
//
// The API is split into multiple steps so that callers can construct detailed
// telemetry from it.
class Validator {
 public:
  // Construct Validator for the given model and compute settings. The
  // compute_settings must be valid for the lifetime of the Validator instance.
  Validator(std::unique_ptr<tools::ModelLoader> model_loader,
            const ComputeSettings* compute_settings)
      : model_loader_(std::move(model_loader)),
        compute_settings_(compute_settings) {}

  // Results from validation.
  struct Results {
    // Are the results correct (metrics below threshold). When validation
    // is not embedded, this field is set to false.
    bool ok = false;
    // What are the accuracy metrics results, for telemetry.
    std::map<std::string, std::vector<float>> metrics;
    // How long did loading the delegate and creating the interpreter take. -1
    // if failed.
    int64_t delegate_prep_time_us = 0;
    // How long did execution (Invoke) take. (Empty in rare cases when reading
    // the system clock fails).
    std::vector<int64_t> execution_time_us;
    // Any possible error from the delegate.
    int delegate_error = 0;
    // Number of delegated kernels.
    int delegated_kernels = 0;
    // Model output with the delegate, in byte format. It is ordered the same as
    // tflite::Interpreter::output_tensor(), i.e. the value of output_tensor(i)
    // is stored in actual_inference_output[i].
    std::vector<std::vector<char>> actual_inference_output;
  };

  // Status from validation run.
  struct Status {
    // Status of the mini benchmark run.
    MinibenchmarkStatus status;
    // Stage during which validation run failed.
    // Unknown in case of a successful run.
    BenchmarkStage stage = BenchmarkStage_UNKNOWN;
  };

  // Run the validation graph and return validation results.
  Status RunValidation(Results* results_out);

  // Get timestamps.
  static int64_t BootTimeMicros();
  static int64_t WallTimeMicros();

  Validator(Validator&) = delete;
  Validator& operator=(Validator&) = delete;
  Validator(Validator&&) = delete;
  Validator& operator=(Validator&&) = delete;

 private:
  // Load delegate plugin and create delegate.
  MinibenchmarkStatus LoadDelegate();

  // Create the interpreter with the delegate. Must be called after
  // LoadDelegate().
  MinibenchmarkStatus CreateInterpreter(int* delegate_error_out,
                                        int* delegated_kernels_out);

  // Only used in embedded validation case. If the golden output is not
  // embedded, run Model on CPU and add golden output to model_.
  MinibenchmarkStatus CheckGoldenOutput(Results* results_out);

  std::unique_ptr<tools::ModelLoader> model_loader_;
  const ComputeSettings* compute_settings_;
  // Optional. Interpreter that runs on CPU.
  std::unique_ptr<Interpreter> golden_interpreter_;
  // Interpreter that runs with delegate enabled, using the compute settings
  // passed to the Validator constructor.
  std::unique_ptr<Interpreter> interpreter_;
  // Op resolver used to create the interpreters. Depending on the
  // compute_settings_, it may or may not include the default delegate.
  std::unique_ptr<::tflite::MutableOpResolver> resolver_;
  std::unique_ptr<FlatBufferModel> model_;
  ::tflite::delegates::TfLiteDelegatePtr delegate_ =
      delegates::TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
  std::unique_ptr<tflite::delegates::DelegatePluginInterface> delegate_plugin_;
  int validation_entrypoint_index_ = -1;
  Subgraph* validation_entrypoint_ = nullptr;
  Subgraph* main_model_ = nullptr;
  // Whether accuracy validation is embedded.
  bool has_accuracy_validation_ = false;
};

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_H_
