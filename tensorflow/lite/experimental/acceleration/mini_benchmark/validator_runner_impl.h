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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_RUNNER_IMPL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_RUNNER_IMPL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/benchmark_result_evaluator.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/model_modifier/custom_validation_embedder.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/nnapi/sl/include/SupportLibrary.h"

namespace tflite {
namespace acceleration {

// This class implements the logic of managing models and triggering validation
// tests in separate processes, so that we can provide blocking and non-blocking
// API of ValidatorRunner.
class ValidatorRunnerImpl {
 public:
  // nnapi_sl should be valid until Init() finishes. error_reporter should be
  // valid during the entire lifetime of the class.
  // TODO(b/246912769): Create a common Context class to store shared params.
  ValidatorRunnerImpl(
      const std::string& fd_or_model_path, const std::string& storage_path,
      const std::string& data_directory_path, int timeout_ms,
      std::unique_ptr<CustomValidationEmbedder> custom_validation_embedder,
      ErrorReporter* error_reporter, const NnApiSLDriverImplFL5* nnapi_sl,
      const TfLiteDelegatePlugin* gpu_plugin_handle,
      const std::string& validation_entrypoint_name,
      AbstractBenchmarkResultEvaluator* benchmark_evaluator)
      : fd_or_model_path_(fd_or_model_path),
        storage_path_(storage_path),
        data_directory_path_(data_directory_path),
        timeout_ms_(timeout_ms),
        custom_validation_embedder_(std::move(custom_validation_embedder)),
        error_reporter_(error_reporter),
        storage_(storage_path_, error_reporter_),
        nnapi_helper_(nnapi_sl),
        gpu_helper_(gpu_plugin_handle),
        validation_entrypoint_helper_(validation_entrypoint_name,
                                      error_reporter_),
        benchmark_evaluator_(benchmark_evaluator) {}

  MinibenchmarkStatus Init();

  // Trigger the test for the given tflite_settings in a new thread. The
  // settings will run sequentially. The storage_path will be used to store
  // intermediate test output.
  void TriggerValidationAsync(
      std::vector<flatbuffers::FlatBufferBuilder> tflite_settings,
      absl::string_view storage_path);

  // Returns the unmodified successful BenchmarkEvent from storage. If a
  // BenchmarkEvent is considered pass with the BenchmarkResultEvaluator, but
  // its result.ok is set to false, the BenchmarkEvent will be returned as-is.
  std::vector<const BenchmarkEvent*> GetSuccessfulResultsFromStorage();
  int GetNumCompletedResults();

  // Returns the completed BenchmarkEvent. BenchmarkResult::ok() will be set to
  // the result from BenchmarkResultEvaluator.
  std::vector<flatbuffers::FlatBufferBuilder> GetCompletedResults();

 private:
  class NnapiHelper {
   public:
    // nnapi_sl should be valid when Load() is called.
    explicit NnapiHelper(const NnApiSLDriverImplFL5* nnapi_sl)
        : nnapi_sl_(nnapi_sl) {}

    // Load the NNAPI SL from dynamic linking loader. Returns the error status
    // if failed.
    MinibenchmarkStatus Load();

    // Returns the pathname of the shared object.
    const std::string& nnapi_sl_path() const { return nnapi_sl_path_; }

   private:
    const NnApiSLDriverImplFL5* nnapi_sl_;
    std::string nnapi_sl_path_;
  };

  // Lookup the GPU Module .so file path if gpu_plugin_handle is provided.
  class GpuHelper {
   public:
    explicit GpuHelper(const TfLiteDelegatePlugin* gpu_plugin_handle)
        : gpu_plugin_handle_(gpu_plugin_handle) {}
    MinibenchmarkStatus Load();
    const std::string& gpu_so_path() const { return gpu_so_path_; }

   private:
    const TfLiteDelegatePlugin* gpu_plugin_handle_;
    std::string gpu_so_path_;
  };

  class ValidationEntrypointHelper {
   public:
    using EntrypointFunc = int(int argc, char** argv);

    // error_reporter should be valid for the entire lifetime.
    explicit ValidationEntrypointHelper(
        const std::string& validation_entrypoint_name,
        ErrorReporter* error_reporter)
        : validation_entrypoint_name_(validation_entrypoint_name),
          error_reporter_(error_reporter) {}

    // Verify that the entrypoint function can be found with dlsym(). Returns
    // the error status if failed.
    MinibenchmarkStatus Validate();

    // Returns the entrypoint function from dlsym(). Returns nullptr if failed.
    // Note this function will perform the lookup each time when it's called.
    EntrypointFunc* LoadEntrypoint();

    // Returns the function name. Lifetime is the same as the helper class
    // itself.
    const std::string& name() { return validation_entrypoint_name_; }

   private:
    std::string validation_entrypoint_name_;
    ErrorReporter* error_reporter_;
  };

  std::string fd_or_model_path_;
  std::string storage_path_;
  std::string data_directory_path_;
  int timeout_ms_ = 0;
  std::unique_ptr<CustomValidationEmbedder> custom_validation_embedder_;
  // Owns the model data if model is created with custom_validation_embedder_.
  // Otherwise this is empty.
  flatbuffers::FlatBufferBuilder model_with_custom_input_;
  // If not null, this allocation points the already-loaded model that need to
  // be copied to the detached thread.
  std::unique_ptr<Allocation> model_allocation_ = nullptr;
  ErrorReporter* error_reporter_;
  FlatbufferStorage<BenchmarkEvent> storage_;
  NnapiHelper nnapi_helper_;
  GpuHelper gpu_helper_;
  ValidationEntrypointHelper validation_entrypoint_helper_;
  AbstractBenchmarkResultEvaluator* benchmark_evaluator_ = nullptr;
};

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_RUNNER_IMPL_H_
