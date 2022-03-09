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

#include <fcntl.h>
#ifndef _WIN32
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif  // !_WIN32

#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/runner.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator.h"
#include "tensorflow/lite/nnapi/sl/include/SupportLibrary.h"

namespace tflite {
namespace acceleration {

constexpr const char* TfLiteValidationFunctionName() {
  return "Java_org_tensorflow_lite_acceleration_validation_entrypoint";
}

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

  // Construct ValidatorRunner for a model and a file for storing results in.
  // The 'storage_path' must be specific for the model.
  // 'data_directory_path' must be suitable for extracting an executable file
  // to.
  // The nnapi_sl pointer can be used to configure the runner to use
  // the NNAPI implementation coming from the Support Library instead of
  // the NNAPI platform drivers.
  // If nnapi_sl is not null we expect the functions referenced by the structure
  // lifetime to be enclosing the one of the mini-benchmark. In particular
  // we expect that if the NnApiSupportLibrary was loaded by a shared library,
  // dlclose is called only after all this mini-benchmark object has been
  // deleted.
  ValidatorRunner(const std::string& model_path,
                  const std::string& storage_path,
                  const std::string& data_directory_path,
                  const NnApiSLDriverImplFL5* nnapi_sl = nullptr,
                  const std::string validation_function_name =
                      TfLiteValidationFunctionName(),
                  ErrorReporter* error_reporter = DefaultErrorReporter());
  ValidatorRunner(int model_fd, size_t model_offset, size_t model_size,
                  const std::string& storage_path,
                  const std::string& data_directory_path,
                  const NnApiSLDriverImplFL5* nnapi_sl = nullptr,
                  const std::string validation_function_name =
                      TfLiteValidationFunctionName(),
                  ErrorReporter* error_reporter = DefaultErrorReporter());
  MinibenchmarkStatus Init();

  // The following methods invalidate previously returned pointers.

  // Run validation for those settings in 'for_settings' where validation has
  // not yet been run. Incomplete validation may be retried a small number of
  // times (e.g., 2).
  // Returns number of runs triggered (this may include runs triggered through a
  // different instance, and is meant for debugging).
  int TriggerMissingValidation(std::vector<const TFLiteSettings*> for_settings);
  // Get results for successfully completed validation runs. The caller can then
  // pick the best configuration based on timings.
  std::vector<const BenchmarkEvent*> GetSuccessfulResults();
  // Get results for completed validation runs regardless whether it is
  // successful or not.
  int GetNumCompletedResults();
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
  std::string model_path_;
  int model_fd_ = -1;
  size_t model_offset_, model_size_;
  std::string storage_path_;
  std::string data_directory_path_;
  FlatbufferStorage<BenchmarkEvent> storage_;
  std::string validation_function_name_;
  ErrorReporter* error_reporter_;
  bool triggered_ = false;
  std::string nnapi_sl_path_;
  const NnApiSLDriverImplFL5* nnapi_sl_;
};

}  // namespace acceleration
}  // namespace tflite

class FileLock {
 public:
  explicit FileLock(const std::string& path) : path_(path) {}
  bool TryLock() {
#ifndef _WIN32  // Validator runner not supported on Windows.
    // O_CLOEXEC is needed for correctness, as another thread may call
    // popen() and the callee inherit the lock if it's not O_CLOEXEC.
    fd_ = open(path_.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0600);
    if (fd_ < 0) {
      return false;
    }
    if (flock(fd_, LOCK_EX | LOCK_NB) == 0) {
      return true;
    }
#endif  // !_WIN32
    return false;
  }
  ~FileLock() {
#ifndef _WIN32  // Validator runner not supported on Windows.
    if (fd_ >= 0) {
      close(fd_);
    }
#endif  // !_WIN32
  }

 private:
  std::string path_;
  int fd_ = -1;
};

extern "C" {
int Java_org_tensorflow_lite_acceleration_validation_entrypoint(int argc,
                                                                char** argv);
}  // extern "C"

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_RUNNER_H_
