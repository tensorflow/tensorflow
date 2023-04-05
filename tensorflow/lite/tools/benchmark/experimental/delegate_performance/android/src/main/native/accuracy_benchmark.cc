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

#include "tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/src/main/native/accuracy_benchmark.h"

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
#include "tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/src/main/native/status_codes.h"

namespace tflite {
namespace benchmark {
namespace accuracy {

flatbuffers::Offset<BenchmarkEvent> Benchmark(
    flatbuffers::FlatBufferBuilder& fbb, const TFLiteSettings& tflite_settings,
    int model_fd, size_t model_offset, size_t model_size,
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
    BenchmarkErrorBuilder error_builder(fbb);
    error_builder.add_stage(BenchmarkStage_INITIALIZATION);
    error_builder.add_exit_code(kBenchmarkRunnerInitializationFailed);
    error_builder.add_mini_benchmark_error_code(status);
    flatbuffers::Offset<BenchmarkError> error = error_builder.Finish();
    BenchmarkEventBuilder builder(fbb);
    builder.add_event_type(BenchmarkEventType_ERROR);
    builder.add_error(error);
    return builder.Finish();
  }

  std::vector<const TFLiteSettings*> settings = {&tflite_settings};
  std::vector<flatbuffers::FlatBufferBuilder> results =
      runner.TriggerValidation(settings);
  if (results.size() != settings.size()) {
    TFLITE_LOG_PROD(
        TFLITE_LOG_ERROR,
        "Number of result events (%zu) doesn't match the expectation (%zu).",
        results.size(), settings.size());
    flatbuffers::Offset<BenchmarkError> error =
        CreateBenchmarkError(fbb, BenchmarkStage_INFERENCE,
                             /*exit_code=*/kBenchmarkResultCountMismatch);
    BenchmarkEventBuilder builder(fbb);
    builder.add_event_type(BenchmarkEventType_ERROR);
    builder.add_error(error);
    return builder.Finish();
  }
  // The settings contains one test only. Therefore, the benchmark checks for
  // the first result only.
  TFLITE_CHECK_EQ(results.size(), 1);
  BenchmarkEventT benchmark_event;
  flatbuffers::GetRoot<tflite::BenchmarkEvent>(results[0].GetBufferPointer())
      ->UnPackTo(&benchmark_event);
  return CreateBenchmarkEvent(fbb, &benchmark_event);
}

}  // namespace accuracy
}  // namespace benchmark
}  // namespace tflite
