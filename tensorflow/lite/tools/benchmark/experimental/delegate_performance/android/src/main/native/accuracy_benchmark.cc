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
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "flatbuffers/base.h"  // from @flatbuffers
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/c/c_api.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/src/main/native/status_codes.h"

namespace tflite {
namespace benchmark {
namespace accuracy {

namespace {
std::vector<const tflite::BenchmarkEvent*> ToBenchmarkEvents(uint8_t* data,
                                                             size_t size) {
  std::vector<const tflite::BenchmarkEvent*> results;
  uint8_t* current_root = data;
  while (current_root < data + size) {
    flatbuffers::uoffset_t current_size =
        flatbuffers::GetPrefixedSize(current_root);
    results.push_back(
        flatbuffers::GetSizePrefixedRoot<tflite::BenchmarkEvent>(current_root));
    current_root += current_size + sizeof(flatbuffers::uoffset_t);
  }
  // Checks the data read is not over the bounds of the buffer.
  TFLITE_CHECK_EQ(current_root, data + size);
  return results;
}
}  // namespace

flatbuffers::Offset<BenchmarkEvent> Benchmark(
    flatbuffers::FlatBufferBuilder& fbb, const TFLiteSettings& tflite_settings,
    int model_fd, size_t model_offset, size_t model_size,
    const char* result_path_chars) {
  std::string result_path(result_path_chars);
  std::string storage_path = result_path + "/storage_path.fb";
  int return_code = std::remove(storage_path.c_str());
  if (return_code) {
    TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                    "Failed to remove storage file (%s): %s.",
                    storage_path.c_str(), strerror(errno));
  }

  flatbuffers::FlatBufferBuilder mini_benchmark_fbb;
  TFLiteSettingsT tflite_settings_t;
  tflite_settings.UnPackTo(&tflite_settings_t);
  flatbuffers::Offset<TFLiteSettings> tflite_settings_offset =
      CreateTFLiteSettings(mini_benchmark_fbb, &tflite_settings_t);
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<TFLiteSettings>>>
      tflite_settings_vector_offset =
          mini_benchmark_fbb.CreateVector({tflite_settings_offset});
  ModelFileBuilder model_file_builder(mini_benchmark_fbb);
  model_file_builder.add_fd(model_fd);
  model_file_builder.add_offset(model_offset);
  model_file_builder.add_length(model_size);
  flatbuffers::Offset<ModelFile> model_file_offset =
      model_file_builder.Finish();
  flatbuffers::Offset<BenchmarkStoragePaths> storage_paths_offset =
      CreateBenchmarkStoragePaths(mini_benchmark_fbb,
                                  mini_benchmark_fbb.CreateString(storage_path),
                                  mini_benchmark_fbb.CreateString(result_path));
  flatbuffers::Offset<ValidationSettings> validation_settings_offset =
      CreateValidationSettings(mini_benchmark_fbb,
                               /*per_test_timeout_ms=*/5000);
  mini_benchmark_fbb.Finish(CreateMinibenchmarkSettings(
      mini_benchmark_fbb, tflite_settings_vector_offset, model_file_offset,
      storage_paths_offset, validation_settings_offset));

  TfLiteMiniBenchmarkSettings* settings = TfLiteMiniBenchmarkSettingsCreate();
  TfLiteMiniBenchmarkSettingsSetFlatBufferData(
      settings, mini_benchmark_fbb.GetBufferPointer(),
      mini_benchmark_fbb.GetSize());

  TfLiteMiniBenchmarkResult* result =
      TfLiteBlockingValidatorRunnerTriggerValidation(settings);
  std::vector<const BenchmarkEvent*> events =
      ToBenchmarkEvents(TfLiteMiniBenchmarkResultFlatBufferData(result),
                        TfLiteMiniBenchmarkResultFlatBufferDataSize(result));
  TfLiteMiniBenchmarkSettingsFree(settings);

  // The settings contains one test only. Therefore, the benchmark checks for
  // the first result only.
  if (events.size() != 1) {
    TfLiteMiniBenchmarkResultFree(result);
    TFLITE_LOG_PROD(
        TFLITE_LOG_ERROR,
        "Number of result events (%zu) doesn't match the expectation (%zu).",
        events.size(), 1);
    flatbuffers::Offset<BenchmarkError> error =
        CreateBenchmarkError(fbb, BenchmarkStage_INFERENCE,
                             /*exit_code=*/kBenchmarkResultCountMismatch);
    BenchmarkEventBuilder builder(fbb);
    builder.add_event_type(BenchmarkEventType_ERROR);
    builder.add_error(error);
    return builder.Finish();
  }
  BenchmarkEventT benchmark_event;
  events[0]->UnPackTo(&benchmark_event);
  TfLiteMiniBenchmarkResultFree(result);
  return CreateBenchmarkEvent(fbb, &benchmark_event);
}

}  // namespace accuracy
}  // namespace benchmark
}  // namespace tflite
