/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/c/c_api.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/benchmark_result_evaluator.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/blocking_validator_runner.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_options.h"

namespace {

using ::flatbuffers::FlatBufferBuilder;
constexpr int kPerBenchmarkEventSize = 300;

// ErrorReporter that invokes C-style error reporting functions.
class CErrorReporter : public tflite::ErrorReporter {
 public:
  using ErrorReporterFunc = int(void* user_data, const char* format,
                                va_list args);

  CErrorReporter(void* user_data, ErrorReporterFunc* func)
      : user_data_(user_data), error_reporter_func_(func) {}

  int Report(const char* format, va_list args) override {
    return error_reporter_func_(user_data_, format, args);
  }

 private:
  void* user_data_;
  ErrorReporterFunc* error_reporter_func_;
};

class CResultEvaluator
    : public tflite::acceleration::AbstractBenchmarkResultEvaluator {
 public:
  using ValidatorFunc = bool(void* user_data, uint8_t* benchmark_result_data,
                             int benchmark_result_data_size);
  explicit CResultEvaluator(
      const TfLiteMiniBenchmarkCustomValidationInfo& custom_validation_info)
      : user_data_(custom_validation_info.accuracy_validator_user_data),
        validator_func_(custom_validation_info.accuracy_validator_func) {}

  ~CResultEvaluator() override = default;

  bool HasPassedAccuracyCheck(
      const tflite::BenchmarkResult& benchmark_result) override {
    flatbuffers::FlatBufferBuilder fbb;
    tflite::BenchmarkResultT result_obj;
    benchmark_result.UnPackTo(&result_obj);
    fbb.Finish(tflite::CreateBenchmarkResult(fbb, &result_obj));
    return validator_func_(user_data_, fbb.GetBufferPointer(), fbb.GetSize());
  }

 private:
  void* user_data_;
  ValidatorFunc* validator_func_;
};

// Allocate memory in minibenchmark_result and serialize benchmark_events to it.
void CreateData(
    const std::vector<const tflite::BenchmarkEvent*>& benchmark_events,
    TfLiteMiniBenchmarkResult& minibenchmark_result) {
  if (benchmark_events.empty()) {
    return;
  }

  std::vector<uint8_t> data;
  data.reserve(kPerBenchmarkEventSize * benchmark_events.size());
  auto cur = data.begin();
  for (auto& event : benchmark_events) {
    FlatBufferBuilder fbb;
    tflite::BenchmarkEventT event_obj;
    event->UnPackTo(&event_obj);
    fbb.FinishSizePrefixed(CreateBenchmarkEvent(fbb, &event_obj));
    data.insert(cur, fbb.GetBufferPointer(),
                fbb.GetBufferPointer() + fbb.GetSize());
    std::advance(cur, fbb.GetSize());
  }
  minibenchmark_result.flatbuffer_data = new uint8_t[data.size()];
  minibenchmark_result.flatbuffer_data_size = data.size();
  memcpy(minibenchmark_result.flatbuffer_data, data.data(), data.size());
}

std::vector<std::vector<uint8_t>> ToCustomInputData(
    const TfLiteMiniBenchmarkCustomValidationInfo& custom_validation_info) {
  std::vector<std::vector<uint8_t>> result(
      custom_validation_info.buffer_dim_size);
  uint8_t* buffer_current = custom_validation_info.buffer;
  for (int i = 0; i < custom_validation_info.buffer_dim_size; i++) {
    uint8_t* buffer_next =
        buffer_current + custom_validation_info.buffer_dim[i];
    result[i].insert(result[i].begin(), buffer_current, buffer_next);

    buffer_current = buffer_next;
  }
  return result;
}

void TfLiteBlockingValidatorRunnerTriggerValidationImpl(
    const TfLiteMinibenchmarkSettings& settings,
    TfLiteMiniBenchmarkResult& result, tflite::ErrorReporter* error_reporter) {
  // Create ValidatorRunnerOptions.
  const tflite::MinibenchmarkSettings* minibenchmark_settings =
      flatbuffers::GetRoot<tflite::MinibenchmarkSettings>(
          settings.flatbuffer_data);
  tflite::acceleration::ValidatorRunnerOptions options =
      tflite::acceleration::CreateValidatorRunnerOptionsFrom(
          *minibenchmark_settings);

  std::unique_ptr<tflite::acceleration::AbstractBenchmarkResultEvaluator>
      result_evaluator;
  if (settings.custom_validation_info.buffer) {
    options.custom_input_batch_size =
        settings.custom_validation_info.batch_size;
    options.custom_input_data =
        ToCustomInputData(settings.custom_validation_info);
    if (settings.custom_validation_info.accuracy_validator_func != nullptr) {
      result_evaluator =
          std::make_unique<CResultEvaluator>(settings.custom_validation_info);
      options.benchmark_result_evaluator = result_evaluator.get();
    }
  }
  if (error_reporter) {
    options.error_reporter = error_reporter;
  }

  tflite::acceleration::BlockingValidatorRunner runner(options);
  result.init_status = runner.Init();
  if (result.init_status != tflite::acceleration::kMinibenchmarkSuccess) {
    return;
  }

  std::vector<const tflite::TFLiteSettings*> tflite_settings;
  for (auto tflite_setting : *minibenchmark_settings->settings_to_test()) {
    tflite_settings.push_back(tflite_setting);
  }

  CreateData(runner.TriggerValidation(tflite_settings), result);
}

}  // namespace

TfLiteMiniBenchmarkResult* TfLiteBlockingValidatorRunnerTriggerValidation(
    TfLiteMinibenchmarkSettings* settings) {
  TfLiteMiniBenchmarkResult* return_value =
      new TfLiteMiniBenchmarkResult{0, nullptr, 0};

  if (!settings) {
    return_value->init_status =
        tflite::acceleration::kMinibenchmarkPreconditionNotMet;
    return return_value;
  }

  std::unique_ptr<CErrorReporter> error_reporter;
  if (settings->error_reporter_func != nullptr) {
    error_reporter = std::make_unique<CErrorReporter>(
        settings->error_reporter_user_data, settings->error_reporter_func);
  }

  if (!settings->flatbuffer_data || settings->flatbuffer_data_size == 0) {
    return_value->init_status =
        tflite::acceleration::kMinibenchmarkPreconditionNotMet;
    if (error_reporter) {
      TF_LITE_REPORT_ERROR(error_reporter.get(),
                           "MinibenchmarkSettings config is not set.");
    }
    return return_value;
  }
  // Verify data is not corrupted.
  flatbuffers::Verifier verifier(settings->flatbuffer_data,
                                 settings->flatbuffer_data_size);
  if (!verifier.VerifyBuffer<tflite::MinibenchmarkSettings>()) {
    return_value->init_status =
        tflite::acceleration::kMinibenchmarkCorruptSizePrefixedFlatbufferFile;
    if (error_reporter) {
      TF_LITE_REPORT_ERROR(error_reporter.get(),
                           "MinibenchmarkSettings is corruprted.");
    }
    return return_value;
  }

  TfLiteBlockingValidatorRunnerTriggerValidationImpl(*settings, *return_value,
                                                     error_reporter.get());
  return return_value;
}

void TfLiteMiniBenchmarkResultFree(TfLiteMiniBenchmarkResult* result) {
  delete[] result->flatbuffer_data;
  delete result;
}
