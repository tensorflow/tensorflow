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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_options.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace acceleration {
constexpr int kMaxAttempts = 2;

ValidatorRunner::ValidatorRunner(const ValidatorRunnerOptions& options)
    : storage_path_(options.storage_path),
      storage_(options.storage_path, options.error_reporter),
      error_reporter_(options.error_reporter) {
  validator_runner_impl_ = std::make_unique<ValidatorRunnerImpl>(
      CreateModelLoaderPath(options), options.storage_path,
      options.data_directory_path, options.per_test_timeout_ms,
      options.custom_input_data.empty()
          ? nullptr
          : std::make_unique<CustomValidationEmbedder>(
                options.custom_input_batch_size, options.custom_input_data,
                options.error_reporter),
      error_reporter_, options.nnapi_sl, options.gpu_plugin_handle,
      options.validation_entrypoint_name, options.benchmark_result_evaluator);
}

MinibenchmarkStatus ValidatorRunner::Init() {
  MinibenchmarkStatus status = storage_.Read();
  if (status != kMinibenchmarkSuccess) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Storage::Read failed");
    return status;
  }
  return validator_runner_impl_->Init();
}

int ValidatorRunner::TriggerMissingValidation(
    const std::vector<const TFLiteSettings*>& for_settings) {
  if (triggered_) {
    return 0;
  }
  triggered_ = true;
  storage_.Read();

  // Filter out settings that have already been tried.
  std::vector<flatbuffers::FlatBufferBuilder> to_be_run;
  for (auto settings : for_settings) {
    TFLiteSettingsT tflite_settings;
    settings->UnPackTo(&tflite_settings);
    int started_count = 0;
    int results_count = 0;
    for (int i = 0; i < storage_.Count(); i++) {
      const BenchmarkEvent* event = storage_.Get(i);
      if (event->event_type() == BenchmarkEventType_LOGGED) {
        continue;
      }
      if (!event->tflite_settings()) {
        TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                        "Got previous event %d type %d with no tflite settings",
                        i, static_cast<int>(event->event_type()));
        continue;
      }
      TFLiteSettingsT event_settings;
      event->tflite_settings()->UnPackTo(&event_settings);
      if (event_settings != tflite_settings) {
        continue;
      }
      if (event->event_type() == BenchmarkEventType_START) {
        started_count++;
      } else if (event->event_type() == BenchmarkEventType_END) {
        results_count++;
      }
    }
    if (results_count > 0 || started_count >= kMaxAttempts) {
      continue;
    }
    flatbuffers::FlatBufferBuilder copy;
    copy.Finish(CreateTFLiteSettings(copy, &tflite_settings));
    to_be_run.emplace_back(std::move(copy));
  }
  int to_be_run_count = to_be_run.size();
  validator_runner_impl_->TriggerValidationAsync(std::move(to_be_run),
                                                 storage_path_);
  return to_be_run_count;
}

std::vector<const BenchmarkEvent*> ValidatorRunner::GetAndFlushEventsToLog(
    int64_t timeout_us) {
  std::vector<const BenchmarkEvent*> events;
  storage_.Read();
  if (storage_.Count() == 0) {
    return events;
  }
  const BenchmarkEvent* last = storage_.Get(storage_.Count() - 1);
  if (!last || last->event_type() == BenchmarkEventType_LOGGED) {
    return events;
  }
  bool has_pending_event = false;
  for (int i = storage_.Count() - 1; i >= 0; i--) {
    const BenchmarkEvent* event = storage_.Get(i);
    if (!event || event->event_type() == BenchmarkEventType_LOGGED) {
      break;
    } else if (event->event_type() == BenchmarkEventType_END ||
               event->event_type() == BenchmarkEventType_ERROR) {
      break;
    } else if (event->event_type() == BenchmarkEventType_START &&
               std::abs(event->boottime_us() - Validator::BootTimeMicros()) <
                   timeout_us) {
      has_pending_event = true;
    }
  }
  if (has_pending_event) {
    return events;
  }

  flatbuffers::FlatBufferBuilder fbb;
  int64_t boottime_us = Validator::BootTimeMicros();
  storage_.Append(
      &fbb, CreateBenchmarkEvent(fbb, 0, BenchmarkEventType_LOGGED,
                                 /* result */ 0, /* error */ 0, boottime_us,
                                 Validator::WallTimeMicros()));
  storage_.Read();
  // Track whether we've seen an end result after a start. We lock around
  // validation so there should be no interleaved events from different runs.
  bool seen_end = false;
  for (int i = storage_.Count() - 1; i >= 0; i--) {
    const BenchmarkEvent* event = storage_.Get(i);
    if (!event || (event->event_type() == BenchmarkEventType_LOGGED &&
                   event->boottime_us() != boottime_us)) {
      // This is the previous log marker, events before it have been already
      // logged.
      // It can happen, that we read at the same time as validation is
      // writing, so that new entries were written after our log marker. In
      // that case those events will be logged twice, which is not terrible.
      break;
    }
    if (event->event_type() == BenchmarkEventType_END ||
        event->event_type() == BenchmarkEventType_ERROR ||
        event->event_type() == BenchmarkEventType_RECOVERED_ERROR) {
      events.push_back(event);
      seen_end = true;
    } else if (event->event_type() == BenchmarkEventType_START) {
      if (!seen_end) {
        // Incomplete test, start without end.
        events.push_back(event);
      } else {
        seen_end = false;
      }
    }
  }
  return events;
}

}  // namespace acceleration
}  // namespace tflite
