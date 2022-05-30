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

#include <fcntl.h>

#ifndef _WIN32
#include <dlfcn.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif  // !_WIN32

#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>  // NOLINT: code only used on Android, where std::thread is allowed
#include <vector>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/runner.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace acceleration {

constexpr int kMaxAttempts = 2;
constexpr int64_t ValidatorRunner::kDefaultEventTimeoutUs;

using ::tflite::nnapi::NnApiSupportLibrary;

ValidatorRunner::ValidatorRunner(const std::string& model_path,
                                 const std::string& storage_path,
                                 const std::string& data_directory_path,
                                 const NnApiSLDriverImplFL5* nnapi_sl,
                                 const std::string validation_function_name,
                                 ErrorReporter* error_reporter)
    : model_path_(model_path),
      storage_path_(storage_path),
      data_directory_path_(data_directory_path),
      storage_(storage_path_, error_reporter),
      validation_function_name_(validation_function_name),
      error_reporter_(error_reporter),
      nnapi_sl_(nnapi_sl) {}

ValidatorRunner::ValidatorRunner(int model_fd, size_t model_offset,
                                 size_t model_size,
                                 const std::string& storage_path,
                                 const std::string& data_directory_path,
                                 const NnApiSLDriverImplFL5* nnapi_sl,
                                 const std::string validation_function_name,
                                 ErrorReporter* error_reporter)
    :
#ifndef _WIN32
      model_fd_(dup(model_fd)),
#else   // _WIN32
      model_fd_(-1),
#endif  // !_WIN32
      model_offset_(model_offset),
      model_size_(model_size),
      storage_path_(storage_path),
      data_directory_path_(data_directory_path),
      storage_(storage_path_, error_reporter),
      validation_function_name_(validation_function_name),
      error_reporter_(error_reporter),
      nnapi_sl_(nnapi_sl) {
}

MinibenchmarkStatus ValidatorRunner::Init() {
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(CreateComputeSettings(fbb, tflite::ExecutionPreference_ANY,
                                   CreateTFLiteSettings(fbb)));
  std::unique_ptr<Validator> check_validator;
  // We are not configuring the validator to use the NNAPI Support Library
  // even if specified since we just want to check that the model can be loaded
  // from disk and we are not interacting with NNAPI.
  if (!model_path_.empty()) {
    check_validator = std::make_unique<Validator>(
        model_path_,
        flatbuffers::GetRoot<ComputeSettings>(fbb.GetBufferPointer()));
  } else {
    check_validator = std::make_unique<Validator>(
        model_fd_, model_offset_, model_size_,
        flatbuffers::GetRoot<ComputeSettings>(fbb.GetBufferPointer()));
  }
  MinibenchmarkStatus load_status =
      check_validator->CheckModel(/* load_only */ true);
  if (load_status != kMinibenchmarkSuccess) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Could not load model %s: %d",
                         model_path_.c_str(), static_cast<int>(load_status));
    return load_status;
  }

#ifndef _WIN32
  int (*validation_entrypoint)(int, char**) =
      reinterpret_cast<int (*)(int, char**)>(
          dlsym(RTLD_DEFAULT, validation_function_name_.c_str()));
  if (!validation_entrypoint) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Could not load symbol '%s': '%s'",
                         validation_function_name_.c_str(), dlerror());
    return kMinibenchmarkValidationEntrypointSymbolNotFound;
  }
  ProcessRunner check_runner(data_directory_path_,
                             validation_function_name_.c_str(),
                             validation_entrypoint);
  MinibenchmarkStatus status = check_runner.Init();
  if (status != kMinibenchmarkSuccess) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Runner::Init returned %d",
                         static_cast<int>(status));
    return status;
  }

  status = storage_.Read();
  if (status != kMinibenchmarkSuccess) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Storage::Read failed");
    return status;
  }

  if (nnapi_sl_) {
    Dl_info dl_info;
    // Looking for the file where the NNAPI SL is loaded from. We are using
    // the ANeuralNetworks_getRuntimeFeatureLevel because it is a required
    // function for NNAPI drivers.
    // If the function is not defined or it wasn't defined in any of the shared
    // libraries loaded by the calling process we fail with a specific error
    // code.
    // This could happen only if the NNAPI Support Library pointer set into
    // our TfLiteSettings comes from an invalid NNAPI SL library or there
    // is some error in the NNAPI loading code.
    if (!nnapi_sl_->ANeuralNetworks_getRuntimeFeatureLevel) {
      return kMiniBenchmarkCannotLoadSupportLibrary;
    }
    int status = dladdr(reinterpret_cast<void*>(
                            nnapi_sl_->ANeuralNetworks_getRuntimeFeatureLevel),
                        &dl_info);
    if (status == 0 || !dl_info.dli_fname) {
      return kMiniBenchmarkCannotLoadSupportLibrary;
    }
    nnapi_sl_path_ = dl_info.dli_fname;
  }

  return kMinibenchmarkSuccess;
#else   // _WIN32
  return kMinibenchmarkUnsupportedPlatform;
#endif  // !_WIN32
}

int ValidatorRunner::TriggerMissingValidation(
    std::vector<const TFLiteSettings*> for_settings) {
  if (triggered_) {
    return 0;
  }
  triggered_ = true;
  storage_.Read();

  // Filter out settings that have already been tried.
  std::vector<const flatbuffers::FlatBufferBuilder*> to_be_run;
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
    flatbuffers::FlatBufferBuilder* copy = new flatbuffers::FlatBufferBuilder;
    copy->Finish(CreateTFLiteSettings(*copy, &tflite_settings));
    to_be_run.push_back(copy);
  }
  if (to_be_run.empty()) {
    // We expect this to be the common case, as validation needs to only be
    // run when the app is updated.
    return 0;
  }

  std::string model_path;
  if (!model_path_.empty()) {
    model_path = model_path_;
  } else {
    std::stringstream ss;
    ss << "fd:" << model_fd_ << ":" << model_offset_ << ":" << model_size_;
    model_path = ss.str();
  }

  // We purposefully detach the thread and have it own all the data. The
  // runner may potentially hang, so we can't wait for it to terminate.
  std::thread detached_thread([model_path = model_path,
                               storage_path = storage_path_,
                               data_directory_path = data_directory_path_,
                               to_be_run,
                               validation_function_name =
                                   validation_function_name_,
                               nnapi_sl_path = nnapi_sl_path_]() {
    FileLock lock(storage_path + ".parent_lock");
    if (!lock.TryLock()) {
      return;
    }
    for (auto one_to_run : to_be_run) {
      FlatbufferStorage<BenchmarkEvent> storage(storage_path);
      TFLiteSettingsT tflite_settings;
      flatbuffers::GetRoot<TFLiteSettings>(one_to_run->GetBufferPointer())
          ->UnPackTo(&tflite_settings);
      int (*validation_entrypoint)(int, char**) = nullptr;
      TFLITE_LOG_PROD(TFLITE_LOG_INFO, "Loading validation entry point '%s'",
                      validation_function_name.c_str());
#ifndef _WIN32
      validation_entrypoint = reinterpret_cast<int (*)(int, char**)>(
          dlsym(RTLD_DEFAULT, validation_function_name.c_str()));
#endif  // !_WIN32
      ProcessRunner runner(data_directory_path,
                           validation_function_name.c_str(),
                           validation_entrypoint);
      int exitcode = 0;
      int signal = 0;
      MinibenchmarkStatus status = runner.Init();
      if (status == kMinibenchmarkSuccess) {
        flatbuffers::FlatBufferBuilder fbb;
        status = storage.Append(
            &fbb,
            CreateBenchmarkEvent(
                fbb, CreateTFLiteSettings(fbb, &tflite_settings),
                BenchmarkEventType_START, /* result */ 0, /* error */ 0,
                Validator::BootTimeMicros(), Validator::WallTimeMicros()));
        if (status == kMinibenchmarkSuccess) {
          std::vector<std::string> args{model_path, storage_path,
                                        data_directory_path};
          if (!nnapi_sl_path.empty() &&
              tflite_settings.delegate == tflite::Delegate_NNAPI) {
            TFLITE_LOG_PROD(
                TFLITE_LOG_INFO,
                "Running benchmark using NNAPI support library at path '%s'",
                nnapi_sl_path.c_str());
            args.push_back(nnapi_sl_path);
          }
          std::string output;
          status = runner.Run(args, &output, &exitcode, &signal);
        }
      }
      if (status != kMinibenchmarkSuccess) {
        std::cout << "Run() returned " << status << std::endl;
        flatbuffers::FlatBufferBuilder fbb;
        storage.Append(
            &fbb,
            CreateBenchmarkEvent(
                fbb, CreateTFLiteSettings(fbb, &tflite_settings),
                BenchmarkEventType_ERROR, /* result */ 0,
                CreateBenchmarkError(fbb, BenchmarkStage_UNKNOWN, status,
                                     signal, {}, exitcode),
                Validator::BootTimeMicros(), Validator::WallTimeMicros()));
      }
      delete one_to_run;
    }
  });
  detached_thread.detach();

  return to_be_run.size();
}

std::vector<const BenchmarkEvent*> ValidatorRunner::GetSuccessfulResults() {
  std::vector<const BenchmarkEvent*> results;
  storage_.Read();
  for (int i = 0; i < storage_.Count(); i++) {
    const BenchmarkEvent* event = storage_.Get(i);
    if (event->event_type() == BenchmarkEventType_END && event->result() &&
        event->result()->ok()) {
      results.push_back(event);
    }
  }
  return results;
}

int ValidatorRunner::GetNumCompletedResults() {
  storage_.Read();
  int num_results = 0;
  for (int i = 0; i < storage_.Count(); i++) {
    const BenchmarkEvent* event = storage_.Get(i);
    if (event->event_type() == BenchmarkEventType_ERROR ||
        (event->event_type() == BenchmarkEventType_END && event->result())) {
      num_results++;
    }
  }
  return num_results;
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
