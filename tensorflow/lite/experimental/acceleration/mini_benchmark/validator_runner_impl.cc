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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_impl.h"

#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <thread>  // NOLINT: code only used on Android, where std::thread is allowed
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/benchmark_result_evaluator.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/file_lock.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/model_modifier/custom_validation_embedder.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/runner.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/tools/model_loader.h"

namespace tflite {
namespace acceleration {
namespace {
using ::flatbuffers::FlatBufferBuilder;

// If input is not null, make a copy of the data pointed by input, and create a
// new Allocation pointing to the copied data.
std::pair<std::unique_ptr<Allocation>, std::vector<uint8_t>> CopyModel(
    const Allocation* input, ErrorReporter* error_reporter) {
  std::vector<uint8_t> copy;
  if (!input) {
    return {nullptr, copy};
  }

  copy.resize(input->bytes());
  memcpy(copy.data(), input->base(), input->bytes());

  return {std::make_unique<MemoryAllocation>(copy.data(), copy.size(),
                                             error_reporter),
          std::move(copy)};
}

// A simple holder for file descriptor that will close the file descriptor at
// destruction time.
class FdHolder {
 public:
  explicit FdHolder(int fd) : fd_(fd) {}

  // Move only.
  FdHolder(FdHolder&& other) = default;
  FdHolder& operator=(FdHolder&& other) = default;

  ~FdHolder() {
    if (fd_ > 0) {
      close(fd_);
    }
  }

 private:
  int fd_;
};

// Returns a FdHolder that will close the duped file descriptor when going out
// of scope. If the model is passed in as a file descriptor, update the
// model_path with a duped file descriptor. The original file descriptor may be
// opened with FD_CLOEXEC, and cannot be read from the child process.
std::unique_ptr<FdHolder> UpdateModelPathIfUsingFd(std::string& model_path) {
  if (!absl::StartsWith(model_path, "fd:")) {
    return nullptr;
  }
  std::vector<std::string> parts = absl::StrSplit(model_path, ':');
  int model_fd;
  if (!absl::SimpleAtoi(parts[1], &model_fd)) {
    TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                    "Failed to parse file descriptor %s from model_path %s",
                    parts[1].c_str(), model_path.c_str());
    return nullptr;
  }
  int new_fd = dup(model_fd);
  if (new_fd < 0) {
    TFLITE_LOG_PROD(
        TFLITE_LOG_ERROR,
        "Failed to dup() file descriptor. Original fd: %d errno: %d", model_fd,
        errno);
    return nullptr;
  }

  parts[1] = std::to_string(new_fd);
  model_path = absl::StrJoin(parts, ":");
  return std::make_unique<FdHolder>(new_fd);
}

}  // namespace

MinibenchmarkStatus ValidatorRunnerImpl::Init() {
  if (storage_path_.empty()) {
    TF_LITE_REPORT_ERROR(error_reporter_, "storage_path is empty.");
    return kMinibenchmarkPreconditionNotMet;
  }
  if (data_directory_path_.empty()) {
    TF_LITE_REPORT_ERROR(error_reporter_, "data_directory_path is empty.");
    return kMinibenchmarkPreconditionNotMet;
  }
  if (benchmark_evaluator_ == nullptr) {
    TF_LITE_REPORT_ERROR(error_reporter_, "benchmark_evaluator is null.");
    return kMinibenchmarkPreconditionNotMet;
  }
  MinibenchmarkStatus status = storage_.Read();
  if (status != kMinibenchmarkSuccess) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Storage::Read failed.");
    return status;
  }

  std::unique_ptr<tools::ModelLoader> model_loader =
      tools::CreateModelLoaderFromPath(fd_or_model_path_);
  if (!model_loader) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Failed to parse model path.");
    return kMinibenchmarkPreconditionNotMet;
  }

  // Check that the model can be loaded from disk.
  if (!model_loader->Init() || !model_loader->GetModel()) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Could not load model.");
    return kMinibenchmarkModelInitFailed;
  }

  if (custom_validation_embedder_) {
    status = custom_validation_embedder_->BuildModel(
        *model_loader->GetModel()->GetModel(), model_with_custom_input_);
    if (status != kMinibenchmarkSuccess) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Failed to embed golden input to model: %d",
                           static_cast<int>(status));
      return status;
    }
    model_allocation_ = std::make_unique<MemoryAllocation>(
        model_with_custom_input_.GetBufferPointer(),
        model_with_custom_input_.GetSize(), error_reporter_);
  } else if (dynamic_cast<tools::BufferModelLoader*>(model_loader.get())) {
    // If model is already loaded, it needs to be copied to the detached thread.
    const Allocation* alloc = model_loader->GetModel()->allocation();
    if (!alloc || !alloc->valid() || !alloc->base() || alloc->bytes() <= 0) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Internal error: BufferModelLoader doesn't have a "
                           "valid allocation.");
      return kMinibenchmarkPreconditionNotMet;
    }

    model_allocation_ = std::make_unique<MemoryAllocation>(
        alloc->base(), alloc->bytes(), error_reporter_);
  }

  status = nnapi_helper_.Load();
  if (status != kMinibenchmarkSuccess) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Failed to load NNAPI SL: %d",
                         static_cast<int>(status));
    return status;
  }

  status = gpu_helper_.Load();
  if (status != kMinibenchmarkSuccess) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Failed to load GPU Module: %d",
                         static_cast<int>(status));
    return status;
  }

  status = validation_entrypoint_helper_.Validate();
  if (status != kMinibenchmarkSuccess) {
    return status;
  }

  ProcessRunner check_runner(data_directory_path_,
                             validation_entrypoint_helper_.name().c_str(),
                             validation_entrypoint_helper_.LoadEntrypoint(),
                             timeout_ms_, error_reporter_);
  status = check_runner.Init();
  if (status != kMinibenchmarkSuccess) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Runner::Init returned %d",
                         static_cast<int>(status));
    return status;
  }
  return kMinibenchmarkSuccess;
}

void ValidatorRunnerImpl::TriggerValidationAsync(
    std::unique_ptr<std::vector<FlatBufferBuilder>> tflite_settings) {
  if (!tflite_settings || tflite_settings->empty()) {
    return;
  }

  // We purposefully detach the thread and have it own all the data. The
  // runner may potentially hang, so we can't wait for it to terminate.
  // error_reporter is not passed in because the ownership cannot be passed to
  // the thread. Model data is copied from model_allocation_ if set and owned by
  // the thread.
  std::thread detached_thread(
      [original_model_path = fd_or_model_path_, storage_path = storage_path_,
       data_directory_path = data_directory_path_,
       tflite_settings = std::move(tflite_settings),
       validation_entrypoint_name =
           validation_entrypoint_helper_.name().c_str(),
       validation_entrypoint = validation_entrypoint_helper_.LoadEntrypoint(),
       nnapi_sl_path = nnapi_helper_.nnapi_sl_path(),
       gpu_so_path = gpu_helper_.gpu_so_path(),
       allocation_and_model =
           CopyModel(model_allocation_.get(), error_reporter_),
       timeout_ms = timeout_ms_]() {
        FileLock lock(storage_path + ".parent_lock");
        if (!lock.TryLock()) {
          return;
        }

        std::string model_path = original_model_path;
        std::unique_ptr<FdHolder> fd_holder =
            UpdateModelPathIfUsingFd(model_path);
        for (auto& one_setting : *tflite_settings) {
          FlatbufferStorage<BenchmarkEvent> storage(storage_path);
          TFLiteSettingsT tflite_settings_obj;
          flatbuffers::GetRoot<TFLiteSettings>(one_setting.GetBufferPointer())
              ->UnPackTo(&tflite_settings_obj);
          TFLITE_LOG_PROD(TFLITE_LOG_INFO,
                          "Run validation with entry point '%s'",
                          validation_entrypoint_name);
          ProcessRunner runner(data_directory_path, validation_entrypoint_name,
                               validation_entrypoint, timeout_ms);
          int exitcode = 0;
          int signal = 0;
          MinibenchmarkStatus status = runner.Init();
          if (status == kMinibenchmarkSuccess) {
            // Write START event to storage.
            flatbuffers::FlatBufferBuilder fbb;
            status = storage.Append(
                &fbb,
                CreateBenchmarkEvent(
                    fbb, CreateTFLiteSettings(fbb, &tflite_settings_obj),
                    BenchmarkEventType_START, /* result */ 0, /* error */ 0,
                    Validator::BootTimeMicros(), Validator::WallTimeMicros()));
          }
          if (status != kMinibenchmarkSuccess) {
            flatbuffers::FlatBufferBuilder fbb;
            storage.Append(
                &fbb,
                CreateBenchmarkEvent(
                    fbb, CreateTFLiteSettings(fbb, &tflite_settings_obj),
                    BenchmarkEventType_ERROR, /* result */ 0,
                    CreateBenchmarkError(fbb, BenchmarkStage_INITIALIZATION,
                                         exitcode, signal, /* error_code */ {},
                                         status),
                    Validator::BootTimeMicros(), Validator::WallTimeMicros()));
            continue;
          }
          std::vector<std::string> args;
          if (!allocation_and_model.first) {
            args.push_back(model_path);
          }
          args.push_back(storage_path);
          args.push_back(data_directory_path);
          // If NNAPI or GPU is provided as a shared object file, pass the file
          // path as a commandline flag.
          if (tflite_settings_obj.delegate == tflite::Delegate_NNAPI &&
              !nnapi_sl_path.empty()) {
            TFLITE_LOG_PROD(
                TFLITE_LOG_INFO,
                "Running benchmark using NNAPI support library at path '%s'",
                nnapi_sl_path.c_str());
            args.push_back(nnapi_sl_path);
          } else if (tflite_settings_obj.delegate == tflite::Delegate_GPU &&
                     !gpu_so_path.empty()) {
            TFLITE_LOG_PROD(
                TFLITE_LOG_INFO,
                "Running benchmark using GPU Delegate Module at path '%s'",
                gpu_so_path.c_str());
            args.push_back(gpu_so_path);
          }

          std::string output;
          status = runner.Run(allocation_and_model.first.get(), args, &output,
                              &exitcode, &signal);
          if (status != kMinibenchmarkSuccess) {
            std::cout << "Run() returned " << status << std::endl;
            flatbuffers::FlatBufferBuilder fbb;
            storage.Append(
                &fbb,
                CreateBenchmarkEvent(
                    fbb, CreateTFLiteSettings(fbb, &tflite_settings_obj),
                    BenchmarkEventType_ERROR, /* result */ 0,
                    CreateBenchmarkError(fbb, BenchmarkStage_UNKNOWN, exitcode,
                                         signal, {}, status),
                    Validator::BootTimeMicros(), Validator::WallTimeMicros()));
          }
        }
      });
  detached_thread.detach();
}

std::vector<const BenchmarkEvent*>
ValidatorRunnerImpl::GetSuccessfulResultsFromStorage() {
  std::vector<const BenchmarkEvent*> results;
  storage_.Read();
  for (int i = 0; i < storage_.Count(); i++) {
    const BenchmarkEvent* event = storage_.Get(i);
    TFLITE_LOG_PROD(TFLITE_LOG_WARNING, "Benchmark event(%d).",
                    event->event_type());

    if (benchmark_evaluator_->IsValidationSuccessEvent(*event)) {
      results.push_back(event);
    } else if (event->event_type() == BenchmarkEventType_ERROR) {
      TFLITE_LOG(
          TFLITE_LOG_WARNING,
          "Benchmark event failed with error code (%d), signal (%d), exit code "
          "(%d), stage (%d), mini benchmark error code (%d).\n",
          event->error()->error_code(), event->error()->signal(),
          event->error()->exit_code(), event->error()->stage(),
          event->error()->mini_benchmark_error_code());
    }
  }
  return results;
}

std::vector<FlatBufferBuilder> ValidatorRunnerImpl::GetCompletedResults() {
  storage_.Read();
  std::vector<FlatBufferBuilder> results;
  for (int i = 0; i < storage_.Count(); i++) {
    const BenchmarkEvent* event = storage_.Get(i);
    if (event->event_type() != BenchmarkEventType_ERROR &&
        event->event_type() != BenchmarkEventType_END) {
      continue;
    }
    BenchmarkEventT event_obj;
    event->UnPackTo(&event_obj);

    if (benchmark_evaluator_->IsValidationSuccessEvent(*event)) {
      event_obj.result->ok = true;
    }

    FlatBufferBuilder fbb;
    fbb.Finish(CreateBenchmarkEvent(fbb, &event_obj));
    results.emplace_back(std::move(fbb));
  }
  return results;
}

int ValidatorRunnerImpl::GetNumCompletedResults() {
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

MinibenchmarkStatus
ValidatorRunnerImpl::ValidationEntrypointHelper::Validate() {
#ifndef _WIN32
  if (!LoadEntrypoint()) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Could not load symbol '%s': '%s'",
                         validation_entrypoint_name_.c_str(), dlerror());
    return kMinibenchmarkValidationEntrypointSymbolNotFound;
  }
  return kMinibenchmarkSuccess;
#else   // _WIN32
  return kMinibenchmarkUnsupportedPlatform;
#endif  // !_WIN32
}

ValidatorRunnerImpl::ValidationEntrypointHelper::EntrypointFunc*
ValidatorRunnerImpl::ValidationEntrypointHelper::LoadEntrypoint() {
#ifndef _WIN32
  // We use dlsym() to lookup the entrypoint function every time because this
  // helper is used in a multi-threaded environment.
  return reinterpret_cast<int (*)(int, char**)>(
      dlsym(RTLD_DEFAULT, validation_entrypoint_name_.c_str()));
#endif  // !_WIN32
  return nullptr;
}

MinibenchmarkStatus ValidatorRunnerImpl::NnapiHelper::Load() {
  if (nnapi_sl_) {
#ifndef _WIN32
    Dl_info dl_info;
    // Looking for the file where the NNAPI SL is loaded from. We are using
    // the ANeuralNetworks_getRuntimeFeatureLevel because it is a required
    // function for NNAPI drivers.
    // If the function is not defined or it wasn't defined in any of the
    // shared libraries loaded by the calling process we fail with a
    // specific error code. This could happen only if the NNAPI Support
    // Library pointer set into our TfLiteSettings comes from an invalid
    // NNAPI SL library or there is some error in the NNAPI loading code.
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
#else   // _WIN32
    return kMinibenchmarkUnsupportedPlatform;
#endif  // !_WIN32
  }
  return kMinibenchmarkSuccess;
}

MinibenchmarkStatus ValidatorRunnerImpl::GpuHelper::Load() {
  if (gpu_plugin_handle_) {
#ifndef _WIN32
    Dl_info dl_info;
    // Looking for the file where GPU is loaded from. This file will be passed
    // to validator in a separate process.
    int status = dladdr(gpu_plugin_handle_, &dl_info);
    if (status == 0 || !dl_info.dli_fname) {
      return kMinibenchmarkCannotLoadGpuModule;
    }
    gpu_so_path_ = dl_info.dli_fname;
  }
#else   // _WIN32
    return kMinibenchmarkUnsupportedPlatform;
  }
#endif  // !_WIN32
  return kMinibenchmarkSuccess;
}
}  // namespace acceleration
}  // namespace tflite
