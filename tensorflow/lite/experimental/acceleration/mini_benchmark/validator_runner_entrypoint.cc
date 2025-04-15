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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_entrypoint.h"

#include <dlfcn.h>

#include <memory>
#include <string>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/constants.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/file_lock.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/set_big_core_affinity.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator.h"
#include "tensorflow/lite/nnapi/sl/include/SupportLibrary.h"
#include "tensorflow/lite/tools/model_loader.h"

namespace tflite {
namespace acceleration {
namespace {

using flatbuffers::Offset;

Validator::Status RunValidator(const std::string& model_path,
                               const std::string& delegate_so_path,
                               const TFLiteSettingsT& tflite_settings,
                               Validator::Results& results) {
  // Make a copy of tflite_settings for modification, so that the original
  // tflite_settings will be written to storage. BlockingValidatorRunner
  // compares the input and output tflite_settings to make sure all configs have
  // corresponding results.
  TFLiteSettingsT copy(tflite_settings);
  // Load Delegate Library if specified.
  std::unique_ptr<const ::tflite::nnapi::NnApiSupportLibrary> nnapi_sl_handle;
  if (!delegate_so_path.empty()) {
    if (tflite_settings.nnapi_settings) {
      // We are not calling dlclose, it will be done once the
      // validator process ends.
      nnapi_sl_handle =
          ::tflite::nnapi::loadNnApiSupportLibrary(delegate_so_path);

      if (!nnapi_sl_handle) {
        return Validator::Status{kMiniBenchmarkCannotLoadSupportLibrary,
                                 BenchmarkStage_INITIALIZATION};
      }

      copy.nnapi_settings->support_library_handle =
          reinterpret_cast<uint64_t>(nnapi_sl_handle->getFL5());
    } else if (tflite_settings.gpu_settings) {
      // Pass the module file path to GpuModulePlugin to load. GPU delegate is
      // not currently using the stable delegate API, but uses the same
      // mechanism to load.
      // TODO(b/266066861): Migrate to stable delegate API once it's launched.
      copy.stable_delegate_loader_settings =
          std::make_unique<StableDelegateLoaderSettingsT>();
      copy.stable_delegate_loader_settings->delegate_path = delegate_so_path;
    }
  }

  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(CreateComputeSettings(fbb, ExecutionPreference_ANY,
                                   CreateTFLiteSettings(fbb, &copy)));
  std::unique_ptr<tools::ModelLoader> model_loader =
      tools::CreateModelLoaderFromPath(model_path);
  if (!model_loader) {
    return Validator::Status{kMinibenchmarkPreconditionNotMet,
                             BenchmarkStage_INITIALIZATION};
  }

  auto validator = std::make_unique<Validator>(
      std::move(model_loader),
      flatbuffers::GetRoot<ComputeSettings>(fbb.GetBufferPointer()));

  return validator->RunValidation(&results);
}

}  // namespace

extern "C" {

int Java_org_tensorflow_lite_acceleration_validation_entrypoint(int argc,
                                                                char** argv) {
  if (argc < 6) return 1;
  // argv[1] is the helper binary name
  // argv[2] is the function name
  const std::string model_path = argv[3];
  const std::string storage_path = argv[4];
  // argv[5] is data directory path.
  // argv[6] if present is the NNAPI SL path
  const std::string nnapi_sl_path = argc > 6 ? argv[6] : "";
  FileLock lock(storage_path + ".child_lock");
  if (!lock.TryLock()) {
    return kMinibenchmarkChildProcessAlreadyRunning;
  }

  // Write subprocess id first after getting the lock. The length is fixed to
  // make sure the first read on the reader side won't block.
  // Note: The ProcessRunner implementation expects the subprocess to not block
  // on write(). It may hang if this function start to write more data to
  // stdout.
  std::string pid = std::to_string(getpid());
  pid.resize(kPidBufferLength);
  if (write(1, pid.data(), kPidBufferLength) != kPidBufferLength) {
    return kMinibenchmarkPreconditionNotMet;
  }

  FlatbufferStorage<BenchmarkEvent> storage(storage_path);
  MinibenchmarkStatus read_status = storage.Read();
  if (read_status != kMinibenchmarkSuccess) {
    return read_status;
  }
  TFLiteSettingsT tflite_settings;

  int32_t set_big_core_affinity_errno = SetBigCoresAffinity();
  if (set_big_core_affinity_errno != 0) {
    flatbuffers::FlatBufferBuilder fbb;
    storage.Append(
        &fbb,
        CreateBenchmarkEvent(
            fbb, CreateTFLiteSettings(fbb, &tflite_settings),
            BenchmarkEventType_RECOVERED_ERROR, /* result = */ 0,
            // There is no dedicated field for the errno, so we pass it as
            // exit_code instead.
            CreateBenchmarkError(fbb, BenchmarkStage_INITIALIZATION,
                                 /* exit_code = */ set_big_core_affinity_errno,
                                 /* signal = */ 0,
                                 /* error_code = */ 0,
                                 /* mini_benchmark_error_code = */
                                 kMinibenchmarkUnableToSetCpuAffinity),
            Validator::BootTimeMicros(), Validator::WallTimeMicros()));
  }

  Validator::Status run_status =
      Validator::Status{kMinibenchmarkNoValidationRequestFound};

  for (int i = storage.Count() - 1; i >= 0; i--) {
    const BenchmarkEvent* event = storage.Get(i);
    if (event->event_type() == BenchmarkEventType_START) {
      event->tflite_settings()->UnPackTo(&tflite_settings);

      Validator::Results results;
      run_status =
          RunValidator(model_path, nnapi_sl_path, tflite_settings, results);
      if (run_status.status != kMinibenchmarkSuccess) {
        break;
      }

      // If succeed, write MiniBenchmark output to file then return.
      flatbuffers::FlatBufferBuilder fbb;
      std::vector<int64_t> delegate_prep_time_us{results.delegate_prep_time_us};
      std::vector<Offset<tflite::BenchmarkMetric>> metrics;
      metrics.reserve(results.metrics.size());
      for (const auto& name_and_values : results.metrics) {
        metrics.push_back(
            CreateBenchmarkMetric(fbb, fbb.CreateString(name_and_values.first),
                                  fbb.CreateVector(name_and_values.second)));
      }
      std::vector<Offset<BenchmarkResult_::InferenceOutput>> actual_output;
      for (const auto& output : results.actual_inference_output) {
        const uint8_t* output_uint8 =
            reinterpret_cast<const uint8_t*>(output.data());
        actual_output.push_back(BenchmarkResult_::CreateInferenceOutput(
            fbb, fbb.CreateVector(output_uint8, output.size())));
      }
      return storage.Append(
          &fbb,
          CreateBenchmarkEvent(
              fbb, CreateTFLiteSettings(fbb, &tflite_settings),
              BenchmarkEventType_END,
              CreateBenchmarkResult(
                  fbb, fbb.CreateVector(delegate_prep_time_us),
                  fbb.CreateVector(results.execution_time_us), 0, results.ok,
                  fbb.CreateVector(metrics), fbb.CreateVector(actual_output)),
              /* error */ 0, Validator::BootTimeMicros(),
              Validator::WallTimeMicros()));
    }
  }
  // Write error to file.
  flatbuffers::FlatBufferBuilder fbb;
  return storage.Append(
      &fbb, CreateBenchmarkEvent(
                fbb, CreateTFLiteSettings(fbb, &tflite_settings),
                BenchmarkEventType_ERROR, /* result */ 0,
                CreateBenchmarkError(fbb, run_status.stage, /* exit_code */ 0,
                                     /* signal */ 0, /* error_code */ 0,
                                     run_status.status),
                Validator::BootTimeMicros(), Validator::WallTimeMicros()));
}

}  // extern "C"
}  // namespace acceleration
}  // namespace tflite

#endif  // !_WIN32
