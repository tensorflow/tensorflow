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

#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_validation_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_nnapi_sl_fake_impl.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/nnapi_sl_fake_impl.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/nnapi/sl/include/SupportLibrary.h"

#ifdef __ANDROID__
#include <dlfcn.h>

#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_validator_runner_entrypoint.h"
#endif  // __ANDROID__

namespace tflite {
namespace acceleration {
namespace {

std::vector<const TFLiteSettings*> BuildBenchmarkSettings(
    const AndroidInfo& android_info, flatbuffers::FlatBufferBuilder& fbb_cpu,
    flatbuffers::FlatBufferBuilder& fbb_nnapi,
    flatbuffers::FlatBufferBuilder& fbb_gpu,
    bool ignore_android_version = false) {
  std::vector<const TFLiteSettings*> settings;
  fbb_cpu.Finish(CreateTFLiteSettings(fbb_cpu, Delegate_NONE,
                                      CreateNNAPISettings(fbb_cpu)));
  settings.push_back(
      flatbuffers::GetRoot<TFLiteSettings>(fbb_cpu.GetBufferPointer()));
  if (ignore_android_version || android_info.android_sdk_version >= "28") {
    fbb_nnapi.Finish(CreateTFLiteSettings(fbb_nnapi, Delegate_NNAPI,
                                          CreateNNAPISettings(fbb_nnapi)));
    settings.push_back(
        flatbuffers::GetRoot<TFLiteSettings>(fbb_nnapi.GetBufferPointer()));
  }

#ifdef __ANDROID__
  fbb_gpu.Finish(CreateTFLiteSettings(fbb_gpu, Delegate_GPU));
  settings.push_back(
      flatbuffers::GetRoot<TFLiteSettings>(fbb_gpu.GetBufferPointer()));
#endif  // __ANDROID__

  return settings;
}

std::string GetTargetDeviceName(const BenchmarkEvent* event) {
  if (event->tflite_settings()->delegate() == Delegate_GPU) {
    return "GPU";
  } else if (event->tflite_settings()->delegate() == Delegate_NNAPI) {
    return "NNAPI";
  }
  return "CPU";
}

class ValidatorRunnerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    MiniBenchmarkTestHelper helper;
    should_perform_test_ = helper.should_perform_test();

    if (!should_perform_test_) {
      return;
    }

    model_path_ = helper.DumpToTempFile(
        "mobilenet_quant_with_validation.tflite",
        g_tflite_acceleration_embedded_mobilenet_validation_model,
        g_tflite_acceleration_embedded_mobilenet_validation_model_len);
    ASSERT_TRUE(!model_path_.empty());

#ifdef __ANDROID__
    // We extract the test files here as that's the only way to get the right
    // architecture when building tests for multiple architectures.
    std::string entry_point_file = MiniBenchmarkTestHelper::DumpToTempFile(
        "libvalidator_runner_entrypoint.so",
        g_tflite_acceleration_embedded_validator_runner_entrypoint,
        g_tflite_acceleration_embedded_validator_runner_entrypoint_len);
    ASSERT_TRUE(!entry_point_file.empty());

    void* module =
        dlopen(entry_point_file.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    EXPECT_TRUE(module) << dlerror();
#endif  // __ANDROID__
  }

  void CheckConfigurations(bool use_path = true) {
    if (!should_perform_test_) {
      std::cerr << "Skipping test";
      return;
    }
    AndroidInfo android_info;
    auto status = RequestAndroidInfo(&android_info);
    ASSERT_TRUE(status.ok());

    ValidatorRunner::Options options;
    options.data_directory_path = ::testing::TempDir();
    options.storage_path = ::testing::TempDir() + "/storage_path.fb";
    (void)unlink(options.storage_path.c_str());
    if (use_path) {
      options.model_path = model_path_;
    } else {
      options.model_fd = open(model_path_.c_str(), O_RDONLY);
      ASSERT_GE(options.model_fd, 0);
      struct stat stat_buf = {0};
      ASSERT_EQ(fstat(options.model_fd, &stat_buf), 0);
      options.model_size = stat_buf.st_size;
      options.model_offset = 0;
    }
    auto validator1 = std::make_unique<ValidatorRunner>(options);
    auto validator2 = std::make_unique<ValidatorRunner>(options);
    ASSERT_EQ(validator1->Init(), kMinibenchmarkSuccess);
    ASSERT_EQ(validator2->Init(), kMinibenchmarkSuccess);

    std::vector<const BenchmarkEvent*> events =
        validator1->GetAndFlushEventsToLog();
    ASSERT_TRUE(events.empty());

    flatbuffers::FlatBufferBuilder fbb_cpu, fbb_nnapi, fbb_gpu;
    std::vector<const TFLiteSettings*> settings =
        BuildBenchmarkSettings(android_info, fbb_cpu, fbb_nnapi, fbb_gpu);

    ASSERT_EQ(validator1->TriggerMissingValidation(settings), settings.size());

    int event_count = 0;
    while (event_count < settings.size()) {
      events = validator1->GetAndFlushEventsToLog();
      event_count += events.size();
      for (const BenchmarkEvent* event : events) {
        std::string delegate_name = GetTargetDeviceName(event);
        if (event->event_type() == BenchmarkEventType_END) {
          if (event->result()->ok()) {
            std::cout << "Validation passed on " << delegate_name << std::endl;
          } else {
            std::cout << "Validation did not pass on " << delegate_name
                      << std::endl;
          }
        } else if (event->event_type() == BenchmarkEventType_ERROR) {
          std::cout << "Failed to run validation on " << delegate_name
                    << std::endl;
        }
      }
#ifndef _WIN32
      sleep(1);
#endif  // !_WIN32
    }

    EXPECT_EQ(validator2->TriggerMissingValidation(settings), 0);
  }

  bool should_perform_test_ = true;
  std::string model_path_;
};

TEST_F(ValidatorRunnerTest, AllConfigurationsWithFilePath) {
  CheckConfigurations(true);
}

TEST_F(ValidatorRunnerTest, AllConfigurationsWithFd) {
  CheckConfigurations(false);
}

// #ifdef __ANDROID__
using ::tflite::nnapi::NnApiSupportLibrary;

std::unique_ptr<const NnApiSupportLibrary> LoadNnApiSupportLibrary() {
  MiniBenchmarkTestHelper helper;
  std::string nnapi_sl_path = helper.DumpToTempFile(
      "libnnapi_fake.so", g_nnapi_sl_fake_impl, g_nnapi_sl_fake_impl_len);

  std::unique_ptr<const NnApiSupportLibrary> nnapi_sl =
      ::tflite::nnapi::loadNnApiSupportLibrary(nnapi_sl_path);

  return nnapi_sl;
}

TEST_F(ValidatorRunnerTest, ShouldUseNnApiSl) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }

  AndroidInfo android_info;
  auto status = RequestAndroidInfo(&android_info);
  ASSERT_TRUE(status.ok());

  InitNnApiSlInvocationStatus();

  std::unique_ptr<const NnApiSupportLibrary> nnapi_sl =
      LoadNnApiSupportLibrary();
  ASSERT_THAT(nnapi_sl.get(), ::testing::NotNull());

  ValidatorRunner::Options options;
  options.model_path = model_path_;
  options.storage_path = ::testing::TempDir() + "/storage_path.fb";
  (void)unlink(options.storage_path.c_str());
  options.data_directory_path = ::testing::TempDir();
  options.nnapi_sl = nnapi_sl->getFL5();
  ValidatorRunner validator(options);

  ASSERT_EQ(validator.Init(), kMinibenchmarkSuccess);

  std::vector<const BenchmarkEvent*> events =
      validator.GetAndFlushEventsToLog();
  ASSERT_TRUE(events.empty());

  flatbuffers::FlatBufferBuilder fbb_cpu, fbb_nnapi, fbb_gpu;
  std::vector<const TFLiteSettings*> settings =
      BuildBenchmarkSettings(android_info, fbb_cpu, fbb_nnapi, fbb_gpu,
                             /*ignore_android_version=*/true);
  ASSERT_EQ(validator.TriggerMissingValidation(settings), settings.size());

  // Waiting for benchmark to complete.
  int event_count = 0;
  while (event_count < settings.size()) {
    events = validator.GetAndFlushEventsToLog();
    event_count += events.size();
  }
  EXPECT_TRUE(WasNnApiSlInvoked());
}

TEST_F(ValidatorRunnerTest, ShouldFailIfItCannotFindNnApiSlPath) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }

  std::string storage_path = ::testing::TempDir() + "/storage_path.fb";
  (void)unlink(storage_path.c_str());

  // Building an NNAPI SL structure with invalid handle.
  NnApiSLDriverImplFL5 wrong_handle_nnapi_sl{};

  ValidatorRunner::Options options;
  options.model_path = model_path_;
  options.storage_path = ::testing::TempDir() + "/storage_path.fb";
  (void)unlink(options.storage_path.c_str());
  options.data_directory_path = ::testing::TempDir();
  options.nnapi_sl = &wrong_handle_nnapi_sl;
  ValidatorRunner validator(options);

  ASSERT_EQ(validator.Init(), kMiniBenchmarkCannotLoadSupportLibrary);
}
// #endif  // ifdef __ANDROID__

}  // namespace
}  // namespace acceleration
}  // namespace tflite
