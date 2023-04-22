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
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_validation_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"

namespace tflite {
namespace acceleration {
namespace {

class ValidatorRunnerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    MiniBenchmarkTestHelper helper;
    should_perform_test_ = helper.should_perform_test();

    if (should_perform_test_) {
      model_path_ = helper.DumpToTempFile(
          "mobilenet_quant_with_validation.tflite",
          g_tflite_acceleration_embedded_mobilenet_validation_model,
          g_tflite_acceleration_embedded_mobilenet_validation_model_len);
      ASSERT_TRUE(!model_path_.empty());
    }
  }

  void CheckConfigurations(bool use_path = true) {
    if (!should_perform_test_) return;

    AndroidInfo android_info;
    auto status = RequestAndroidInfo(&android_info);
    ASSERT_TRUE(status.ok());
    std::unique_ptr<ValidatorRunner> validator1, validator2;
    std::string storage_path = ::testing::TempDir() + "/storage_path.fb";
    (void)unlink(storage_path.c_str());
    if (use_path) {
      validator1 = std::make_unique<ValidatorRunner>(model_path_, storage_path,
                                                     ::testing::TempDir());
      validator2 = std::make_unique<ValidatorRunner>(model_path_, storage_path,
                                                     ::testing::TempDir());
    } else {
      int fd = open(model_path_.c_str(), O_RDONLY);
      ASSERT_GE(fd, 0);
      struct stat stat_buf = {0};
      ASSERT_EQ(fstat(fd, &stat_buf), 0);
      validator1 = std::make_unique<ValidatorRunner>(
          fd, 0, stat_buf.st_size, storage_path, ::testing::TempDir());
      validator2 = std::make_unique<ValidatorRunner>(
          fd, 0, stat_buf.st_size, storage_path, ::testing::TempDir());
    }
    ASSERT_EQ(validator1->Init(), kMinibenchmarkSuccess);
    ASSERT_EQ(validator2->Init(), kMinibenchmarkSuccess);

    std::vector<const BenchmarkEvent*> events =
        validator1->GetAndFlushEventsToLog();
    ASSERT_TRUE(events.empty());

    std::vector<const TFLiteSettings*> settings;
    flatbuffers::FlatBufferBuilder fbb_cpu, fbb_nnapi, fbb_gpu;
    fbb_cpu.Finish(CreateTFLiteSettings(fbb_cpu, Delegate_NONE,
                                        CreateNNAPISettings(fbb_cpu)));
    settings.push_back(
        flatbuffers::GetRoot<TFLiteSettings>(fbb_cpu.GetBufferPointer()));
    if (android_info.android_sdk_version >= "28") {
      fbb_nnapi.Finish(CreateTFLiteSettings(fbb_nnapi, Delegate_NNAPI,
                                            CreateNNAPISettings(fbb_nnapi)));
      settings.push_back(
          flatbuffers::GetRoot<TFLiteSettings>(fbb_nnapi.GetBufferPointer()));
    }
    fbb_gpu.Finish(CreateTFLiteSettings(fbb_gpu, Delegate_GPU));
#ifdef __ANDROID__
    settings.push_back(
        flatbuffers::GetRoot<TFLiteSettings>(fbb_gpu.GetBufferPointer()));
#endif  // __ANDROID__

    ASSERT_EQ(validator1->TriggerMissingValidation(settings), settings.size());

    int event_count = 0;
    while (event_count < settings.size()) {
      events = validator1->GetAndFlushEventsToLog();
      event_count += events.size();
      for (const BenchmarkEvent* event : events) {
        std::string delegate_name = "CPU";
        if (event->tflite_settings()->delegate() == Delegate_GPU) {
          delegate_name = "GPU";
        } else if (event->tflite_settings()->delegate() == Delegate_NNAPI) {
          delegate_name = "NNAPI";
        }
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

}  // namespace
}  // namespace acceleration
}  // namespace tflite
