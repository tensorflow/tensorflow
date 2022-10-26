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

#include <sys/types.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator.h"

// Note that these tests are not meant to be completely exhaustive, but to test
// error propagation.

namespace tflite {
namespace acceleration {

static int32_t big_core_affinity_result;
int32_t SetBigCoresAffinity() { return big_core_affinity_result; }

namespace {

class ValidatorRunnerEntryPointTest : public ::testing::Test {
 protected:
  ValidatorRunnerEntryPointTest()
      : storage_path_(::testing::TempDir() + "/events.fb"),
        storage_(storage_path_) {}

  std::vector<const tflite::BenchmarkEvent*> GetEvents() {
    std::vector<const tflite::BenchmarkEvent*> result;

    storage_.Read();
    int storage_size = storage_.Count();
    if (storage_size == 0) {
      return result;
    }

    for (int i = 0; i < storage_size; i++) {
      const ::tflite::BenchmarkEvent* event = storage_.Get(i);
      result.push_back(event);
    }
    return result;
  }

  void ClearEvents() { (void)unlink(storage_path_.c_str()); }

  void SetUp() override {
    ClearEvents();
    SetBigCoreAffinityReturns(0);
  }

  int CallEntryPoint(std::string cpu_affinity = "0") {
    std::vector<std::string> args = {
        "test",
        "binary_name",
        "Java_org_tensorflow_lite_acceleration_validation_entrypoint",
        "model_path",
        storage_path_,
        "data_dir"};
    std::vector<std::vector<char>> mutable_args(args.size());
    std::vector<char*> argv(args.size());
    for (int i = 0; i < mutable_args.size(); i++) {
      mutable_args[i] = {args[i].data(), args[i].data() + args[i].size()};
      mutable_args[i].push_back('\0');
      argv[i] = mutable_args[i].data();
    }
    return Java_org_tensorflow_lite_acceleration_validation_entrypoint(
        argv.size(), argv.data());
  }

  void SetBigCoreAffinityReturns(int32_t value) {
    big_core_affinity_result = value;
  }

  std::string storage_path_;
  FlatbufferStorage<BenchmarkEvent> storage_;
};

TEST_F(ValidatorRunnerEntryPointTest, NotEnoughArguments) {
  std::vector<std::string> args = {
      "test", "binary_name",
      "Java_org_tensorflow_lite_acceleration_validation_entrypoint",
      "model_path", storage_path_};
  std::vector<std::vector<char>> mutable_args(args.size());
  std::vector<char*> argv(args.size());
  for (int i = 0; i < mutable_args.size(); i++) {
    mutable_args[i] = {args[i].data(), args[i].data() + args[i].size()};
    mutable_args[i].push_back('\0');
    argv[i] = mutable_args[i].data();
  }
  EXPECT_EQ(1, Java_org_tensorflow_lite_acceleration_validation_entrypoint(
                   5, argv.data()));
}

TEST_F(ValidatorRunnerEntryPointTest, NoValidationRequestFound) {
  EXPECT_EQ(kMinibenchmarkSuccess, CallEntryPoint());

  std::vector<const tflite::BenchmarkEvent*> events = GetEvents();
  ASSERT_THAT(events, testing::SizeIs(1));
  const tflite::BenchmarkEvent* event = events[0];

  EXPECT_EQ(BenchmarkEventType_ERROR, event->event_type());
  EXPECT_EQ(kMinibenchmarkNoValidationRequestFound,
            event->error()->exit_code());
}

TEST_F(ValidatorRunnerEntryPointTest, CannotSetCpuAffinity) {
  SetBigCoreAffinityReturns(10);
  EXPECT_EQ(kMinibenchmarkSuccess, CallEntryPoint("invalid_cpu_affinity"));

  std::vector<const tflite::BenchmarkEvent*> events = GetEvents();
  ASSERT_THAT(events, testing::SizeIs(2));
  // The last event is the notification of NoValidationRequestFound.
  const tflite::BenchmarkEvent* event = events[0];

  EXPECT_EQ(BenchmarkEventType_RECOVERED_ERROR, event->event_type());
  EXPECT_EQ(kMinibenchmarkUnableToSetCpuAffinity, event->error()->exit_code());
  EXPECT_EQ(10, event->error()->mini_benchmark_error_code());
}

TEST_F(ValidatorRunnerEntryPointTest, CannotLoadNnapi) {
  // Write TFLiteSettings to storage_.
  flatbuffers::FlatBufferBuilder fbb;
  TFLiteSettingsT tflite_settings;
  NNAPISettingsT nnapi_settings;
  ASSERT_EQ(
      storage_.Append(
          &fbb,
          CreateBenchmarkEvent(
              fbb,
              CreateTFLiteSettings(fbb, Delegate_NNAPI,
                                   CreateNNAPISettings(fbb, &nnapi_settings)),
              BenchmarkEventType_START, /* result */ 0, /* error */ 0,
              Validator::BootTimeMicros(), Validator::WallTimeMicros())),
      kMinibenchmarkSuccess);
  // Prep argv.
  std::vector<std::string> args = {
      "test",
      "binary_name",
      "Java_org_tensorflow_lite_acceleration_validation_entrypoint",
      "model_path",
      storage_path_,
      "data_directory_path",
      "nnapi_path"};
  std::vector<std::vector<char>> mutable_args(args.size());
  std::vector<char*> argv(args.size());
  for (int i = 0; i < mutable_args.size(); i++) {
    mutable_args[i] = {args[i].data(), args[i].data() + args[i].size()};
    mutable_args[i].push_back('\0');
    argv[i] = mutable_args[i].data();
  }
  EXPECT_EQ(kMinibenchmarkSuccess,
            Java_org_tensorflow_lite_acceleration_validation_entrypoint(
                7, argv.data()));

  // Verify.
  std::vector<const tflite::BenchmarkEvent*> events = GetEvents();
  ASSERT_THAT(events, testing::SizeIs(2));
  const tflite::BenchmarkEvent* event = events[1];
  EXPECT_EQ(BenchmarkEventType_ERROR, event->event_type());
  EXPECT_EQ(kMiniBenchmarkCannotLoadSupportLibrary,
            event->error()->exit_code());
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite
