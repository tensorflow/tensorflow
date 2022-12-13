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
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/benchmark_result_evaluator.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_validation_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_nnapi_sl_fake_impl.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/model_modifier/custom_validation_embedder.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/nnapi_sl_fake_impl.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/nnapi/sl/include/SupportLibrary.h"
#include "tensorflow/lite/stderr_reporter.h"
#ifdef __ANDROID__
#include <dlfcn.h>

#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_validator_runner_entrypoint.h"
#endif  // __ANDROID__

namespace tflite {
namespace acceleration {
namespace {

constexpr absl::Duration kWaitBetweenRefresh = absl::Milliseconds(20);

class ValidatorRunnerImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
    (void)unlink(storage_path_.c_str());
    MiniBenchmarkTestHelper helper;
    should_perform_test_ = helper.should_perform_test();
    model_path_ = helper.DumpToTempFile(
        "mobilenet_quant_with_validation.tflite",
        g_tflite_acceleration_embedded_mobilenet_validation_model,
        g_tflite_acceleration_embedded_mobilenet_validation_model_len);
    ASSERT_TRUE(!model_path_.empty());

    plain_model_path_ = MiniBenchmarkTestHelper::DumpToTempFile(
        "mobilenet_quant.tflite",
        g_tflite_acceleration_embedded_mobilenet_model,
        g_tflite_acceleration_embedded_mobilenet_model_len);
    ASSERT_TRUE(!plain_model_path_.empty());

    nnapi_sl_dump_path_ = helper.DumpToTempFile(
        "libnnapi_fake.so", g_nnapi_sl_fake_impl, g_nnapi_sl_fake_impl_len);
  }

  ValidatorRunnerImpl CreateValidator() {
    return ValidatorRunnerImpl(model_path_, storage_path_, data_directory_path_,
                               0, std::move(custom_validation_embedder_),
                               error_reporter_, nnapi_sl_, entrypoint_name_,
                               EmbeddedResultEvaluator::GetInstance());
  }

  bool should_perform_test_;
  std::string data_directory_path_ = ::testing::TempDir();
  std::string storage_path_ = ::testing::TempDir() + "/storage_path.fb";
  std::string model_path_;
  std::string plain_model_path_;
  std::unique_ptr<CustomValidationEmbedder> custom_validation_embedder_ =
      nullptr;
  std::string nnapi_sl_dump_path_;
  const NnApiSLDriverImplFL5* nnapi_sl_ = nullptr;
  std::string entrypoint_name_ =
      "Java_org_tensorflow_lite_acceleration_validation_entrypoint";
  ErrorReporter* error_reporter_ = tflite::DefaultErrorReporter();
};

TEST_F(ValidatorRunnerImplTest, SucceedWithNnApiSl) {
  // Setup.
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }

  AndroidInfo android_info;
  auto status = RequestAndroidInfo(&android_info);
  ASSERT_TRUE(status.ok());

  InitNnApiSlInvocationStatus();

  std::unique_ptr<const ::tflite::nnapi::NnApiSupportLibrary> fake_nnapi_sl =
      ::tflite::nnapi::loadNnApiSupportLibrary(nnapi_sl_dump_path_);
  ASSERT_THAT(fake_nnapi_sl.get(), ::testing::NotNull());
  nnapi_sl_ = fake_nnapi_sl->getFL5();

  ValidatorRunnerImpl validator = CreateValidator();
  ASSERT_EQ(validator.Init(), kMinibenchmarkSuccess);

  std::vector<flatbuffers::FlatBufferBuilder> tflite_settings(1);
  tflite_settings[0].Finish(
      CreateTFLiteSettings(tflite_settings[0], Delegate_NNAPI,
                           CreateNNAPISettings(tflite_settings[0])));

  // Run.
  validator.TriggerValidationAsync(
      std::make_unique<std::vector<flatbuffers::FlatBufferBuilder>>(
          std::move(tflite_settings)));

  // Validate.
  FlatbufferStorage<BenchmarkEvent> storage(storage_path_, error_reporter_);
  while (validator.GetNumCompletedResults() < 1) {
    usleep(absl::ToInt64Microseconds(kWaitBetweenRefresh));
  }
  std::vector<const BenchmarkEvent*> results = validator.GetSuccessfulResults();
  for (auto& result : results) {
    ASSERT_THAT(result, testing::Property(&BenchmarkEvent::event_type,
                                          testing::Eq(BenchmarkEventType_END)));
    EXPECT_THAT(result->result()->actual_output(),
                testing::Pointee(testing::SizeIs(0)));
  }
  EXPECT_TRUE(WasNnApiSlInvoked());
}

TEST_F(ValidatorRunnerImplTest, SucceedWithCustomValidation) {
  // Setup.
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }
  int batch_size = 3;
  custom_validation_embedder_ = std::make_unique<CustomValidationEmbedder>(
      batch_size, std::vector<std::vector<uint8_t>>{
                      std::vector<uint8_t>(batch_size * 224 * 224 * 3, 1)});
  model_path_ = plain_model_path_;
  ValidatorRunnerImpl validator = CreateValidator();
  ASSERT_EQ(validator.Init(), kMinibenchmarkSuccess);

  std::vector<flatbuffers::FlatBufferBuilder> tflite_settings(1);
  tflite_settings[0].Finish(CreateTFLiteSettings(tflite_settings[0]));

  // Run.
  validator.TriggerValidationAsync(
      std::make_unique<std::vector<flatbuffers::FlatBufferBuilder>>(
          std::move(tflite_settings)));

  // Validate.
  FlatbufferStorage<BenchmarkEvent> storage(storage_path_, error_reporter_);
  while (validator.GetNumCompletedResults() < 1) {
    usleep(absl::ToInt64Microseconds(kWaitBetweenRefresh));
  }
  std::vector<const BenchmarkEvent*> results = validator.GetSuccessfulResults();
  for (auto result : results) {
    ASSERT_THAT(result, testing::Property(&BenchmarkEvent::event_type,
                                          testing::Eq(BenchmarkEventType_END)));
    EXPECT_THAT(result->result()->actual_output(),
                testing::Pointee(testing::SizeIs(1)));
    EXPECT_THAT(result->result()->actual_output()->Get(0)->value(),
                testing::Pointee(testing::SizeIs(batch_size * 1001)));
  }
}

TEST_F(ValidatorRunnerImplTest, FailIfItCannotFindNnApiSlPath) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }

  // Building an NNAPI SL structure with invalid handle.
  NnApiSLDriverImplFL5 wrong_handle_nnapi_sl{};
  nnapi_sl_ = &wrong_handle_nnapi_sl;
  ValidatorRunnerImpl validator = CreateValidator();

  EXPECT_EQ(validator.Init(), kMiniBenchmarkCannotLoadSupportLibrary);
}

TEST_F(ValidatorRunnerImplTest, FailWithInvalidEntrypoint) {
  entrypoint_name_ = "invalid_name()";
  EXPECT_EQ(CreateValidator().Init(),
            kMinibenchmarkValidationEntrypointSymbolNotFound);
}

TEST_F(ValidatorRunnerImplTest, FailIfCannotLoadModel) {
  model_path_ = "invalid/path";
  EXPECT_EQ(CreateValidator().Init(), kMinibenchmarkModelInitFailed);
}

TEST_F(ValidatorRunnerImplTest, FailIfCannotEmbedInputData) {
  model_path_ = plain_model_path_;
  custom_validation_embedder_ = std::make_unique<CustomValidationEmbedder>(
      1, std::vector<std::vector<uint8_t>>(2));
  EXPECT_EQ(CreateValidator().Init(),
            kMinibenchmarkValidationSubgraphBuildFailed);
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite
