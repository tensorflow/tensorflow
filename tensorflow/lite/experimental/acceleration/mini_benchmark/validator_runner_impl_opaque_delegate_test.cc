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
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/core/acceleration/configuration/stable_delegate_registry.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/delegate_loader.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/benchmark_result_evaluator.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_validation_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_impl.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_options.h"
#include "tensorflow/lite/stderr_reporter.h"

namespace tflite {
namespace acceleration {
namespace {

using ::flatbuffers::FlatBufferBuilder;

constexpr absl::Duration kWaitBetweenRefresh = absl::Milliseconds(20);

class ValidatorRunnerImplTest : public ::testing::Test {
 protected:
  static constexpr char kDelegateName[] = "test_xnnpack_delegate";
  static constexpr char kDelegateVersion[] = "1.0.0";
  static constexpr char kDelegateBinaryPath[] =
      "tensorflow/lite/delegates/utils/experimental/"
      "stable_delegate/libtensorflowlite_stable_xnnpack_delegate.so";

  void SetUp() override {
    MiniBenchmarkTestHelper helper(
        /*should_load_entrypoint_dynamically=*/false);
    should_perform_test_ = helper.should_perform_test();

    options_.data_directory_path = ::testing::TempDir();
    options_.storage_path = ::testing::TempDir() + "/storage_path.fb";
    options_.validation_entrypoint_name =
        "Java_org_tensorflow_lite_acceleration_validation_entrypoint";
    options_.error_reporter = tflite::DefaultErrorReporter();
    options_.benchmark_result_evaluator =
        EmbeddedResultEvaluator::GetInstance();
    options_.per_test_timeout_ms = 0;

    options_.model_path = helper.DumpToTempFile(
        "mobilenet_quant_with_validation.tflite",
        g_tflite_acceleration_embedded_mobilenet_validation_model,
        g_tflite_acceleration_embedded_mobilenet_validation_model_len);
    ASSERT_TRUE(!options_.model_path.empty());
  }

  void TearDown() override {
    if (should_perform_test_) {
      ASSERT_EQ(unlink(options_.storage_path.c_str()), 0);
    }
  }

  ValidatorRunnerImpl CreateValidator() {
    return ValidatorRunnerImpl(
        CreateModelLoaderPath(options_), options_.storage_path,
        options_.data_directory_path, options_.per_test_timeout_ms,
        /*custom_validation_embedder=*/nullptr, options_.error_reporter,
        options_.nnapi_sl, options_.gpu_plugin_handle,
        options_.validation_entrypoint_name,
        options_.benchmark_result_evaluator);
  }

  bool should_perform_test_;
  ValidatorRunnerOptions options_{};
};

TEST_F(
    ValidatorRunnerImplTest,
    GetSuccessfulResultsSucceedWithOpaqueXnnpackDelegateAndEmbeddedValidation) {
  // Setup.
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }

  // Retrieves the stable delegate plugin from the shared library and register
  // it to the stable delegate registry. This allows the stable delegate
  // module plugin to fetch the stable delegate plugin from the registry.
  const TfLiteStableDelegate* stable_delegate_handle =
      delegates::utils::LoadDelegateFromSharedLibrary(kDelegateBinaryPath);
  TfLiteStableDelegate stable_delegate = {
      TFL_STABLE_DELEGATE_ABI_VERSION, kDelegateName, kDelegateVersion,
      stable_delegate_handle->delegate_plugin};
  delegates::StableDelegateRegistry::RegisterStableDelegate(&stable_delegate);
  AndroidInfo android_info;
  auto status = RequestAndroidInfo(&android_info);
  ASSERT_TRUE(status.ok());
  ValidatorRunnerImpl validator = CreateValidator();
  ASSERT_EQ(validator.Init(), kMinibenchmarkSuccess);
  std::vector<flatbuffers::FlatBufferBuilder> tflite_settings(1);
  flatbuffers::Offset<flatbuffers::String> stable_delegate_name_offset =
      tflite_settings[0].CreateString(kDelegateName);
  tflite_settings[0].Finish(CreateTFLiteSettings(
      tflite_settings[0], Delegate_XNNPACK,
      /*nnapi_settings=*/0,
      /*gpu_settings=*/0,
      /*hexagon_settings=*/0, CreateXNNPackSettings(tflite_settings[0]),
      /*coreml_settings=*/0,
      /*cpu_settings=*/0,
      /*max_delegated_partitions=*/0,
      /*edgetpu_settings=*/0,
      /*coral_settings=*/0,
      /*fallback_settings=*/0,
      /*disable_default_delegates=*/true,
      CreateStableDelegateLoaderSettings(tflite_settings[0],
                                         /*delegate_path=*/0,
                                         stable_delegate_name_offset)));

  // Run.
  validator.TriggerValidationAsync(std::move(tflite_settings),
                                   options_.storage_path);

  // Validate.
  FlatbufferStorage<BenchmarkEvent> storage(options_.storage_path,
                                            options_.error_reporter);
  while (validator.GetNumCompletedResults() < 1) {
    usleep(absl::ToInt64Microseconds(kWaitBetweenRefresh));
  }
  std::vector<const BenchmarkEvent*> results =
      validator.GetSuccessfulResultsFromStorage();
  ASSERT_THAT(results, testing::Not(testing::IsEmpty()));
  for (auto& result : results) {
    ASSERT_THAT(result, testing::Property(&BenchmarkEvent::event_type,
                                          testing::Eq(BenchmarkEventType_END)));
    EXPECT_THAT(result->result()->actual_output(),
                testing::Pointee(testing::SizeIs(0)));
  }
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite
