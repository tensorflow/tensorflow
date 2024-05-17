/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/core/experimental/acceleration/mini_benchmark/c/c_api.h"

#include <stdint.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/base.h"  // from @flatbuffers
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_validation_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_simple_addition_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"

namespace {

using ::testing::_;

std::vector<const tflite::BenchmarkEvent*> ToBenchmarkEvents(uint8_t* data,
                                                             size_t size) {
  std::vector<const tflite::BenchmarkEvent*> results;
  uint8_t* current_root = data;
  while (current_root < data + size) {
    flatbuffers::uoffset_t current_size =
        flatbuffers::GetPrefixedSize(current_root);
    results.push_back(
        flatbuffers::GetSizePrefixedRoot<tflite::BenchmarkEvent>(current_root));
    current_root += current_size + sizeof(flatbuffers::uoffset_t);
  }
  return results;
}

class MockErrorReporter {
 public:
  static int InvokeErrorReporter(void* user_data, const char* format,
                                 va_list args) {
    MockErrorReporter* error_reporter =
        static_cast<MockErrorReporter*>(user_data);
    return error_reporter->Log(format, args);
  }
  MOCK_METHOD(int, Log, (const char* format, va_list args));
};

class MockResultEvaluator {
 public:
  static bool Invoke(void* user_data, uint8_t* benchmark_result_data,
                     int benchmark_result_data_size) {
    MockResultEvaluator* evaluator =
        static_cast<MockResultEvaluator*>(user_data);
    return evaluator->HasPassedAccuracyCheck(benchmark_result_data,
                                             benchmark_result_data_size);
  }
  MOCK_METHOD(bool, HasPassedAccuracyCheck,
              (uint8_t * benchmark_result_data,
               int benchmark_result_data_size));
};

class CApiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tflite::acceleration::MiniBenchmarkTestHelper helper;
    should_perform_test_ = helper.should_perform_test();

    if (!should_perform_test_) {
      return;
    }
    embedded_model_path_ = helper.DumpToTempFile(
        "mobilenet_quant_with_validation.tflite",
        g_tflite_acceleration_embedded_mobilenet_validation_model,
        g_tflite_acceleration_embedded_mobilenet_validation_model_len);
    ASSERT_TRUE(!embedded_model_path_.empty());

    plain_model_path_ = helper.DumpToTempFile(
        "mobilenet_quant.tflite",
        g_tflite_acceleration_embedded_mobilenet_model,
        g_tflite_acceleration_embedded_mobilenet_model_len);

    simple_addition_model_path_ = helper.DumpToTempFile(
        "add.bin", g_tflite_acceleration_embedded_simple_addition_model,
        g_tflite_acceleration_embedded_simple_addition_model_len);
  }

  flatbuffers::Offset<tflite::BenchmarkStoragePaths> CreateStoragePaths() {
    return tflite::CreateBenchmarkStoragePaths(
        mini_benchmark_fbb_,
        mini_benchmark_fbb_.CreateString(::testing::TempDir() +
                                         "/storage_path.fb"),
        mini_benchmark_fbb_.CreateString(::testing::TempDir()));
  }

  flatbuffers::Offset<tflite::ValidationSettings> CreateValidationSettings() {
    return tflite::CreateValidationSettings(mini_benchmark_fbb_, 5000);
  }

  flatbuffers::Offset<tflite::ModelFile> CreateModelFile(
      const std::string& model_path) {
    return tflite::CreateModelFile(
        mini_benchmark_fbb_, mini_benchmark_fbb_.CreateString(model_path));
  }

  flatbuffers::Offset<
      flatbuffers::Vector<flatbuffers::Offset<tflite::TFLiteSettings>>>
  CreateTFLiteSettings() {
    return mini_benchmark_fbb_.CreateVector(
        {tflite::CreateTFLiteSettings(mini_benchmark_fbb_,
#ifdef ANDROID
                                      tflite::Delegate_GPU
#else
                                      tflite::Delegate_NONE
#endif

                                      )});
  }

  flatbuffers::FlatBufferBuilder mini_benchmark_fbb_;
  std::string embedded_model_path_;
  std::string plain_model_path_;
  std::string simple_addition_model_path_;
  bool should_perform_test_ = true;
};

TEST_F(CApiTest, SucceedWithEmbeddedValidation) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }

  mini_benchmark_fbb_.Finish(tflite::CreateMinibenchmarkSettings(
      mini_benchmark_fbb_, CreateTFLiteSettings(),
      CreateModelFile(embedded_model_path_), CreateStoragePaths(),
      CreateValidationSettings()));
  TfLiteMiniBenchmarkSettings* settings = TfLiteMiniBenchmarkSettingsCreate();
  TfLiteMiniBenchmarkSettingsSetFlatBufferData(
      settings, mini_benchmark_fbb_.GetBufferPointer(),
      mini_benchmark_fbb_.GetSize());
  TfLiteMiniBenchmarkResult* result =
      TfLiteBlockingValidatorRunnerTriggerValidation(settings);
  std::vector<const tflite::BenchmarkEvent*> events =
      ToBenchmarkEvents(TfLiteMiniBenchmarkResultFlatBufferData(result),
                        TfLiteMiniBenchmarkResultFlatBufferDataSize(result));

  EXPECT_THAT(TfLiteMiniBenchmarkResultInitStatus(result),
              tflite::acceleration::kMinibenchmarkSuccess);
  EXPECT_THAT(events, testing::Not(testing::IsEmpty()));
  for (auto& event : events) {
    EXPECT_EQ(event->event_type(), tflite::BenchmarkEventType_END);
    EXPECT_TRUE(event->result()->ok());
  }
  TfLiteMiniBenchmarkResultFree(result);
  TfLiteMiniBenchmarkSettingsFree(settings);
}

TEST_F(CApiTest, SucceedWithCustomValidationAndPassingRule) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }

  const int batch_size = 5;
  size_t input_size[] = {batch_size * 224 * 224 * 3};
  std::vector<uint8_t> custom_input_data(input_size[0], 1);
  mini_benchmark_fbb_.Finish(tflite::CreateMinibenchmarkSettings(
      mini_benchmark_fbb_, CreateTFLiteSettings(),
      CreateModelFile(plain_model_path_), CreateStoragePaths(),
      CreateValidationSettings()));
  MockResultEvaluator mock_evaluator;
  EXPECT_CALL(mock_evaluator, HasPassedAccuracyCheck(_, _))
      .Times(1)
      .WillRepeatedly(testing::Return(true));

  TfLiteMiniBenchmarkSettings* settings = TfLiteMiniBenchmarkSettingsCreate();
  TfLiteMiniBenchmarkSettingsSetFlatBufferData(
      settings, mini_benchmark_fbb_.GetBufferPointer(),
      mini_benchmark_fbb_.GetSize());
  TfLiteMiniBenchmarkCustomValidationInfo* custom_validation =
      TfLiteMiniBenchmarkSettingsCustomValidationInfo(settings);
  TfLiteMiniBenchmarkCustomValidationInfoSetBuffer(custom_validation,
                                                   /*batch_size=*/batch_size,
                                                   custom_input_data.data(),
                                                   /*buffer_dim=*/input_size,
                                                   /*buffer_dim_size=*/1);
  TfLiteMiniBenchmarkCustomValidationInfoSetAccuracyValidator(
      custom_validation, &mock_evaluator, MockResultEvaluator::Invoke);

  TfLiteMiniBenchmarkResult* result =
      TfLiteBlockingValidatorRunnerTriggerValidation(settings);
  std::vector<const tflite::BenchmarkEvent*> events =
      ToBenchmarkEvents(TfLiteMiniBenchmarkResultFlatBufferData(result),
                        TfLiteMiniBenchmarkResultFlatBufferDataSize(result));

  EXPECT_THAT(TfLiteMiniBenchmarkResultInitStatus(result),
              tflite::acceleration::kMinibenchmarkSuccess);

  EXPECT_THAT(events, testing::Not(testing::IsEmpty()));
  for (auto& event : events) {
    EXPECT_EQ(event->event_type(), tflite::BenchmarkEventType_END);
    EXPECT_TRUE(event->result()->ok());
  }
  TfLiteMiniBenchmarkResultFree(result);
  TfLiteMiniBenchmarkSettingsFree(settings);
}

TEST_F(CApiTest, SucceedWithMultipleTfliteSettings) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }

  const int batch_size = 5;
  size_t input_size[] = {batch_size * 224 * 224 * 3};
  std::vector<uint8_t> custom_input_data(input_size[0], 1);
  mini_benchmark_fbb_.Finish(tflite::CreateMinibenchmarkSettings(
      mini_benchmark_fbb_,
      mini_benchmark_fbb_.CreateVector(
          {tflite::CreateTFLiteSettings(mini_benchmark_fbb_),
           tflite::CreateTFLiteSettings(mini_benchmark_fbb_),
           tflite::CreateTFLiteSettings(mini_benchmark_fbb_)}),
      CreateModelFile(plain_model_path_), CreateStoragePaths(),
      CreateValidationSettings()));
  MockResultEvaluator mock_evaluator;
  EXPECT_CALL(mock_evaluator, HasPassedAccuracyCheck(_, _))
      .Times(9)
      .WillRepeatedly(testing::Return(true));

  TfLiteMiniBenchmarkSettings* settings = TfLiteMiniBenchmarkSettingsCreate();
  TfLiteMiniBenchmarkSettingsSetFlatBufferData(
      settings, mini_benchmark_fbb_.GetBufferPointer(),
      mini_benchmark_fbb_.GetSize());
  TfLiteMiniBenchmarkCustomValidationInfo* custom_validation =
      TfLiteMiniBenchmarkSettingsCustomValidationInfo(settings);
  TfLiteMiniBenchmarkCustomValidationInfoSetBuffer(custom_validation,
                                                   /*batch_size=*/batch_size,
                                                   custom_input_data.data(),
                                                   /*buffer_dim=*/input_size,
                                                   /*buffer_dim_size=*/1);
  TfLiteMiniBenchmarkCustomValidationInfoSetAccuracyValidator(
      custom_validation, &mock_evaluator, MockResultEvaluator::Invoke);
  for (int i = 0; i < 3; i++) {
    TfLiteMiniBenchmarkResult* result =
        TfLiteBlockingValidatorRunnerTriggerValidation(settings);
    std::vector<const tflite::BenchmarkEvent*> events =
        ToBenchmarkEvents(TfLiteMiniBenchmarkResultFlatBufferData(result),
                          TfLiteMiniBenchmarkResultFlatBufferDataSize(result));

    EXPECT_THAT(TfLiteMiniBenchmarkResultInitStatus(result),
                tflite::acceleration::kMinibenchmarkSuccess);

    EXPECT_THAT(events, testing::Not(testing::IsEmpty()));
    for (auto& event : events) {
      EXPECT_EQ(event->event_type(), tflite::BenchmarkEventType_END);
      EXPECT_TRUE(event->result()->ok());
    }
    TfLiteMiniBenchmarkResultFree(result);
  }
  TfLiteMiniBenchmarkSettingsFree(settings);
}

TEST_F(CApiTest, ReturnNotOkWhenAccuracyCheckFail) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }

  const int batch_size = 5;
  size_t input_size[] = {batch_size * 224 * 224 * 3};
  std::vector<uint8_t> custom_input_data(input_size[0], 1);
  mini_benchmark_fbb_.Finish(tflite::CreateMinibenchmarkSettings(
      mini_benchmark_fbb_, CreateTFLiteSettings(),
      CreateModelFile(plain_model_path_), CreateStoragePaths(),
      CreateValidationSettings()));
  MockResultEvaluator mock_evaluator;
  EXPECT_CALL(mock_evaluator, HasPassedAccuracyCheck(_, _))
      .Times(1)
      .WillRepeatedly(testing::Return(false));

  TfLiteMiniBenchmarkSettings* settings = TfLiteMiniBenchmarkSettingsCreate();
  TfLiteMiniBenchmarkSettingsSetFlatBufferData(
      settings, mini_benchmark_fbb_.GetBufferPointer(),
      mini_benchmark_fbb_.GetSize());
  TfLiteMiniBenchmarkCustomValidationInfo* custom_validation =
      TfLiteMiniBenchmarkSettingsCustomValidationInfo(settings);
  TfLiteMiniBenchmarkCustomValidationInfoSetBuffer(custom_validation,
                                                   /*batch_size=*/batch_size,
                                                   custom_input_data.data(),
                                                   /*buffer_dim=*/input_size,
                                                   /*buffer_dim_size=*/1);
  TfLiteMiniBenchmarkCustomValidationInfoSetAccuracyValidator(
      custom_validation, &mock_evaluator, MockResultEvaluator::Invoke);

  TfLiteMiniBenchmarkResult* result =
      TfLiteBlockingValidatorRunnerTriggerValidation(settings);
  std::vector<const tflite::BenchmarkEvent*> events =
      ToBenchmarkEvents(TfLiteMiniBenchmarkResultFlatBufferData(result),
                        TfLiteMiniBenchmarkResultFlatBufferDataSize(result));

  EXPECT_THAT(TfLiteMiniBenchmarkResultInitStatus(result),
              tflite::acceleration::kMinibenchmarkSuccess);

  EXPECT_THAT(events, testing::Not(testing::IsEmpty()));
  for (auto& event : events) {
    EXPECT_EQ(event->event_type(), tflite::BenchmarkEventType_END);
    EXPECT_FALSE(event->result()->ok());
  }
  TfLiteMiniBenchmarkResultFree(result);
  TfLiteMiniBenchmarkSettingsFree(settings);
}

TEST_F(CApiTest, ReturnFailStatusWhenModelPathInvalid) {
  mini_benchmark_fbb_.Finish(tflite::CreateMinibenchmarkSettings(
      mini_benchmark_fbb_, CreateTFLiteSettings(),
      CreateModelFile("invalid/path"), CreateStoragePaths(),
      CreateValidationSettings()));
  TfLiteMiniBenchmarkSettings* settings = TfLiteMiniBenchmarkSettingsCreate();
  TfLiteMiniBenchmarkSettingsSetFlatBufferData(
      settings, mini_benchmark_fbb_.GetBufferPointer(),
      mini_benchmark_fbb_.GetSize());

  TfLiteMiniBenchmarkResult* result =
      TfLiteBlockingValidatorRunnerTriggerValidation(settings);

  EXPECT_THAT(TfLiteMiniBenchmarkResultInitStatus(result),
              tflite::acceleration::kMinibenchmarkModelInitFailed);
  EXPECT_EQ(TfLiteMiniBenchmarkResultFlatBufferData(result), nullptr);
  EXPECT_EQ(TfLiteMiniBenchmarkResultFlatBufferDataSize(result), 0);
  TfLiteMiniBenchmarkResultFree(result);
  TfLiteMiniBenchmarkSettingsFree(settings);
}

TEST_F(CApiTest, ReturnErrorWhenTestTimedOut) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }

  mini_benchmark_fbb_.Finish(tflite::CreateMinibenchmarkSettings(
      mini_benchmark_fbb_, CreateTFLiteSettings(),
      CreateModelFile(embedded_model_path_), CreateStoragePaths(),
      tflite::CreateValidationSettings(mini_benchmark_fbb_,
                                       /*per_test_timeout_ms=*/2)));
  TfLiteMiniBenchmarkSettings* settings = TfLiteMiniBenchmarkSettingsCreate();
  TfLiteMiniBenchmarkSettingsSetFlatBufferData(
      settings, mini_benchmark_fbb_.GetBufferPointer(),
      mini_benchmark_fbb_.GetSize());

  TfLiteMiniBenchmarkResult* result =
      TfLiteBlockingValidatorRunnerTriggerValidation(settings);

  EXPECT_THAT(TfLiteMiniBenchmarkResultInitStatus(result),
              tflite::acceleration::kMinibenchmarkSuccess);
  std::vector<const tflite::BenchmarkEvent*> events =
      ToBenchmarkEvents(TfLiteMiniBenchmarkResultFlatBufferData(result),
                        TfLiteMiniBenchmarkResultFlatBufferDataSize(result));
  EXPECT_THAT(events, testing::Not(testing::IsEmpty()));
  for (auto& event : events) {
    EXPECT_EQ(event->event_type(), tflite::BenchmarkEventType_ERROR);
  }
  TfLiteMiniBenchmarkResultFree(result);
  TfLiteMiniBenchmarkSettingsFree(settings);
}

TEST_F(CApiTest, UseProvidedErrorReporterWhenFail) {
  mini_benchmark_fbb_.Finish(tflite::CreateMinibenchmarkSettings(
      mini_benchmark_fbb_, CreateTFLiteSettings(), 0, 0, 0));
  MockErrorReporter reporter;
  EXPECT_CALL(reporter, Log(_, _)).Times(testing::AtLeast(1));

  TfLiteMiniBenchmarkSettings* settings = TfLiteMiniBenchmarkSettingsCreate();
  TfLiteMiniBenchmarkSettingsSetFlatBufferData(
      settings, mini_benchmark_fbb_.GetBufferPointer(),
      mini_benchmark_fbb_.GetSize());
  TfLiteMiniBenchmarkSettingsSetErrorReporter(
      settings, &reporter, &MockErrorReporter::InvokeErrorReporter);

  TfLiteMiniBenchmarkResult* result =
      TfLiteBlockingValidatorRunnerTriggerValidation(settings);
  EXPECT_THAT(TfLiteMiniBenchmarkResultInitStatus(result),
              tflite::acceleration::kMinibenchmarkPreconditionNotMet);
  EXPECT_EQ(TfLiteMiniBenchmarkResultFlatBufferData(result), nullptr);
  EXPECT_EQ(TfLiteMiniBenchmarkResultFlatBufferDataSize(result), 0);
  TfLiteMiniBenchmarkResultFree(result);
  TfLiteMiniBenchmarkSettingsFree(settings);
}

TEST_F(CApiTest, ReturnFailStatusWhenSettingsCorrupted) {
  std::vector<uint8_t> settings_corrupted(10, 1);
  TfLiteMiniBenchmarkSettings* settings = TfLiteMiniBenchmarkSettingsCreate();
  TfLiteMiniBenchmarkSettingsSetFlatBufferData(
      settings, settings_corrupted.data(), settings_corrupted.size());
  TfLiteMiniBenchmarkResult* result =
      TfLiteBlockingValidatorRunnerTriggerValidation(settings);

  EXPECT_THAT(
      TfLiteMiniBenchmarkResultInitStatus(result),
      tflite::acceleration::kMinibenchmarkCorruptSizePrefixedFlatbufferFile);
  TfLiteMiniBenchmarkResultFree(result);
  TfLiteMiniBenchmarkSettingsFree(settings);
}

TEST_F(CApiTest,
       SucceedOnSimpleAdditionModelWithCustomValidationAndPassingRule) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }

  const int batch_size = 1;
  size_t input_size[] = {batch_size * 8 * 8 * 3 * 4};
  std::vector<uint8_t> custom_input_data(input_size[0]);

  float test_input_float = 55.25f;
  float test_output_float = test_input_float * 3;

  uint8_t* test_input_bytes = reinterpret_cast<uint8_t*>(&test_input_float);
  uint8_t* test_output_bytes = reinterpret_cast<uint8_t*>(&test_output_float);

  for (int i = 0; i < input_size[0]; i += 4) {
    custom_input_data[i] = test_input_bytes[0] & 0xff;
    custom_input_data[i + 1] = test_input_bytes[1] & 0xff;
    custom_input_data[i + 2] = test_input_bytes[2] & 0xff;
    custom_input_data[i + 3] = test_input_bytes[3] & 0xff;
  }

  mini_benchmark_fbb_.Finish(tflite::CreateMinibenchmarkSettings(
      mini_benchmark_fbb_, CreateTFLiteSettings(),
      CreateModelFile(simple_addition_model_path_), CreateStoragePaths(),
      CreateValidationSettings()));
  MockResultEvaluator mock_evaluator;
  EXPECT_CALL(mock_evaluator, HasPassedAccuracyCheck)
      .Times(1)
      .WillRepeatedly(testing::Return(true));

  TfLiteMiniBenchmarkSettings* settings = TfLiteMiniBenchmarkSettingsCreate();
  TfLiteMiniBenchmarkSettingsSetFlatBufferData(
      settings, mini_benchmark_fbb_.GetBufferPointer(),
      mini_benchmark_fbb_.GetSize());
  TfLiteMiniBenchmarkCustomValidationInfo* custom_validation =
      TfLiteMiniBenchmarkSettingsCustomValidationInfo(settings);
  TfLiteMiniBenchmarkCustomValidationInfoSetBuffer(custom_validation,
                                                   /*batch_size=*/batch_size,
                                                   custom_input_data.data(),
                                                   /*buffer_dim=*/input_size,
                                                   /*buffer_dim_size=*/1);
  TfLiteMiniBenchmarkCustomValidationInfoSetAccuracyValidator(
      custom_validation, &mock_evaluator, MockResultEvaluator::Invoke);

  TfLiteMiniBenchmarkResult* result =
      TfLiteBlockingValidatorRunnerTriggerValidation(settings);
  std::vector<const tflite::BenchmarkEvent*> events =
      ToBenchmarkEvents(TfLiteMiniBenchmarkResultFlatBufferData(result),
                        TfLiteMiniBenchmarkResultFlatBufferDataSize(result));

  EXPECT_THAT(TfLiteMiniBenchmarkResultInitStatus(result),
              tflite::acceleration::kMinibenchmarkSuccess);

  EXPECT_THAT(events, testing::Not(testing::IsEmpty()));
  for (auto& event : events) {
    const tflite::BenchmarkResult* result = event->result();
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(event->event_type(), tflite::BenchmarkEventType_END);
    EXPECT_TRUE(event->result()->ok());
    EXPECT_THAT(event->result()->actual_output(),
                testing::Pointee(testing::SizeIs(1)));

    const flatbuffers::Vector<unsigned char>* inference_output =
        event->result()->actual_output()->Get(0)->value();

    EXPECT_EQ(inference_output->size(), input_size[0]);

    for (int i = 0; i < inference_output->size(); i += 4) {
      float f;
      uint8_t b[] = {inference_output->Get(i), inference_output->Get(i + 1),
                     inference_output->Get(i + 2),
                     inference_output->Get(i + 3)};
      memcpy(&f, &b, sizeof(f));
      EXPECT_EQ(f, test_output_float);
      EXPECT_EQ(b[0], test_output_bytes[0]);
      EXPECT_EQ(b[1], test_output_bytes[1]);
      EXPECT_EQ(b[2], test_output_bytes[2]);
      EXPECT_EQ(b[3], test_output_bytes[3]);
    }
  }
  TfLiteMiniBenchmarkResultFree(result);
  TfLiteMiniBenchmarkSettingsFree(settings);
}
}  // namespace
