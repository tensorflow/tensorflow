/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/telemetry.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/profiling/profile_buffer.h"

namespace tflite {
namespace delegates {
namespace {

constexpr int32_t kDummyCode = 2;
constexpr bool kDummyGpuPrecisionLossAllowed = true;
constexpr tflite::Delegate kDummyDelegate = tflite::Delegate_GPU;
constexpr DelegateStatusSource kDummySource =
    DelegateStatusSource::TFLITE_NNAPI;

TEST(TelemetryTest, StatusConversion) {
  DelegateStatus status(kDummySource, kDummyCode);
  int64_t serialized_int = status.full_status();
  DelegateStatus deserialized_status(serialized_int);

  EXPECT_EQ(kDummyCode, deserialized_status.code());
  EXPECT_EQ(kDummySource, deserialized_status.source());
  EXPECT_EQ(serialized_int, deserialized_status.full_status());
}

// Dummy profiler to test delegate reporting.
class DelegateProfiler : public Profiler {
 public:
  DelegateProfiler() {}
  ~DelegateProfiler() override = default;

  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1,
                      int64_t event_metadata2) override {
    int event_handle = -1;
    if (event_type ==
            Profiler::EventType::GENERAL_RUNTIME_INSTRUMENTATION_EVENT &&
        std::string(tag) == kDelegateSettingsTag) {
      event_buffer_.emplace_back();
      event_handle = event_buffer_.size();

      // event_metadata1 is a pointer to a TfLiteDelegate.
      EXPECT_NE(event_metadata1, 0);
      auto* delegate = reinterpret_cast<TfLiteDelegate*>(event_metadata1);
      EXPECT_EQ(delegate->flags, kTfLiteDelegateFlagsNone);
      // event_metadata2 is a pointer to TFLiteSettings.
      EXPECT_NE(event_metadata2, 0);
      auto* settings = reinterpret_cast<TFLiteSettings*>(event_metadata2);
      EXPECT_EQ(settings->delegate(), kDummyDelegate);
      EXPECT_EQ(settings->gpu_settings()->is_precision_loss_allowed(),
                kDummyGpuPrecisionLossAllowed);
    } else if (event_type ==
                   Profiler::EventType::GENERAL_RUNTIME_INSTRUMENTATION_EVENT &&
               std::string(tag) == kDelegateStatusTag) {
      event_buffer_.emplace_back();
      event_handle = event_buffer_.size();

      EXPECT_EQ(event_metadata2, static_cast<int64_t>(kTfLiteOk));
      DelegateStatus reported_status(event_metadata1);
      EXPECT_EQ(reported_status.source(), kDummySource);
      EXPECT_EQ(reported_status.code(), kDummyCode);
    }

    EXPECT_NE(-1, event_handle);
    return event_handle;
  }

  void EndEvent(uint32_t event_handle) override {
    EXPECT_EQ(event_handle, event_buffer_.size());
  }

  int NumRecordedEvents() { return event_buffer_.size(); }

 private:
  std::vector<profiling::ProfileEvent> event_buffer_;
};

TEST(TelemetryTest, DelegateStatusReport) {
  DelegateProfiler profiler;
  TfLiteDelegate delegate = TfLiteDelegateCreate();
  TfLiteContext context;
  context.profiler = &profiler;
  DelegateStatus status(kDummySource, kDummyCode);

  EXPECT_EQ(ReportDelegateStatus(&context, &delegate, status), kTfLiteOk);
  EXPECT_EQ(ReportDelegateStatus(&context, &delegate, status), kTfLiteOk);
  EXPECT_EQ(profiler.NumRecordedEvents(), 2);
}

TEST(TelemetryTest, DelegateSettingsReport) {
  DelegateProfiler profiler;
  TfLiteDelegate delegate = TfLiteDelegateCreate();
  TfLiteContext context;
  context.profiler = &profiler;

  flatbuffers::FlatBufferBuilder flatbuffer_builder;
  flatbuffers::Offset<tflite::GPUSettings> gpu_settings =
      tflite::CreateGPUSettings(
          flatbuffer_builder,
          /**is_precision_loss_allowed**/ kDummyGpuPrecisionLossAllowed);
  auto* tflite_settings_ptr = flatbuffers::GetTemporaryPointer(
      flatbuffer_builder,
      CreateTFLiteSettings(flatbuffer_builder, kDummyDelegate,
                           /*nnapi_settings=*/0,
                           /*gpu_settings=*/gpu_settings));

  EXPECT_EQ(ReportDelegateSettings(&context, &delegate, *tflite_settings_ptr),
            kTfLiteOk);
  EXPECT_EQ(profiler.NumRecordedEvents(), 1);

  // Also report status to simulate typical use-case.
  DelegateStatus status(kDummySource, kDummyCode);
  EXPECT_EQ(ReportDelegateStatus(&context, &delegate, status), kTfLiteOk);
  EXPECT_EQ(ReportDelegateStatus(&context, &delegate, status), kTfLiteOk);
  EXPECT_EQ(profiler.NumRecordedEvents(), 3);
}

}  // namespace
}  // namespace delegates
}  // namespace tflite
