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
#include "tensorflow/lite/profiling/telemetry/telemetry.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/profiling/telemetry/c/telemetry_setting_internal.h"
#include "tensorflow/lite/profiling/telemetry/profiler.h"
#include "tensorflow/lite/profiling/telemetry/telemetry_status.h"

namespace tflite::telemetry {
namespace {

constexpr char kEventName[] = "event_name";
constexpr char kSettingName[] = "setting_name";

class MockTelemetryProfiler : public TelemetryProfiler {
 public:
  MOCK_METHOD(void, ReportTelemetryEvent,
              (const char* event_name, TelemetryStatusCode status), (override));
  MOCK_METHOD(void, ReportTelemetryOpEvent,
              (const char* event_name, int64_t op_idx, int64_t subgraph_idx,
               TelemetryStatusCode status),
              (override));
  MOCK_METHOD(void, ReportSettings,
              (const char* setting_name,
               const TfLiteTelemetrySettings* settings),
              (override));
  MOCK_METHOD(uint32_t, ReportBeginOpInvokeEvent,
              (const char* op_name, int64_t op_idx, int64_t subgraph_idx),
              (override));
  MOCK_METHOD(void, ReportEndOpInvokeEvent, (uint32_t event_handle),
              (override));
  MOCK_METHOD(void, ReportOpInvokeEvent,
              (const char* op_name, uint64_t elapsed_time, int64_t op_idx,
               int64_t subgraph_idx),
              (override));
};

class TelemetryTest : public ::testing::Test {
 protected:
  TelemetryTest() { context_.profiler = &profiler_; }

  MockTelemetryProfiler profiler_;
  TfLiteContext context_;
};

TEST_F(TelemetryTest, TelemetryReportEvent) {
  EXPECT_CALL(profiler_,
              ReportTelemetryEvent(kEventName, TelemetryStatusCode(kTfLiteOk)));

  TelemetryReportEvent(&context_, kEventName, kTfLiteOk);
}

TEST_F(TelemetryTest, TelemetryReportOpEvent) {
  EXPECT_CALL(profiler_, ReportTelemetryOpEvent(
                             kEventName, 1, 2, TelemetryStatusCode(kTfLiteOk)));

  TelemetryReportOpEvent(&context_, kEventName, 1, 2, kTfLiteOk);
}

TEST_F(TelemetryTest, TelemetryReportDelegateEvent) {
  EXPECT_CALL(profiler_, ReportTelemetryEvent(
                             kEventName, TelemetryStatusCode(
                                             TelemetrySource::TFLITE_GPU, 21)));

  TelemetryReportDelegateEvent(&context_, kEventName,
                               TelemetrySource::TFLITE_GPU, 21);
}

TEST_F(TelemetryTest, TelemetryReportDelegateOpEvent) {
  EXPECT_CALL(profiler_,
              ReportTelemetryOpEvent(
                  kEventName, 1, 2,
                  TelemetryStatusCode(TelemetrySource::TFLITE_GPU, 21)));

  TelemetryReportDelegateOpEvent(&context_, kEventName, 1, 2,
                                 TelemetrySource::TFLITE_GPU, 21);
}

TEST_F(TelemetryTest, TelemetryReportSettings) {
  EXPECT_CALL(profiler_, ReportSettings(kSettingName, testing::_));
  TfLiteTelemetryInterpreterSettings settings{};

  TelemetryReportSettings(&context_, kSettingName, &settings);
}

TEST_F(TelemetryTest, TelemetryReportDelegateSettings) {
  std::string settings = "gpu delegate settings";
  EXPECT_CALL(profiler_, ReportSettings(kSettingName, testing::_));

  TelemetryReportDelegateSettings(&context_, kSettingName,
                                  TelemetrySource::TFLITE_GPU, &settings);
}

}  // namespace
}  // namespace tflite::telemetry
