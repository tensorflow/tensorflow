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
#include "tensorflow/lite/profiling/telemetry/profiler.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/profiling/telemetry/c/telemetry_setting.h"
#include "tensorflow/lite/profiling/telemetry/telemetry_status.h"

namespace tflite::telemetry {
namespace {

constexpr char kEventName[] = "event_name";
constexpr char kSettingName[] = "setting_name";

class MockTelemtryProfiler : public TelemetryProfiler {
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

class TelemetryStructTest : public ::testing::Test {
 protected:
  TelemetryStructTest() {
    context_.profiler = &profiler_;

    profiler_struct_.data = &mock_profiler_;
    profiler_struct_.ReportTelemetryEvent =
        [](struct TfLiteTelemetryProfilerStruct* profiler,
           const char* event_name, uint64_t status) {
          static_cast<MockTelemtryProfiler*>(profiler->data)
              ->ReportTelemetryEvent(
                  event_name, tflite::telemetry::TelemetryStatusCode(status));
        };
    profiler_struct_.ReportTelemetryOpEvent =
        [](struct TfLiteTelemetryProfilerStruct* profiler,
           const char* event_name, int64_t op_idx, int64_t subgraph_idx,
           uint64_t status) {
          static_cast<MockTelemtryProfiler*>(profiler->data)
              ->ReportTelemetryOpEvent(
                  event_name, op_idx, subgraph_idx,
                  tflite::telemetry::TelemetryStatusCode(status));
        };
    profiler_struct_.ReportSettings =
        [](struct TfLiteTelemetryProfilerStruct* profiler,
           const char* setting_name, const TfLiteTelemetrySettings* settings) {
          static_cast<MockTelemtryProfiler*>(profiler->data)
              ->ReportSettings(setting_name, settings);
        };
    profiler_struct_.ReportBeginOpInvokeEvent =
        [](struct TfLiteTelemetryProfilerStruct* profiler, const char* op_name,
           int64_t op_idx, int64_t subgraph_idx) -> uint32_t {
      return static_cast<MockTelemtryProfiler*>(profiler->data)
          ->ReportBeginOpInvokeEvent(op_name, op_idx, subgraph_idx);
    };
    profiler_struct_.ReportEndOpInvokeEvent =
        [](struct TfLiteTelemetryProfilerStruct* profiler,
           uint32_t event_handle) {
          return static_cast<MockTelemtryProfiler*>(profiler->data)
              ->ReportEndOpInvokeEvent(event_handle);
        };
    profiler_struct_.ReportOpInvokeEvent =
        [](struct TfLiteTelemetryProfilerStruct* profiler, const char* op_name,
           uint64_t elapsed_time, int64_t op_idx, int64_t subgraph_idx) {
          return static_cast<MockTelemtryProfiler*>(profiler->data)
              ->ReportOpInvokeEvent(op_name, elapsed_time, op_idx,
                                    subgraph_idx);
        };
    profiler_.reset(telemetry::MakeTfLiteTelemetryProfiler(&profiler_struct_));
  }

  MockTelemtryProfiler mock_profiler_;
  std::unique_ptr<TelemetryProfiler> profiler_;
  TfLiteContext context_;
  TfLiteTelemetryProfilerStruct profiler_struct_;
};

TEST_F(TelemetryStructTest, TelemetryReportEvent) {
  EXPECT_CALL(mock_profiler_,
              ReportTelemetryEvent(kEventName, TelemetryStatusCode(kTfLiteOk)));

  profiler_->ReportTelemetryEvent(kEventName, TelemetryStatusCode(kTfLiteOk));
}

TEST_F(TelemetryStructTest, TelemetryReportOpEvent) {
  EXPECT_CALL(
      mock_profiler_,
      ReportTelemetryOpEvent(kEventName, 1, 2, TelemetryStatusCode(kTfLiteOk)));

  profiler_->ReportTelemetryOpEvent(kEventName, 1, 2,
                                    TelemetryStatusCode(kTfLiteOk));
}

TEST_F(TelemetryStructTest, TelemetryReportSettings) {
  EXPECT_CALL(mock_profiler_, ReportSettings(kSettingName, testing::_));
  TfLiteTelemetrySettings settings{};

  profiler_->ReportSettings(kSettingName, &settings);
}

TEST_F(TelemetryStructTest, TelemetryReportBeginOpInvokeEvent) {
  EXPECT_CALL(mock_profiler_, ReportBeginOpInvokeEvent(kSettingName, 1, 2));

  profiler_->ReportBeginOpInvokeEvent(kSettingName, 1, 2);
}

TEST_F(TelemetryStructTest, TelemetryReportEndOpInvokeEvent) {
  EXPECT_CALL(mock_profiler_, ReportEndOpInvokeEvent(1));

  profiler_->ReportEndOpInvokeEvent(1);
}

TEST_F(TelemetryStructTest, TelemetryReportOpInvokeEvent) {
  EXPECT_CALL(mock_profiler_, ReportOpInvokeEvent(kSettingName, 1, 2, 3));

  profiler_->ReportOpInvokeEvent(kSettingName, 1, 2, 3);
}

}  // namespace
}  // namespace tflite::telemetry
