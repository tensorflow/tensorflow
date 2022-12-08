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

#include "tensorflow/lite/core/api/profiler.h"

namespace tflite {

void TelemetryProfiler::AddEvent(const char* tag, EventType event_type,
                                 uint64_t metric, int64_t event_metadata1,
                                 int64_t event_metadata2) {
  switch (event_type) {
    case EventType::TELEMETRY_EVENT:
    case EventType::TELEMETRY_DELEGATE_EVENT: {
      // When the event_metadata1 is set to -1, the event is not associated
      // with a particular op. See telemetry.cc.
      if (event_metadata1 == -1) {
        ReportTelemetryEvent(tag, TelemetryStatusCode(metric));
      } else {
        ReportTelemetryOpEvent(tag, event_metadata1, event_metadata2,
                               TelemetryStatusCode(metric));
      }
      break;
    }
    case EventType::OPERATOR_INVOKE_EVENT:
    case EventType::DELEGATE_OPERATOR_INVOKE_EVENT:
    case EventType::DELEGATE_PROFILED_OPERATOR_INVOKE_EVENT: {
      ReportOpInvokeEvent(tag, metric, event_metadata1, event_metadata2);
      break;
    }
    default:
      // Rejects other event types.
      return;
  }
}

void TelemetryProfiler::AddEventWithData(const char* tag, EventType event_type,
                                         const void* data) {
  switch (event_type) {
    case EventType::TELEMETRY_REPORT_SETTINGS:
    case EventType::TELEMETRY_DELEGATE_REPORT_SETTINGS: {
      auto* settings = reinterpret_cast<const TelemetrySettings*>(data);
      if (settings) {
        ReportSettings(tag, *settings);
      }
      break;
    }
    default:
      // No other AddEventWithData will be accepted for telemetry.
      return;
  }
}

uint32_t TelemetryProfiler::BeginEvent(const char* tag, EventType event_type,
                                       int64_t event_metadata1,
                                       int64_t event_metadata2) {
  switch (event_type) {
    case EventType::OPERATOR_INVOKE_EVENT:
    case EventType::DELEGATE_OPERATOR_INVOKE_EVENT:
    case EventType::DELEGATE_PROFILED_OPERATOR_INVOKE_EVENT: {
      return ReportBeginOpInvokeEvent(tag, event_metadata1, event_metadata2);
    }
    default:
      // Telemetry Profiler does not accept other event types with BeginEvent.
      return UINT32_MAX;
  }
}

void TelemetryProfiler::EndEvent(uint32_t event_handle) {
  if (event_handle == UINT32_MAX) return;
  ReportEndOpInvokeEvent(event_handle);
}

}  // namespace tflite
