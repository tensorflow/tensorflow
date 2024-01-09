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
#ifndef TENSORFLOW_LITE_PROFILING_TELEMETRY_C_PROFILER_H_
#define TENSORFLOW_LITE_PROFILING_TELEMETRY_C_PROFILER_H_

#include <stdint.h>

#include "tensorflow/lite/profiling/telemetry/c/telemetry_setting.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// C API for TFLite telemetry profiler.
// See C++ interface in tflite::telemetry::TelemetryProfiler.
// Note: This struct does not comply with ABI stability.
typedef struct TfLiteTelemetryProfilerStruct {
  // Data that profiler needs to identify itself. This data is owned by the
  // profiler. The profiler is owned in the user code, so the profiler is
  // responsible for deallocating this when it is destroyed.
  void* data;

  // Reports a telemetry event with status.
  // `event_name` indicates the name of the event (e.g. "Invoke") and should not
  // be nullptr.
  // `status`: uint64_t representation of TelemetryStatusCode.
  void (*ReportTelemetryEvent)(  // NOLINT
      struct TfLiteTelemetryProfilerStruct* profiler, const char* event_name,
      uint64_t status);

  // Reports an op telemetry event with status.
  // Same as `ReportTelemetryEvent`, with additional args `op_idx` and
  // `subgraph_idx`.
  // `status`: uint64_t representation of TelemetryStatusCode.
  void (*ReportTelemetryOpEvent)(  // NOLINT
      struct TfLiteTelemetryProfilerStruct* profiler, const char* event_name,
      int64_t op_idx, int64_t subgraph_idx, uint64_t status);

  // Reports the model and interpreter settings.
  // `setting_name` indicates the name of the setting and should not be nullptr.
  // `settings`'s lifespan is not guaranteed outside the scope of
  // `ReportSettings` call.
  void (*ReportSettings)(  // NOLINT
      struct TfLiteTelemetryProfilerStruct* profiler, const char* setting_name,
      const TfLiteTelemetrySettings* settings);

  // Signals the beginning of an operator invocation.
  // `op_name` is the name of the operator and should not be nullptr.
  // Op invoke event are triggered with OPERATOR_INVOKE_EVENT type for TfLite
  // ops and delegate kernels, and DELEGATE_OPERATOR_INVOKE_EVENT for delegate
  // ops within a delegate kernels, if the instrumentation is in place.
  // Returns event handle which can be passed to `EndOpInvokeEvent` later.
  uint32_t (*ReportBeginOpInvokeEvent)(  // NOLINT
      struct TfLiteTelemetryProfilerStruct* profiler, const char* op_name,
      int64_t op_idx, int64_t subgraph_idx);

  // Signals the end to the event specified by `event_handle`.
  void (*ReportEndOpInvokeEvent)(  // NOLINT
      struct TfLiteTelemetryProfilerStruct* profiler, uint32_t event_handle);

  // For op / delegate op with built-in performance measurements, they
  // are able to report the elapsed time directly.
  // `elapsed_time` is in microsecond.
  void (*ReportOpInvokeEvent)(  // NOLINT
      struct TfLiteTelemetryProfilerStruct* profiler, const char* op_name,
      uint64_t elapsed_time, int64_t op_idx, int64_t subgraph_idx);
} TfLiteTelemetryProfilerStruct;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_PROFILING_TELEMETRY_C_PROFILER_H_
