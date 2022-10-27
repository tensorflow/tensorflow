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

#ifndef TENSORFLOW_LITE_PROFILING_TELEMETRY_PROFILER_H_
#define TENSORFLOW_LITE_PROFILING_TELEMETRY_PROFILER_H_

#include <cstdint>

#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/profiling/telemetry/telemetry_settings.h"
#include "tensorflow/lite/profiling/telemetry/telemetry_status.h"

namespace tflite {

// Telemetry profiler interface.
// When installed, the telemetry profilers accepts profiler events exported from
// TFLite runtime profiler instrumentation points, interprets the events
// based on the event type and forward to corresponding `Report` function.
// The implementation of the `Report` functions are responsible for dumping the
// profiling events to the data sink.
// The implementation of TelemetryProfiler is required to be thread safe.
class TelemetryProfiler : public Profiler {
 public:
  // General Telemetry events.

  // Reports a telemetry event with status.
  // `event_name` indicates the name of the event (e.g. "Invoke") and should not
  // be nullptr.
  // `status` shows 1) the source of the event, interpreter or which delegate,
  // 2) the return status of the event.
  virtual void ReportTelemetryEvent(const char* event_name,
                                    TelemetryStatusCode status) = 0;

  // Reports an op telemetry event with status.
  // Same as `ReportTelemetryEvent`, with additional args `op_idx` and
  // `subgraph_idx`.
  virtual void ReportTelemetryOpEvent(const char* event_name, int64_t op_idx,
                                      int64_t subgraph_idx,
                                      TelemetryStatusCode status) = 0;

  // Telemetry ReportSettings events.

  // Reports the model and interpreter settings.
  // `setting_name` indicates the name of the setting and should not be nullptr.
  // `settings`'s lifespan is not guaranteed outside the scope of
  // `ReportSettings` call.
  virtual void ReportSettings(const char* setting_name,
                              const TelemetrySettings& settings) = 0;

  // Performance measurement events.

  // Signals the beginning of an operator invocation.
  // `op_name` is the name of the operator and should not be nullptr.
  // Op invoke event are triggered with OPERATOR_INVOKE_EVENT type for TfLite
  // ops and delegate kernels, and DELEGATE_OPERATOR_INVOKE_EVENT for delegate
  // ops within a delegate kernels, if the instrumentation is in place.
  // Returns event handle which can be passed to `EndOpInvokeEvent` later.
  virtual uint32_t ReportBeginOpInvokeEvent(const char* op_name, int64_t op_idx,
                                            int64_t subgraph_idx) = 0;

  // Signals the end to the event specified by `event_handle`.
  virtual void ReportEndOpInvokeEvent(uint32_t event_handle) = 0;

  // For op / delegate op with built-in performance measurements, they
  // are able to report the elapsed time directly.
  virtual void ReportOpInvokeEvent(const char* op_name, uint64_t elapsed_time,
                                   int64_t op_idx, int64_t subgraph_idx) = 0;

 private:
  // Methods inherited from TfLite Profiler.
  // TelemetryProfiler will dispatch the event signals to appropriate `Report`
  // functinos defined above based on the event type.
  // Subclasses should not override those following methods.
  void AddEvent(const char* tag, EventType event_type, uint64_t metric,
                int64_t event_metadata1, int64_t event_metadata2) final;
  void AddEventWithData(const char* tag, EventType event_type,
                        const void* data) final;
  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1, int64_t event_metadata2) final;
  void EndEvent(uint32_t event_handle) final;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_TELEMETRY_PROFILER_H_
