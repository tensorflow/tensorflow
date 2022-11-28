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

#ifndef TENSORFLOW_LITE_PROFILING_TELEMETRY_TELEMETRY_H_
#define TENSORFLOW_LITE_PROFILING_TELEMETRY_TELEMETRY_H_

#include <cstdint>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/profiling/telemetry/telemetry_settings.h"
#include "tensorflow/lite/profiling/telemetry/telemetry_status.h"

namespace tflite {

// Methods for instrumenting TFLite runtime to export telemetry events to
// profilers.

// Reports an interpreter telemetry event.
// `event_name` indicates the name of the event (e.g. "Invoke") and should not
// be nullptr.
void TelemetryReportEvent(TfLiteContext* context, const char* event_name,
                          TfLiteStatus status);

// Reports an interpreter telemetry event associated with an op.
// `op_name` indicates the name of the op and should not be nullptr.
void TelemetryReportOpEvent(TfLiteContext* context, const char* op_name,
                            int64_t op_index, int64_t subgraph_index,
                            TfLiteStatus status);

// Reports a delegate telemetry event.
// `event_name` indicates the name of the event (e.g. "Invoke") and should not
// be nullptr.
// `source` indicates which delegate the event is from.
// `code` is the error code from the delegate.
void TelemetryReportDelegateEvent(TfLiteContext* context,
                                  const char* event_name,
                                  TelemetrySource source, uint32_t code);

// Reports a delegate telemetry event associated with an op.
// `op_name` indicates the name of the op and should not be nullptr.
void TelemetryReportDelegateOpEvent(TfLiteContext* context, const char* op_name,
                                    int64_t op_index, int64_t subgraph_index,
                                    TelemetrySource source, uint32_t code);

// Reports model and interpreter level settings.
// `setting_name` indicates the name of the setting.
void TelemetryReportSettings(TfLiteContext* context, const char* setting_name,
                             const TelemetryInterpreterSettings& settings);

// Reports delegate settings.
// `setting_name` indicates the name of the setting.
// `source` indicates which delegate the event is from.
// `settings` is the delegate provided settings and should not be nullptr.
void TelemetryReportDelegateSettings(TfLiteContext* context,
                                     const char* setting_name,
                                     TelemetrySource source,
                                     const void* settings);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_TELEMETRY_TELEMETRY_H_
