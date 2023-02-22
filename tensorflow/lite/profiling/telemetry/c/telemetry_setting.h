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
#ifndef TENSORFLOW_LITE_PROFILING_TELEMETRY_C_TELEMETRY_SETTING_H_
#define TENSORFLOW_LITE_PROFILING_TELEMETRY_C_TELEMETRY_SETTING_H_

#include <stddef.h>
#include <stdint.h>

#include "tensorflow/lite/core/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// TFLite model, interpreter or delegate settings that will be reported by
// telemetry.
// Note: This struct does not comply with ABI stability.
typedef struct TfLiteTelemetrySettings {
  // Source of the settings. Determines how `data` is interpreted.
  // See tflite::telemetry::TelemetrySource for definition.
  uint32_t source;

  // Settings data. Interpretation based on `source`.
  // If `source` is TFLITE_INTERPRETER, the type of `data` will
  // be `TelemetryInterpreterSettings`.
  // Otherwise, the data is provided by the individual delegate.
  // Owned by the caller that exports TelemetrySettings (e.g. Interpreter).
  const void* data;
} TfLiteTelemetrySettings;

typedef struct TfLiteTelemetryConversionMetadata
    TfLiteTelemetryConversionMetadata;

const int32_t* TfLiteTelemetryConversionMetadataGetModelOptimizationModes(
    const TfLiteTelemetryConversionMetadata* metadata);

size_t TfLiteTelemetryConversionMetadataGetNumModelOptimizationModes(
    const TfLiteTelemetryConversionMetadata* metadata);

// TfLite model information and settings of the interpreter.
// Note: This struct does not comply with ABI stability.
typedef struct TfLiteTelemetryInterpreterSettings
    TfLiteTelemetryInterpreterSettings;

const TfLiteTelemetryConversionMetadata*
TfLiteTelemetryInterpreterSettingsGetConversionMetadata(
    const TfLiteTelemetryInterpreterSettings* settings);

// Telemetry data for a specific TFLite subgraph.
typedef struct TfLiteTelemetrySubgraphInfo TfLiteTelemetrySubgraphInfo;

size_t TfLiteTelemetryInterpreterSettingsGetNumSubgraphInfo(
    const TfLiteTelemetryInterpreterSettings* settings);

const TfLiteTelemetrySubgraphInfo*
TfLiteTelemetryInterpreterSettingsGetSubgraphInfo(
    const TfLiteTelemetryInterpreterSettings* settings);

size_t TfLiteTelemetrySubgraphInfoGetNumOpTypes(
    TfLiteTelemetrySubgraphInfo* subgraph_info);

const int32_t* TfLiteTelemetrySubgraphInfoGetOpTypes(
    TfLiteTelemetrySubgraphInfo* subgraph_info);

size_t TfLiteTelemetrySubgraphInfoGetNumQuantizations(
    TfLiteTelemetrySubgraphInfo* subgraph_info);

const TfLiteQuantization* TfLiteTelemetrySubgraphInfoGetQuantizations(
    TfLiteTelemetrySubgraphInfo* subgraph_info);

size_t TfLiteTelemetrySubgraphInfoGetNumCustomOpNames(
    TfLiteTelemetrySubgraphInfo* subgraph_info);

const char** TfLiteTelemetrySubgraphInfoGetCustomOpNames(
    TfLiteTelemetrySubgraphInfo* subgraph_info);

// Telemetry information for GPU delegate.
typedef struct TfLiteTelemetryGpuDelegateSettings
    TfLiteTelemetryGpuDelegateSettings;

size_t TfLiteTelemetryGpuDelegateSettingsGetNumNodesDelegated(
    const TfLiteTelemetryGpuDelegateSettings* settings);

int TfLiteTelemetryGpuDelegateSettingsGetBackend(
    const TfLiteTelemetryGpuDelegateSettings* settings);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_PROFILING_TELEMETRY_C_TELEMETRY_SETTING_H_
