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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_TELEMETRY_TELEMETRY_STATUS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_TELEMETRY_TELEMETRY_STATUS_H_

#include <cstdint>

#include "tensorflow/lite/c/c_api_types.h"

namespace tflite {

// The source of a telemetry event.
enum class TelemetrySource : uint32_t {
  TFLITE_INTERPRETER = 0,

  // For external delegate.
  // External delegate should identify themselves in telemetry event names by
  // prefixing the delegame name to it.
  TFLITE_CUSTOM_DELEGATE = 1,

  TFLITE_GPU = 2,
  TFLITE_NNAPI = 3,
  TFLITE_HEXAGON = 4,
  TFLITE_XNNPACK = 5,
  TFLITE_COREML = 6,
};

// A namespaced status code for telemetry events.
struct TelemetryStatusCode {
  TelemetrySource source = TelemetrySource::TFLITE_INTERPRETER;
  uint32_t status_code = 0;

  // Helper constructors to build the status code from various types.
  TelemetryStatusCode() = default;
  TelemetryStatusCode(TelemetrySource source, uint32_t status_code)
      : source(source), status_code(status_code) {}
  explicit TelemetryStatusCode(TfLiteStatus status)
      : TelemetryStatusCode(TelemetrySource::TFLITE_INTERPRETER, status) {}
  explicit TelemetryStatusCode(uint64_t code)
      : TelemetryStatusCode(static_cast<TelemetrySource>(code >> 32),
                            static_cast<uint32_t>(code)) {}

  // Returns the uint64_t representation of the status code.
  uint64_t code() const {
    return (static_cast<uint64_t>(source) << 32 | status_code);
  }

  bool operator==(const TelemetryStatusCode& other) const {
    return source == other.source && status_code == other.status_code;
  }
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_TELEMETRY_TELEMETRY_STATUS_H_
