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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_TELEMETRY_TELEMETRY_SETTINGS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_TELEMETRY_TELEMETRY_SETTINGS_H_

#include <map>
#include <string>

#include "tensorflow/lite/experimental/telemetry/telemetry_status.h"

namespace tflite {

// TFLite model and interpreter settings that will be reported by telemetry.
struct TelemetrySettings {
  // Source of the settings. Determines how `data` is interpreted.
  TelemetrySource source;
  // Settings data. Interpretation based on `source`.
  // If `source` is TFLITE_INTERPRETER, the type of `data` will
  // be `TelemetryInterpreterSettings`.
  // Otherwise, the data is provided by the individual delegate.
  const void* data = nullptr;
};

// TfLite model information and settings of the interpreter.
struct TelemetryInterpreterSettings {
  TelemetryInterpreterSettings() = default;

  // Metadata from the TfLite model.
  std::map<std::string, std::string> model_metadata;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_TELEMETRY_TELEMETRY_SETTINGS_H_
