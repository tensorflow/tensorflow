/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_ACCELERATION_CONFIGURATION_FLATBUFFER_TO_PROTO_H_
#define TENSORFLOW_LITE_ACCELERATION_CONFIGURATION_FLATBUFFER_TO_PROTO_H_

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/acceleration/configuration/configuration.pb.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"

namespace tflite {

// Converts the provided ComputeSettings from flatbuffer to proto format.
proto::ComputeSettings ConvertFromFlatbuffer(
    const ComputeSettings& settings, bool skip_mini_benchmark_settings = false);

proto::ComputeSettings ConvertFromFlatbuffer(
    const ComputeSettingsT& settings,
    bool skip_mini_benchmark_settings = false);

// Converts the provided MiniBenchmarkEvent from flatbuffer to proto format.
proto::MiniBenchmarkEvent ConvertFromFlatbuffer(
    const MiniBenchmarkEvent& event);

proto::MiniBenchmarkEvent ConvertFromFlatbuffer(
    const MiniBenchmarkEventT& event);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_ACCELERATION_CONFIGURATION_FLATBUFFER_TO_PROTO_H_
