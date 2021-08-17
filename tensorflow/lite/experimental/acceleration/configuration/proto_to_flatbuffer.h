/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_PROTO_TO_FLATBUFFER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_PROTO_TO_FLATBUFFER_H_

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/configuration/configuration.pb.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"

namespace tflite {

// Converts the provided ComputeSettings from proto to flatbuffer format.
const ComputeSettings* ConvertFromProto(
    const proto::ComputeSettings& proto_settings,
    flatbuffers::FlatBufferBuilder* builder);

// Converts the provided MiniBenchmarkSettings from proto to flatbuffer format.
const MinibenchmarkSettings* ConvertFromProto(
    const proto::MinibenchmarkSettings& proto_settings,
    flatbuffers::FlatBufferBuilder* builder);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_PROTO_TO_FLATBUFFER_H_
