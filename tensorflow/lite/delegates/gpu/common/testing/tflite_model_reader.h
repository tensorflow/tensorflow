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
#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TESTING_TFLITE_MODEL_READER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TESTING_TFLITE_MODEL_READER_H_

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/model_builder.h"

namespace tflite {
namespace gpu {

// Generates GraphFloat32 basing on the FlatBufferModel without specifying a
// delegate.
absl::Status BuildFromFlatBuffer(const tflite::FlatBufferModel& flatbuffer,
                                 GraphFloat32* graph);
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TESTING_TFLITE_MODEL_READER_H_
