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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MODEL_MODIFIER_INPUT_EMBEDDER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MODEL_MODIFIER_INPUT_EMBEDDER_H_

#include <vector>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace acceleration {

// Generate a new Model by fill the input data of plain_model with
// new_input_buffer. The new model version is set to 3.
MinibenchmarkStatus GenerateModelWithInput(
    const tflite::Model& plain_model,
    const std::vector<std::vector<uint8_t>>& new_input_buffer,
    flatbuffers::FlatBufferBuilder& output_model);

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MODEL_MODIFIER_INPUT_EMBEDDER_H_
