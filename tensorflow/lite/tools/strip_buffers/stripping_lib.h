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

#ifndef TENSORFLOW_LITE_TOOLS_STRIP_BUFFERS_STRIPPING_LIB_H_
#define TENSORFLOW_LITE_TOOLS_STRIP_BUFFERS_STRIPPING_LIB_H_

#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// Strips eligible buffers from Flatbuffer, to generate a 'leaner' model.
// Buffers for tensors that satisfy the following constraints are stripped out:
// 1. Are either of: Float32, Int32, UInt8, Int8
// 2. If Int32, the tensor should have a min of 10 elements
// NOTE: This only supports a single Subgraph for now.
TfLiteStatus StripWeightsFromFlatbuffer(
    const Model* input_model,
    flatbuffers::FlatBufferBuilder* new_model_builder);

// The same function as above, but takes in the input model flatbuffer in a
// string and returns the stripped version in a string.
// Returns empty string on error.
// NOTE: This only supports a single Subgraph for now.
std::string StripWeightsFromFlatbuffer(
    const absl::string_view input_flatbuffer);

// Generates buffers with random data, for tensors that were mutated using
// strip_buffers_from_fb.
// The modified flatbuffer is built into new_model_builder.
// NOTE: This only supports a single Subgraph for now.
TfLiteStatus ReconstituteConstantTensorsIntoFlatbuffer(
    const Model* input_model,
    flatbuffers::FlatBufferBuilder* new_model_builder);

// The same function as above but takes in the input model flatbuffer in a
// string and returns the reconstituded version in a string.
// Returns empty string on error.
// NOTE: This only supports a single Subgraph for now.
std::string ReconstituteConstantTensorsIntoFlatbuffer(
    const absl::string_view input_flatbuffer);

// Return true if the input model has been stripped before.
bool FlatbufferHasStrippedWeights(const Model* input_model);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_STRIP_BUFFERS_STRIPPING_LIB_H_
