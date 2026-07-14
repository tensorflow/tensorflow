/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_QUANTIZATION_WRAPPER_UTILS_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_QUANTIZATION_WRAPPER_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {

// Load a tflite model from path.
TfLiteStatus LoadModel(const string& path, ModelT* model);

// Going through the model and add intermediates tensors if the ops have any.
// Returns early if the model has already intermediate tensors. This is to
// support cases where a model is initialized multiple times.
TfLiteStatus AddIntermediateTensorsToFusedOp(
    flatbuffers::FlatBufferBuilder* builder, ModelT* model);

// Write model to a given location.
bool WriteFile(const std::string& out_file, const uint8_t* bytes,
               size_t num_bytes);

}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_QUANTIZATION_WRAPPER_UTILS_H_
