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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_QUANTIZE_MODEL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_QUANTIZE_MODEL_H_

#include <memory>
#include <unordered_set>

#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mlir {
namespace lite {

// Quantize the `input_model` and write the result to a flatbuffer `builder`.
// The `input_type` and `output_type` can be float32/qint8/int8.
// Return partially quantized model if `allow_float` is true.
TfLiteStatus QuantizeModel(
    const tflite::ModelT& input_model, const tflite::TensorType& input_type,
    const tflite::TensorType& output_type,
    const std::unordered_set<std::string>& operator_names, bool allow_float,
    flatbuffers::FlatBufferBuilder* builder,
    tflite::ErrorReporter* error_reporter);
}  // namespace lite
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_QUANTIZE_MODEL_H_
