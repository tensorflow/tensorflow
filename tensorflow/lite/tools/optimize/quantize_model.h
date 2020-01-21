/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_QUANTIZE_MODEL_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_QUANTIZE_MODEL_H_

#include <memory>
#include <unordered_set>

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace optimize {

// Quantizes input_model and populates the provided builder with the new model.
// input_model is required to have min/max information populated in its
// quantization params.
//
// Inputs and output types default to float instead of a quantized type.
//
// Note: This is a private API, subject to change.
TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* input_model,
                           const TensorType& activations_type,
                           ErrorReporter* error_reporter);

// Same as above, but the types of quantized inputs and outputs are
// configurable.
//
// Note: This is a private API, subject to change.
TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* input_model, const TensorType& input_type,
                           const TensorType& output_type,
                           const TensorType& activations_type,
                           ErrorReporter* error_reporter);

// Same as above, but can enable allowing float intermediate operations for ops
// that do not yet support quantizable.
//
// Note: This is a private API, subject to change.
TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* input_model, const TensorType& input_type,
                           const TensorType& output_type, bool allow_float,
                           const TensorType& activations_type,
                           ErrorReporter* error_reporter);

// Same as above, but enables only quantizing a whitelist of operations,
// specified by their operator output name.
//
// Note: This is a private API, subject to change.
TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* input_model, const TensorType& input_type,
                           const TensorType& output_type, bool allow_float,
                           const std::unordered_set<string>& operator_names,
                           const TensorType& activations_type,
                           ErrorReporter* error_reporter);

}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_QUANTIZE_MODEL_H_
