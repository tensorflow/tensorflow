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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_SPARSITY_SPARSIFY_MODEL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_SPARSITY_SPARSIFY_MODEL_H_

#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/c/c_api_types.h"

namespace mlir {
namespace lite {

// Sparsify the `input_model` and write the result to a flatbuffer `builder`.
TfLiteStatus SparsifyModel(const tflite::ModelT& input_model,
                           flatbuffers::FlatBufferBuilder* builder,
                           tflite::ErrorReporter* error_reporter);
}  // namespace lite
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_SPARSITY_SPARSIFY_MODEL_H_
