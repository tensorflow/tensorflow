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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_SAVED_MODEL_TO_TFL_FLATBUFFER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_SAVED_MODEL_TO_TFL_FLATBUFFER_H_

#include "tensorflow/compiler/mlir/lite/converter_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/model_flags.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Converts the given saved_model(either v1 or v2) to a TF Lite FlatBuffer
// string according to the given model flags, converter flags and tags. Returns
// error status if it fails to convert the input.
absl::Status ConvertSavedModelToTFLiteFlatBuffer(
    const tflite::ModelFlags& model_flags,
    tflite::ConverterFlags& converter_flags, string* result,
    const quantization::PyFunctionLibrary* quantization_py_function_lib);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_SAVED_MODEL_TO_TFL_FLATBUFFER_H_
