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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_SCHEMA_SCHEMA_CONVERSION_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_SCHEMA_SCHEMA_CONVERSION_UTILS_H_

#include "flatbuffers/flatbuffers.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

namespace tflite {

int8_t ConvertBuiltinCodeToDeprecatedBuiltinCode(BuiltinOperator builtin_code);

// The following methods are for backward compatibility for the early version
// three, which does not have an extended builtin code.
flatbuffers::Offset<OperatorCode> CreateOperatorCode(
    flatbuffers::FlatBufferBuilder &_fbb,
    BuiltinOperator builtin_code = BuiltinOperator_ADD,
    flatbuffers::Offset<flatbuffers::String> custom_code = 0,
    int32_t version = 1);

flatbuffers::Offset<OperatorCode> CreateOperatorCodeDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    BuiltinOperator builtin_code = BuiltinOperator_ADD,
    const char *custom_code = nullptr, int32_t version = 1);

}  // namespace tflite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_SCHEMA_SCHEMA_CONVERSION_UTILS_H_
