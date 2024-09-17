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
#include "tensorflow/compiler/mlir/lite/schema/schema_conversion_utils.h"

#include <algorithm>

namespace tflite {

int8_t ConvertBuiltinCodeToDeprecatedBuiltinCode(
    const BuiltinOperator builtin_code) {
  return (builtin_code < BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES)
             ? static_cast<int8_t>(builtin_code)
             : static_cast<int8_t>(
                   BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES);
}

// The following methods are the following `OperatorCode` table object creation
// methods for backward compatibility.  These are manually copied from the
// flatbuffer generated code from schema v3. They serve as overloads for the
// v3a's CreateOperatorCode functions in schema_generated.h and enable code that
// still assumes flatbuffer schema v3 to be unchanged with the inclusion of the
// schema_utils header.
// TODO(b/162392898): remove once all callers are updated to use schema v3a
// functions.

flatbuffers::Offset<OperatorCode> CreateOperatorCode(
    flatbuffers::FlatBufferBuilder &_fbb, BuiltinOperator builtin_code,
    flatbuffers::Offset<flatbuffers::String> custom_code, int32_t version) {
  OperatorCodeBuilder builder_(_fbb);
  builder_.add_version(version);

  int8_t deprecated_builtin_code =
      static_cast<int8_t>(BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES);
  if (builtin_code < BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES) {
    deprecated_builtin_code = static_cast<int8_t>(builtin_code);
  }
  builder_.add_deprecated_builtin_code(deprecated_builtin_code);
  builder_.add_custom_code(custom_code);
  builder_.add_builtin_code(builtin_code);
  return builder_.Finish();
}

flatbuffers::Offset<OperatorCode> CreateOperatorCodeDirect(
    flatbuffers::FlatBufferBuilder &_fbb, BuiltinOperator builtin_code,
    const char *custom_code, int32_t version) {
  auto custom_code__ = custom_code ? _fbb.CreateString(custom_code) : 0;
  int8_t deprecated_builtin_code =
      static_cast<int8_t>(BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES);
  if (builtin_code < BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES) {
    deprecated_builtin_code = static_cast<int8_t>(builtin_code);
  }
  return CreateOperatorCode(_fbb, deprecated_builtin_code, custom_code__,
                            version, builtin_code);
}

}  // namespace tflite
