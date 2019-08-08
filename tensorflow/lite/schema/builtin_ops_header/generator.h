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
// An utility library to generate pure C header for builtin ops definition.
#ifndef TENSORFLOW_LITE_SCHEMA_BUILTIN_OPS_HEADER_GENERATOR_H_
#define TENSORFLOW_LITE_SCHEMA_BUILTIN_OPS_HEADER_GENERATOR_H_

#include <iostream>
#include <string>

namespace tflite {
namespace builtin_ops_header {

// Check if the input enum name (from the Flatbuffer definition) is valid.
bool IsValidInputEnumName(const std::string& name);

// Convert the enum name from Flatbuffer convention to C enum name convention.
// E.g. `L2_POOL_2D` becomes `kTfLiteBuiltinL2Pool2d`.
std::string ConstantizeVariableName(const std::string& name);

// The function generates a pure C header for builtin ops definition, and write
// it to the output stream.
bool GenerateHeader(std::ostream& os);

}  // namespace builtin_ops_header
}  // namespace tflite

#endif  // TENSORFLOW_LITE_SCHEMA_BUILTIN_OPS_HEADER_GENERATOR_H_
