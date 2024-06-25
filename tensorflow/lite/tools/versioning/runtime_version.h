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
#ifndef TENSORFLOW_LITE_TOOLS_VERSIONING_RUNTIME_VERSION_H_
#define TENSORFLOW_LITE_TOOLS_VERSIONING_RUNTIME_VERSION_H_

#include <string>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"

namespace tflite {
// Update minimum runtime version of the given TFL flatbuffer model.
void UpdateMinimumRuntimeVersionForModel(uint8_t* model_buffer_pointer);

// Find the minimum runtime version of a given op version. Return an empty
// string the version is not registered.
std::string FindMinimumRuntimeVersionForOp(tflite::BuiltinOperator op_code,
                                           int op_version);

// Returns true if the first version string precedes the second.
// For example, '1.9' should precede '1.14', also '1.14' should precede
// '1.14.1'. If two version string is equal, then false will be returned.
bool CompareRuntimeVersion(const std::string&, const std::string&);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_VERSIONING_RUNTIME_VERSION_H_
