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
#ifndef TENSORFLOW_LITE_TOOLS_VERSIONING_OP_VERSION_H_
#define TENSORFLOW_LITE_TOOLS_VERSIONING_OP_VERSION_H_

#include <vector>

#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"
#include "tensorflow/lite/tools/versioning/op_signature.h"

namespace tflite {

// Returns version of builtin ops by the given signature.
int GetBuiltinOperatorVersion(const OpSignature& op_sig);

// Update operator's version of the given TFL flatbuffer model.
void UpdateOpVersion(uint8_t* model_buffer_pointer);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_VERSIONING_OP_VERSION_H_
