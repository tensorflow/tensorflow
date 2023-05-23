/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
/// WARNING: Users of TensorFlow Lite should not include this file directly,
/// but should instead include
/// "third_party/tensorflow/lite/tools/verifier_internal.h".
/// Only the TensorFlow Lite implementation itself should include this
/// file directly.
#ifndef TENSORFLOW_LITE_CORE_TOOLS_VERIFIER_INTERNAL_H_
#define TENSORFLOW_LITE_CORE_TOOLS_VERIFIER_INTERNAL_H_

#include <stddef.h>

#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace internal {

// Verifies that the buffer is a valid TF Lite Model flatbuffer
// (without checking the consistency of the flatbuffer contents,
// just that it is a valid flatbuffer).
// Returns the FlatBuffer Model on success, or nullptr if the buffer does not
// contain a valid TF Lite Model flatbuffer.
const Model* VerifyFlatBufferAndGetModel(const void* buf, size_t len);

}  // namespace internal
}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_TOOLS_VERIFIER_INTERNAL_H_
