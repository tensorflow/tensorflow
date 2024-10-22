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

#include "tensorflow/lite/core/tools/verifier_internal.h"

#include <stddef.h>
#include <stdint.h>

#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace internal {

// Verifies flatbuffer format of the model contents and returns the in-memory
// model.
const Model* VerifyFlatBufferAndGetModel(const void* buf, size_t len) {
  ::flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buf), len);
  if (VerifyModelBuffer(verifier)) {
    return ::tflite::GetModel(buf);
  } else {
    return nullptr;
  }
}

}  // namespace internal
}  // namespace tflite
