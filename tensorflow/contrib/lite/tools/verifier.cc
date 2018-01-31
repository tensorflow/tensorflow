/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/lite/tools/verifier.h"
#include "tensorflow/contrib/lite/schema/schema_generated.h"
#include "tensorflow/contrib/lite/version.h"

namespace tflite {

namespace {

const Model* VerifyFlatbufferAndGetModel(const void* buf, size_t len) {
  ::flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buf), len);
  if (VerifyModelBuffer(verifier)) {
    return ::tflite::GetModel(buf);
  } else {
    return nullptr;
  }
}

}  // namespace

bool Verify(const void* buf, size_t len) {
  const Model* model = VerifyFlatbufferAndGetModel(buf, len);
  if (model == nullptr) {
    return false;
  }

  return model->version() == TFLITE_SCHEMA_VERSION;
}
}  // namespace tflite
