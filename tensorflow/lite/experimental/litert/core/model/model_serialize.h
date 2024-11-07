// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_MODEL_SERIALIZE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_MODEL_SERIALIZE_H_

#include "tensorflow/lite/experimental/litert/c/litert_model.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Serializes model to bytes. NOTE this destroys the model before it returns.
// NOTE: Caller takes ownership of `buf`. Flatbuffers are packed into their
// arrays back to front, so the valid flatbuffer is buf[offset, size].
LiteRtStatus LiteRtSerializeModel(LiteRtModel model, uint8_t** buf,
                                  size_t* size, size_t* offset);

#ifdef __cplusplus
}

#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_support.h"
#include "tensorflow/lite/experimental/litert/core/util/buffer_ref.h"

namespace litert::internal {

LiteRtResult<litert::OwningBufferRef<uint8_t>> SerializeModel(Model&& model);

}  // namespace litert::internal

#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_MODEL_SERIALIZE_H_
