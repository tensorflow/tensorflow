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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_LITERT_MODEL_INIT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_LITERT_MODEL_INIT_H_

#include <memory>

#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_support.h"
#include "tensorflow/lite/experimental/litert/core/util/buffer_ref.h"

namespace litert::internal {

// Load model from flatbuffer file.
LiteRtStatus LoadModelFromFile(const char* path, LiteRtModel* model);

// Load model from flatbuffer memory.
LiteRtStatus LoadModel(const uint8_t* buf, size_t buf_size, LiteRtModel* model);

// Add a new custom code to the registry in this model. This will be associated
// with all custom ops and should only can be set once.
// TODO consider expanding this to allow for "custom op builder" hook.
LiteRtStatus RegisterCustomOpCode(LiteRtModel model, const char* new_op_code);

// Destroy model and any associated storage.
void DestroyModel(LiteRtModel model);

// Adds given metadata buffer to be serialized with the flatbuffer. Weights can
// be retrieved at runtime under `metadata_name`.
LiteRtStatus AppendMetadata(LiteRtModel model, const void* metadata,
                            size_t metadata_size, const char* metadata_name);

// Serializes model to bytes. NOTE this destroys the model before it returns.
// NOTE: Caller takes ownership of `buf`. Flatbuffers are packed into their
// arrays back to front, so the valid flatbuffer is buf[offset, size].
LiteRtStatus SerializeModel(LiteRtModel model, uint8_t** buf, size_t* size,
                            size_t* offset);

struct LiteRtModelDeleter {
  void operator()(LiteRtModel model) { DestroyModel(model); }
};

using UniqueLiteRtModel = std::unique_ptr<LiteRtModelT, LiteRtModelDeleter>;

LiteRtResult<OwningBufferRef<uint8_t>> SerializeModel(UniqueLiteRtModel model);

LiteRtResult<UniqueLiteRtModel> LoadModel(BufferRef<uint8_t> serialized);

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_LITERT_MODEL_INIT_H_
