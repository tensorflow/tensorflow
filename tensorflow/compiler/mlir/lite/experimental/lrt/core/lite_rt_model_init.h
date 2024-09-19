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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_CORE_LITE_RT_MODEL_INIT_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_CORE_LITE_RT_MODEL_INIT_H_

#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_model.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Load model from flatbuffer file.
LrtStatus LoadModelFromFile(const char* path, LrtModel* model);

// Load model from flatbuffer memory.
LrtStatus LoadModel(const uint8_t* buf, size_t buf_size, LrtModel* model);

// Add a new custom code to the registry in this model. This will be associated
// with all custom ops and should only can be set once.
// TODO consider expanding this to allow for "custom op builder" hook.
LrtStatus RegisterCustomOpCode(LrtModel model, const char* new_op_code);

// Destroy model and any associated storage.
void ModelDestroy(LrtModel model);

// Adds given metadata buffer to be serialized with the flatbuffer. Buffer can
// be retrieved at runtime under `metadata_name`.
LrtStatus AppendMetadata(LrtModel model, const void* metadata,
                         size_t metadata_size, const char* metadata_name);

// Serializes model to bytes. NOTE this destroys the model before it returns.
// NOTE: Caller takes ownership of `buf`. Flatbuffers are packed into their
// arrays back to front, so the valid flatbuffer is buf[offset, size].
LrtStatus SerializeModel(LrtModel model, uint8_t** buf, size_t* size,
                         size_t* offset);

#ifdef __cplusplus
}

#include <memory>

struct LrtModelDeleter {
  void operator()(LrtModel model) {
    if (model != nullptr) {
      ModelDestroy(model);
    }
  }
};

using UniqueLrtModel = std::unique_ptr<LrtModelT, LrtModelDeleter>;

#endif  // __cplusplus

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_CORE_LITE_RT_MODEL_INIT_H_
