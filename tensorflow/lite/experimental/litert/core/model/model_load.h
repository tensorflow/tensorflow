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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_MODEL_LOAD_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_MODEL_LOAD_H_

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Load model from flatbuffer file.
LiteRtStatus LiteRtLoadModelFromFile(const char* path, LiteRtModel* model);

// Load model from flatbuffer memory.
LiteRtStatus LiteRtLoadModelFromMemory(const uint8_t* buf, size_t buf_size,
                                       LiteRtModel* model);

#ifdef __cplusplus
}

#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"

namespace litert::internal {

Expected<Model> LoadModelFromFile(absl::string_view path);

Expected<Model> LoadModelFromMemory(BufferRef<uint8_t> serialized);

}  // namespace litert::internal

#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_MODEL_LOAD_H_
