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

#include <cstdint>
#include <memory>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"

namespace litert::internal {

Expected<std::unique_ptr<LiteRtModelT>> LoadModelFromFile(
    absl::string_view filename);

Expected<std::unique_ptr<LiteRtModelT>> LoadModelFromBuffer(
    BufferRef<uint8_t> buffer);

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_MODEL_LOAD_H_
