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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_MODEL_BUFFER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_MODEL_BUFFER_H_

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"

namespace litert::internal {

// Get a buffer that is the concatenation of given tflite file and one or more
// NPU byte code files. Adds metadata containing the offset/size of npu byte
// code. TFL custom ops are mapped to NPU byte code by their custom code, which
// must be non-null.
//
// NOTE: this is intended to be used for testing and tools and may be removed in
// the future.
Expected<OwningBufferRef<uint8_t>> GetModelBufWithByteCode(
    absl::string_view tfl_file,
    const absl::flat_hash_map<std::string, std::string>&
        custom_code_to_npu_file);

// Same as above, but with a map specifying NPU byte code buffers.
Expected<OwningBufferRef<uint8_t>> GetModelBufWithByteCode(
    LiteRtModelT&& model,
    const absl::flat_hash_map<std::string, OwningBufferRef<uint8_t>>&
        custom_code_to_npu_bytecode);

// Same as above, but only a single NPU byte code file is specified.
Expected<OwningBufferRef<uint8_t>> GetModelBufWithByteCode(
    absl::string_view tfl_file, absl::string_view npu_file);

// Same as above, but only a single NPU byte code buffer is specified.
Expected<OwningBufferRef<uint8_t>> GetModelBufWithByteCode(
    LiteRtModelT&& model, BufferRef<uint8_t> npu_byte_code);

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_MODEL_BUFFER_H_
