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

#include "tensorflow/lite/experimental/litert/core/model/model.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/schema/schema_generated.h"

using ::litert::BufferRef;
using ::litert::Expected;
using ::litert::Unexpected;

Expected<BufferRef<uint8_t>> LiteRtModelT::FindMetadata(
    const absl::string_view key) const {
  return ::litert::internal::GetMetadata(key, *flatbuffer_model);
}

LiteRtStatus LiteRtModelT::PushMetadata(absl::string_view key,
                                        BufferRef<uint8_t> data) {
  return ::litert::internal::PushMetadata(key, *flatbuffer_model, data);
}

litert::Expected<LiteRtSignatureT*> LiteRtModelT::FindSignature(
    absl::string_view signature_key) const {
  for (auto& signature : signatures) {
    if (signature->key == signature_key) {
      return signature.get();
    }
  }
  return Unexpected(kLiteRtStatusErrorNotFound, "Signature not found");
}

litert::Expected<const LiteRtSubgraphT*> LiteRtModelT::FindSubgraph(
    absl::string_view signature_key) const {
  for (auto& signature : signatures) {
    if (signature->key == signature_key) {
      return &(subgraphs[signature->subgraph_index]);
    }
  }
  return Unexpected(kLiteRtStatusErrorNotFound, "Signature not found");
}
