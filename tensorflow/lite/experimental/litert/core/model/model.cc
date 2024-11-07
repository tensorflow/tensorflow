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
#include "tensorflow/lite/experimental/litert/cc/litert_support.h"
#include "tensorflow/lite/experimental/litert/core/util/buffer_ref.h"
#include "tensorflow/lite/schema/schema_generated.h"

using litert::BufferRef;
using litert::MutableBufferRef;


LiteRtStatus LiteRtModelT::FindMetadataInd(absl::string_view key,
                                           uint32_t& ind) const {
  tflite::MetadataT* fb_metadata = nullptr;
  for (auto& m : flatbuffer_model->metadata) {
    if (m->name == key) {
      fb_metadata = m.get();
      break;
    }
  }
  if (fb_metadata == nullptr) {
    return kLiteRtStatusErrorNotFound;
  }

  ind = fb_metadata->buffer;
  return kLiteRtStatusOk;
}

LiteRtResult<MutableBufferRef<uint8_t>> LiteRtModelT::FindMetadata(
    const absl::string_view key) const {
  using ResT = MutableBufferRef<uint8_t>;

  uint32_t m_buffer_idx;
  LITERT_RETURN_RESULT_IF_NOT_OK(FindMetadataInd(key, m_buffer_idx), ResT);

  if (m_buffer_idx >= flatbuffer_model->buffers.size()) {
    return LiteRtResult<ResT>::FromStatus(kLiteRtStatusErrorIndexOOB);
  }
  tflite::BufferT* m_buffer = flatbuffer_model->buffers.at(m_buffer_idx).get();

  return LiteRtResult<ResT>::FromValue(
      MutableBufferRef(m_buffer->data.data(), m_buffer->data.size()));
}

LiteRtStatus LiteRtModelT::PushMetadata(absl::string_view key,
                                        BufferRef<uint8_t> data) {
  {
    uint32_t m_buffer_ind;
    if (FindMetadataInd(key, m_buffer_ind) == kLiteRtStatusOk) {
      return kLiteRtStatusErrorNotFound;
    }
  }

  auto& new_metadata = flatbuffer_model->metadata.emplace_back(
      std::make_unique<tflite::MetadataT>());
  new_metadata->name.assign(key.data(), key.size());

  const size_t new_m_buffer_ind = flatbuffer_model->buffers.size();
  new_metadata->buffer = new_m_buffer_ind;

  auto& new_buffer = flatbuffer_model->buffers.emplace_back(
      std::make_unique<tflite::BufferT>());
  new_buffer->data.assign(data.Data(), data.Data() + data.Size());

  return kLiteRtStatusOk;
}

// TODO model delete, register custom code
