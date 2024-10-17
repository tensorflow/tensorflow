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

#include "tensorflow/lite/experimental/lrt/core/litert_model_serialize.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/lrt/c/litert_common.h"
#include "tensorflow/lite/experimental/lrt/c/litert_model.h"
#include "tensorflow/lite/experimental/lrt/cc/litert_support.h"
#include "tensorflow/lite/experimental/lrt/core/litert_model_init.h"
#include "tensorflow/lite/experimental/lrt/core/util/buffer_ref.h"

//
// METADATA Strategy
//

LiteRtStatus LiteRtModelAddByteCodeMetadata(LiteRtModel model,
                                            const char* soc_manufacturer,
                                            const char* soc_model,
                                            const void* byte_code,
                                            size_t byte_code_size) {
  // Register custom code shared by all NPU dispatch ops.
  LITERT_RETURN_STATUS_IF_NOT_OK(
      RegisterCustomOpCode(model, kLiteRtDispatchOpCustomCode));

  // Add the build tag to the model.
  const std::string m_buffer =
      absl::StrFormat(kLiteRtBuildTagTpl, soc_manufacturer, soc_model,
                      kLiteRtMetadataSerializationStrategy);
  LITERT_RETURN_STATUS_IF_NOT_OK(AppendMetadata(
      model, m_buffer.data(), m_buffer.size(), kLiteRtBuildTagKey));

  // Add the raw byte code.
  LITERT_RETURN_STATUS_IF_NOT_OK(AppendMetadata(
      model, byte_code, byte_code_size, kLiteRtMetadataByteCodeKey));

  return kLiteRtStatusOk;
}

//
// APPEND Strategy
//

LiteRtStatus LiteRtModelPrepareForByteCodeAppend(LiteRtModel model,
                                                 const char* soc_manufacturer,
                                                 const char* soc_model) {
  // Register custom code shared by all NPU dispatch ops.
  LITERT_RETURN_STATUS_IF_NOT_OK(
      RegisterCustomOpCode(model, kLiteRtDispatchOpCustomCode));

  // Add the build tag to the model.
  const std::string m_buffer =
      absl::StrFormat(kLiteRtBuildTagTpl, soc_manufacturer, soc_model,
                      kLiteRtAppendSerializationStrategy);
  LITERT_RETURN_STATUS_IF_NOT_OK(AppendMetadata(
      model, m_buffer.data(), m_buffer.size(), kLiteRtBuildTagKey));

  // Add the byte code placeholder.
  const std::string placeholder = absl::StrCat(
      kLiteRtAppendedByteCodePrefix, kLiteRtAppendedByteCodePlaceholder);
  LITERT_RETURN_STATUS_IF_NOT_OK(AppendMetadata(model, placeholder.data(),
                                                placeholder.size(),
                                                kLiteRtMetadataByteCodeKey));

  return kLiteRtStatusOk;
}

namespace litert::internal {

namespace {

using ::litert::MutableBufferRef;

static constexpr absl::string_view kBreak = ",";
static constexpr absl::string_view kFiller = "*";

// Use string_views to normalize any null-terminated strings.
static absl::string_view kPref = kLiteRtAppendedByteCodePrefix;
static absl::string_view kPlaceholder = kLiteRtAppendedByteCodePlaceholder;

bool ParseNum(absl::string_view piece, size_t& out) {
  const auto num_start = piece.find_first_not_of(kFiller);
  if (num_start == absl::string_view::npos) {
    return false;
  }
  return absl::SimpleAtoi<size_t>(piece.substr(num_start), &out);
}

}  // namespace

LiteRtStatus FinishByteCodeAppend(MutableBufferRef<uint8_t> serialized_model,
                                  size_t byte_code_size) {
  auto placeholder_tag_start = serialized_model.StrView().rfind(kPref);

  if (placeholder_tag_start == absl::string_view::npos) {
    return kLiteRtStatusErrorNotFound;
  }
  placeholder_tag_start += kPref.size();

  const auto offset_str_start = placeholder_tag_start + 1;
  const auto offset_str_end =
      serialized_model.StrView().find_first_not_of(kFiller, offset_str_start);

  {
    const auto offset_str_size = offset_str_end - offset_str_start;
    const std::string offset_str =
        absl::StrFormat("%lu", serialized_model.Size());
    LITERT_ENSURE(
        offset_str.size() <= offset_str_size, kLiteRtStatusErrorNotFound,
        "Model is too large to be supported with this placeholder technique.");
    LITERT_ENSURE(serialized_model.WriteInto(
                      offset_str, offset_str_end - offset_str.size()),
                  kLiteRtStatusErrorNotFound,
                  "Failed to write byte code offset into raw model.");
  }

  {
    const auto byte_code_size_str_start = offset_str_end + 1;
    const auto byte_code_size_str_end =
        serialized_model.StrView().find_first_not_of(kFiller,
                                                     byte_code_size_str_start);
    const auto byte_code_size_str_size =
        byte_code_size_str_end - byte_code_size_str_start;
    const std::string byte_code_size_str =
        absl::StrFormat("%lu", byte_code_size);
    LITERT_ENSURE(byte_code_size_str.size() <= byte_code_size_str_size,
                  kLiteRtStatusErrorNotFound,
                  "Bytecode is too large to be supported with this placeholder "
                  "technique.");
    LITERT_ENSURE(serialized_model.WriteInto(
                      byte_code_size_str,
                      byte_code_size_str_end - byte_code_size_str.size()),
                  kLiteRtStatusErrorNotFound,
                  "Failed to write byte code size into raw model.");
  }

  return kLiteRtStatusOk;
}

LiteRtResult<std::pair<size_t, size_t>> ParseByteCodeOffsetFromMetadata(
    BufferRef<uint8_t> metadata_buffer) {
  using ResT = LiteRtResult<std::pair<size_t, size_t>>;

  if (!metadata_buffer.StrView().starts_with(kPref)) {
    return ResT::FromStatus(kLiteRtStatusErrorInvalidArgument);
  }

  auto view = metadata_buffer.StrView().substr(kPref.size());
  if (!view.starts_with("[") || !view.ends_with("]")) {
    return ResT::FromStatus(kLiteRtStatusErrorInvalidArgument);
  }
  view = view.substr(1, kPlaceholder.size() - 2);

  const auto break_offset = view.find_first_of(kBreak);
  if (break_offset == absl::string_view::npos) {
    return ResT::FromStatus(kLiteRtStatusErrorInvalidArgument);
  }

  size_t offset_num;
  {
    // Offset of byte code piece
    const auto piece_view = view.substr(0, break_offset);
    if (!ParseNum(piece_view, offset_num)) {
      return ResT::FromStatus(kLiteRtStatusErrorInvalidArgument);
    }
  }

  size_t size_num;
  {
    // Size of byte code piece
    const auto piece_view = view.substr(break_offset + 1);
    if (!ParseNum(piece_view, size_num)) {
      return ResT::FromStatus(kLiteRtStatusErrorInvalidArgument);
    }
  }

  return ResT::FromValue({offset_num, size_num});
}

}  // namespace litert::internal
