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

#include "tensorflow/lite/experimental/litert/core/byte_code_util.h"

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"

namespace litert::internal {

namespace {
// Simple metadata added to the flatbuffer related to compiler plugin.
struct BuildStamp {
  char soc_manufacturer[kSocManufacturerMaxLen + 1] = {};
  char soc_model[kSocModelMaxLen + 1] = {};
  Serialization serialization = kUnknown;
};

// Structure of serialized byte code placeholder.
struct ByteCodePlaceholder {
  char offset_str[kByteCodeOffsetStrMaxLen + 1] = {};
  char size_str[kByteCodeSizeStrMaxLen + 1] = {};
};

// Structure of serialized per-custom op data.
struct ExecInfo {
  char entrypoint_name[kEntryPointNameMaxLen + 1] = {};
  char metadata_key[kMetadataKeyMaxLen + 1] = {};
};

static constexpr size_t kByteCodePlaceholderBufSize =
    sizeof(ByteCodePlaceholder) + kByteCodePrefix.size();
}  // namespace

Expected<OwningBufferRef<uint8_t>> MakeBuildStamp(
    absl::string_view soc_manufacturer, absl::string_view soc_model,
    Serialization serialization) {
  if (soc_manufacturer.size() >= kSocManufacturerMaxLen ||
      soc_model.size() >= kSocModelMaxLen) {
    LITERT_LOG(LITERT_ERROR, "%s", "Soc Make/Model strings too large\n");
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  BuildStamp stamp;
  soc_manufacturer.copy(stamp.soc_manufacturer, soc_manufacturer.size());
  soc_model.copy(stamp.soc_model, soc_model.size());
  stamp.serialization = serialization;
  return OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(&stamp),
                                  sizeof(stamp));
}

// Parse a serialized build stamp from the given buf.
Expected<std::tuple<absl::string_view, absl::string_view, Serialization>>
ParseBuildStamp(BufferRef<uint8_t> buf) {
  if (buf.Size() != sizeof(BuildStamp)) {
    LITERT_LOG(LITERT_ERROR, "%s", "Build stamp size mismatch\n");
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  const BuildStamp* stamp = reinterpret_cast<const BuildStamp*>(buf.Data());
  return std::make_tuple(absl::string_view(stamp->soc_manufacturer),
                         absl::string_view(stamp->soc_model),
                         stamp->serialization);
}

OwningBufferRef<uint8_t> MakeByteCodePlaceholder() {
  OwningBufferRef<uint8_t> buf(kByteCodePlaceholderBufSize);
  buf.WriteInto(kByteCodePrefix);
  ByteCodePlaceholder* placeholder = reinterpret_cast<ByteCodePlaceholder*>(
      buf.Data() + kByteCodePrefix.size());
  *placeholder = ByteCodePlaceholder();
  return buf;
}

Expected<std::pair<size_t, size_t>> ParseByteCodePlaceholder(
    BufferRef<uint8_t> buf) {
  if (buf.Size() != kByteCodePlaceholderBufSize ||
      buf.StrView().compare(0, kByteCodePrefix.size(), kByteCodePrefix) != 0) {
    LITERT_LOG(LITERT_ERROR, "%s", "Byte code placeholder size mismatch\n");
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }

  const ByteCodePlaceholder* placeholder =
      reinterpret_cast<const ByteCodePlaceholder*>(buf.Data() +
                                                   kByteCodePrefix.size());
  const absl::string_view offset_str(placeholder->offset_str);
  const absl::string_view size_str(placeholder->size_str);

  size_t offset, size;
  if (!absl::SimpleAtoi(offset_str, &offset) ||
      !absl::SimpleAtoi(size_str, &size)) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "Byte code placeholder offset/size invalid\n");
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }

  return std::make_pair(offset, size);
}

LiteRtStatus FinishByteCodePlaceholders(
    MutableBufferRef<uint8_t> seralized_model, size_t byte_code_size) {
  const size_t placeholder_start =
      seralized_model.StrView().rfind(kByteCodePrefix);
  LITERT_ENSURE(placeholder_start != absl::string_view::npos,
                kLiteRtStatusErrorInvalidArgument,
                "Cannot find any bytecode placeholders in the model");

  ByteCodePlaceholder* placeholder = reinterpret_cast<ByteCodePlaceholder*>(
      seralized_model.Data() + kByteCodePrefix.size() + placeholder_start);

  const int offset_written =
      absl::SNPrintF(placeholder->offset_str, kByteCodeOffsetStrMaxLen, "%lu",
                     seralized_model.Size());
  LITERT_ENSURE(
      offset_written > -1 && offset_written <= kByteCodeOffsetStrMaxLen,
      kLiteRtStatusErrorInvalidArgument, "Offset too large");

  const int size_written = absl::SNPrintF(
      placeholder->size_str, kByteCodeSizeStrMaxLen, "%lu", byte_code_size);
  LITERT_ENSURE(size_written > -1 && size_written <= kByteCodeSizeStrMaxLen,
                kLiteRtStatusErrorInvalidArgument, "Size too large");
  return kLiteRtStatusOk;
}

Expected<std::pair<absl::string_view, absl::string_view>> ParseExecInfo(
    BufferRef<uint8_t> buf) {
  if (buf.Size() != sizeof(ExecInfo)) {
    LITERT_LOG(LITERT_ERROR, "%s", "Exec info size mismatch\n");
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  const ExecInfo* exec_info = reinterpret_cast<const ExecInfo*>(buf.Data());
  return std::make_pair(absl::string_view(exec_info->entrypoint_name),
                        absl::string_view(exec_info->metadata_key));
}

Expected<OwningBufferRef<uint8_t>> MakeExecInfo(
    absl::string_view entrypoint_name, absl::string_view metadata_key) {
  if (entrypoint_name.size() >= kEntryPointNameMaxLen ||
      metadata_key.size() >= kMetadataKeyMaxLen) {
    LITERT_LOG(LITERT_ERROR, "%s", "Exec info strings too large\n");
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  ExecInfo exec_info;
  entrypoint_name.copy(exec_info.entrypoint_name, entrypoint_name.size());
  metadata_key.copy(exec_info.metadata_key, metadata_key.size());
  return OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(&exec_info),
                                  sizeof(exec_info));
}

}  // namespace litert::internal
