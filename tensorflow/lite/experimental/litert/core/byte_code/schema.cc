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

#include "tensorflow/lite/experimental/litert/core/byte_code/schema.h"

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert::internal {
namespace {

static constexpr size_t kBackendIdMaxLen = 124;
static constexpr size_t kEntryPointNameMaxLen = 124;
static constexpr size_t kMetadataKeyMaxLen = 124;

struct ExecInfoSchema {
  char backend_id[kBackendIdMaxLen + 1] = {};
  char entrypoint_name[kEntryPointNameMaxLen + 1] = {};
  char metadata_key[kMetadataKeyMaxLen + 1] = {};
};

}  // namespace

Expected<OwningBufferRef<uint8_t>> SerializeExecInfo(ExecInfo exec_info) {
  if (exec_info.backend_id.size() >= kBackendIdMaxLen ||
      exec_info.entrypoint_name.size() >= kEntryPointNameMaxLen ||
      exec_info.metadata_key.size() >= kMetadataKeyMaxLen) {
    LITERT_LOG(LITERT_ERROR, "%s", "Exec info strings too large\n");
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  ExecInfoSchema exec_info_schema;
  exec_info.backend_id.copy(exec_info_schema.backend_id,
                            exec_info.backend_id.size());
  exec_info.entrypoint_name.copy(exec_info_schema.entrypoint_name,
                                 exec_info.entrypoint_name.size());
  exec_info.metadata_key.copy(exec_info_schema.metadata_key,
                              exec_info.metadata_key.size());
  return OwningBufferRef<uint8_t>(
      reinterpret_cast<const uint8_t*>(&exec_info_schema),
      sizeof(exec_info_schema));
}

Expected<ExecInfo> ExecInfoFromBuf(BufferRef<uint8_t> buf) {
  if (buf.Size() != sizeof(ExecInfoSchema)) {
    LITERT_LOG(LITERT_ERROR, "%s", "Exec info size mismatch\n");
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  const ExecInfoSchema* exec_info_schema =
      reinterpret_cast<const ExecInfoSchema*>(buf.Data());
  absl::string_view backend_id(exec_info_schema->backend_id);
  if (backend_id.size() > kBackendIdMaxLen) {
    LITERT_LOG(LITERT_ERROR, "%s", "Exec info backend id too large\n");
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  absl::string_view entrypoint_name(exec_info_schema->entrypoint_name);
  if (entrypoint_name.size() > kEntryPointNameMaxLen) {
    LITERT_LOG(LITERT_ERROR, "%s", "Exec info entrypoint name too large\n");
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  absl::string_view metadata_key(exec_info_schema->metadata_key);
  if (metadata_key.size() > kMetadataKeyMaxLen) {
    LITERT_LOG(LITERT_ERROR, "%s", "Exec info metadata key too large\n");
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  return ExecInfo{backend_id, entrypoint_name, metadata_key};
}

}  // namespace litert::internal
