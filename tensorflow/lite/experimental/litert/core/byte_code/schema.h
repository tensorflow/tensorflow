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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_BYTE_CODE_SCHEMA_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_BYTE_CODE_SCHEMA_H_

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert::internal {

// TODO consider using a flatbuffer schema for this.

// Parsed representation of the custom options of an op that uses byte code
// assets.
struct ExecInfo {
  // Backend id of the dispach op.
  absl::string_view backend_id;

  // Entry point name into the byte code asset.
  absl::string_view entrypoint_name;

  // Metadata key of the byte code asset.
  absl::string_view metadata_key;
};

// Parse the custom options of an op that uses byte code assets.
Expected<ExecInfo> ExecInfoFromBuf(BufferRef<uint8_t> buf);

// Serialize the given exec info into a buffer.
Expected<OwningBufferRef<uint8_t>> SerializeExecInfo(ExecInfo exec_info);

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_BYTE_CODE_SCHEMA_H_
