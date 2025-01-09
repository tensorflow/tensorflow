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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_BYTE_CODE_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_BYTE_CODE_UTIL_H_

#include <stddef.h>

#include <cstdint>
#include <tuple>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"

namespace litert::internal {

// Shared "custom_code" for all dispatch ops.
static constexpr absl::string_view kLiteRtDispatchOpCustomCode = "DISPATCH_OP";

//
// Build Stamp
//

// Maximum size of string for soc_manufacturer.
static constexpr size_t kSocManufacturerMaxLen = 124;

// Maximum size of string for soc_model.
static constexpr size_t kSocModelMaxLen = 124;

// The method used for packing byte code with flatbuffer.
enum Serialization : uint8_t {
  kUnknown = 0,
  // Byte code is appended to back of .tflite.
  kAppend = 1,
  // Byte code is stored in a metadata buffer [FOR TESTING ONLY].
  kMetadata = 2
};

// Metadata key to lookup the build stamp.
static constexpr absl::string_view kLiteRtBuildStampKey = "LiteRtStamp";

// Make a serialized build stamp that can go directly in the flatbuffer.
Expected<OwningBufferRef<uint8_t>> MakeBuildStamp(
    absl::string_view soc_manufacturer, absl::string_view soc_model,
    Serialization serialization);

// Parse a serialized build stamp from the given buf.
Expected<std::tuple<absl::string_view, absl::string_view, Serialization>>
ParseBuildStamp(BufferRef<uint8_t> buf);

//
// METADATA
//

// Metadata key for looking up byte code that is directly packed.
static constexpr absl::string_view kByteCodeMetadataKey = "NPU_BYTE_CODE";

//
// APPEND: Placeholder for bytecode offset and size.
//

// Maximum number of digits the byte code size can be base 10.
static constexpr size_t kByteCodeSizeStrMaxLen = 10;

// Maximum number of digits the byte code offset can be base 10.
static constexpr size_t kByteCodeOffsetStrMaxLen = 10;

// Prefix before serialized [offset, size, function name].
static constexpr absl::string_view kByteCodePrefix = "<npu_byte_code>";

// Get a new serialized byte code placeholder buffer with prefix.
OwningBufferRef<uint8_t> MakeByteCodePlaceholder();

// Parse byte code offset and size serialized as a ByteCodePlaceholder in buf.
Expected<std::pair<size_t, size_t>> ParseByteCodePlaceholder(
    BufferRef<uint8_t> buf);

// Replace all byte code placeholders with actual values. This happens directly
// on a serialized model without changing its size.
LiteRtStatus FinishByteCodePlaceholders(
    MutableBufferRef<uint8_t> seralized_model, size_t byte_code_size);

//
// APPEND: ExecInfo for per-custom op info.
//

// Maximum length of string for the entry point name.
static constexpr size_t kEntryPointNameMaxLen = 124;

// Maximum length of a metadata key stored per custom op.
static constexpr size_t kMetadataKeyMaxLen = 124;

// Make a serialized exec info from the given values.
Expected<OwningBufferRef<uint8_t>> MakeExecInfo(
    absl::string_view entrypoint_name, absl::string_view metadata_key);

// Parse serialized exec info from buffer.
Expected<std::pair<absl::string_view, absl::string_view>> ParseExecInfo(
    BufferRef<uint8_t> buf);

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_BYTE_CODE_UTIL_H_
