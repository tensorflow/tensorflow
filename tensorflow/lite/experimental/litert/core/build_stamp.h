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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_BUILD_STAMP_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_BUILD_STAMP_H_

#include <stddef.h>

#include <cstdint>
#include <tuple>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert::internal {

// TODO update this library to use the flexbuffers api.

// Shared "custom_code" for all dispatch ops.
static constexpr absl::string_view kLiteRtDispatchOpCustomCode = "DISPATCH_OP";

//
// Build Stamp
//

// Maximum size of string for soc_manufacturer.
static constexpr size_t kSocManufacturerMaxLen = 124;

// Maximum size of string for soc_model.
static constexpr size_t kSocModelMaxLen = 124;

// Metadata key to lookup the build stamp.
static constexpr absl::string_view kLiteRtBuildStampKey = "LiteRtStamp";

// Make a serialized build stamp that can go directly in the flatbuffer.
Expected<OwningBufferRef<uint8_t>> MakeBuildStamp(
    absl::string_view soc_manufacturer, absl::string_view soc_model);

// Parse a serialized build stamp from the given buf.
Expected<std::tuple<absl::string_view, absl::string_view>> ParseBuildStamp(
    BufferRef<uint8_t> buf);

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_BUILD_STAMP_H_
