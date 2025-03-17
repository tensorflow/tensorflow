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

#include "tensorflow/lite/experimental/litert/core/build_stamp.h"

#include <cstdint>
#include <tuple>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert::internal {

namespace {
// Simple metadata added to the flatbuffer related to compiler plugin.
struct BuildStamp {
  char soc_manufacturer[kSocManufacturerMaxLen + 1] = {};
  char soc_model[kSocModelMaxLen + 1] = {};
};

}  // namespace

Expected<OwningBufferRef<uint8_t>> MakeBuildStamp(
    absl::string_view soc_manufacturer, absl::string_view soc_model) {
  if (soc_manufacturer.size() >= kSocManufacturerMaxLen ||
      soc_model.size() >= kSocModelMaxLen) {
    LITERT_LOG(LITERT_ERROR, "%s", "Soc Make/Model strings too large\n");
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  BuildStamp stamp;
  soc_manufacturer.copy(stamp.soc_manufacturer, soc_manufacturer.size());
  soc_model.copy(stamp.soc_model, soc_model.size());
  return OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(&stamp),
                                  sizeof(stamp));
}

// Parse a serialized build stamp from the given buf.
Expected<std::tuple<absl::string_view, absl::string_view>> ParseBuildStamp(
    BufferRef<uint8_t> buf) {
  if (buf.Size() != sizeof(BuildStamp)) {
    LITERT_LOG(LITERT_ERROR, "%s", "Build stamp size mismatch\n");
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  const BuildStamp* stamp = reinterpret_cast<const BuildStamp*>(buf.Data());
  return std::make_tuple(absl::string_view(stamp->soc_manufacturer),
                         absl::string_view(stamp->soc_model));
}

}  // namespace litert::internal
