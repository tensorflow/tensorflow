/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/tsl/platform/cloud/compute_engine_zone_provider.h"

#include <utility>

#include "tsl/platform/str_util.h"
namespace tsl {

namespace {
constexpr char kGceMetadataZonePath[] = "instance/zone";
}  // namespace

ComputeEngineZoneProvider::ComputeEngineZoneProvider(
    std::shared_ptr<ComputeEngineMetadataClient> google_metadata_client)
    : google_metadata_client_(std::move(google_metadata_client)) {}

absl::Status ComputeEngineZoneProvider::GetZone(string* zone) {
  if (!cached_zone.empty()) {
    *zone = cached_zone;
    return absl::OkStatus();
  }
  std::vector<char> response_buffer;
  TF_RETURN_IF_ERROR(google_metadata_client_->GetMetadata(kGceMetadataZonePath,
                                                          &response_buffer));
  absl::string_view location(&response_buffer[0], response_buffer.size());

  std::vector<string> elems = str_util::Split(location, "/");
  if (elems.size() == 4) {
    cached_zone = elems.back();
    *zone = cached_zone;
  } else {
    LOG(ERROR) << "Failed to parse the zone name from location: "
               << string(location);
  }

  return absl::OkStatus();
}
ComputeEngineZoneProvider::~ComputeEngineZoneProvider() {}

}  // namespace tsl
