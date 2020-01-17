/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/utils/metadata_matcher.h"

#include "absl/strings/string_view.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::XPlane;

absl::flat_hash_map<int64, int> CreateEventMetadataMap(
    const XPlane& xplane,
    const std::vector<std::pair<const absl::Span<const absl::string_view>,
                                /*first_event_type*/ int>>&
        event_type_metadata_maps) {
  absl::flat_hash_map<int64, int> id_to_event_type_map;
  for (const auto& id_and_event_metadata : xplane.event_metadata()) {
    int64 id = id_and_event_metadata.first;
    absl::string_view event_name = id_and_event_metadata.second.name();
    for (const auto& event_type_metadata_map_and_first_event_type :
         event_type_metadata_maps) {
      auto event_type_metadata_map =
          event_type_metadata_map_and_first_event_type.first;
      int first_event_type =
          event_type_metadata_map_and_first_event_type.second;
      for (int i = 0; i < event_type_metadata_map.size(); ++i) {
        if (event_type_metadata_map[i] == event_name) {
          id_to_event_type_map[id] = first_event_type + i;
          break;
        }
      }
    }
  }
  return id_to_event_type_map;
}

}  // namespace

MetadataMatcher::MetadataMatcher(
    const XPlane& xplane,
    const std::vector<std::pair<const absl::Span<const absl::string_view>,
                                /*first_event_type*/ int>>&
        event_type_metadata_maps)
    : id_to_event_type_map_(
          CreateEventMetadataMap(xplane, event_type_metadata_maps)),
      event_type_to_id_map_(gtl::ReverseMap<decltype(event_type_to_id_map_)>(
          id_to_event_type_map_)) {}

}  // namespace profiler
}  // namespace tensorflow
