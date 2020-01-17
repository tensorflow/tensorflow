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

using ::tensorflow::profiler::XEvent;
using ::tensorflow::profiler::XPlane;
using ::tensorflow::profiler::XStat;

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

absl::flat_hash_map<int64, int> CreateStatMetadataMap(
    const XPlane& xplane,
    const absl::Span<const absl::string_view> stat_type_str_map) {
  absl::flat_hash_map<int64, int> id_to_stat_type_map;
  for (const auto& id_and_stat_metadata : xplane.stat_metadata()) {
    int64 id = id_and_stat_metadata.first;
    absl::string_view stat_name = id_and_stat_metadata.second.name();
    for (int stat_type = 0; stat_type < stat_type_str_map.size(); ++stat_type) {
      if (stat_type_str_map[stat_type] == stat_name) {
        id_to_stat_type_map[id] = stat_type;
        break;
      }
    }
  }
  return id_to_stat_type_map;
}

}  // namespace

MetadataMatcher::MetadataMatcher(
    const XPlane& xplane,
    const std::vector<std::pair<const absl::Span<const absl::string_view>,
                                /*first_event_type*/ int>>&
        event_type_metadata_maps,
    const absl::Span<const absl::string_view> stat_type_str_map)
    : id_to_event_type_map_(
          CreateEventMetadataMap(xplane, event_type_metadata_maps)),
      id_to_stat_type_map_(CreateStatMetadataMap(xplane, stat_type_str_map)),
      event_type_to_id_map_(gtl::ReverseMap<decltype(event_type_to_id_map_)>(
          id_to_event_type_map_)),
      stat_type_to_id_map_(gtl::ReverseMap<decltype(stat_type_to_id_map_)>(
          id_to_stat_type_map_)) {}

const XStat* MetadataMatcher::GetStat(const XEvent& event,
                                      int stat_type) const {
  for (const auto& stat : event.stats()) {
    if (GetStatType(stat) == stat_type) {
      return &stat;
    }
  }
  return nullptr;
}

absl::optional<std::tuple<const XStat*, const XStat*>>
MetadataMatcher::GetStats(const XEvent& event, int first_stat_type,
                          int second_stat_type) const {
  const XStat* first_stat = nullptr;
  const XStat* second_stat = nullptr;
  for (const auto& stat : event.stats()) {
    if (GetStatType(stat) == first_stat_type) {
      first_stat = &stat;
    } else if (GetStatType(stat) == second_stat_type) {
      second_stat = &stat;
    }
  }
  if (first_stat && second_stat) {
    return std::make_tuple(first_stat, second_stat);
  }
  return absl::nullopt;
}

absl::optional<std::tuple<const XStat*, const XStat*, const XStat*>>
MetadataMatcher::GetStats(const XEvent& event, int first_stat_type,
                          int second_stat_type, int third_stat_type) const {
  const XStat* first_stat = nullptr;
  const XStat* second_stat = nullptr;
  const XStat* third_stat = nullptr;
  for (const auto& stat : event.stats()) {
    if (GetStatType(stat) == first_stat_type) {
      first_stat = &stat;
    } else if (GetStatType(stat) == second_stat_type) {
      second_stat = &stat;
    } else if (GetStatType(stat) == third_stat_type) {
      third_stat = &stat;
    }
  }
  if (first_stat && second_stat && third_stat) {
    return std::make_tuple(first_stat, second_stat, third_stat);
  }
  return absl::nullopt;
}

absl::optional<int64> MetadataMatcher::GetIntStatValue(const XEvent& event,
                                                       int stat_type) const {
  if (const XStat* stat = GetStat(event, stat_type)) {
    return stat->int64_value();
  }
  return absl::nullopt;
}

}  // namespace profiler
}  // namespace tensorflow
