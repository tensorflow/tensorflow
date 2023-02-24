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
#include "tensorflow/tsl/profiler/utils/xplane_builder.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/tsl/platform/types.h"
#include "tensorflow/tsl/profiler/protobuf/xplane.pb.h"
#include "tensorflow/tsl/profiler/utils/math_utils.h"

namespace tsl {
namespace profiler {

XPlaneBuilder::XPlaneBuilder(XPlane* plane)
    : XStatsBuilder<XPlane>(plane, this), plane_(plane) {
  for (auto& id_and_metadata : *plane->mutable_event_metadata()) {
    auto& metadata = id_and_metadata.second;
    last_event_metadata_id_ =
        std::max<int64_t>(last_event_metadata_id_, metadata.id());
    if (!metadata.name().empty()) {
      event_metadata_by_name_.try_emplace(metadata.name(), &metadata);
    }
  }
  for (auto& id_and_metadata : *plane->mutable_stat_metadata()) {
    auto& metadata = id_and_metadata.second;
    last_stat_metadata_id_ =
        std::max<int64_t>(last_stat_metadata_id_, metadata.id());
    if (!metadata.name().empty()) {
      stat_metadata_by_name_.try_emplace(metadata.name(), &metadata);
    }
  }
  for (XLine& line : *plane->mutable_lines()) {
    lines_by_id_.try_emplace(line.id(), &line);
  }
}

XEventMetadata* XPlaneBuilder::GetOrCreateEventMetadata(int64_t metadata_id) {
  XEventMetadata& metadata = (*plane_->mutable_event_metadata())[metadata_id];
  metadata.set_id(metadata_id);
  return &metadata;
}

XEventMetadata* XPlaneBuilder::CreateEventMetadata() {
  return GetOrCreateEventMetadata(++last_event_metadata_id_);
}

XEventMetadata* XPlaneBuilder::GetOrCreateEventMetadata(
    absl::string_view name) {
  XEventMetadata*& metadata = event_metadata_by_name_[name];
  if (metadata == nullptr) {
    metadata = CreateEventMetadata();
    metadata->set_name(std::string(name));
  }
  return metadata;
}

XEventMetadata* XPlaneBuilder::GetOrCreateEventMetadata(std::string&& name) {
  XEventMetadata*& metadata = event_metadata_by_name_[name];
  if (metadata == nullptr) {
    metadata = CreateEventMetadata();
    metadata->set_name(std::move(name));
  }
  return metadata;
}

std::vector<XEventMetadata*> XPlaneBuilder::GetOrCreateEventsMetadata(
    const std::vector<absl::string_view>& names) {
  std::vector<XEventMetadata*> metadata;
  metadata.reserve(names.size());
  for (absl::string_view name : names) {
    metadata.push_back(GetOrCreateEventMetadata(name));
  }
  return metadata;
}

XEventMetadata* XPlaneBuilder::GetEventMetadata(absl::string_view name) const {
  auto result = event_metadata_by_name_.find(name);
  if (result == event_metadata_by_name_.end()) return nullptr;
  return result->second;
}

XStatMetadata* XPlaneBuilder::GetStatMetadata(absl::string_view name) const {
  auto result = stat_metadata_by_name_.find(name);
  if (result == stat_metadata_by_name_.end()) return nullptr;
  return result->second;
}

XStatMetadata* XPlaneBuilder::GetOrCreateStatMetadata(int64_t metadata_id) {
  XStatMetadata& metadata = (*plane_->mutable_stat_metadata())[metadata_id];
  metadata.set_id(metadata_id);
  return &metadata;
}

const XStatMetadata* XPlaneBuilder::GetStatMetadata(int64_t metadata_id) const {
  auto result = plane_->stat_metadata().find(metadata_id);
  if (result == plane_->stat_metadata().end()) return nullptr;
  return &(result->second);
}

XStatMetadata* XPlaneBuilder::CreateStatMetadata() {
  return GetOrCreateStatMetadata(++last_stat_metadata_id_);
}

XStatMetadata* XPlaneBuilder::GetOrCreateStatMetadata(absl::string_view name) {
  XStatMetadata*& metadata = stat_metadata_by_name_[name];
  if (metadata == nullptr) {
    metadata = CreateStatMetadata();
    metadata->set_name(std::string(name));
  }
  return metadata;
}

XStatMetadata* XPlaneBuilder::GetOrCreateStatMetadata(std::string&& name) {
  XStatMetadata*& metadata = stat_metadata_by_name_[name];
  if (metadata == nullptr) {
    metadata = CreateStatMetadata();
    metadata->set_name(std::move(name));
  }
  return metadata;
}

XLineBuilder XPlaneBuilder::GetOrCreateLine(int64_t line_id) {
  XLine*& line = lines_by_id_[line_id];
  if (line == nullptr) {
    line = plane_->add_lines();
    line->set_id(line_id);
  }
  return XLineBuilder(line, this);
}

XEventBuilder XLineBuilder::AddEvent(const XEventMetadata& metadata) {
  XEvent* event = line_->add_events();
  event->set_metadata_id(metadata.id());
  return XEventBuilder(line_, plane_, event);
}

XEventBuilder XLineBuilder::AddEvent(const XEvent& event) {
  XEvent* new_event = line_->add_events();
  *new_event = event;
  return XEventBuilder(line_, plane_, new_event);
}

void XLineBuilder::SetTimestampNsAndAdjustEventOffsets(int64_t timestamp_ns) {
  int64_t offset_ps = NanoToPico(line_->timestamp_ns() - timestamp_ns);
  line_->set_timestamp_ns(timestamp_ns);
  if (offset_ps) {
    for (auto& event : *line_->mutable_events()) {
      event.set_offset_ps(event.offset_ps() + offset_ps);
    }
  }
}

}  // namespace profiler
}  // namespace tsl
