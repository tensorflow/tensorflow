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
#include "tensorflow/core/profiler/utils/xplane_builder.h"

#include "absl/strings/numbers.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"

namespace tensorflow {
namespace profiler {

XPlaneBuilder::XPlaneBuilder(XPlane* plane) : plane_(plane) {
  for (auto& iter : *plane->mutable_event_metadata()) {
    last_event_metadata_id_ =
        std::max<int64>(last_event_metadata_id_, iter.second.id());
    event_metadata_by_name_.try_emplace(iter.second.name(), &iter.second);
  }
  for (auto& iter : *plane->mutable_stat_metadata()) {
    last_stat_metadata_id_ =
        std::max<int64>(last_stat_metadata_id_, iter.second.id());
    stat_metadata_by_name_.try_emplace(iter.second.name(), &iter.second);
  }
  for (XLine& line : *plane->mutable_lines()) {
    lines_by_id_.try_emplace(line.id(), &line);
  }
}

XEventMetadata* XPlaneBuilder::GetOrCreateEventMetadata(int64 metadata_id) {
  XEventMetadata& metadata = (*plane_->mutable_event_metadata())[metadata_id];
  metadata.set_id(metadata_id);
  return &metadata;
}

// Returns XEventMetadata for the given event name.
XEventMetadata* XPlaneBuilder::GetOrCreateEventMetadata(
    absl::string_view name) {
  XEventMetadata*& metadata = event_metadata_by_name_[name];
  if (metadata == nullptr) {
    metadata =
        XPlaneBuilder::GetOrCreateEventMetadata(++last_event_metadata_id_);
    metadata->set_name(std::string(name));
    std::string event_name = TfOpEventName(name);
    if (event_name != name) {
      metadata->set_display_name(std::move(event_name));
    }
  }
  return metadata;
}

XStatMetadata* XPlaneBuilder::GetOrCreateStatMetadata(int64 metadata_id) {
  XStatMetadata& metadata = (*plane_->mutable_stat_metadata())[metadata_id];
  metadata.set_id(metadata_id);
  return &metadata;
}

// Returns XStatMetadata for the given stat name.
XStatMetadata* XPlaneBuilder::GetOrCreateStatMetadata(absl::string_view name) {
  XStatMetadata*& metadata = stat_metadata_by_name_[name];
  if (metadata == nullptr) {
    metadata = XPlaneBuilder::GetOrCreateStatMetadata(++last_stat_metadata_id_);
    metadata->set_name(std::string(name));
  }
  return metadata;
}

XLine* XPlaneBuilder::AddLine(int64 line_id) {
  XLine*& line = lines_by_id_[line_id];
  if (line == nullptr) {
    line = RawPlane()->add_lines();
    line->set_id(line_id);
  }
  return line;
}

// Returns a builder for the line with the given id. Creates a new line if the
// id was unused, otherwise the builder will add events to an existing line.
XLineBuilder XPlaneBuilder::GetOrCreateLine(int64 line_id) {
  return XLineBuilder(AddLine(line_id));
}

XEventBuilder XLineBuilder::AddEvent(const XEventMetadata& metadata) {
  XEvent* event = line_->add_events();
  event->set_metadata_id(metadata.id());
  return XEventBuilder(line_, event);
}

XStat* XEventBuilder::AddStat(const XStatMetadata& metadata) {
  XStat* stat = event_->add_stats();
  stat->set_metadata_id(metadata.id());
  return stat;
}

void XEventBuilder::ParseAndAddStatValue(const XStatMetadata& metadata,
                                         absl::string_view value) {
  int64 int_value;
  uint64 uint_value;
  double double_value;
  if (absl::SimpleAtoi(value, &int_value)) {
    AddStatValue(metadata, int_value);
  } else if (absl::SimpleAtoi(value, &uint_value)) {
    AddStatValue(metadata, uint_value);
  } else if (absl::SimpleAtod(value, &double_value)) {
    AddStatValue(metadata, double_value);
  } else {
    AddStatValue(metadata, value);
  }
}

}  // namespace profiler
}  // namespace tensorflow
