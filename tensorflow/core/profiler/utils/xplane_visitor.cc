/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

XStatVisitor::XStatVisitor(const XPlaneVisitor* plane, const XStat* stat)
    : stat_(stat),
      metadata_(plane->GetStatMetadata(stat->metadata_id())),
      plane_(plane),
      type_(plane->GetStatType(stat->metadata_id())) {}

std::string XStatVisitor::ToString() const {
  switch (stat_->value_case()) {
    case XStat::kInt64Value:
      return absl::StrCat(stat_->int64_value());
    case XStat::kUint64Value:
      return absl::StrCat(stat_->uint64_value());
    case XStat::kDoubleValue:
      return absl::StrCat(stat_->double_value());
    case XStat::kStrValue:
      return stat_->str_value();
    case XStat::kBytesValue:
      return "<opaque bytes>";
    case XStat::kRefValue:
      return plane_->GetStatMetadata(stat_->ref_value())->name();
    case XStat::VALUE_NOT_SET:
      return "";
  }
}

absl::string_view XStatVisitor::StrOrRefValue() const {
  switch (stat_->value_case()) {
    case XStat::kStrValue:
      return stat_->str_value();
    case XStat::kRefValue:
      return plane_->GetStatMetadata(stat_->ref_value())->name();
    case XStat::kInt64Value:
    case XStat::kUint64Value:
    case XStat::kDoubleValue:
    case XStat::kBytesValue:
    case XStat::VALUE_NOT_SET:
      return absl::string_view();
  }
}

XEventVisitor::XEventVisitor(const XPlaneVisitor* plane, const XLine* line,
                             const XEvent* event)
    : XStatsOwner<XEvent>(plane, event),
      plane_(plane),
      line_(line),
      event_(event),
      metadata_(plane->GetEventMetadata(event_->metadata_id())),
      type_(plane->GetEventType(event_->metadata_id())) {}

XPlaneVisitor::XPlaneVisitor(const XPlane* plane,
                             const TypeGetterList& event_type_getter_list,
                             const TypeGetterList& stat_type_getter_list)
    : XStatsOwner<XPlane>(this, plane), plane_(plane) {
  for (const auto& event_type_getter : event_type_getter_list) {
    BuildEventTypeMap(plane, event_type_getter);
  }
  for (const auto& stat_type_getter : stat_type_getter_list) {
    BuildStatTypeMap(plane, stat_type_getter);
  }
}

void XPlaneVisitor::BuildEventTypeMap(const XPlane* plane,
                                      const TypeGetter& event_type_getter) {
  for (const auto& event_metadata : plane->event_metadata()) {
    uint64 metadata_id = event_metadata.first;
    const auto& metadata = event_metadata.second;
    absl::optional<int64> event_type = event_type_getter(metadata.name());
    if (event_type.has_value()) {
      auto result = event_metadata_id_map_.emplace(metadata_id, *event_type);
      DCHECK(result.second);  // inserted
      event_type_map_.emplace(*event_type, &metadata);
    }
  }
}

void XPlaneVisitor::BuildStatTypeMap(const XPlane* plane,
                                     const TypeGetter& stat_type_getter) {
  for (const auto& stat_metadata : plane->stat_metadata()) {
    uint64 metadata_id = stat_metadata.first;
    const auto& metadata = stat_metadata.second;
    absl::optional<int64> stat_type = stat_type_getter(metadata.name());
    if (stat_type.has_value()) {
      auto result = stat_metadata_id_map_.emplace(metadata_id, *stat_type);
      DCHECK(result.second);  // inserted
      stat_type_map_.emplace(*stat_type, &metadata);
    }
  }
}

const XStatMetadata* XPlaneVisitor::GetStatMetadata(
    int64 stat_metadata_id) const {
  const auto& stat_metadata_map = plane_->stat_metadata();
  const auto it = stat_metadata_map.find(stat_metadata_id);
  if (it != stat_metadata_map.end()) return &it->second;
  return &XStatMetadata::default_instance();
}

absl::optional<int64> XPlaneVisitor::GetStatType(int64 stat_metadata_id) const {
  const auto it = stat_metadata_id_map_.find(stat_metadata_id);
  if (it != stat_metadata_id_map_.end()) return it->second;
  return absl::nullopt;
}

absl::optional<int64> XPlaneVisitor::GetStatMetadataId(int64 stat_type) const {
  const auto it = stat_type_map_.find(stat_type);
  if (it != stat_type_map_.end()) return it->second->id();
  return absl::nullopt;
}

const XEventMetadata* XPlaneVisitor::GetEventMetadata(
    int64 event_metadata_id) const {
  const auto& event_metadata_map = plane_->event_metadata();
  const auto it = event_metadata_map.find(event_metadata_id);
  if (it != event_metadata_map.end()) return &it->second;
  return &XEventMetadata::default_instance();
}

absl::optional<int64> XPlaneVisitor::GetEventType(
    int64 event_metadata_id) const {
  const auto it = event_metadata_id_map_.find(event_metadata_id);
  if (it != event_metadata_id_map_.end()) return it->second;
  return absl::nullopt;
}

}  // namespace profiler
}  // namespace tensorflow
