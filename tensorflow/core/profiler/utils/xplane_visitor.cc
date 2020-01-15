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

#include "absl/types/optional.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {
namespace profiler {

XStatVisitor::XStatVisitor(const XPlaneVisitor* plane, const XStat* stat)
    : stat_(stat),
      metadata_(plane->GetStatMetadata(stat->metadata_id())),
      type_(plane->GetStatType(stat->metadata_id())) {}

XEventVisitor::XEventVisitor(const XPlaneVisitor* plane, const XLine* line,
                             const XEvent* event)
    : XStatsOwner<XEvent>(plane, event),
      plane_(plane),
      line_(line),
      event_(event),
      metadata_(plane->GetEventMetadata(event_->metadata_id())) {}

XPlaneVisitor::XPlaneVisitor(const XPlane* plane)
    : XStatsOwner<XPlane>(this, plane), plane_(plane) {
  for (const auto& stat_metadata : plane->stat_metadata()) {
    StatType type =
        tensorflow::profiler::GetStatType(stat_metadata.second.name());
    stat_metadata_id_map_.emplace(stat_metadata.first,
                                  std::make_pair(&stat_metadata.second, type));
    stat_type_map_.emplace(type, &stat_metadata.second);
  }
}

const XStatMetadata* XPlaneVisitor::GetStatMetadata(
    int64 stat_metadata_id) const {
  const auto* it = gtl::FindOrNull(stat_metadata_id_map_, stat_metadata_id);
  return it ? it->first : &XStatMetadata::default_instance();
}

StatType XPlaneVisitor::GetStatType(int64 stat_metadata_id) const {
  const auto* it = gtl::FindOrNull(stat_metadata_id_map_, stat_metadata_id);
  return it ? it->second : kUnknownStatType;
}

absl::optional<int64> XPlaneVisitor::GetStatMetadataId(
    StatType stat_type) const {
  const auto* it = gtl::FindOrNull(stat_type_map_, stat_type);
  if (!it) return absl::nullopt;
  return (*it)->id();
}

const XEventMetadata* XPlaneVisitor::GetEventMetadata(
    int64 event_metadata_id) const {
  return &gtl::FindWithDefault(plane_->event_metadata(), event_metadata_id,
                               XEventMetadata::default_instance());
}
}  // namespace profiler
}  // namespace tensorflow
