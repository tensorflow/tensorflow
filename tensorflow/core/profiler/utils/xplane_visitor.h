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
#ifndef TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_VISITOR_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_VISITOR_H_

#include <functional>
#include <unordered_map>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/time_utils.h"
#include "tensorflow/core/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {

class XPlaneVisitor;

class XStatVisitor {
 public:
  // REQUIRED: plane and stat cannot be nullptr.
  XStatVisitor(const XPlaneVisitor* plane, const XStat* stat);

  int64 Id() const { return stat_->metadata_id(); }

  absl::string_view Name() const { return metadata_->name(); }

  absl::optional<int64> Type() const { return type_; }

  absl::string_view Description() const { return metadata_->description(); }

  XStat::ValueCase ValueCase() const { return stat_->value_case(); }

  int64 IntValue() const { return stat_->int64_value(); }

  uint64 UintValue() const { return stat_->uint64_value(); }

  double DoubleValue() const { return stat_->double_value(); }

  absl::string_view StrValue() const { return stat_->str_value(); }

  absl::string_view RefValue() const;

  // Returns a string view.
  // REQUIRED: the value type should be string type or reference type.
  absl::string_view StrOrRefValue() const;

  const XStat& RawStat() const { return *stat_; }

  // Return a string representation of all value type.
  std::string ToString() const;

 private:
  const XStat* stat_;
  const XStatMetadata* metadata_;
  const XPlaneVisitor* plane_;
  absl::optional<int64> type_;
};

template <class T>
class XStatsOwner {
 public:
  // REQUIRED: metadata and stats_owner cannot be nullptr.
  XStatsOwner(const XPlaneVisitor* metadata, const T* stats_owner)
      : stats_owner_(stats_owner), metadata_(metadata) {}

  // For each plane level stats, call the specified lambda.
  template <typename ForEachStatFunc>
  void ForEachStat(ForEachStatFunc&& for_each_stat) const {
    for (const XStat& stat : stats_owner_->stats()) {
      for_each_stat(XStatVisitor(metadata_, &stat));
    }
  }

  // Shortcut to get a specfic stat type, nullptr if it is absent.
  const XStat* GetStats(int64 stat_type) const;

 private:
  const T* stats_owner_;
  const XPlaneVisitor* metadata_;
};

class XEventVisitor : public XStatsOwner<XEvent> {
 public:
  // REQUIRED: plane, line and event cannot be nullptr.
  XEventVisitor(const XPlaneVisitor* plane, const XLine* line,
                const XEvent* event);
  int64 Id() const { return event_->metadata_id(); }

  absl::string_view Name() const { return metadata_->name(); }

  absl::optional<int64> Type() const { return type_; }

  bool HasDisplayName() const { return !metadata_->display_name().empty(); }

  absl::string_view DisplayName() const { return metadata_->display_name(); }

  absl::string_view Metadata() const { return metadata_->metadata(); }

  double OffsetNs() const { return PicosToNanos(event_->offset_ps()); }

  int64 OffsetPs() const { return event_->offset_ps(); }

  int64 LineTimestampNs() const { return line_->timestamp_ns(); }

  double TimestampNs() const { return line_->timestamp_ns() + OffsetNs(); }

  int64 TimestampPs() const {
    return NanosToPicos(line_->timestamp_ns()) + event_->offset_ps();
  }

  double DurationNs() const { return PicosToNanos(event_->duration_ps()); }

  int64 DurationPs() const { return event_->duration_ps(); }

  int64 EndOffsetPs() const {
    return event_->offset_ps() + event_->duration_ps();
  }

  int64 NumOccurrences() const { return event_->num_occurrences(); }

  bool operator<(const XEventVisitor& other) const {
    return GetTimespan() < other.GetTimespan();
  }

  const XEventMetadata* metadata() const { return metadata_; }

  Timespan GetTimespan() const { return Timespan(TimestampPs(), DurationPs()); }

 private:
  const XPlaneVisitor* plane_;
  const XLine* line_;
  const XEvent* event_;
  const XEventMetadata* metadata_;
  absl::optional<int64> type_;
};

class XLineVisitor {
 public:
  // REQUIRED: plane and line cannot be nullptr.
  XLineVisitor(const XPlaneVisitor* plane, const XLine* line)
      : plane_(plane), line_(line) {}

  int64 Id() const { return line_->id(); }

  int64 DisplayId() const {
    return line_->display_id() ? line_->display_id() : line_->id();
  }

  absl::string_view Name() const { return line_->name(); }

  absl::string_view DisplayName() const {
    return !line_->display_name().empty() ? line_->display_name()
                                          : line_->name();
  }

  double TimestampNs() const { return line_->timestamp_ns(); }

  int64 DurationPs() const { return line_->duration_ps(); }

  size_t NumEvents() const { return line_->events_size(); }

  template <typename ForEachEventFunc>
  void ForEachEvent(ForEachEventFunc&& for_each_event) const {
    for (const XEvent& event : line_->events()) {
      for_each_event(XEventVisitor(plane_, line_, &event));
    }
  }

 private:
  const XPlaneVisitor* plane_;
  const XLine* line_;
};

using TypeGetter = std::function<absl::optional<int64>(absl::string_view)>;
using TypeGetterList = std::vector<TypeGetter>;

class XPlaneVisitor : public XStatsOwner<XPlane> {
 public:
  // REQUIRED: plane cannot be nullptr.
  explicit XPlaneVisitor(
      const XPlane* plane,
      const TypeGetterList& event_type_getter_list = TypeGetterList(),
      const TypeGetterList& stat_type_getter_list = TypeGetterList());

  int64 Id() const { return plane_->id(); }

  absl::string_view Name() const { return plane_->name(); }

  size_t NumLines() const { return plane_->lines_size(); }

  template <typename ForEachLineFunc>
  void ForEachLine(ForEachLineFunc&& for_each_line) const {
    for (const XLine& line : plane_->lines()) {
      for_each_line(XLineVisitor(this, &line));
    }
  }

  // TODO(jiesun): use single map look up for both StatMetadata and StatType.
  const XStatMetadata* GetStatMetadata(int64 stat_metadata_id) const;
  absl::optional<int64> GetStatType(int64 stat_metadata_id) const;
  absl::optional<int64> GetStatType(const XStat& stat) const {
    return GetStatType(stat.metadata_id());
  }
  absl::optional<int64> GetStatMetadataId(int64 stat_type) const;
  const XEventMetadata* GetEventMetadata(int64 event_metadata_id) const;
  absl::optional<int64> GetEventType(int64 event_metadata_id) const;
  absl::optional<int64> GetEventType(const XEvent& event) const {
    return GetEventType(event.metadata_id());
  }

 private:
  void BuildEventTypeMap(const XPlane* plane,
                         const TypeGetter& event_type_getter);
  void BuildStatTypeMap(const XPlane* plane,
                        const TypeGetter& stat_type_getter);

  const XPlane* plane_;

  absl::flat_hash_map<int64 /*metadata_id*/, int64 /*StatType*/>
      stat_metadata_id_map_;
  absl::flat_hash_map<int64 /*StatType*/, const XStatMetadata*> stat_type_map_;
  absl::flat_hash_map<int64 /*metadata_id*/, int64 /*EventType*/>
      event_metadata_id_map_;
  absl::flat_hash_map<int64 /*EventType*/, const XEventMetadata*>
      event_type_map_;
};

template <class T>
const XStat* XStatsOwner<T>::GetStats(int64 stat_type) const {
  absl::optional<int64> stat_metadata_id =
      metadata_->GetStatMetadataId(stat_type);
  if (!stat_metadata_id) return nullptr;  // type does not exist in the XPlane.
  for (const XStat& stat : stats_owner_->stats()) {
    if (stat.metadata_id() == *stat_metadata_id) return &stat;
  }
  return nullptr;  // type does not exist in this owner.
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_VISITOR_H_
