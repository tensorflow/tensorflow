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
#ifndef XLA_TSL_PROFILER_UTILS_XPLANE_VISITOR_H_
#define XLA_TSL_PROFILER_UTILS_XPLANE_VISITOR_H_

#include <stddef.h>

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

using tensorflow::profiler::XEvent;          // NOLINT
using tensorflow::profiler::XEventMetadata;  // NOLINT
using tensorflow::profiler::XLine;           // NOLINT
using tensorflow::profiler::XPlane;          // NOLINT
using tensorflow::profiler::XSpace;          // NOLINT
using tensorflow::profiler::XStat;           // NOLINT
using tensorflow::profiler::XStatMetadata;   // NOLINT

class XPlaneVisitor;

class XStatVisitor {
 public:
  // REQUIRED: plane and stat cannot be nullptr.
  XStatVisitor(const XPlaneVisitor* plane, const XStat* stat);

  // REQUIRED: plane, stat and metadata cannot be nullptr.
  XStatVisitor(const XPlaneVisitor* plane, const XStat* stat,
               const XStatMetadata* metadata, std::optional<int64_t> type);

  int64_t Id() const { return stat_->metadata_id(); }

  absl::string_view Name() const { return metadata_->name(); }

  std::optional<int64_t> Type() const { return type_; }

  absl::string_view Description() const { return metadata_->description(); }

  XStat::ValueCase ValueCase() const { return stat_->value_case(); }

  bool BoolValue() const { return static_cast<bool>(IntValue()); }

  int64_t IntValue() const { return stat_->int64_value(); }

  uint64 UintValue() const { return stat_->uint64_value(); }

  absl::string_view BytesValue() const { return stat_->bytes_value(); }

  uint64 IntOrUintValue() const {
    return ValueCase() == XStat::kUint64Value ? UintValue()
                                              : static_cast<uint64>(IntValue());
  }

  double DoubleValue() const { return stat_->double_value(); }

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
  std::optional<int64_t> type_;
};

template <class T>
class XStatsOwner {
 public:
  // REQUIRED: plane and stats_owner cannot be nullptr.
  XStatsOwner(const XPlaneVisitor* plane, const T* stats_owner)
      : plane_(plane), stats_owner_(stats_owner) {}

  // For each stat, call the specified lambda.
  template <typename ForEachStatFunc>
  void ForEachStat(ForEachStatFunc&& for_each_stat) const {
    for (const XStat& stat : stats_owner_->stats()) {
      for_each_stat(XStatVisitor(plane_, &stat));
    }
  }

  // Shortcut to get a specific stat type, nullopt if absent.
  // This function performs a linear search for the requested stat value.
  // Prefer ForEachStat above when multiple stat values are necessary.
  std::optional<XStatVisitor> GetStat(int64_t stat_type) const;

  // Same as above that skips searching for the stat.
  std::optional<XStatVisitor> GetStat(
      int64_t stat_type, const XStatMetadata& stat_metadata) const {
    for (const XStat& stat : stats_owner_->stats()) {
      if (stat.metadata_id() == stat_metadata.id()) {
        return XStatVisitor(plane_, &stat, &stat_metadata, stat_type);
      }
    }
    return std::nullopt;  // type does not exist in this owner.
  }

 protected:
  const XPlaneVisitor* plane() const { return plane_; }
  const T* stats_owner() const { return stats_owner_; }

 private:
  const XPlaneVisitor* plane_;
  const T* stats_owner_;
};

class XEventMetadataVisitor : public XStatsOwner<XEventMetadata> {
 public:
  // REQUIRED: plane and metadata cannot be nullptr.
  XEventMetadataVisitor(const XPlaneVisitor* plane,
                        const XEventMetadata* metadata)
      : XStatsOwner(plane, metadata) {}

  int64_t Id() const { return metadata()->id(); }

  absl::string_view Name() const { return metadata()->name(); }

  bool HasDisplayName() const { return !metadata()->display_name().empty(); }

  absl::string_view DisplayName() const { return metadata()->display_name(); }

  // For each child event metadata, call the specified lambda.
  template <typename ForEachChildFunc>
  void ForEachChild(ForEachChildFunc&& for_each_child) const;

 private:
  const XEventMetadata* metadata() const { return stats_owner(); }
};

class XEventVisitor : public XStatsOwner<XEvent> {
 public:
  // REQUIRED: plane, line and event cannot be nullptr.
  XEventVisitor(const XPlaneVisitor* plane, const XLine* line,
                const XEvent* event);

  const XPlaneVisitor& Plane() const { return *plane_; }

  const XEvent& RawEvent() const { return *event_; }

  int64_t Id() const { return event_->metadata_id(); }

  absl::string_view Name() const { return metadata_->name(); }

  std::optional<int64_t> Type() const { return type_; }

  bool HasDisplayName() const { return !metadata_->display_name().empty(); }

  absl::string_view DisplayName() const { return metadata_->display_name(); }

  double OffsetNs() const { return PicoToNano(event_->offset_ps()); }

  int64_t OffsetPs() const { return event_->offset_ps(); }

  int64_t LineTimestampNs() const { return line_->timestamp_ns(); }

  int64_t TimestampNs() const { return line_->timestamp_ns() + OffsetNs(); }

  int64_t TimestampPs() const {
    return NanoToPico(line_->timestamp_ns()) + event_->offset_ps();
  }

  double DurationNs() const { return PicoToNano(event_->duration_ps()); }

  int64_t DurationPs() const { return event_->duration_ps(); }

  int64_t EndOffsetPs() const {
    return event_->offset_ps() + event_->duration_ps();
  }

  int64_t EndTimestampNs() const { return TimestampNs() + DurationNs(); }

  int64_t EndTimestampPs() const { return TimestampPs() + DurationPs(); }

  int64_t NumOccurrences() const { return event_->num_occurrences(); }

  bool IsAggregatedEvent() const {
    return event_->data_case() == XEvent::kNumOccurrences;
  }

  bool operator<(const XEventVisitor& other) const {
    return GetTimespan() < other.GetTimespan();
  }

  const XEventMetadata* metadata() const { return metadata_; }

  XEventMetadataVisitor Metadata() const {
    return XEventMetadataVisitor(plane_, metadata_);
  }

  Timespan GetTimespan() const { return Timespan(TimestampPs(), DurationPs()); }

 private:
  const XPlaneVisitor* plane_;
  const XLine* line_;
  const XEvent* event_;
  const XEventMetadata* metadata_;
  std::optional<int64_t> type_;
};

class XLineVisitor {
 public:
  // REQUIRED: plane and line cannot be nullptr.
  XLineVisitor(const XPlaneVisitor* plane, const XLine* line)
      : plane_(plane), line_(line) {}

  int64_t Id() const { return line_->id(); }

  int64_t DisplayId() const {
    return line_->display_id() ? line_->display_id() : line_->id();
  }

  absl::string_view Name() const { return line_->name(); }

  absl::string_view DisplayName() const {
    return !line_->display_name().empty() ? line_->display_name()
                                          : line_->name();
  }

  int64_t TimestampNs() const { return line_->timestamp_ns(); }

  int64_t DurationPs() const { return line_->duration_ps(); }

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

using TypeGetter = std::function<std::optional<int64_t>(absl::string_view)>;
using TypeGetterList = std::vector<TypeGetter>;

class XPlaneVisitor : public XStatsOwner<XPlane> {
 public:
  // REQUIRED: plane cannot be nullptr.
  explicit XPlaneVisitor(
      const XPlane* plane,
      const TypeGetterList& event_type_getter_list = TypeGetterList(),
      const TypeGetterList& stat_type_getter_list = TypeGetterList());

  int64_t Id() const { return plane_->id(); }

  absl::string_view Name() const { return plane_->name(); }

  size_t NumLines() const { return plane_->lines_size(); }

  template <typename ForEachLineFunc>
  void ForEachLine(ForEachLineFunc&& for_each_line) const {
    for (const XLine& line : plane_->lines()) {
      for_each_line(XLineVisitor(this, &line));
    }
  }
  template <typename ThreadBundle, typename ForEachLineFunc>
  void ForEachLineInParallel(ForEachLineFunc&& for_each_line) const {
    ThreadBundle bundle;
    for (const XLine& line : plane_->lines()) {
      bundle.Add([this, line = &line, &for_each_line] {
        for_each_line(XLineVisitor(this, line));
      });
    }
    bundle.JoinAll();
  }

  template <typename ForEachEventMetadataFunc>
  void ForEachEventMetadata(
      ForEachEventMetadataFunc&& for_each_event_metadata) const {
    for (const auto& [id, event_metadata] : plane_->event_metadata()) {
      for_each_event_metadata(XEventMetadataVisitor(this, &event_metadata));
    }
  }

  // Returns event metadata given its id. Returns a default value if not found.
  const XEventMetadata* GetEventMetadata(int64_t event_metadata_id) const;

  // Returns the type of an event given its id.
  std::optional<int64_t> GetEventType(int64_t event_metadata_id) const;

  // Returns stat metadata given its id. Returns a default value if not found.
  const XStatMetadata* GetStatMetadata(int64_t stat_metadata_id) const;

  // Returns stat metadata given its type. Returns nullptr if not found.
  // Use as an alternative to GetStatMetadata above.
  const XStatMetadata* GetStatMetadataByType(int64_t stat_type) const;

  // Returns the type of an stat given its id.
  std::optional<int64_t> GetStatType(int64_t stat_metadata_id) const;

 private:
  void BuildEventTypeMap(const XPlane* plane,
                         const TypeGetterList& event_type_getter_list);
  void BuildStatTypeMap(const XPlane* plane,
                        const TypeGetterList& stat_type_getter_list);

  const XPlane* plane_;

  absl::flat_hash_map<int64_t /*metadata_id*/, int64_t /*EventType*/>
      event_type_by_id_;
  absl::flat_hash_map<int64_t /*metadata_id*/, int64_t /*StatType*/>
      stat_type_by_id_;
  absl::flat_hash_map<int64_t /*StatType*/, const XStatMetadata*>
      stat_metadata_by_type_;
};

template <class T>
std::optional<XStatVisitor> XStatsOwner<T>::GetStat(int64_t stat_type) const {
  const auto* stat_metadata = plane_->GetStatMetadataByType(stat_type);
  if (stat_metadata != nullptr) {
    return GetStat(stat_type, *stat_metadata);
  }
  return std::nullopt;  // type does not exist in this owner.
}

template <typename ForEachChildFunc>
void XEventMetadataVisitor::ForEachChild(
    ForEachChildFunc&& for_each_child) const {
  for (int64_t child_id : metadata()->child_id()) {
    const auto* event_metadata = plane()->GetEventMetadata(child_id);
    if (event_metadata != nullptr) {
      for_each_child(XEventMetadataVisitor(plane(), event_metadata));
    }
  }
}

}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_UTILS_XPLANE_VISITOR_H_
