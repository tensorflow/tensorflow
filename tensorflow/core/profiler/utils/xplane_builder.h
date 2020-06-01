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
#ifndef TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_BUILDER_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_BUILDER_H_

#include <stddef.h>

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/time_utils.h"

namespace tensorflow {
namespace profiler {

class XPlaneBuilder;

template <class T>
class XStatsBuilder {
 public:
  explicit XStatsBuilder(T* stats_owner, XPlaneBuilder* stats_metadata_owner)
      : stats_owner_(stats_owner),
        stats_metadata_owner_(stats_metadata_owner) {}

  void AddStatValue(const XStatMetadata& metadata, uint32 value) {
    AddStat(metadata)->set_uint64_value(value);
  }
  void AddStatValue(const XStatMetadata& metadata, uint64 value) {
    AddStat(metadata)->set_uint64_value(value);
  }
  void AddStatValue(const XStatMetadata& metadata, int32 value) {
    AddStat(metadata)->set_int64_value(value);
  }
  void AddStatValue(const XStatMetadata& metadata, int64 value) {
    AddStat(metadata)->set_int64_value(value);
  }
  void AddStatValue(const XStatMetadata& metadata, double value) {
    AddStat(metadata)->set_double_value(value);
  }
  void AddStatValue(const XStatMetadata& metadata, absl::string_view value) {
    AddStat(metadata)->set_str_value(std::string(value));
  }
  void AddStatValue(const XStatMetadata& metadata, std::string&& value) {
    AddStat(metadata)->set_str_value(std::move(value));
  }
  void AddStatValue(const XStatMetadata& key, const XStatMetadata& value) {
    AddStat(key)->set_ref_value(value.id());
  }
  void AddStatValue(const XStatMetadata& metadata,
                    const protobuf::MessageLite& proto) {
    auto* bytes = AddStat(metadata)->mutable_bytes_value();
    proto.SerializeToString(bytes);
  }

  void AddStat(const XStatMetadata& key, const XStat& stat, const XPlane& src);

  XStat* FindOrAddMutableStat(int64 metadata_id) {
    for (auto& stat : *stats_owner_->mutable_stats()) {
      if (stat.metadata_id() == metadata_id) {
        return &stat;
      }
    }
    return stats_owner_->add_stats();
  }

  void ParseAndAddStatValue(const XStatMetadata& metadata,
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
  void ReserveStats(size_t num_stats) {
    stats_owner_->mutable_stats()->Reserve(num_stats);
  }

 private:
  XStat* AddStat(const XStatMetadata& metadata) {
    XStat* stat = stats_owner_->add_stats();
    stat->set_metadata_id(metadata.id());
    return stat;
  }

  T* stats_owner_;
  XPlaneBuilder* stats_metadata_owner_;
};

class XEventBuilder : public XStatsBuilder<XEvent> {
 public:
  XEventBuilder(const XLine* line, XPlaneBuilder* plane, XEvent* event)
      : XStatsBuilder<XEvent>(event, plane), line_(line), event_(event) {}

  int64 OffsetPs() const { return event_->offset_ps(); }
  int64 MetadataId() const { return event_->metadata_id(); }

  void SetOffsetPs(int64 offset_ps) { event_->set_offset_ps(offset_ps); }

  void SetOffsetNs(int64 offset_ns) { SetOffsetPs(NanosToPicos(offset_ns)); }

  void SetTimestampNs(int64 timestamp_ns) {
    SetOffsetPs(NanosToPicos(timestamp_ns - line_->timestamp_ns()));
  }

  void SetNumOccurrences(int64 num_occurrences) {
    event_->set_num_occurrences(num_occurrences);
  }

  void SetDurationPs(int64 duration_ps) {
    event_->set_duration_ps(duration_ps);
  }

  void SetDurationNs(int64 duration_ns) {
    SetDurationPs(NanosToPicos(duration_ns));
  }

  void SetEndTimestampNs(int64 end_timestamp_ns) {
    SetDurationPs(NanosToPicos(end_timestamp_ns - line_->timestamp_ns()) -
                  event_->offset_ps());
  }

 private:
  const XLine* line_;
  XEvent* event_;
};

class XLineBuilder {
 public:
  explicit XLineBuilder(XLine* line, XPlaneBuilder* plane)
      : line_(line), plane_(plane) {}

  int64 Id() { return line_->id(); }
  void SetId(int64 id) { line_->set_id(id); }

  int64 NumEvents() { return line_->events_size(); }

  void SetName(absl::string_view name) { line_->set_name(std::string(name)); }

  void SetNameIfEmpty(absl::string_view name) {
    if (line_->name().empty()) SetName(name);
  }

  int64 TimestampNs() { return line_->timestamp_ns(); }
  // This will set the line start timestamp.
  // WARNING: The offset_ps of existing events will not be altered.
  void SetTimestampNs(int64 timestamp_ns) {
    line_->set_timestamp_ns(timestamp_ns);
  }
  // This will set the line start timestamp to specific time, and adjust
  // the offset_ps of all existing events.
  void SetTimestampNsAndAdjustEventOffsets(int64 timestamp_ns);

  void SetDurationPs(int64 duration_ps) { line_->set_duration_ps(duration_ps); }

  void ReserveEvents(size_t num_events) {
    line_->mutable_events()->Reserve(num_events);
  }

  void SetDisplayNameIfEmpty(absl::string_view display_name) {
    if (line_->display_name().empty()) {
      line_->set_display_name(std::string(display_name));
    }
  }

  XEventBuilder AddEvent(const XEventMetadata& metadata);
  XEventBuilder AddEvent(const XEvent& event);

 private:
  XLine* line_;
  XPlaneBuilder* plane_;
};

// Provides methods to build an XPlane.
// NOTE: avoid to use two builders to wrap the same XPlane.
class XPlaneBuilder : public XStatsBuilder<XPlane> {
 public:
  explicit XPlaneBuilder(XPlane* plane);

  int64 Id() { return plane_->id(); }
  void SetId(int64 id) { plane_->set_id(id); }

  void SetName(absl::string_view name) { plane_->set_name(std::string(name)); }

  void ReserveLines(size_t num_lines) {
    plane_->mutable_lines()->Reserve(num_lines);
  }

  template <typename ForEachLineFunc>
  void ForEachLine(ForEachLineFunc&& for_each_line) {
    for (XLine& line : *plane_->mutable_lines()) {
      for_each_line(XLineBuilder(&line, this));
    }
  }

  XLineBuilder GetOrCreateLine(int64 line_id);

  XEventMetadata* GetOrCreateEventMetadata(int64 metadata_id);
  XEventMetadata* GetOrCreateEventMetadata(absl::string_view name);
  XEventMetadata* GetOrCreateEventMetadata(std::string&& name);
  inline XEventMetadata* GetOrCreateEventMetadata(const char* name) {
    return GetOrCreateEventMetadata(absl::string_view(name));
  }

  XStatMetadata* GetOrCreateStatMetadata(int64 metadata_id);
  XStatMetadata* GetOrCreateStatMetadata(absl::string_view name);

 protected:
  XPlane* RawPlane() const { return plane_; }
  XLine* AddLine(int64 line_id);

 private:
  XPlane* plane_;

  // Artifacts to accelerate the builders.
  int64 last_event_metadata_id_ = 0LL;
  int64 last_stat_metadata_id_ = 0LL;
  absl::flat_hash_map<std::string, XEventMetadata*> event_metadata_by_name_;
  absl::flat_hash_map<std::string, XStatMetadata*> stat_metadata_by_name_;
  absl::flat_hash_map<int64, XLine*> lines_by_id_;
};

template <class T>
void XStatsBuilder<T>::AddStat(const XStatMetadata& key, const XStat& stat,
                               const XPlane& src) {
  if (stat.value_case() == XStat::kRefValue) {
    const auto& stat_metadata_map = src.stat_metadata();
    const auto it = stat_metadata_map.find(stat.ref_value());
    if (TF_PREDICT_FALSE(it == stat_metadata_map.end())) {
      // the reference value in stat is not found in XStatMetadata from src.
      return;
    }
    XStatMetadata* value =
        stats_metadata_owner_->GetOrCreateStatMetadata(it->second.name());
    AddStatValue(key, *value);
  } else {
    XStat* new_stat = stats_owner_->add_stats();
    *new_stat = stat;
    new_stat->set_metadata_id(key.id());
  }
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_BUILDER_H_
