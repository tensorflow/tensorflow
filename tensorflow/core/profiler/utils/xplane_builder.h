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

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/time_utils.h"

namespace tensorflow {
namespace profiler {

class XEventBuilder {
 public:
  XEventBuilder(const XLine* line, XEvent* event)
      : line_(line), event_(event) {}

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

  void ReserveStats(size_t num_stats) {
    event_->mutable_stats()->Reserve(num_stats);
  }

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
    AddStat(metadata)->set_str_value(string(value));
  }
  void AddStatValue(const XStatMetadata& metadata, string&& value) {
    AddStat(metadata)->set_str_value(std::move(value));
  }

  void ParseAndAddStatValue(const XStatMetadata& metadata,
                            absl::string_view value);

  void AddStat(const XStatMetadata& metadata, const XStat& stat) {
    DCHECK_EQ(metadata.id(), stat.metadata_id());
    *event_->add_stats() = stat;
  }

 private:
  XStat* AddStat(const XStatMetadata& metadata);

  const XLine* line_;
  XEvent* event_;
};

class XLineBuilder {
 public:
  explicit XLineBuilder(XLine* line) : line_(line) {}

  void SetId(int64 id) { line_->set_id(id); }

  void SetName(absl::string_view name) { line_->set_name(string(name)); }

  void SetNameIfEmpty(absl::string_view name) {
    if (line_->name().empty()) SetName(name);
  }

  void SetTimestampNs(int64 timestamp_ns) {
    line_->set_timestamp_ns(timestamp_ns);
  }

  void SetDurationPs(int64 duration_ps) { line_->set_duration_ps(duration_ps); }

  void ReserveEvents(size_t num_events) {
    line_->mutable_events()->Reserve(num_events);
  }

  XEventBuilder AddEvent(const XEventMetadata& metadata);

 private:
  XLine* line_;
};

// Provides methods to build an XPlane.
class XPlaneBuilder {
 public:
  explicit XPlaneBuilder(XPlane* plane) : plane_(plane) {}

  void SetId(int64 id) { plane_->set_id(id); }

  void SetName(absl::string_view name) { plane_->set_name(string(name)); }

  void ReserveLines(size_t num_lines) {
    plane_->mutable_lines()->Reserve(num_lines);
  }

  XLineBuilder AddLine() { return XLineBuilder(plane_->add_lines()); }

  XEventMetadata* GetOrCreateEventMetadata(int64 metadata_id);

  XStatMetadata* GetOrCreateStatMetadata(int64 metadata_id);

 protected:
  XPlane* RawPlane() const { return plane_; }

 private:
  XPlane* plane_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_BUILDER_H_
