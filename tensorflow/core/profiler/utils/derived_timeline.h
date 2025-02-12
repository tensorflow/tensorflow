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
#ifndef TENSORFLOW_CORE_PROFILER_UTILS_DERIVED_TIMELINE_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_DERIVED_TIMELINE_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

// Store the mapping from child scope range id to parent scope range id, which
// logically form a scope range call stack tree/forest.
typedef absl::flat_hash_map<int64_t /* child_scope_range_id */,
                            int64_t /* parent_scope_range_id */>
    ScopeRangeIdTree;

// Helper for deriving XEvents.
class DerivedXEventBuilder {
 public:
  DerivedXEventBuilder(XEventBuilder event, std::optional<int64_t> group_id,
                       std::optional<int64_t> scope_range_id = std::nullopt);

  bool ShouldExpand(const XEventMetadata& event_metadata,
                    std::optional<int64_t> group_id,
                    std::optional<int64_t> scope_range_id = std::nullopt) const;

  void Expand(tsl::profiler::Timespan event_span);
  tsl::profiler::Timespan GetTimespan() const { return event_.GetTimespan(); }
  void SetTimespan(tsl::profiler::Timespan event_span) {
    event_.SetTimespan(event_span);
  }

  template <typename ValueT>
  void SetOrAddStatValue(const XStatMetadata& metadata, ValueT&& value) {
    event_.SetOrAddStatValue(metadata, std::forward<ValueT>(value));
  }

 private:
  XEventBuilder event_;
  std::optional<int64_t> group_id_;
  std::optional<int64_t> scope_range_id_;
};

// Helper for deriving an XLine from events in another XLine.
class DerivedXLineBuilder {
 public:
  DerivedXLineBuilder(XPlaneBuilder* plane, int64_t line_id,
                      absl::string_view name, int64_t timestamp_ns,
                      std::vector<DerivedXLineBuilder*> dependent_lines);

  XLineBuilder& Line() { return line_; }

  // Either merges event with the last event or creates a new event on this
  // XLine. group_id and low_level_event_name may be passed to separate
  // consecutive invocations of the same event, depending on the XEvent type:
  //   TF-op, TF name scope: both group_id and low_level_event_name are used.
  //   HLO-op, step: only group_id is used.
  //   HLO module, source: both group_id and low_level_event_name are NOT used.
  // If scope_range_id is provided, it will be compared with the one in the
  // event which is to be merged with. If they are different, merging is not
  // allowed.
  void ExpandOrAddEvent(const XEventMetadata& event_metadata,
                        tsl::profiler::Timespan event_span,
                        std::optional<int64_t> group_id,
                        std::optional<int64_t> scope_range_id = std::nullopt);

  // The multi-level version of ExpandOrAddEvent. Here, the XEvents at different
  // levels all share the same group_id and low_level_event_name.
  // Conceptually, the scope_range_ids should be of same length as the
  // events_metadata_per_level. However, if it is shorter, this function will
  // assume the missing elements at the end of scope_range_ids vector with the
  // value of std::nullopt; and if it is longer, the extra elements in
  // scope_range_ids will be ignored.
  void ExpandOrAddEvents(
      const std::vector<XEventMetadata*>& events_metadata_per_level,
      tsl::profiler::Timespan event_span, std::optional<int64_t> group_id,
      absl::Span<std::optional<int64_t>> scope_range_ids = {});

  // Reset the last events lower than or equal to the given level.
  void ResetLastEvents(int level = 0);

  // To avoid using templates while need hide its implementation in .cc file,
  // use two functions to set stat value for int64_t and uint64_t here.
  void AddStatToLevelEvent(int level, const XStatMetadata& metadata,
                           int64_t value);

  void AddStatToLevelEvent(int level, const XStatMetadata& metadata,
                           uint64_t value);

  const XStatMetadata* GetCorrelationIdMetadata() const {
    return correlation_id_metadata_;
  }

  const XStatMetadata* GetCudaGraphIdMetadata() const {
    return cuda_graph_id_metadata_;
  }

 private:
  // If the last event of the given level has the same metadata, expands it to
  // include the time until the given event's end time.
  // Otherwise, adds a new event and clears last_event_by_level_ for the levels
  // below the given level and all levels of the dependent lines. Clearing
  // last_event_by_level_ prevents a nested event from growing larger than the
  // parent event(s).
  void ExpandOrAddLevelEvent(const XEventMetadata& event_metadata,
                             tsl::profiler::Timespan event_span,
                             std::optional<int64_t> group_id,
                             std::optional<int64_t> scope_range_id, int level);
  void AdjustDurationForTraceViewer(int level);

  const XStatMetadata* group_id_stat_metadata_ = nullptr;
  const XStatMetadata* correlation_id_metadata_ = nullptr;
  const XStatMetadata* cuda_graph_id_metadata_ = nullptr;

  XLineBuilder line_;
  absl::flat_hash_map<int, std::optional<DerivedXEventBuilder>>
      last_event_by_level_;
  std::vector<DerivedXLineBuilder*> dependent_lines_;
  bool is_gpu_plane_ = false;
};

struct Symbol {
  absl::string_view tf_op_name;
  std::string source_info;
  std::string hlo_text;
};

using SymbolResolver = std::function<Symbol(std::optional<uint64_t> program_id,
                                            absl::string_view hlo_module_name,
                                            absl::string_view hlo_op)>;

// Derives TF name scope and op events from the TF op's fully qualified name
// with the name of the originating low-level event.
void ProcessTfOpEvent(absl::string_view tf_op_full_name,
                      tsl::profiler::Timespan event_span,
                      std::optional<int64_t> group_id,
                      XPlaneBuilder& plane_builder,
                      DerivedXLineBuilder& tf_name_scope_line_builder,
                      DerivedXLineBuilder& tf_op_line_builder);

// Derives "Steps" line from group_id XStat in XEvents.
void DeriveStepEventsFromGroups(
    const tsl::profiler::GroupMetadataMap& group_metadata_map,
    XPlane* device_trace);

// Derives "TensorFlow Ops", "TensorFlow Name Scope", "XLA Ops" and "XLA Module"
// lines in an NVIDIA_GPU device trace from data passed as ScopedAnnotations and
// stored as XStats in XEvents corresponding to GPU Kernels. Consecutive
// annotations with the same value are merged into a single event except for XLA
// modules. The device_trace is both input and output.
void DeriveEventsFromAnnotations(
    const SymbolResolver& symbol_resolver, XPlane* device_trace,
    const ScopeRangeIdTree* scope_range_id_tree = nullptr);

// Derives "Launch Activities Summary" line from host trace.
void DeriveEventsFromHostTrace(
    const XPlane* host_trace,
    const tsl::profiler::GroupMetadataMap& group_metadata_map,
    std::vector<XPlane*> device_traces);

// Loops through XPlanes of input XSpace, if it is "device" XPlane, generating
// derived timelines for the plane by calling DeriveEventsFromAnnotations.
void GenerateDerivedTimeLines(
    const tsl::profiler::GroupMetadataMap& group_metadata_map, XSpace* space);

// Derives `Tensorflow Ops`, `Tensorflow Name Scope` and `Source Code` lines
// from device_trace.
void DeriveLinesFromStats(tensorflow::profiler::XPlane* device_trace);

// Devices Framework Op and Module lines for XLA:CPU ops.
void DeriveLinesForXlaCpuOps(tensorflow::profiler::XPlane* host_trace);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_DERIVED_TIMELINE_H_
