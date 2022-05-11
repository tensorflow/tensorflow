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

#include <functional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/group_events.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"

namespace tensorflow {
namespace profiler {

// Additional information of an XEvent used to separate consecutive invocations
// of the same Op on the XLine.
struct XEventInfo {
  int64_t group_id;  // group ID of this XEvent or kInvalidGroupId.

  // The set of low level events associated with this XEvent.
  // For a TF op that is compiled by XLA, these are its composing HLO op names.
  // For a TF op that is not compiled by XLA, these are its composing kernel
  // names.
  absl::flat_hash_set<std::string> low_level_event_names;

  XEventInfo(int64_t gid, absl::string_view low_level_event_name) {
    group_id = gid;
    if (!low_level_event_name.empty()) {
      low_level_event_names.insert(std::string(low_level_event_name));
    }
  }
};

// Helper for deriving an XLine from events in another XLine.
class DerivedXLineBuilder {
 public:
  static constexpr int64_t kInvalidGroupId = -1;

  DerivedXLineBuilder(XPlaneBuilder* plane, int64_t line_id,
                      absl::string_view name, int64_t timestamp_ns,
                      std::vector<DerivedXLineBuilder*> dependent_lines);

  // Either merges event with the last event or creates a new event on this
  // XLine. group_id and low_level_event_name may be passed to separate
  // consecutive invocations of the same event, depending on the XEvent type:
  //   TF-op, TF name scope: both group_id and low_level_event_name are used.
  //   HLO-op, step: only group_id is used.
  //   HLO module, source: both group_id and low_level_event_name are NOT used.
  void ExpandOrAddEvent(const XEvent& event, int64_t group_id = kInvalidGroupId,
                        absl::string_view low_level_event_name = "") {
    ExpandOrAddLevelEvent(event, group_id, low_level_event_name,
                          /*level=*/0);
  }

  // The multi-level version of ExpandOrAddEvent. Here, the XEvents at different
  // levels all share the same group_id and low_level_event_name.
  void ExpandOrAddEvents(const std::vector<XEvent>& event_per_level,
                         int64_t group_id = kInvalidGroupId,
                         absl::string_view low_level_event_name = "") {
    size_t current_nested_level = event_per_level.size();
    for (size_t level = 0; level < current_nested_level; ++level) {
      ExpandOrAddLevelEvent(event_per_level[level], group_id,
                            low_level_event_name, level);
    }
    if (current_nested_level) ResetLastEvents(current_nested_level);
  }

  // Reset the last events lower than or equal to the given level.
  void ResetLastEvents(int level = 0);

 private:
  // If the last event of the given level has the same metadata, expands it to
  // include the time until the given event's (offset_ps + duration_ps).
  // Otherwise, adds a new event and clears last_event_by_level_ for the levels
  // below the given level and all levels of the dependent lines. Clearing
  // last_event_by_level_ prevents a nested event from growing larger than the
  // parent event(s).
  void ExpandOrAddLevelEvent(const XEvent& event, int64_t group_id,
                             absl::string_view low_level_event_name, int level);

  void ResetDependentLines() {
    for (DerivedXLineBuilder* line : dependent_lines_) {
      line->ResetLastEvents();
    }
  }

  const XStatMetadata* level_stats_ = nullptr;
  XLineBuilder line_;
  absl::flat_hash_map<int, absl::optional<XEventBuilder>> last_event_by_level_;
  absl::flat_hash_map<int, absl::optional<XEventInfo>> last_eventinfo_by_level_;
  std::vector<DerivedXLineBuilder*> dependent_lines_;
};

struct Symbol {
  absl::string_view tf_op_name;
  std::string source_info;
};

using SymbolResolver = std::function<Symbol(absl::optional<uint64_t> program_id,
                                            absl::string_view hlo_module_name,
                                            absl::string_view hlo_op)>;

// Derives TF name scope and op events from the TF op's fully qualified name
// with the name of the originating low-level event.
void ProcessTfOpEvent(absl::string_view tf_op_full_name,
                      absl::string_view low_level_event_name, int64_t offset_ps,
                      int64_t duration_ps, absl::optional<int64_t> group_id,
                      XPlaneBuilder* plane_builder,
                      DerivedXLineBuilder* tf_name_scope_line_builder,
                      DerivedXLineBuilder* tf_op_line_builder);

// Derives "Step Info", "Tensorflow Ops", "XLA Ops" and "XLA Module" lines in
// an NVIDIA_GPU device trace from data passed as ScopedAnnotations and stored
// as XStats in XEvents corresponding to GPU Kernels. Consecutive annotations
// with the same value are merged into a single event except for XLA modules.
// The device_trace is both input and output.
void DeriveEventsFromAnnotations(const SymbolResolver& symbol_resolver,
                                 const GroupMetadataMap& group_metadata_map,
                                 XPlane* device_trace,
                                 bool step_info_only = false);

// Derives "Launch Activities Summary" line from host trace.
void DeriveEventsFromHostTrace(const XPlane* host_trace,
                               const GroupMetadataMap& group_metadata_map,
                               std::vector<XPlane*> device_traces);

// Loops through XPlanes of input XSpace, if it is "device" XPlane, generating
// derived timelines for the plane by calling DeriveEventsFromAnnotations.
void GenerateDerivedTimeLines(const GroupMetadataMap& group_metadata_map,
                              XSpace* space, bool step_info_only = false);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_DERIVED_TIMELINE_H_
