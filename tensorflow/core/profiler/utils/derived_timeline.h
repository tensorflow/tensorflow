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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/group_events.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"

namespace tensorflow {
namespace profiler {

// Helper for deriving an XLine from events in another XLine.
class DerivedXLineBuilder {
 public:
  DerivedXLineBuilder(XPlaneBuilder* plane, int64 line_id,
                      absl::string_view name, int64 timestamp_ns,
                      std::vector<DerivedXLineBuilder*> dependent_lines);

  void ExpandOrAddEvents(const std::vector<XEvent>& event_per_level) {
    for (size_t level = 0; level < event_per_level.size(); ++level) {
      ExpandOrAddLevelEvent(event_per_level[level], level);
    }
  }

  void ExpandOrAddEvent(const XEvent& event) {
    ExpandOrAddLevelEvent(event, /*level=*/0);
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
  void ExpandOrAddLevelEvent(const XEvent& event, int level);

  void ResetDependentLines() {
    for (DerivedXLineBuilder* line : dependent_lines_) {
      line->ResetLastEvents();
    }
  }

  XLineBuilder line_;
  absl::flat_hash_map<int, absl::optional<XEventBuilder>> last_event_by_level_;
  std::vector<DerivedXLineBuilder*> dependent_lines_;
};

using SymbolResolver = std::function<absl::string_view(
    absl::string_view hlo_module_name, absl::string_view hlo_op)>;

// Derives TF name scope and op events from the TF op's fully qualified name.
void ProcessTfOpEvent(absl::string_view tf_op_full_name, int64 offset_ps,
                      int64 duration_ps, absl::optional<int64> group_id,
                      XPlaneBuilder* plane_builder,
                      DerivedXLineBuilder* tf_name_scope_line_builder,
                      DerivedXLineBuilder* tf_op_line_builder);

// Derives "Step Info", "Tensorflow Ops", "XLA Ops" and "XLA Module" lines in
// an NVIDIA_GPU device trace from data passed as ScopedAnnotations and stored
// as XStats in XEvents corresponding to GPU Kernels. Consecutive annotations
// with the same value are merged into a single event except for XLA modules.
// The device_trace is both input and output.
void DeriveEventsFromAnnotations(const SymbolResolver& symbol_resolver,
                                 const EventGroupNameMap& event_group_name_map,
                                 XPlane* device_trace,
                                 bool step_info_only = false);

// Derives "Launch Activities Summary" line from host trace.
void DeriveEventsFromHostTrace(const XPlane* host_trace,
                               const EventGroupNameMap& event_group_name_map,
                               std::vector<XPlane*> device_traces);

// Loops through XPlanes of input XSpace, if it is "device" XPlane, generating
// derived timelines for the plane by calling DeriveEventsFromAnnotations.
void GenerateDerivedTimeLines(const EventGroupNameMap& event_group_name_map,
                              XSpace* space, bool step_info_only = false);
void GenerateDerivedTimeLines(const EventGroupNameMap& event_group_name_map,
                              const std::vector<XPlane*>& device_traces,
                              bool step_info_only = false);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_DERIVED_TIMELINE_H_
