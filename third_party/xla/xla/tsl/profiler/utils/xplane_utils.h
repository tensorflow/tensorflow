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
#ifndef XLA_TSL_PROFILER_UTILS_XPLANE_UTILS_H_
#define XLA_TSL_PROFILER_UTILS_XPLANE_UTILS_H_

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

// Returns a Timespan from an XEvent.
// WARNING: This should only be used when comparing events from the same XLine.
inline Timespan XEventTimespan(const XEvent& event) {
  return Timespan(event.offset_ps(), event.duration_ps());
}

// Returns the planes with the given predicate.
template <typename F>
std::vector<const XPlane*> FindPlanes(const XSpace& space, const F& predicate) {
  std::vector<const XPlane*> result;
  for (const XPlane& plane : space.planes()) {
    if (predicate(plane)) {
      result.push_back(&plane);
    }
  }
  return result;
}

// Returns mutable planes with the given predicate.
template <typename F>
std::vector<XPlane*> FindMutablePlanes(XSpace* space, const F& predicate) {
  std::vector<XPlane*> result;
  for (XPlane& plane : *space->mutable_planes()) {
    if (predicate(plane)) {
      result.push_back(&plane);
    }
  }
  return result;
}

// Returns the plane with the given name or nullptr if not found.
const XPlane* FindPlaneWithName(const XSpace& space, absl::string_view name);
XPlane* FindMutablePlaneWithName(XSpace* space, absl::string_view name);

// Returns the planes with the given names, if found.
std::vector<const XPlane*> FindPlanesWithNames(
    const XSpace& space, const std::vector<absl::string_view>& names);

// Returns the plane with the given name in the container. If necessary, adds a
// new plane to the container.
XPlane* FindOrAddMutablePlaneWithName(XSpace* space, absl::string_view name);

// Returns all the planes with a given prefix.
std::vector<const XPlane*> FindPlanesWithPrefix(const XSpace& space,
                                                absl::string_view prefix);
std::vector<XPlane*> FindMutablePlanesWithPrefix(XSpace* space,
                                                 absl::string_view prefix);

// Returns the plane with the given id/name or nullptr if not found.
const XLine* FindLineWithId(const XPlane& plane, int64_t id);
std::vector<const XLine*> FindLinesWithId(const XPlane& plane, int64_t id);
const XLine* FindLineWithName(const XPlane& plane, absl::string_view name);

XStat* FindOrAddMutableStat(const XStatMetadata& stat_metadata, XEvent* event);

void RemovePlane(XSpace* space, const XPlane* plane);
void RemovePlanes(XSpace* space, const std::vector<const XPlane*>& planes);
void RemoveLine(XPlane* plane, const XLine* line);
void RemoveEvents(XLine* line,
                  const absl::flat_hash_set<const XEvent*>& events);

void RemoveEmptyPlanes(XSpace* space);
void RemoveEmptyLines(XPlane* plane);

// Sort lines in plane with a provided comparator.
template <class Compare>
void SortXLinesBy(XPlane* plane, Compare comp) {
  std::sort(plane->mutable_lines()->pointer_begin(),
            plane->mutable_lines()->pointer_end(), comp);
}

class XLinesComparatorByName {
 public:
  bool operator()(const XLine* a, const XLine* b) const {
    auto& line_a = a->display_name().empty() ? a->name() : a->display_name();
    auto& line_b = b->display_name().empty() ? b->name() : b->display_name();
    return line_a < line_b;
  }
};

// Sorts each XLine's XEvents by offset_ps (ascending) and duration_ps
// (descending) so nested events are sorted from outer to innermost.
void SortXPlane(XPlane* plane);
// Sorts each plane of the XSpace.
void SortXSpace(XSpace* space);

// Functor that compares XEvents for sorting by timespan.
struct XEventsComparator {
  bool operator()(const XEvent* a, const XEvent* b) const;
};

// Returns a sorted vector of all XEvents in the given XPlane.
// This template can be used with either XPlaneVisitor or XPlaneBuilder.
// If line_ids is empty, all lines could be used to collect events. Otherwise,
// only lines whose id exists in the line_ids will be used to collect events.
template <typename Event, typename Plane>
std::vector<Event> GetSortedEvents(Plane& plane,
                                   bool include_derived_events = false,
                                   absl::Span<const int64_t> line_ids = {}) {
  std::vector<Event> events;
  plane.ForEachLine([&events, include_derived_events, line_ids](auto line) {
    if (!include_derived_events && IsDerivedThreadId(line.Id())) return;
    if (!line_ids.empty() && std::find(line_ids.begin(), line_ids.end(),
                                       line.Id()) == line_ids.end())
      return;
    line.ForEachEvent(
        [&events](auto event) { events.emplace_back(std::move(event)); });
  });
  absl::c_sort(events);
  return events;
}

// Normalize timestamps by time-shifting to start_time_ns_ as origin.
void NormalizeTimestamps(XPlane* plane, uint64 start_time_ns);
void NormalizeTimestamps(XSpace* space, uint64 start_time_ns);

// Merges src_plane into dst_plane. Both plane level stats, lines, events and
// event level stats are merged. If src_plane and dst_plane both have the same
// line, which have different start timestamps, we will normalize the events
// offset timestamp correspondingly.
void MergePlanes(const XPlane& src_plane, XPlane* dst_plane);

// Merges each plane with a src_planes, into the dst_plane.
void MergePlanes(const std::vector<const XPlane*>& src_planes,
                 XPlane* dst_plane);

// Plane's start timestamp is defined as the minimum of all lines' start
// timestamps. If zero line exists, return 0;
int64_t GetStartTimestampNs(const XPlane& plane);

// Returns true if there are no XEvents.
bool IsEmpty(const XSpace& space);

// Return true if grouping/step-tracking is done on the Xspace already.
bool IsXSpaceGrouped(const XSpace& space);

// Mutate the XPlane by adding predefined XFlow. e.g. GPU kernel launches =>
// GPU kernel events.
void AddFlowsToXplane(int32_t host_id, bool is_host_plane, bool connect_traceme,
                      XPlane* plane);

// Get a fingerprint of device plane for deduplicating derived lines in similar
// device planes. The fingerprint is a hash of sorted HLO modules name which
// were appeared on current plane.
// Returns 0 when such "Xla Modules" line don't exist.
uint64_t GetDevicePlaneFingerprint(const XPlane& plane);
template <typename XPlanePointerIterator>
void SortPlanesById(XPlanePointerIterator begin, XPlanePointerIterator end) {
  std::sort(begin, end, [&](const XPlane* a, const XPlane* b) {
    return a->id() < b->id();  // ascending order of device xplane id.
  });
}

// When certain event context only exists from event from other line, which
// "encloses" current event in timeline, we need to find out quickly which
// enclosing event is (or if there is one).
// To Avoid O(N) search overhead, assume the event are processed in the order
// of "XLine default sorting order".
class XEventContextTracker {
 public:
  // The events on line need to be sorted and disjointed.
  XEventContextTracker(const XPlaneVisitor* plane, const XLine* line)
      : plane_(plane), line_(line) {}

  // Returns the event that encloses/contains the specified input event.
  // Expects called with events with start timestamps sorted incrementingly.
  std::optional<XEventVisitor> GetContainingEvent(const Timespan& event);

  // Returns the event that overlaps the specified input event.
  // Expects called with events with start timestamps sorted incrementingly.
  std::optional<XEventVisitor> GetOverlappingEvent(const Timespan& event);

 private:
  const XPlaneVisitor* plane_;
  const XLine* line_;
  int64_t current_index_ = -1;
};

// Aggregate traces on op_line in the full_trace xplane and add them onto the
// aggregated_trace xplane. The function also copies the step line from the
// full_trace into the aggregated_trace.
void AggregateXPlane(const XPlane& full_trace, XPlane& aggregated_trace);

// Return whether this is a custom plan.
bool IsCustomPlane(const XPlane& plane);

// Return whether this is a host plan.
bool IsHostPlane(const XPlane& plane);

// Return whether this is a device plan.
bool IsDevicePlane(const XPlane& plane);

}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_UTILS_XPLANE_UTILS_H_
