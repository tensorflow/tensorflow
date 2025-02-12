/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_EVENTS_UTIL_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_EVENTS_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"

namespace tensorflow {
namespace profiler {

// Returns the resource name for the given (device_id, resource_id) in trace.
inline absl::string_view ResourceName(const Trace& trace, uint32_t device_id,
                                      uint32_t resource_id) {
  return trace.devices().at(device_id).resources().at(resource_id).name();
}

// Returns the resource name for the given event in trace.
inline absl::string_view ResourceName(const Trace& trace,
                                      const TraceEvent& event) {
  return ResourceName(trace, event.device_id(), event.resource_id());
}

// Functor that compares trace events for sorting.
// Trace events are sorted by timestamp_ps (ascending) and duration_ps
// (descending) so nested events are sorted from outer to innermost.
struct TraceEventsComparator {
  bool operator()(const TraceEvent* a, const TraceEvent* b) const {
    if (a->timestamp_ps() < b->timestamp_ps()) return true;
    if (a->timestamp_ps() > b->timestamp_ps()) return false;
    return (a->duration_ps() > b->duration_ps());
  }
};

// Creates a tsl::profiler::Timespan from a TraceEvent.
inline tsl::profiler::Timespan EventSpan(const TraceEvent& event) {
  return tsl::profiler::Timespan(event.timestamp_ps(), event.duration_ps());
}

// Creates a tsl::profiler::Timespan from a Trace.
inline tsl::profiler::Timespan TraceSpan(const Trace& trace) {
  return tsl::profiler::Timespan::FromEndPoints(trace.min_timestamp_ps(),
                                                trace.max_timestamp_ps());
}

// A flow of events in the trace-viewer.
// All events in the flow have the same flow_id.
using TraceEventFlow = std::vector<TraceEvent*>;

// In case the flow_id was re-used, split into individual flows based on the
// flow_entry_type.
std::vector<TraceEventFlow> SplitEventFlow(TraceEventFlow&& flow);

// Returns whether the flow is complete.
inline bool IsCompleteFlow(const TraceEventFlow& flow) {
  DCHECK(!flow.empty());
  return flow.front()->flow_entry_type() == TraceEvent::FLOW_START &&
         flow.back()->flow_entry_type() == TraceEvent::FLOW_END;
}

// Updates the timestamps of a Trace to ensure it includes the given
// tsl::profiler::Timespan.
void ExpandTraceSpan(const tsl::profiler::Timespan& span, Trace* trace);

// Nway-merge implementation.

// Reorders the elements of the range [first, last) to restore the heap
// condition (i.e. `std::is_heap(first, last, comp)`) following a change
// in the value of `*first`.
//
// REQUIRES: `first < last`, and [first, last) would be a valid heap if `*first`
// had a suitable value.
template <typename RandIt, typename Compare>
void push_down_root(RandIt first, RandIt last, Compare comp) {
  size_t size = last - first;
  size_t hole = 0;  // root.
  auto value = std::move(*first);
  while (true) {
    size_t l_child = 2 * hole + 1;
    size_t r_child = l_child + 1;
    size_t max_child = l_child;
    if (r_child < size && comp(first[l_child], first[r_child])) {
      max_child = r_child;
    }
    if (max_child >= size) break;
    if (!comp(value, first[max_child])) break;
    first[hole] = std::move(first[max_child]);
    hole = max_child;
  }
  first[hole] = std::move(value);
}

// ContainerContainer could be a container of pointers to container.
template <typename ContainerContainer, typename Out, typename Cmp>
Out nway_merge(const ContainerContainer& containers, Out out, Cmp cmp) {
  using std::begin;
  using std::end;
  using In = decltype(begin(**begin(containers)));  // The input iterator type.
  using Range = std::pair<In, In>;
  std::vector<Range> sources;
  for (const auto& container : containers) {
    Range r(begin(*container), end(*container));
    if (r.first != r.second) {
      sources.push_back(r);
    }
  }
  if (sources.empty()) return out;
  // Take a comparator for T and produce an inverse comparator
  // for std::pair<In<T>, In<T>>, inverted so as to produce a min-heap.
  auto heap_cmp = [&](const Range& a, const Range& b) {
    // Compares b < a instead of a < b.
    return cmp(*b.first, *a.first);
  };
  std::make_heap(sources.begin(), sources.end(), heap_cmp);
  while (true) {
    Range& r = sources.front();
    *out = *r.first;
    ++r.first;
    ++out;
    if (r.first == r.second) {
      if (sources.size() == 1) return out;
      r = std::move(sources.back());
      sources.pop_back();
    }
    push_down_root(sources.begin(), sources.end(), heap_cmp);
  }
}

// Interface that allows defining classes that map XLines within a single XPlane
// to multiple virtual devices in trace viewer.
class ResourceGrouperInterface {
 public:
  virtual ~ResourceGrouperInterface() = default;

  virtual std::vector<std::pair<uint32_t /*resource_id*/, absl::string_view>>
  Devices() const = 0;

  virtual uint32_t GetDeviceId(uint32_t resource_id) const = 0;
};

std::unique_ptr<ResourceGrouperInterface> CreateDefaultResourceGrouper(
    uint32_t device_id, absl::string_view name);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_EVENTS_UTIL_H_
