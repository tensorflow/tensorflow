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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_EVENTS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_EVENTS_H_

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/bind_front.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/profiler/convert/trace_viewer/trace_events_filter_interface.h"
#include "tensorflow/core/profiler/convert/trace_viewer/trace_events_util.h"
#include "tensorflow/core/profiler/convert/trace_viewer/trace_viewer_visibility.h"
#include "tensorflow/core/profiler/lib/context_types.h"
#include "tensorflow/core/profiler/protobuf/task.pb.h"
#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"
#include "tensorflow/core/profiler/utils/timespan.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
namespace profiler {

// A track of events in the trace-viewer.
using TraceEventTrack = std::vector<TraceEvent*>;

// Merge-sorts the given event tracks. Each track must be sorted.
std::vector<TraceEvent*> MergeEventTracks(
    const std::vector<const TraceEventTrack*>& event_tracks);

tsl::Status DoStoreAsLevelDbTable(
    const std::string& filename, const Trace& trace,
    const std::vector<std::vector<const TraceEvent*>>& events_by_level);

tsl::Status DoLoadFromLevelDbTable(
    const std::string& filename,
    std::unique_ptr<TraceEventsFilterInterface> filter,
    std::unique_ptr<TraceVisibilityFilter> visibility,
    int64_t filter_by_visibility_threshold, Trace& trace,
    bool& filter_by_visibility,
    const std::function<TraceEvent*(const TraceEvent&)>& copy_event_to_arena,
    const std::function<void(TraceEvent*)>& add_arena_event);

std::vector<std::vector<const TraceEvent*>> GetEventsByLevel(
    const Trace& trace, std::vector<TraceEvent*>& events);

struct EventFactory {
  TraceEvent* Create() {
    events.push_back(std::make_unique<TraceEvent>());
    return events.back().get();
  }
  std::vector<std::unique_ptr<TraceEvent>> events;
};

template <typename EventFactory, typename RawData,
          typename Hash = std::hash<absl::string_view>()>
class TraceEventsContainer {
 public:
  TraceEventsContainer() { arenas_.insert(std::make_shared<EventFactory>()); }

  // Movable but non-copyable.
  TraceEventsContainer(TraceEventsContainer&&) = default;
  TraceEventsContainer& operator=(TraceEventsContainer&&) = default;
  TraceEventsContainer(const TraceEventsContainer&) = delete;
  TraceEventsContainer& operator=(const TraceEventsContainer&) = delete;

  // Creates a TraceEvent prefilled with the given values.
  void AddCompleteEvent(absl::string_view name, uint32_t resource_id,
                        uint32_t device_id, Timespan timespan,
                        RawData* raw_data = nullptr,
                        std::optional<int64_t> group_id = std::nullopt) {
    TraceEvent* event = CreateArenaEvent();
    MaybeInternEventName(event, name);
    event->set_resource_id(resource_id);
    event->set_device_id(device_id);
    event->set_timestamp_ps(timespan.begin_ps());
    if (timespan.duration_ps() != 0) {
      event->set_duration_ps(timespan.duration_ps());
    }
    if (raw_data) {
      MaybeInternTraceArgument(raw_data);
      raw_data->SerializePartialToString(event->mutable_raw_data());
      if (event->raw_data().empty()) event->clear_raw_data();
    }
    if (group_id) {
      event->set_group_id(*group_id);
    }
    AddArenaEvent(event);
  }

  // Similar to above, but the TraceEvent also has an associated flow_id and
  // flow_entry_type, to make it part of a flow.
  void AddFlowEvent(absl::string_view name, uint32_t resource_id,
                    uint32_t device_id, Timespan timespan, uint64_t flow_id,
                    TraceEvent::FlowEntryType flow_entry_type,
                    ContextType flow_category = ContextType::kGeneric,
                    RawData* raw_data = nullptr,
                    std::optional<int64_t> group_id = std::nullopt) {
    TraceEvent* event = CreateArenaEvent();
    MaybeInternEventName(event, name);
    event->set_resource_id(resource_id);
    event->set_device_id(device_id);
    event->set_timestamp_ps(timespan.begin_ps());
    if (timespan.duration_ps() != 0) {
      event->set_duration_ps(timespan.duration_ps());
    }
    event->set_flow_id(flow_id);
    event->set_flow_entry_type(flow_entry_type);
    event->set_flow_category(static_cast<uint32_t>(flow_category));
    if (raw_data) {
      MaybeInternTraceArgument(raw_data);
      raw_data->SerializePartialToString(event->mutable_raw_data());
      if (event->raw_data().empty()) event->clear_raw_data();
    }
    if (group_id) {
      event->set_group_id(*group_id);
    }
    AddArenaEvent(event);
  }

  // Similar to above, but the "async" TraceEvent don't have a resource id, its
  // name is used as "async channel" which are used as "thread" name. It has an
  // associated unique flow_id and flow_entry_type to signal asynchronous
  // start and end events and match up between them.
  void AddAsyncEvent(absl::string_view name, uint32_t device_id,
                     Timespan timespan, uint64_t flow_id,
                     TraceEvent::FlowEntryType flow_entry_type,
                     ContextType flow_category = ContextType::kGeneric,
                     RawData* raw_data = nullptr,
                     std::optional<int64_t> group_id = std::nullopt) {
    TraceEvent* event = CreateArenaEvent();
    MaybeInternEventName(event, name);
    event->set_device_id(device_id);
    event->set_timestamp_ps(timespan.begin_ps());
    if (timespan.duration_ps() != 0) {
      event->set_duration_ps(timespan.duration_ps());
    }
    event->set_flow_id(flow_id);
    event->set_flow_entry_type(flow_entry_type);
    event->set_flow_category(static_cast<uint32_t>(flow_category));
    if (raw_data) {
      MaybeInternTraceArgument(raw_data);
      raw_data->SerializePartialToString(event->mutable_raw_data());
      if (event->raw_data().empty()) event->clear_raw_data();
    }
    if (group_id) {
      event->set_group_id(*group_id);
    }
    AddArenaEvent(event);
  }

  // Similar to above, but the TraceEvent also has an associated counter name
  // and value in RawData.args. Counter events are per device, so no resource_id
  // is passed.
  void AddCounterEvent(absl::string_view name, uint32_t device_id,
                       uint64_t timestamp_ps, const RawData& raw_data) {
    TraceEvent* event = CreateArenaEvent();
    event->set_name(name);
    event->set_device_id(device_id);
    // Do not set resource_id for counter events, they are per device.
    event->set_timestamp_ps(timestamp_ps);
    DCHECK(raw_data.has_args());
    DCHECK_EQ(raw_data.args().arg_size(), 1);
    DCHECK(raw_data.args().arg(0).has_uint_value());
    raw_data.SerializePartialToString(event->mutable_raw_data());
    AddArenaEvent(event);
  }

  // Returns a device descriptor.
  Device* MutableDevice(uint32_t device_id) {
    return &(*trace_.mutable_devices())[device_id];
  }

  // Returns a resource descriptor,
  Resource* MutableResource(uint32_t resource_id, uint32_t device_id) {
    Device* device = MutableDevice(device_id);
    return &(*device->mutable_resources())[resource_id];
  }

  // Adds metadata events to set the name of each device and resource.
  // The arguments are callbacks that return the names given ids.
  // This must be called after all AddEvent calls, and no more AddEvent
  // calls should be made after calling AddMetadataEvents.
  void AddMetadataEvents(
      const std::function<std::string(uint32_t /*device_id*/)>& device_name,
      const std::function<std::string(
          uint32_t /*device_id*/, uint32_t /*resource_id*/)>& resource_name) {
    for (const auto& id_and_device : events_by_device_) {
      uint32_t device_id = id_and_device.first;
      auto& device = (*trace_.mutable_devices())[device_id];
      device.set_device_id(device_id);
      device.set_name(device_name(device_id));
      const DeviceEvents& device_events = id_and_device.second;
      for (const auto& id_and_resource : device_events.events_by_resource) {
        uint32_t resource_id = id_and_resource.first;
        auto& resource = (*device.mutable_resources())[resource_id];
        resource.set_resource_id(resource_id);
        resource.set_name(resource_name(device_id, resource_id));
        resource.set_num_events(id_and_resource.second.size());
      }
    }
  }

  // Adds task metadata for the given host.
  void AddTask(int host_id, const Task& task) {
    (*trace_.mutable_tasks())[host_id] = task;
  }

  // Stores the contents of this container in a level-db sstable file.
  tsl::Status StoreAsLevelDbTable(const std::string& filename) const {
    Trace trace = trace_;
    trace.set_num_events(NumEvents());
    auto events_by_level = EventsByLevel();
    return DoStoreAsLevelDbTable(filename, trace, events_by_level);
  }

  // Loads the contents of this container from a level-db sstable file.
  // In order to be efficient, requires resolution__ to be set.
  // If span_ is not set, it is initialized from the loaded trace_.
  tsl::Status LoadFromLevelDbTable(
      const std::string& filename,
      std::unique_ptr<TraceEventsFilterInterface> filter = nullptr,
      std::unique_ptr<TraceVisibilityFilter> visibility = nullptr,
      int64_t filter_by_visibility_threshold = -1LL) {
    return DoLoadFromLevelDbTable(
        filename, std::move(filter), std::move(visibility),
        filter_by_visibility_threshold, trace_, filter_by_visibility_,
        absl::bind_front(&TraceEventsContainer::CopyEventToArena, this),
        absl::bind_front(&TraceEventsContainer::AddArenaEvent, this));
  }

  // Calls 'callback' with all events stored in this container.
  template <typename Callback>
  void ForAllEvents(Callback callback) const {
    for (const auto& [device_id, device] : events_by_device_) {
      for (const auto& [counter_name, events] : device.counter_events_by_name) {
        for (auto* event : events) {
          callback(*event);
        }
      }
      for (const auto& [resource_id, events] : device.events_by_resource) {
        for (auto* event : events) {
          callback(*event);
        }
      }
    }
  }

  // Calls 'callback' with all event tracks stored in this container.
  template <typename Callback>
  void ForAllTracks(Callback callback) const {
    for (const auto& [device_id, device] : events_by_device_) {
      for (const auto& [counter_name, events] : device.counter_events_by_name) {
        if (!events.empty()) {
          if (ABSL_PREDICT_FALSE(!callback(device_id, counter_name, events)))
            return;
        }
      }
      for (const auto& [resource_id, events] : device.events_by_resource) {
        if (!events.empty()) {
          if (ABSL_PREDICT_FALSE(!callback(device_id, resource_id, events)))
            return;
        }
      }
    }
  }

  // Calls 'callback' with all event tracks stored in this container.
  template <typename Callback>
  void ForAllMutableTracks(Callback callback) const {
    for (auto& [device_id, device] : events_by_device_) {
      for (auto& [counter_name, events] : device.counter_events_by_name) {
        if (!events.empty()) {
          callback(device_id, counter_name, &events);
        }
      }
      for (auto& [resource_id, events] : device.events_by_resource) {
        if (!events.empty()) {
          callback(device_id, resource_id, &events);
        }
      }
    }
  }

  // Calls 'callback' with all event flows stored in this container.
  template <typename Callback>
  void ForAllFlows(Callback callback) const {
    absl::flat_hash_map<uint64_t /*flow_id*/, TraceEventFlow> flows;
    for (const auto& [device_id, device] : events_by_device_) {
      // Counter events are not flow events.
      for (const auto& [resource_id, events] : device.events_by_resource) {
        for (auto* event : events) {
          if (event->has_flow_id()) flows[event->flow_id()].push_back(event);
        }
      }
    }
    for (auto& [flow_id, combined_flow] : flows) {
      // If the flow_id is reused, split into individual flows.
      for (auto& flow : SplitEventFlow(std::move(combined_flow))) {
        callback(flow_id, flow);
      }
    }
  }

  // Returns the metadata for this trace container.
  const Trace& trace() const { return trace_; }

  // Returns the number of events.
  size_t NumEvents() const {
    size_t count = 0;
    for (const auto& [device_id, device] : events_by_device_) {
      for (const auto& [counter_name, events] : device.counter_events_by_name) {
        count += events.size();
      }
      for (const auto& [resource_id, events] : device.events_by_resource) {
        count += events.size();
      }
    }
    return count;
  }

  // Returns the number of tracks.
  size_t NumTracks() const {
    size_t num_tracks = 0;
    for (auto& [device_id, device] : events_by_device_) {
      num_tracks += device.counter_events_by_name.size() +
                    device.events_by_resource.size();
    }
    return num_tracks;
  }

  bool FilterByVisibility() const { return filter_by_visibility_; }

 protected:
  // Allocates an event in the first of the arenas_.
  TraceEvent* CreateArenaEvent() { return (*arenas_.begin())->Create(); }

  // Copies event into arenas_.
  TraceEvent* CopyEventToArena(const TraceEvent& event) {
    TraceEvent* copy = CreateArenaEvent();
    *copy = event;
    return copy;
  }

  // Adds an event from arenas_ to events_by_device_.
  void AddArenaEvent(TraceEvent* event) {
    ExpandTraceSpan(EventSpan(*event), &trace_);
    DeviceEvents& device_events = events_by_device_[event->device_id()];
    if (!event->has_resource_id()) {
      device_events.counter_events_by_name[event->name()].push_back(event);
    } else {
      device_events.events_by_resource[event->resource_id()].push_back(event);
    }
  }

  // Returns all events grouped by visibility level.
  std::vector<std::vector<const TraceEvent*>> EventsByLevel() const {
    std::vector<TraceEvent*> events = SortedEvents();
    return GetEventsByLevel(trace_, events);
  }

  // Returns all events sorted using TraceEventsComparator.
  // Helper for EventsByLevel().
  // REQUIRED: All events have been added and SortTracks() has been called.
  std::vector<TraceEvent*> SortedEvents() const {
    std::vector<const TraceEventTrack*> event_tracks;
    event_tracks.reserve(NumTracks());
    ForAllMutableTracks(
        [&event_tracks](uint32_t device_id,
                        std::variant<uint32_t, absl::string_view> resource_id,
                        TraceEventTrack* events) {
          event_tracks.push_back(events);
        });
    return MergeEventTracks(event_tracks);
  }

  uint64_t MaybeInternString(absl::string_view name) {
    uint64_t fp = hash_(name);
    auto& it = (*trace_.mutable_name_table())[fp];
    if (it.empty()) {
      it = name;
    }
    return fp;
  }

  void MaybeInternEventName(TraceEvent* event, absl::string_view name) {
    static constexpr size_t kNameInternThreshold = 32;
    if (name.size() > kNameInternThreshold) {
      event->set_name_ref(MaybeInternString(name));
    } else {
      event->set_name(name);
    }
  }

  void MaybeInternTraceArgument(RawData* raw_data) {
    if (raw_data->has_args()) {
      for (auto& arg : *raw_data->mutable_args()->mutable_arg()) {
        constexpr size_t kTraceArgInternThreshold = 16;
        if (arg.has_str_value() &&
            arg.str_value().size() > kTraceArgInternThreshold) {
          // Use name table to string intern the trace argument.
          if (arg.name() == "long_name" || arg.name() == "hlo_text") {
            // Also mark it as potential stack frame.
            arg.set_ref_value(MaybeInternString("@@" + arg.str_value()));
          } else {
            arg.set_ref_value(MaybeInternString(arg.str_value()));
          }
        }
      }
    }
  }

  // Events shown within a single device.
  struct DeviceEvents {
    // Counter events, which are per-device (don't have resource_id), and are
    // plotted in different tracks for each counter name.
    absl::flat_hash_map<std::string, TraceEventTrack> counter_events_by_name;

    // Complete events and flow events, mapped by resource_id.
    std::map<uint32_t, TraceEventTrack> events_by_resource;
  };

  // Events, mapped by device_id.
  mutable std::map<uint32_t, DeviceEvents> events_by_device_;

  // Indicator on if visibility filtering is applied or not
  // Currently skip visibility filtering only applies to ssTable
  bool filter_by_visibility_ = true;

  // The arenas containing events constructed in this container or in containers
  // that have been merged into this container.
  using Arenas = absl::flat_hash_set<std::shared_ptr<EventFactory>>;
  Arenas arenas_;

  Trace trace_;
  Hash hash_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_EVENTS_H_
