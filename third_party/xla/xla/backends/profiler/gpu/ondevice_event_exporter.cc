/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/profiler/gpu/ondevice_event_exporter.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/profiler/gpu/cupti_buffer_events.h"
#include "xla/tsl/profiler/backends/gpu/ondevice_trace_event.h"
#include "xla/tsl/profiler/utils/lock_free_queue.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

using ::tensorflow::profiler::XPlane;
using ::tensorflow::profiler::XSpace;
using ::tsl::profiler::GpuOnDeviceTraceEvent;
using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XEventMetadata;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlaneBuilder;

namespace xla {
namespace profiler {
namespace {

class EventQueueWithStringSilo {
 public:
  using EventQueue = ::tsl::profiler::BlockedQueue<GpuOnDeviceTraceEvent>;

  void Clear() {
    absl::MutexLock lock(m_);
    string_silo_.Clear();
    events_.Clear();
  }

  void AddEvent(GpuOnDeviceTraceEvent&& event);

  absl::flat_hash_map<int64_t, EventQueue> GroupPerInstanceEvents();

 private:
  absl::Mutex m_;
  StringDeduper string_silo_ ABSL_GUARDED_BY(m_);
  EventQueue events_ ABSL_GUARDED_BY(m_);
};

using EventQueue = EventQueueWithStringSilo::EventQueue;

void EventQueueWithStringSilo::AddEvent(GpuOnDeviceTraceEvent&& event) {
  absl::MutexLock lock(m_);
  event.tag_name = string_silo_.Dedup(event.tag_name);
  events_.Push(std::move(event));
}

absl::flat_hash_map<int64_t, EventQueue>
EventQueueWithStringSilo::GroupPerInstanceEvents() {
  absl::flat_hash_map<int64_t, EventQueue> grouped;
  // Note: after GroupPerInstanceEvents, the events_ is empty.
  absl::MutexLock lock(m_);
  for (std::optional<GpuOnDeviceTraceEvent> event = events_.Pop();
       event.has_value(); event = events_.Pop()) {
    grouped[event->injection_instance_id].Push(std::move(event.value()));
  }
  return grouped;
}

class OndeviceLineIdAllocator {
 public:
  int64_t operator()(uint32_t pid, uint32_t tid) {
    uint64_t key = Key(pid, tid);
    auto it = pid_tid_to_line_id_.find(key);
    if (it != pid_tid_to_line_id_.end()) {
      return it->second;
    }
    int64_t line_id = pid_tid_to_line_id_.size();
    pid_tid_to_line_id_[key] = line_id;
    return line_id;
  }

 private:
  static inline uint64_t Key(uint32_t pid, uint32_t tid) {
    return static_cast<uint64_t>(pid) << 32 | tid;
  }

  absl::flat_hash_map<uint64_t, int64_t> pid_tid_to_line_id_;
};

class GpuOnDeviceTraceEventCollectorImpl final
    : public GpuOnDeviceTraceEventExporter {
 public:
  GpuOnDeviceTraceEventCollectorImpl(
      const GpuOnDeviceTraceEventCollectorOptions& options,
      uint64_t start_walltime_ns, uint64_t start_gputime_ns)
      : options_(options), start_gputime_ns_(start_gputime_ns) {}

  ~GpuOnDeviceTraceEventCollectorImpl() override = default;

  absl::Status AddEvent(GpuOnDeviceTraceEvent&& event) override;

  absl::Status Export(XSpace* space, uint64_t end_gpu_ns) override;

 private:
  const GpuOnDeviceTraceEventCollectorOptions& options() const {
    return options_;
  };

  void CreateXEvent(GpuOnDeviceTraceEvent& event, XPlaneBuilder* plane,
                    XLineBuilder* line, int64_t adjust_time_ns);

  size_t FlushPerInstanceEvents(uint32_t instance_id, uint64_t end_gpu_ns,
                                XSpace* space);

  const GpuOnDeviceTraceEventCollectorOptions options_;
  uint64_t start_gputime_ns_;
  EventQueueWithStringSilo all_events_ = {};
  absl::flat_hash_map<int64_t, EventQueue> instance_events_ = {};
};

void GpuOnDeviceTraceEventCollectorImpl::CreateXEvent(
    GpuOnDeviceTraceEvent& event, XPlaneBuilder* plane, XLineBuilder* line,
    int64_t adjust_time_ns) {
  XEventMetadata* event_metadata =
      plane->GetOrCreateEventMetadata(event.tag_name);
  XEventBuilder xevent = line->AddEvent(*event_metadata);
  xevent.SetTimestampNs(event.start_time_ns + adjust_time_ns);
  xevent.SetDurationPs(static_cast<int64_t>(event.duration_ps));
}

size_t GpuOnDeviceTraceEventCollectorImpl::FlushPerInstanceEvents(
    uint32_t instance_id, uint64_t end_gpu_ns, XSpace* space) {
  // Create a plane for each instance.
  if (instance_id >= ::tsl::profiler::kNumGpuOnDeviceCustomPlanesPerHost) {
    LOG(WARNING) << "Instance id " << instance_id
                 << " is larger than kNumGpuOnDeviceCustomPlanesPerHost("
                 << ::tsl::profiler::kNumGpuOnDeviceCustomPlanesPerHost << ")";
    return 0;
  }

  auto& events = instance_events_[instance_id];
  // As uint32 is used for the mosaic event time, no absolute time could be
  // calculated. So we use the earliest time stamp for this instance to align
  // with the collector start time. Note such alignment only happens when the
  // earliest time less than the collector start time.
  int64_t earliest_time_ns = std::numeric_limits<int64_t>::max();
  for (auto it = events.begin(), ite = events.end(); it != ite; ++it) {
    earliest_time_ns = std::min(earliest_time_ns, it->start_time_ns);
  }
  int64_t adjust_time_ns =
      static_cast<int64_t>(start_gputime_ns_) <= earliest_time_ns
          ? 0
          : static_cast<int64_t>(start_gputime_ns_) - earliest_time_ns;

  std::string plane_name =
      ::tsl::profiler::GpuOnDeviceTracePlaneName(instance_id);
  XPlane* plane =
      ::tsl::profiler::FindOrAddMutablePlaneWithName(space, plane_name);
  plane->set_id(::tsl::profiler::kFirstGpuOnDeviceCustomPlaneId + instance_id);
  XPlaneBuilder plane_builder(plane);

  size_t num_events = 0;
  OndeviceLineIdAllocator line_id_allocator;
  for (auto it = events.begin(), ite = events.end(); it != ite; ++it) {
    auto& event = *it;
    int64_t line_id = line_id_allocator(event.pid, event.tid);
    XLineBuilder line = plane_builder.GetOrCreateLine(line_id);
    if (line.Name().empty()) {
      line.SetName(absl::StrFormat("PID#%9u, TID#%9u", it->pid, it->tid));
      line.SetTimestampNs(start_gputime_ns_);
    }
    CreateXEvent(event, &plane_builder, &line, adjust_time_ns);
    num_events++;
  }
  events.Clear();
  return num_events;
}

absl::Status GpuOnDeviceTraceEventCollectorImpl::Export(XSpace* space,
                                                        uint64_t end_gpu_ns) {
  instance_events_ = all_events_.GroupPerInstanceEvents();
  for (auto& [instance_id, events] : instance_events_) {
    FlushPerInstanceEvents(instance_id, end_gpu_ns, space);
  }
  return absl::OkStatus();
}

absl::Status GpuOnDeviceTraceEventCollectorImpl::AddEvent(
    GpuOnDeviceTraceEvent&& event) {
  if (event.injection_instance_id > options_.max_injection_instance) {
    LOG_FIRST_N(WARNING, 32) << "Injection instance id "
                             << event.injection_instance_id << " is too large.";
    return absl::InvalidArgumentError("Injection instance id is too large.");
  }
  if (event.pid >= options_.max_pid && options_.max_pid > 0) {
    LOG_FIRST_N(WARNING, 32) << "Pid " << event.pid << " is too large.";
    return absl::InvalidArgumentError("Pid is too large.");
  }
  if (event.tid >= options_.max_tid && options_.max_tid > 0) {
    LOG_FIRST_N(WARNING, 32) << "Tid " << event.tid << " is too large.";
    return absl::InvalidArgumentError("Tid is too large.");
  }
  all_events_.AddEvent(std::move(event));
  return absl::OkStatus();
}

}  // namespace

std::unique_ptr<GpuOnDeviceTraceEventExporter>
CreateGpuOnDeviceTraceEventExporter(
    const GpuOnDeviceTraceEventCollectorOptions& options,
    uint64_t start_walltime_ns, uint64_t start_gputime_ns) {
  return std::make_unique<GpuOnDeviceTraceEventCollectorImpl>(
      options, start_walltime_ns, start_gputime_ns);
}

}  // namespace profiler
}  // namespace xla
