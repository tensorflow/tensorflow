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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_EVENT_SPAN_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_EVENT_SPAN_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {

// The various event types. Enumerations are numbered such that a bigger number
// has a higher priority than a smaller number when used in execution-time
// breakdown.
enum EventType {
  // No event associated with the time. It could be that the machine was idle or
  // executing some events which were not traced.
  UNKNOWN_TIME = 0,
  // Host is computing.
  HOST_COMPUTE = 10,
  // Host is preprocessing the data before the execution on device.
  HOST_PREPROCESS = 20,
  // Host is postprocessing the data after the execution on device.
  HOST_POSTPROCESS = 30,
  // Host is batching data (for inference).
  HOST_BATCH_FORMATION = 40,
  // Host runtime, like memory allocation and etc.
  HOST_RUNTIME = 50,
  // Host is compiling.
  HOST_COMPILE = 60,
  // Host-to-host communication.
  HOST_TO_HOST = 70,
  // Host-to-device communication.
  HOST_TO_DEVICE = 80,
  // Host is preparing to launch a computation on device.
  HOST_PREPARE = 90,
  // Assigns a smaller priority to DEVICE_COLLECTIVES than HOST_WAIT_INPUT,
  // because if an all-reduce event is overlapped with an host-wait-input event,
  // we want to count it as waiting for input.
  // Collective Ops such as All-Reduce.
  DEVICE_COLLECTIVES = 100,
  // Host is waiting for input.
  HOST_WAIT_INPUT = 110,
  // Device-to-device communication.
  DEVICE_TO_DEVICE = 120,
  // Device-to-host communication.
  DEVICE_TO_HOST = 130,
  // Device is computing with 32-bit precision.
  DEVICE_COMPUTE_32 = 140,
  // Device is computing with 16-bit precision.
  DEVICE_COMPUTE_16 = 150,
  // Device is waiting for another device.
  DEVICE_WAIT_DEVICE = 160,
  // Device is waiting for host.
  DEVICE_WAIT_HOST = 170,
  LAST_EVENT_TYPE = DEVICE_WAIT_HOST
};

// Generic event types that shown to the user.
enum GenericEventType {
  kFirstGenericEventType = 1,
  // Device is computing.
  kDeviceCompute = kFirstGenericEventType,
  // Device-to-device communication.
  kDeviceToDevice,
  // Collective Ops such as All-Reduce and NCCL.
  kDeviceCollectives,
  // Host is computing.
  kHostCompute,
  // Host is preparing to launch a computation on device.
  kHostPrepare,
  // Device waiting for input from the host.
  kInput,
  // Device sending output to the host.
  kOutput,
  // Host is compling.
  kCompile,
  // No recognized event associated with the time.
  kAllOthers,
  kLastGenericEventType = kAllOthers,
};

// Contains the type and timespan of an event.
struct EventTypeSpan {
  EventType type;  // type of this event.
  Timespan span;   // timespan of this event.
  EventTypeSpan(EventType t, Timespan s) : type(t), span(s) {}
  // Equality test.
  bool operator==(const EventTypeSpan& other) const {
    return type == other.type && span == other.span;
  }
  // Inequality test.
  bool operator!=(const EventTypeSpan& other) const {
    return !(*this == other);
  }
};

enum class StepMarkerType {
  // "TraceContext" TraceMe events.
  kExplicitHostStepMarker,
  // Identified by group_events (e.g., FunctionRun, SessionRun).
  kImplicitHostStepMarker,
  // Derived from the result of group_events. A device step marker starts with
  // the first device event of the group and ends with the last event of the
  // group.
  kDeviceStepMarker,
};

// Record of an event that is used as a step marker.
struct StepMarker {
  StepMarkerType type;
  std::string event_name;  // name of this event.
  std::string step_name;
  Timespan span;           // timespan of this event.
  StepMarker(StepMarkerType step_marker_type, absl::string_view name,
             Timespan s)
      : type(step_marker_type), event_name(name), span(s) {}
  // Equality test.
  bool operator==(const StepMarker& other) const {
    return type == other.type && event_name == other.event_name &&
           span == other.span;
  }
  // Inequality test.
  bool operator!=(const StepMarker& other) const { return !(*this == other); }
};

// Details of a step. Note that this could be the result of combining the
// StepDetails of the same step executed on different cores.
class StepDetails {
 public:
  StepDetails() : device_memory_transfers_(3) {}

  const std::vector<StepMarker>& Markers() const { return markers_; }
  const std::vector<EventTypeSpan>& Events() const { return events_; }
  const absl::flat_hash_map<uint32, AllReduceDbResult>& Collectives() const {
    return collectives_;
  }
  const std::vector<DeviceMemoryTransfer>& DeviceMemoryTransfers() const {
    return device_memory_transfers_;
  }
  // Returns the step time.
  Timespan StepTime() const;
  std::vector<StepMarker>* MutableMarkers() { return &markers_; }
  std::vector<EventTypeSpan>* MutableEvents() { return &events_; }
  absl::flat_hash_map<uint32, AllReduceDbResult>* MutableCollectives() {
    return &collectives_;
  }
  std::vector<DeviceMemoryTransfer>* MutableDeviceMemoryTransfers() {
    return &device_memory_transfers_;
  }
  // Adds a step-marker to this step.
  void AddMarker(const StepMarker& m);
  // Adds an EventTypeSpan to this step.
  void AddEvent(const EventTypeSpan& e);
  // Adds a collective op to this step.
  void AddCollectiveOpEvent(uint64 core_id, const AllReduceInfo& e);
  // Appends device memory transfer events to this step.
  // Only event type of HOST_TO_DEVICE/DEVICE_TO_DEVICE/DEVICE_TO_HOST are
  // allowed.
  void AddDeviceMemoryTransferEvent(EventType event_type,
                                    const Timespan& time_span, uint64 bytes);
  // Appends the step-markers from another step to this step.
  void AppendMarkers(const std::vector<StepMarker>& other_markers);
  // Appends the events from another step to this step.
  void AppendEvents(const std::vector<EventTypeSpan>& other_events);
  // Appends the collectives from another step to this step.
  void AppendCollectives(
      const absl::flat_hash_map<uint32, AllReduceDbResult>& collectives);
  // Accumulates the device memory transfers from another step to this step.
  void AggregateDeviceMemoryTransfers(
      const std::vector<DeviceMemoryTransfer> device_memory_transfers);
  // Returns the step name.
  std::string StepName() const { return step_name_; }
  // Sets the name of this step.
  void SetStepName(std::string step_name) { step_name_ = step_name; }
  // Equality test.
  bool operator==(const StepDetails& other) const;
  // Inequality test.
  bool operator!=(const StepDetails& other) const { return !(*this == other); }
  // Returns a string that prints the content of this object.
  std::string DebugString() const;

 private:
  // All step-markers found for marking this step in the traces. There could be
  // multiple step-markers for a single step for different reasons. One such
  // reason is that there may be one step-marker for the same step on each core;
  // so after combining the StepDetails from multiple cores, there would be
  // multiple step-markers for the same step.
  std::vector<StepMarker> markers_;
  // All events belonging to this step.
  std::vector<EventTypeSpan> events_;
  // Collective operation related events such as all-reduce etc.
  absl::flat_hash_map<uint32, AllReduceDbResult> collectives_;
  // Device memory transfers (including time and bytes involved).
  // TODO(jiesun): Consider to use IntervalSet instead of just sum up the event
  // durations.
  std::vector<DeviceMemoryTransfer> device_memory_transfers_;
  std::string step_name_;
};

// Map from step_id to the events happened in that step.
using StepEvents = absl::flat_hash_map<int64 /*step_id*/, StepDetails>;

// Equality test for StepEvents.
bool operator==(const StepEvents& a, const StepEvents& b);

// Returns the event type of the given CPU event.
EventType ClassifyCpuEvent(absl::string_view event_name, int64 correlation_id,
                           bool has_device);

// Returns the event type of the given GPU event and tensor shapes.
EventType ClassifyGpuEvent(absl::string_view event_name,
                           absl::string_view tensor_shapes);

// Returns the name of the given EventType.
std::string PrintEventType(EventType event_type);

// Returns the string of the given GenericEventType.
absl::string_view GetGenericEventTypeStr(GenericEventType event_type);

// Returns a string that prints the given EventTypeSpan.
std::string PrintEventTypeSpan(const EventTypeSpan& event_type_span);

// Returns a string that prints the given StepMarker.
std::string PrintStepMarker(const StepMarker& step_marker);

// Returns a string that prints the given StepEvents.
std::string PrintStepEvents(const StepEvents& step_events);

// Combines the src StepEvents into dst.
void CombineStepEvents(const StepEvents& src, StepEvents* dst);

// Converts from overlapped events to non-overlapped events.
std::vector<EventTypeSpan> ToNonOverlappedEvents(
    const std::vector<EventTypeSpan>& overlapped_events);

// Converts from overlapped step-events to non-overlapped step events.
StepEvents ToNonOverlappedStepEvents(const StepEvents& overlapped_step_events);

// Returns the precision stats of the given non-overlapped step events.
PrecisionStats ComputePrecisionStats(
    const StepEvents& nonoverlapped_step_events);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_EVENT_SPAN_H_
