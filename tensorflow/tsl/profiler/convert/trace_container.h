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
#ifndef TENSORFLOW_TSL_PROFILER_CONVERT_TRACE_CONTAINER_H_
#define TENSORFLOW_TSL_PROFILER_CONVERT_TRACE_CONTAINER_H_

#include <string>
#include <string_view>
#include <vector>

#include "tensorflow/tsl/profiler/protobuf/trace_events.pb.h"

namespace tsl {
namespace profiler {

using tensorflow::profiler::Device;
using tensorflow::profiler::Trace;
using tensorflow::profiler::TraceEvent;

template <typename /*Comparable*/ Event>
class AnyTraceContainer {
 public:
  virtual ~AnyTraceContainer() = default;
  virtual TraceEvent* CreateEvent() = 0;
  virtual const std::vector<TraceEvent*>& UnsortedEvents() const = 0;
};

class TraceContainer : public AnyTraceContainer<TraceEvent> {
 public:
  TraceContainer() = default;
  ~TraceContainer() final {
    for (const TraceEvent* event : events_) {
      delete event;
    }
  }

  // Returns the metadata for this trace container.
  const Trace& trace() const { return metadata_; }

  const std::vector<TraceEvent*>& UnsortedEvents() const final {
    return events_;
  }

  // Caps the number of stored trace events to the specified limit,
  // keeping the `max_count` earliest trace events by timestamp
  // if there are more events than the limit. The sortedness of
  // the trace events after calling this function is currently unspecified.
  void CapEvents(uint32_t max_count);

  // Returns a device descriptor.
  Device* MutableDevice(uint32_t device_id) {
    return &(*metadata_.mutable_devices())[device_id];
  }

  // Allocates and returns a pointer to a trace event owned by this
  // container. Do not persist the pointer; it will be invalidated
  // on `FlushAndSerializeEvents(output:)`, or when the container is
  // deinitialized, whichever comes first.
  TraceEvent* CreateEvent() final {
    TraceEvent* event = new TraceEvent;
    events_.push_back(event);
    return event;
  }

  // Removes all stored trace events from the container, and serializes
  // them as a protobuf string, along with the device metadata. This
  // function does not clear the device metadata.
  void FlushAndSerializeEvents(std::string* output);

  // Used for testing
  bool ParseMetadataFromString(const std::string& description);

 private:
  Trace metadata_;
  std::vector<TraceEvent*> events_;
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_CONVERT_TRACE_CONTAINER_H_
