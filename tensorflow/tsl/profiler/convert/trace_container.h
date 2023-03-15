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

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
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

  // noncopyable
  TraceContainer(TraceContainer&&) = default;
  TraceContainer& operator=(TraceContainer&&) = default;
  TraceContainer(const TraceContainer&) = delete;
  TraceContainer& operator=(const TraceContainer&) = delete;

  // Returns the metadata for this trace container.
  const Trace& trace() const { return trace_; }

  const std::vector<TraceEvent*>& UnsortedEvents() const final {
    return events_;
  }

  std::vector<TraceEvent*>& YieldUnsortedEvents() { return events_; }

  // Returns a device descriptor.
  Device* MutableDevice(uint32_t device_id) {
    return &(*trace_.mutable_devices())[device_id];
  }

  TraceEvent* CreateEvent() final {
    TraceEvent* event = new TraceEvent;
    events_.push_back(event);
    return event;
  }

  // Used for testing
  bool ParseMetadataFromString(const std::string& description);

 private:
  Trace trace_;
  std::vector<TraceEvent*> events_;
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_CONVERT_TRACE_CONTAINER_H_
