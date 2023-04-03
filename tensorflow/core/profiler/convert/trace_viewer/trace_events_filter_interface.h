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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_EVENTS_FILTER_INTERFACE_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_EVENTS_FILTER_INTERFACE_H_

namespace tensorflow {
namespace profiler {

// Trace event filter interface.
template <typename Trace, typename TraceEvent>
class TraceEventsFilterInterface {
 public:
  virtual ~TraceEventsFilterInterface() = default;

  // Allow sub-classes to set up filtering by processing the trace, e.g., by
  // capturing the names of devices and resources that need to be filtered.
  virtual void SetUp(const Trace& trace) = 0;

  // Returns true if event should not be added to a TraceEventsContainer.
  virtual bool Filter(const TraceEvent& event) = 0;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_EVENTS_FILTER_INTERFACE_H_
