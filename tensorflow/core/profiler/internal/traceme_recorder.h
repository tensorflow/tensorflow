/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_TRACEME_RECORDER_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_TRACEME_RECORDER_H_

#include <atomic>
#include <vector>
#include "absl/base/optimization.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace profiler {

namespace internal {
extern std::atomic<int> g_trace_level;
}  // namespace internal

// TraceMeRecorder is a singleton repository of TraceMe events.
// It can be safely and cheaply appended to by multiple threads.
//
// Start() and Stop() must be called in pairs, Stop() returns the events added
// since the previous Start().
//
// This is the backend for TraceMe instrumentation.
// The profiler starts the recorder, the TraceMe constructor records begin
// events, and the destructor records end events.
// The profiler then stops the recorder and finds start/end pairs. (Unpaired
// start/end events are discarded at that point).
class TraceMeRecorder {
 public:
  // An Event is either the start of a TraceMe, the end of a TraceMe, or both.
  // Times are in ns since the Unix epoch.
  struct Event {
    uint64 activity_id;
    string name;
    uint64 start_time;  // 0 = missing
    uint64 end_time;    // 0 = missing
  };
  struct ThreadInfo {
    int64 tid;
    string name;
  };
  struct ThreadEvents {
    const ThreadInfo thread;
    std::vector<Event> events;
  };
  using Events = std::vector<ThreadEvents>;

  // Starts recording of TraceMe().
  // Only traces <= level will be recorded.
  // Level must be >= 0.
  // If level is 0, no traces will be recorded.
  static bool Start(int level);

  // Stops recording and returns events recorded since Start().
  static Events Stop();

  // Returns events recorded till now without stopping the recording. Empty
  // container is returned if the recorder was already stopped.
  static Events Collect();

  // Returns whether we're currently recording. Racy, but cheap!
  static inline bool Active(int level = 1) {
    return ABSL_PREDICT_FALSE(
        internal::g_trace_level.load(std::memory_order_acquire) >= level);
  }

  static void Record(Event);

 private:
  // No copy and assignment
  TraceMeRecorder(const TraceMeRecorder&) = delete;
  TraceMeRecorder& operator=(const TraceMeRecorder&) = delete;

  // Implementation of g_trace_level must be lock-free for faster execution
  // of the TraceMe() public API. This can be commented (if compilation is
  // failing) but execution might be slow (even when host tracing is disabled).
  static_assert(ATOMIC_INT_LOCK_FREE == 2, "Assumed atomic<int> was lock free");
};

}  // namespace profiler
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_TRACEME_RECORDER_H_
