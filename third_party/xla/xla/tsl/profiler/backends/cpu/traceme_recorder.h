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
#ifndef XLA_TSL_PROFILER_BACKENDS_CPU_TRACEME_RECORDER_H_
#define XLA_TSL_PROFILER_BACKENDS_CPU_TRACEME_RECORDER_H_

#include <sys/types.h>

#include <atomic>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "tsl/platform/macros.h"
#include "tsl/platform/types.h"

namespace tsl {
namespace profiler {

namespace internal {

// Current trace level.
// Static atomic so TraceMeRecorder::Active can be fast and non-blocking.
// Modified by TraceMeRecorder singleton when tracing starts/stops.
TF_EXPORT extern std::atomic<int> g_trace_level;
TF_EXPORT extern uint64_t g_trace_filter_bitmap;

}  // namespace internal

// TraceMeRecorder is a singleton repository of TraceMe events.
// It can be safely and cheaply appended to by multiple threads.
//
// Start() and Stop() must be called in pairs, Stop() returns the events added
// since the previous Start().
//
// This is the backend for TraceMe instrumentation.
// The profiler starts the recorder, the TraceMe destructor records complete
// events. TraceMe::ActivityStart records start events, and TraceMe::ActivityEnd
// records end events. The profiler then stops the recorder and finds start/end
// pairs. (Unpaired start/end events are discarded at that point).
class TraceMeRecorder {
 public:
  // An Event is either the start of a TraceMe, the end of a TraceMe, or both.
  // Times are in ns since the Unix epoch.
  // A negative time encodes the activity_id used to pair up the start of an
  // event with its end.
  struct Event {
    bool IsComplete() const { return start_time > 0 && end_time > 0; }
    bool IsStart() const { return end_time < 0; }
    bool IsEnd() const { return start_time < 0; }

    int64_t ActivityId() const {
      if (IsStart()) return -end_time;
      if (IsEnd()) return -start_time;
      return 1;  // complete
    }

    std::string name;
    int64_t start_time;
    int64_t end_time;
  };
  struct ThreadInfo {
    uint32 tid;
    std::string name;
  };
  struct ThreadEvents {
    ThreadInfo thread;
    std::deque<Event> events;
  };
  using Events = std::vector<ThreadEvents>;

  // Starts recording of TraceMe().
  // Only traces <= level will be recorded.
  // Level must be >= 0. If level is 0, no traces will be recorded.
  static bool Start(int level);

  // Starts recording of TraceMe() with filter.
  // Only traces <= level will be recorded.
  // Level must be >= 0. If level is 0, no traces will be recorded.
  // filter_mask is a bitmap that will be used to filter out traces during
  // recording. Filter will be applied only if record function (e.g. TraceMe,
  // ActivityStart, InstantActivity etc.) with filter_mask is called.
  static bool Start(int level, uint64_t filter_mask);

  // Stops recording and returns events recorded since Start().
  // Events passed to Record after Stop has started will be dropped.
  static Events Stop();

  // Returns whether we're currently recording. Racy, but cheap!
  static inline bool Active(int level = 1) {
    return internal::g_trace_level.load(std::memory_order_acquire) >= level;
  }

  // Returns whether the filter is enabled.
  static inline bool CheckFilter(uint64_t filter) {
    return internal::g_trace_filter_bitmap & filter;
  }

  // Default value for trace_level_ when tracing is disabled
  static constexpr int kTracingDisabled = -1;

  // Records an event. Non-blocking.
  static void Record(Event&& event);

  // Returns an activity_id for TraceMe::ActivityStart.
  static int64_t NewActivityId();

 private:
  TraceMeRecorder() = delete;
  ~TraceMeRecorder() = delete;

  // Clears events from all active threads that were added due to Record
  // racing with Stop.
  static void Clear();

  // Gathers events from all active threads, and clears their buffers.
  static TF_MUST_USE_RESULT Events Consume();
};

}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_BACKENDS_CPU_TRACEME_RECORDER_H_
