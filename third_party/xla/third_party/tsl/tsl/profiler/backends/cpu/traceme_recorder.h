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
#ifndef TENSORFLOW_TSL_PROFILER_BACKENDS_CPU_TRACEME_RECORDER_H_
#define TENSORFLOW_TSL_PROFILER_BACKENDS_CPU_TRACEME_RECORDER_H_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace tsl {
namespace profiler {

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
    int32_t tid;
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

  // Stops recording and returns events recorded since Start().
  // Events passed to Record after Stop has started will be dropped.
  static Events Stop();

  // Returns whether we're currently recording. Racy, but cheap!
  static inline bool Active(int level = 1) {
    return trace_level_.Get() >= level;
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

  class TraceLevel {
   public:
    TraceLevel() : level_(kTracingDisabled) {}

    // Returns the current trace level.
    int Get() const { return level_.load(std::memory_order_acquire); }

    // Sets the current trace level to the given level if tracing is
    // currently stopped.
    // If the given level is negative, it is treated as zero.
    // Does nothing and returns false if tracing is already started.
    bool Set(int level) {
      int expected = kTracingDisabled;
      return level_.compare_exchange_strong(expected, std::max(0, level),
                                            std::memory_order_acq_rel);
    }

    // Clears the current trace level to stop tracing.
    // Returns false if tracing is already stopped.
    bool Clear() {
      int level = level_.exchange(kTracingDisabled, std::memory_order_acq_rel);
      return level != kTracingDisabled;
    }

   private:
    // Current trace level.
    std::atomic<int> level_;

    // TraceLevel implementation must be lock-free for faster execution of the
    // TraceMe API. This can be commented (if compilation is failing) but
    // execution might be slow (even when tracing is disabled).
    static_assert(std::atomic<int>::is_always_lock_free,
                  "Assumed atomic<int> was lock free");
  };

  static inline TraceLevel trace_level_;
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_BACKENDS_CPU_TRACEME_RECORDER_H_
