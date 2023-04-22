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
#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_CPU_TRACEME_RECORDER_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_CPU_TRACEME_RECORDER_H_

#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace profiler {
namespace internal {

// Current trace level.
// Static atomic so TraceMeRecorder::Active can be fast and non-blocking.
// Modified by TraceMeRecorder singleton when tracing starts/stops.
TF_EXPORT extern std::atomic<int> g_trace_level;

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

    int64 ActivityId() const {
      if (IsStart()) return -end_time;
      if (IsEnd()) return -start_time;
      return 1;  // complete
    }

    std::string name;
    int64 start_time;
    int64 end_time;
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
  static bool Start(int level) { return Get()->StartRecording(level); }

  // Stops recording and returns events recorded since Start().
  // Events passed to Record after Stop has started will be dropped.
  static Events Stop() { return Get()->StopRecording(); }

  // Returns whether we're currently recording. Racy, but cheap!
  static inline bool Active(int level = 1) {
    return internal::g_trace_level.load(std::memory_order_acquire) >= level;
  }

  // Default value for trace_level_ when tracing is disabled
  static constexpr int kTracingDisabled = -1;

  // Records an event. Non-blocking.
  static void Record(Event&& event);

  // Returns an activity_id for TraceMe::ActivityStart.
  static int64 NewActivityId();

 private:
  class ThreadLocalRecorder;
  class ThreadLocalRecorderWrapper;

  // Returns singleton.
  static TraceMeRecorder* Get();

  TraceMeRecorder() = default;

  TF_DISALLOW_COPY_AND_ASSIGN(TraceMeRecorder);

  void RegisterThread(uint32 tid, std::shared_ptr<ThreadLocalRecorder> thread);
  void UnregisterThread(uint32 tid);

  bool StartRecording(int level);
  Events StopRecording();

  // Clears events from all active threads that were added due to Record
  // racing with StopRecording.
  void Clear() TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Gathers events from all active threads, and clears their buffers.
  TF_MUST_USE_RESULT Events Consume() TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  mutex mutex_;
  // A ThreadLocalRecorder stores trace events. Ownership is shared with
  // ThreadLocalRecorderWrapper, which is allocated in thread_local storage.
  // ThreadLocalRecorderWrapper creates the ThreadLocalRecorder and registers it
  // with TraceMeRecorder on the first TraceMe executed on a thread while
  // tracing is active. If the thread is destroyed during tracing, the
  // ThreadLocalRecorder is marked inactive but remains alive until tracing
  // stops so the events can be retrieved.
  absl::flat_hash_map<uint32, std::shared_ptr<ThreadLocalRecorder>> threads_
      TF_GUARDED_BY(mutex_);
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_CPU_TRACEME_RECORDER_H_
