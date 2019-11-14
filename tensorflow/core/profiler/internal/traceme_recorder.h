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

#include <stddef.h>

#include <atomic>
#include <unordered_map>
#include <vector>

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
    int32 tid;
    string name;
  };
  struct ThreadEvents {
    ThreadInfo thread;
    std::vector<Event> events;
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
  static void Record(Event event);

 private:
  class ThreadLocalRecorder;

  // Returns singleton.
  static TraceMeRecorder* Get();

  TraceMeRecorder() = default;

  TF_DISALLOW_COPY_AND_ASSIGN(TraceMeRecorder);

  void RegisterThread(int32 tid, ThreadLocalRecorder* thread);
  void UnregisterThread(ThreadEvents&& events);

  bool StartRecording(int level);
  Events StopRecording();

  // Gathers events from all active threads, and clears their buffers.
  Events Clear() EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  mutex mutex_;
  // Map of the static container instances (thread_local storage) for each
  // thread. While active, a ThreadLocalRecorder stores trace events.
  std::unordered_map<int32, ThreadLocalRecorder*> threads_ GUARDED_BY(mutex_);
  // Events from threads that died during recording.
  TraceMeRecorder::Events orphaned_events_ GUARDED_BY(mutex_);
};

}  // namespace profiler
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_TRACEME_RECORDER_H_
