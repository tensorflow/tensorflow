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
#include "tsl/profiler/backends/cpu/traceme_recorder.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tsl/platform/env.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/utils/lock_free_queue.h"
#include "tsl/profiler/utils/per_thread.h"

namespace tsl {
namespace profiler {
namespace internal {

#ifdef _WIN32
#define DECL_DLL_EXPORT __declspec(dllexport)
#else
#define DECL_DLL_EXPORT
#endif
// DLL imported variables cannot be initialized on Windows. This file is
// included only on DLL exports.
DECL_DLL_EXPORT std::atomic<int> g_trace_level(
    TraceMeRecorder::kTracingDisabled);

// g_trace_level implementation must be lock-free for faster execution of the
// TraceMe API. This can be commented (if compilation is failing) but execution
// might be slow (even when tracing is disabled).
static_assert(ATOMIC_INT_LOCK_FREE == 2, "Assumed atomic<int> was lock free");

}  // namespace internal

namespace {

// Track events created by ActivityStart and merge their data into events
// created by ActivityEnd. TraceMe records events in its destructor, so this
// results in complete events sorted by their end_time in the thread they ended.
// Within the same thread, the record created by ActivityStart must appear
// before the record created by ActivityEnd. Cross-thread events must be
// processed in a separate pass. A single map can be used because the
// activity_id is globally unique.
class SplitEventTracker {
 public:
  void AddStart(TraceMeRecorder::Event&& event) {
    DCHECK(event.IsStart());
    start_events_.emplace(event.ActivityId(), std::move(event));
  }

  void AddEnd(TraceMeRecorder::Event* event) {
    DCHECK(event->IsEnd());
    if (!FindStartAndMerge(event)) {
      end_events_.push_back(event);
    }
  }

  void HandleCrossThreadEvents() {
    for (auto* event : end_events_) {
      FindStartAndMerge(event);
    }
  }

 private:
  // Finds the start of the given event and merges data into it.
  bool FindStartAndMerge(TraceMeRecorder::Event* event) {
    auto iter = start_events_.find(event->ActivityId());
    if (iter == start_events_.end()) return false;
    auto& start_event = iter->second;
    event->name = std::move(start_event.name);
    event->start_time = start_event.start_time;
    start_events_.erase(iter);
    return true;
  }

  // Start events are collected from each ThreadLocalRecorder::Consume() call.
  // Their data is merged into end_events.
  absl::flat_hash_map<int64_t, TraceMeRecorder::Event> start_events_;

  // End events are stored in the output of TraceMeRecorder::Consume().
  std::vector<TraceMeRecorder::Event*> end_events_;
};

// To avoid unnecessary synchronization between threads, each thread has a
// ThreadLocalRecorder that independently records its events.
class ThreadLocalRecorder {
 public:
  // The recorder is created the first time TraceMeRecorder::Record() is called
  // on a thread.
  ThreadLocalRecorder() {
    auto* env = Env::Default();
    info_.tid = env->GetCurrentThreadId();
    env->GetCurrentThreadName(&info_.name);
  }

  const TraceMeRecorder::ThreadInfo& Info() const { return info_; }

  // Record is only called from the producer thread.
  void Record(TraceMeRecorder::Event&& event) { queue_.Push(std::move(event)); }

  // Clear is called from the control thread when tracing starts to remove any
  // elements added due to Record racing with Consume.
  void Clear() { queue_.Clear(); }

  // Consume is called from the control thread when tracing stops.
  TF_MUST_USE_RESULT std::deque<TraceMeRecorder::Event> Consume(
      SplitEventTracker* split_event_tracker) {
    std::deque<TraceMeRecorder::Event> events;
    std::optional<TraceMeRecorder::Event> event;
    while ((event = queue_.Pop())) {
      if (event->IsStart()) {
        split_event_tracker->AddStart(*std::move(event));
        continue;
      }
      events.push_back(*std::move(event));
      if (events.back().IsEnd()) {
        split_event_tracker->AddEnd(&events.back());
      }
    }
    return events;
  }

 private:
  TraceMeRecorder::ThreadInfo info_;
  LockFreeQueue<TraceMeRecorder::Event> queue_;
};

}  // namespace

// This method is performance critical and should be kept fast. It is called
// when tracing starts.
/* static */ void TraceMeRecorder::Clear() {
  auto recorders = PerThread<ThreadLocalRecorder>::StartRecording();
  for (auto& recorder : recorders) {
    recorder->Clear();
  };
}

// This method is performance critical and should be kept fast. It is called
// when tracing stops.
/* static */ TraceMeRecorder::Events TraceMeRecorder::Consume() {
  TraceMeRecorder::Events result;
  SplitEventTracker split_event_tracker;
  auto recorders = PerThread<ThreadLocalRecorder>::StopRecording();
  for (auto& recorder : recorders) {
    auto events = recorder->Consume(&split_event_tracker);
    if (!events.empty()) {
      result.push_back({recorder->Info(), std::move(events)});
    }
  };
  split_event_tracker.HandleCrossThreadEvents();
  return result;
}

/* static */ bool TraceMeRecorder::Start(int level) {
  level = std::max(0, level);
  int expected = kTracingDisabled;
  bool started = internal::g_trace_level.compare_exchange_strong(
      expected, level, std::memory_order_acq_rel);
  if (started) {
    // We may have old events in buffers because Record() raced with Stop().
    Clear();
  }
  return started;
}

/* static */ void TraceMeRecorder::Record(Event&& event) {
  PerThread<ThreadLocalRecorder>::Get().Record(std::move(event));
}

/* static */ TraceMeRecorder::Events TraceMeRecorder::Stop() {
  TraceMeRecorder::Events events;
  if (internal::g_trace_level.exchange(
          kTracingDisabled, std::memory_order_acq_rel) != kTracingDisabled) {
    events = Consume();
  }
  return events;
}

/*static*/ int64_t TraceMeRecorder::NewActivityId() {
  // Activity IDs: To avoid contention over a counter, the top 32 bits identify
  // the originating thread, the bottom 32 bits name the event within a thread.
  // IDs may be reused after 4 billion events on one thread, or 2 billion
  // threads.
  static std::atomic<int32> thread_counter(1);  // avoid kUntracedActivity
  const thread_local static int32_t thread_id =
      thread_counter.fetch_add(1, std::memory_order_relaxed);
  thread_local static uint32 per_thread_activity_id = 0;
  return static_cast<int64_t>(thread_id) << 32 | per_thread_activity_id++;
}

}  // namespace profiler
}  // namespace tsl
