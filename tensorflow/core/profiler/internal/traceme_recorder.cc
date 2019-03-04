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
#include "tensorflow/core/profiler/internal/traceme_recorder.h"

// To avoid unneccesary synchronization between threads, each thread has a
// ThreadLocalRecorder that independently records its events.
//
// Events are stored in an EventQueue implemented as a linked-list of blocks,
// with start and end pointers:
//  [ events........ | next-]--> [ events......... | next ]
//  ^start_block  ^start         ^end_block  ^end
//
// Record() writes at end, and then advances it, allocating a block if needed.
// Clear() takes ownership of events in the range [start, end).
// The end pointer is atomic so these can be concurrent.
//
// If a thread dies, the ThreadLocalRecorder's destructor hands its data off to
// the orphaned_events list.

#include <string>
#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/stream_executor/lib/initialize.h"

namespace tensorflow {
namespace profiler {

// Default value for g_trace_level when tracing is disabled
constexpr static int kTracingDisabled = -1;

namespace internal {
std::atomic<int> g_trace_level = ATOMIC_VAR_INIT(kTracingDisabled);
}  // namespace internal

namespace {

class ThreadLocalRecorder;

struct Data {
  // Lock for only rare events - start/stop, thread death.
  mutex global_lock;
  // Map of the static container instances (thread_local storage) for each
  // thread, that store the trace events.
  absl::flat_hash_map<uint64, ThreadLocalRecorder*> threads
      GUARDED_BY(global_lock);
  // Events traced from threads that died during tracing.
  TraceMeRecorder::Events orphaned_events GUARDED_BY(global_lock);
}* g_data = nullptr;

// A single-producer single-consumer queue of Events.
// Only the owner thread can write events, writing is lock-free.
// Consume is also lock-free in this class.
//
// Internally, we have a linked list of blocks containing numbered slots.
// start is the first occupied slot, end is the first unoccupied slot.
class EventQueue {
 public:
  EventQueue()
      : start_block_(new Block{0, nullptr}), end_block_(start_block_) {}

  // REQUIRES: Consume() was called since the last Push().
  // Memory should be deallocated and trace events destroyed on destruction.
  // This doesn't require global lock as this discards all the stored trace
  // events and we assume of destruction of this class only after the last
  // Push() has been called.
  ~EventQueue() {
    DCHECK_EQ(start_, end_.load()) << "EventQueue destroyed without Consume()";
    delete end_block_;
  }

  // Add a new event to the back of the queue. Fast and lock-free.
  void Push(TraceMeRecorder::Event&& event) {
    uint64 end = end_.load(std::memory_order_relaxed);
    new (&end_block_->events[end++ - end_block_->start].event)
        TraceMeRecorder::Event(std::move(event));
    if (ABSL_PREDICT_FALSE(end - end_block_->start == Block::kLength)) {
      auto* new_block = new Block{end, nullptr};
      end_block_->next = new_block;
      end_block_ = new_block;
    }
    end_.store(end, std::memory_order_release);  // Write index after contents.
  }

  // Retrieve and remove all events in the queue.
  std::vector<TraceMeRecorder::Event> Consume() {
    // Read index before contents.
    uint64 end = end_.load(std::memory_order_acquire);
    std::vector<TraceMeRecorder::Event> result;
    result.reserve(end - start_);
    while (start_ != end) {
      Shift(&result);
    }
    return result;
  }

 private:
  // Shift one event off the front of the queue into *out.
  void Shift(std::vector<TraceMeRecorder::Event>* out) {
    // Move the next event into the output.
    auto& event = start_block_->events[start_++ - start_block_->start].event;
    out->push_back(std::move(event));
    event.~Event();  // Events must be individually destroyed.
    // If we reach the end of a block, we own it and should delete it.
    // The next block is present: end always points to something.
    if (start_ - start_block_->start == Block::kLength) {
      auto* next_block = start_block_->next;
      delete start_block_;
      start_block_ = next_block;
    }
  }

  // The number of slots in a block. Chosen so that the block fits in 64k.
  struct Block {
    static constexpr size_t kLength =
        ((1 << 16) - (sizeof(uint64) + sizeof(std::atomic<Block*>))) /
        sizeof(TraceMeRecorder::Event);

    const uint64 start;  // The number of the first slot.
    Block* next;
    // Defer construction of Event until the data is available.
    // Must also destroy manually, as the block may not fill entirely.
    union MaybeEvent {
      MaybeEvent() {}
      ~MaybeEvent() {}
      TraceMeRecorder::Event event;
    } events[kLength];
  };

  // Head of list for reading. Only accessed by consumer thread.
  Block* start_block_;
  uint64 start_ = 0;
  // Tail of list for writing. Accessed by producer thread.
  Block* end_block_;
  std::atomic<uint64> end_ = {0};  // Atomic: also read by consumer thread.
};

class ThreadLocalRecorder {
 public:
  // The recorder is created the first time Record() is called on a thread.
  ThreadLocalRecorder() {
    auto* env = Env::Default();
    info_.tid = env->GetCurrentThreadId();
    env->GetCurrentThreadName(&info_.name);
    mutex_lock lock(g_data->global_lock);
    g_data->threads.emplace(info_.tid, this);
  }

  // The destructor is called when the thread shuts down early.
  // We unregister this thread, and move its events to orphaned_events.
  ~ThreadLocalRecorder() {
    mutex_lock lock(g_data->global_lock);
    g_data->threads.erase(info_.tid);
    g_data->orphaned_events.push_back(Clear());
  }

  // This is the performance-critical part!
  void Record(TraceMeRecorder::Event&& event) { queue_.Push(std::move(event)); }

  TraceMeRecorder::ThreadEvents Clear()
      EXCLUSIVE_LOCKS_REQUIRED(g_data->global_lock) {
    return {info_, queue_.Consume()};
  }

 private:
  TraceMeRecorder::ThreadInfo info_;
  EventQueue queue_;
};

// Gather events from all active threads, and clear their buffers. The global
// lock is held, so no threads can be added/removed for the duration while we
// consume the collected trace entries. This will block any new thread and also
// the starting and stopping of TraceMeRecorder, hence, this is performance
// critical and should be kept fast.
TraceMeRecorder::Events Clear() EXCLUSIVE_LOCKS_REQUIRED(g_data->global_lock) {
  TraceMeRecorder::Events result;
  std::swap(g_data->orphaned_events, result);
  for (const auto& entry : g_data->threads) {
    auto* recorder = entry.second;
    result.push_back(recorder->Clear());
  }
  return result;
}

}  // namespace

bool TraceMeRecorder::Start(int level) {
  level = std::max(0, level);
  mutex_lock lock(g_data->global_lock);
  int expected = kTracingDisabled;
  if (!internal::g_trace_level.compare_exchange_strong(
          expected, level, std::memory_order_acq_rel)) {
    return false;
  }
  // We may have old events in buffers because Record() raced with Stop().
  Clear();
  return true;
}


void TraceMeRecorder::Record(Event event) {
  static thread_local ThreadLocalRecorder thread_local_recorder;
  thread_local_recorder.Record(std::move(event));
}

// Only one thread is expected to call Stop() as first instance of XprofSession
// prevents another XprofSession from doing any profiling.
TraceMeRecorder::Events TraceMeRecorder::Stop() {
  mutex_lock lock(g_data->global_lock);
  if (internal::g_trace_level.exchange(
          kTracingDisabled, std::memory_order_acq_rel) == kTracingDisabled) {
    return {};
  }
  return Clear();
}

TraceMeRecorder::Events TraceMeRecorder::Collect() {
  mutex_lock lock(g_data->global_lock);
  if (internal::g_trace_level.load(std::memory_order_acquire) ==
      kTracingDisabled) {
    return {};
  }
  return Clear();
}

}  // namespace profiler
}  // namespace tensorflow

REGISTER_MODULE_INITIALIZER(traceme_recorder, {
  tensorflow::profiler::g_data = new tensorflow::profiler::Data();

  // Workaround for b/35097229, the first block-scoped thread_local can
  // trigger false positives in the heap checker. Currently triggered by
  // //perftools/accelerators/xprof/xprofilez/integration_tests:xla_hlo_trace_test
  static thread_local tensorflow::string fix_deadlock ABSL_ATTRIBUTE_UNUSED;
});
