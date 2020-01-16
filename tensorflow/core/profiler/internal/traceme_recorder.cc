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

#include <stddef.h>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace profiler {

namespace internal {
std::atomic<int> g_trace_level =
    ATOMIC_VAR_INIT(TraceMeRecorder::kTracingDisabled);
}  // namespace internal

// Implementation of TraceMeRecorder::trace_level_ must be lock-free for faster
// execution of the TraceMe() public API. This can be commented (if compilation
// is failing) but execution might be slow (even when host tracing is disabled).
static_assert(ATOMIC_INT_LOCK_FREE == 2, "Assumed atomic<int> was lock free");

namespace {

// A single-producer single-consumer queue of Events.
//
// Implemented as a linked-list of blocks containing numbered slots, with start
// and end pointers:
//
//  [ events........ | next-]--> [ events......... | next ]
//  ^start_block_ ^start_         ^end_block_ ^end_
//
// start_ is the first occupied slot, end_ is the first unoccupied slot.
//
// Push writes at end_, and then advances it, allocating a block if needed.
// PopAll takes ownership of events in the range [start_, end_).
// The end_ pointer is atomic so Push and PopAll can be concurrent.
//
// Push and PopAll are lock free and each might be called from at most one
// thread. Push is only called by the owner thread. PopAll is called by the
// owner thread when it shuts down, or by the tracing control thread.
//
// Thus, PopAll might race with Push, so PopAll only removes events that were
// in the queue when it was invoked. If Push is called while PopAll is active,
// the new event remains in the queue. Thus, the tracing control thread should
// call PopAll when tracing stops to remove events created during tracing, but
// also when tracing starts again to clear any remaining events.
class EventQueue {
 public:
  EventQueue()
      : start_block_(new Block{/*start=*/0, /*next=*/nullptr}),
        start_(start_block_->start),
        end_block_(start_block_),
        end_(start_) {}

  // REQUIRES: PopAll() was called since the last Push().
  // Memory should be deallocated and trace events destroyed on destruction.
  // This doesn't require global lock as this discards all the stored trace
  // events and we assume of destruction of this instance only after the last
  // Push() has been called.
  ~EventQueue() {
    DCHECK(Empty()) << "EventQueue destroyed without PopAll()";
    delete end_block_;
  }

  // Add a new event to the back of the queue. Fast and lock-free.
  void Push(TraceMeRecorder::Event&& event) {
    size_t end = end_.load(std::memory_order_relaxed);
    new (&end_block_->events[end++ - end_block_->start].event)
        TraceMeRecorder::Event(std::move(event));
    if (TF_PREDICT_FALSE(end - end_block_->start == Block::kNumSlots)) {
      auto* new_block = new Block{end, nullptr};
      end_block_->next = new_block;
      end_block_ = new_block;
    }
    end_.store(end, std::memory_order_release);  // Write index after contents.
  }

  // Retrieve and remove all events in the queue at the time of invocation.
  // If Push is called while PopAll is active, the new event will not be
  // removed from the queue.
  // PopAll is only called from ThreadLocalRecorder::Clear, which in turn is
  // only called while holding TraceMeRecorder::Mutex, so PopAll has a single
  // caller at a time.
  std::vector<TraceMeRecorder::Event> PopAll() {
    // Read index before contents.
    size_t end = end_.load(std::memory_order_acquire);
    std::vector<TraceMeRecorder::Event> result;
    result.reserve(end - start_);
    while (start_ != end) {
      result.emplace_back(Pop());
    }
    return result;
  }

 private:
  // Returns true if the queue is empty at the time of invocation.
  bool Empty() const {
    return (start_ == end_.load(std::memory_order_acquire));
  }

  // Remove one event off the front of the queue and return it.
  // REQUIRES: The queue must not be empty.
  TraceMeRecorder::Event Pop() {
    DCHECK(!Empty());
    // Move the next event into the output.
    auto& event = start_block_->events[start_++ - start_block_->start].event;
    TraceMeRecorder::Event out = std::move(event);
    event.~Event();  // Events must be individually destroyed.
    // If we reach the end of a block, we own it and should delete it.
    // The next block is present: end always points to something.
    if (TF_PREDICT_FALSE(start_ - start_block_->start == Block::kNumSlots)) {
      auto* next_block = start_block_->next;
      delete start_block_;
      start_block_ = next_block;
      DCHECK_EQ(start_, start_block_->start);
    }
    return out;
  }

  struct Block {
    // The number of slots in a block is chosen so the block fits in 64 KiB.
    static constexpr size_t kSize = 1 << 16;
    static constexpr size_t kNumSlots =
        (kSize - (sizeof(size_t) + sizeof(Block*))) /
        sizeof(TraceMeRecorder::Event);

    size_t start;  // The number of the first slot.
    Block* next;
    // Defer construction of Event until the data is available.
    // Must also destroy manually, as the block may not fill entirely.
    union MaybeEvent {
      MaybeEvent() {}
      ~MaybeEvent() {}
      TraceMeRecorder::Event event;
    } events[kNumSlots];
  };

  static_assert(sizeof(Block) <= Block::kSize, "");

  // Head of list for reading. Only accessed by consumer thread.
  Block* start_block_;
  size_t start_;
  // Tail of list for writing. Accessed by producer thread.
  Block* end_block_;
  std::atomic<size_t> end_;  // Atomic: also read by consumer thread.
};

}  // namespace

// To avoid unnecessary synchronization between threads, each thread has a
// ThreadLocalRecorder that independently records its events.
class TraceMeRecorder::ThreadLocalRecorder {
 public:
  // The recorder is created the first time TraceMeRecorder::Record() is called
  // on a thread.
  ThreadLocalRecorder() {
    auto* env = Env::Default();
    info_.tid = env->GetCurrentThreadId();
    env->GetCurrentThreadName(&info_.name);
    TraceMeRecorder::Get()->RegisterThread(info_.tid, this);
  }

  // The destructor is called when the thread shuts down early.
  ~ThreadLocalRecorder() {
    // Unregister the thread. Clear() will be called from TraceMeRecorder.
    TraceMeRecorder::Get()->UnregisterThread(info_.tid);
  }

  // Record is only called from the owner thread.
  void Record(TraceMeRecorder::Event&& event) { queue_.Push(std::move(event)); }

  // Clear is called from the control thread when tracing starts/stops, or from
  // the owner thread when it shuts down (see destructor).
  TraceMeRecorder::ThreadEvents Clear() { return {info_, queue_.PopAll()}; }

 private:
  TraceMeRecorder::ThreadInfo info_;
  EventQueue queue_;
};

/*static*/ TraceMeRecorder* TraceMeRecorder::Get() {
  static TraceMeRecorder* singleton = new TraceMeRecorder;
  return singleton;
}

void TraceMeRecorder::RegisterThread(int32 tid, ThreadLocalRecorder* thread) {
  mutex_lock lock(mutex_);
  threads_.emplace(tid, thread);
}

void TraceMeRecorder::UnregisterThread(int32 tid) {
  mutex_lock lock(mutex_);
  auto it = threads_.find(tid);
  if (it != threads_.end()) {
    auto events = it->second->Clear();
    if (!events.events.empty()) {
      orphaned_events_.push_back(std::move(events));
    }
    threads_.erase(it);
  }
}

// This method is performance critical and should be kept fast. It is called
// when tracing starts/stops. The mutex is held, so no threads can be
// registered/unregistered. This prevents calling ThreadLocalRecorder::Clear
// from two different threads.
TraceMeRecorder::Events TraceMeRecorder::Clear() {
  TraceMeRecorder::Events result;
  std::swap(orphaned_events_, result);
  for (const auto& entry : threads_) {
    auto* recorder = entry.second;
    TraceMeRecorder::ThreadEvents events = recorder->Clear();
    if (!events.events.empty()) {
      result.push_back(std::move(events));
    }
  }
  return result;
}

bool TraceMeRecorder::StartRecording(int level) {
  level = std::max(0, level);
  mutex_lock lock(mutex_);
  // Change trace_level_ while holding mutex_.
  int expected = kTracingDisabled;
  bool started = internal::g_trace_level.compare_exchange_strong(
      expected, level, std::memory_order_acq_rel);
  if (started) {
    // We may have old events in buffers because Record() raced with Stop().
    Clear();
  }
  return started;
}

void TraceMeRecorder::Record(Event event) {
  static thread_local ThreadLocalRecorder thread_local_recorder;
  thread_local_recorder.Record(std::move(event));
}

TraceMeRecorder::Events TraceMeRecorder::StopRecording() {
  TraceMeRecorder::Events events;
  mutex_lock lock(mutex_);
  // Change trace_level_ while holding mutex_.
  if (internal::g_trace_level.exchange(
          kTracingDisabled, std::memory_order_acq_rel) != kTracingDisabled) {
    events = Clear();
  }
  return events;
}

/*static*/ uint64 TraceMeRecorder::NewActivityId() {
  // Activity IDs: To avoid contention over a counter, the top 32 bits identify
  // the originating thread, the bottom 32 bits name the event within a thread.
  // IDs may be reused after 4 billion events on one thread, or 4 billion
  // threads.
  static std::atomic<uint32> thread_counter(1);  // avoid kUntracedActivity
  const thread_local static uint32 thread_id =
      thread_counter.fetch_add(1, std::memory_order_relaxed);
  thread_local static uint32 per_thread_activity_id = 0;
  return static_cast<uint64>(thread_id) << 32 | per_thread_activity_id++;
}

}  // namespace profiler
}  // namespace tensorflow
