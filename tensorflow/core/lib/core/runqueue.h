// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//==============================================================================

#ifndef TENSORFLOW_LIB_CORE_RUNQUEUE_H_
#define TENSORFLOW_LIB_CORE_RUNQUEUE_H_

#include <atomic>
#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace thread {

// RunQueue is a fixed-size, partially non-blocking deque or Work items.
// Operations on front of the queue must be done by a single thread (owner),
// operations on back of the queue can be done by multiple threads concurrently.
//
// Algorithm outline:
// All remote threads operating on the queue back are serialized by a mutex.
// This ensures that at most two threads access state: owner and one remote
// thread (Size aside). The algorithm ensures that the occupied region of the
// underlying array is logically continuous (can wraparound, but no stray
// occupied elements). Owner operates on one end of this region, remote thread
// operates on the other end. Synchronization between these threads
// (potential consumption of the last element and take up of the last empty
// element) happens by means of state variable in each element. States are:
// empty, busy (in process of insertion of removal) and ready. Threads claim
// elements (empty->busy and ready->busy transitions) by means of a CAS
// operation. The finishing transition (busy->empty and busy->ready) are done
// with plain store as the element is exclusively owned by the current thread.
//
// Note: we could permit only pointers as elements, then we would not need
// separate state variable as null/non-null pointer value would serve as state,
// but that would require malloc/free per operation for large, complex values
// (and this is designed to store std::function<()>).
template <typename Work, unsigned kSize>
class RunQueue {
 public:
  RunQueue() : front_(), back_() {
    CHECK_EQ(kSize & (kSize - 1), 0);  // require power-of-two for fast masking
    CHECK_GE(kSize, 2);                // why would you do this?
    CHECK_LE(kSize, (64 << 10));       // leave enough space for counter
    for (unsigned i = 0; i < kSize; i++)
      array_[i].state.store(kEmpty, std::memory_order_relaxed);
  }

  ~RunQueue() {
    CHECK(mutex_.try_lock());
    CHECK_EQ(Size(), 0);
  }

  // PushFront inserts w at the beginning of the queue.
  // If queue is full returns w, otherwise returns default-constructed Work.
  Work PushFront(Work w) {
    unsigned front = front_.load(std::memory_order_relaxed);
    Elem* e = &array_[front & kMask];
    uint8_t s = e->state.load(std::memory_order_relaxed);
    if (s != kEmpty ||
        !e->state.compare_exchange_strong(s, kBusy, std::memory_order_acquire))
      return w;
    front_.store(front + 1 + (kSize << 1), std::memory_order_relaxed);
    e->w = std::move(w);
    e->state.store(kReady, std::memory_order_release);
    return Work();
  }

  // PopFront removes and returns the first element in the queue.
  // If the queue was empty returns default-constructed Work.
  Work PopFront() {
    unsigned front = front_.load(std::memory_order_relaxed);
    Elem* e = &array_[(front - 1) & kMask];
    uint8_t s = e->state.load(std::memory_order_relaxed);
    if (s != kReady ||
        !e->state.compare_exchange_strong(s, kBusy, std::memory_order_acquire))
      return Work();
    Work w = std::move(e->w);
    e->state.store(kEmpty, std::memory_order_release);
    front = ((front - 1) & kMask2) | (front & ~kMask2);
    front_.store(front, std::memory_order_relaxed);
    return w;
  }

  // PushBack adds w at the end of the queue.
  // If queue is full returns w, otherwise returns default-constructed Work.
  Work PushBack(Work w) {
    mutex_lock lock(mutex_);
    unsigned back = back_.load(std::memory_order_relaxed);
    Elem* e = &array_[(back - 1) & kMask];
    uint8_t s = e->state.load(std::memory_order_relaxed);
    if (s != kEmpty ||
        !e->state.compare_exchange_strong(s, kBusy, std::memory_order_acquire))
      return w;
    back = ((back - 1) & kMask2) | (back & ~kMask2);
    back_.store(back, std::memory_order_relaxed);
    e->w = std::move(w);
    e->state.store(kReady, std::memory_order_release);
    return Work();
  }

  // PopBack removes and returns half last elements in the queue.
  // Returns number of elements removed. But can also fail spuriously.
  unsigned PopBack(std::vector<Work>* result) {
    if (Empty()) return 0;
    mutex_lock lock(mutex_, std::try_to_lock);
    if (!lock) return 0;
    unsigned back = back_.load(std::memory_order_relaxed);
    unsigned size = Size();
    unsigned mid = back;
    if (size > 1) mid = back + (size - 1) / 2;
    unsigned n = 0;
    unsigned start = 0;
    for (; static_cast<int>(mid - back) >= 0; mid--) {
      Elem* e = &array_[mid & kMask];
      uint8_t s = e->state.load(std::memory_order_relaxed);
      if (n == 0) {
        if (s != kReady ||
            !e->state.compare_exchange_strong(s, kBusy,
                                              std::memory_order_acquire))
          continue;
        start = mid;
      } else {
        // Note: no need to store temporal kBusy, we exclusively own these
        // elements.
        CHECK_EQ(s, kReady);
      }
      result->push_back(std::move(e->w));
      e->state.store(kEmpty, std::memory_order_release);
      n++;
    }
    if (n != 0)
      back_.store(start + 1 + (kSize << 1), std::memory_order_relaxed);
    return n;
  }

  // Size returns current queue size.
  // Can be called by any thread at any time.
  unsigned Size() const {
    // Emptiness plays critical role in thread pool blocking. So we go to great
    // effort to not produce false positives (claim non-empty queue as empty).
    for (;;) {
      // Capture a consistent snapshot of front/tail.
      unsigned front = front_.load(std::memory_order_acquire);
      unsigned back = back_.load(std::memory_order_acquire);
      unsigned front1 = front_.load(std::memory_order_relaxed);
      if (front != front1) continue;
      int size = (front & kMask2) - (back & kMask2);
      // Fix overflow.
      if (size < 0) size += 2 * kSize;
      // Order of modification in push/pop is crafted to make the queue look
      // larger than it is during concurrent modifications. E.g. pop can
      // decrement size before the corresponding push has incremented it.
      // So the computed size can be up to kSize + 1, fix it.
      if (size > kSize) size = kSize;
      return size;
    }
  }

  // Empty tests whether container is empty.
  // Can be called by any thread at any time.
  bool Empty() const { return Size() == 0; }

 private:
  static const unsigned kMask = kSize - 1;
  static const unsigned kMask2 = (kSize << 1) - 1;
  struct Elem {
    std::atomic<uint8_t> state;
    Work w;
  };
  enum {
    kEmpty,
    kBusy,
    kReady,
  };
  mutex mutex_;
  std::atomic<unsigned> front_;
  std::atomic<unsigned> back_;
  Elem array_[kSize];
  TF_DISALLOW_COPY_AND_ASSIGN(RunQueue);
};

}  // namespace thread
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_CORE_RUNQUEUE_H_
