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

#ifndef TENSORFLOW_LIB_CORE_EVENTCOUNT_H_
#define TENSORFLOW_LIB_CORE_EVENTCOUNT_H_

#include <atomic>
#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace thread {

// EventCount allows to wait for arbitrary predicates in non-blocking
// algorithms. Think of condition variable, but wait predicate does not need to
// be protected by a mutex. Usage:
// Waiting thread does:
//
//   if (predicate)
//     return act();
//   EventCount::Waiter w;
//   ec.Prewait(&w);
//   if (predicate)
//     return act();
//   ec.Wait(&w);
//
// Notifying thread does:
//
//   predicate = true;
//   ec.Notify(&count, true);
//
// Notify is cheap if there are no waiting threads. Prewait/Wait are not cheap,
// but they are executed only if the preceeding predicate check has failed.
//
// Algorihtm outline:
// There are two main variables: predicate (managed by user) and state_.
// Operation closely resembles Dekker mutual algorithm:
// https://en.wikipedia.org/wiki/Dekker%27s_algorithm
// Waiting thread sets state_ then checks predicate, Notifying thread sets
// predicate then checks state_. Due to seq_cst fences in between these
// operations it is guaranteed than either waiter will see predicate change
// and won't block, or notifying thread will see state_ change and will unblock
// the waiter, or both. But it can't happen that both threads don't see each
// other changes, which would lead to deadlock.
// The rest is pretty straightforward blocking/signaling using mutex+condvar.
class EventCount {
 public:
  EventCount() : state_(), waiters_() {}

  ~EventCount() {
    CHECK(mutex_.try_lock());
    CHECK_EQ(waiters_.size(), 0);
  }

  class Waiter {
    friend class EventCount;
    Waiter* next;
    mutex mu;
    condition_variable cv;
    unsigned state;
    unsigned epoch;
    enum {
      kNotSignaled,
      kWaiting,
      kSignaled,
    };
  };

  // Prewait prepares for waiting.
  // The returned value must be passed to a subsequent Wait invocation,
  // after re-checking wait predicate. It is OK to skip Wait invocation,
  // if the predicate becomes true in the meantime.
  void Prewait(Waiter* w) {
    w->epoch = state_.fetch_or(kWaiter, std::memory_order_relaxed) & ~kWaiter;
    std::atomic_thread_fence(std::memory_order_seq_cst);
  }

  // Wait commits waiting.
  // Returns true if the thread was notified by another thread.
  bool Wait(Waiter* w) {
    w->state = Waiter::kNotSignaled;
    {
      mutex_lock lock(mutex_);
      if (w->epoch != (state_.load(std::memory_order_seq_cst) & ~kWaiter))
        return false;
      waiters_.push_back(w);
    }
    mutex_lock lock(w->mu);
    while (w->state != Waiter::kSignaled) {
      w->state = Waiter::kWaiting;
      w->cv.wait(lock);
    }
    return true;
  }

  // Notify wakes one or all waiting threads.
  // Must be called after changing the associated wait predicate.
  // Count is incremented by the number of unblocked threads before the actual
  // unblocking. Count increments precisely match true's returned from Wait.
  void Notify(std::atomic<unsigned>* count, bool all) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    unsigned state = state_.load(std::memory_order_relaxed);
    if (!(state & kWaiter)) return;
    Waiter* waiters = nullptr;
    unsigned nwaiter = 0;
    {
      mutex_lock lock(mutex_);
      if (all || waiters_.size() <= 1) {
        nwaiter = waiters_.size();
        for (auto w : waiters_) {
          w->next = waiters;
          waiters = w;
        }
        waiters_.clear();
        while (!state_.compare_exchange_weak(state, (state & ~kWaiter) + kEpoch,
                                             std::memory_order_relaxed)) {
        }
      } else {
        waiters = waiters_.back();
        waiters_.pop_back();
        waiters->next = nullptr;
        nwaiter = 1;
      }
    }
    count->fetch_add(nwaiter, std::memory_order_seq_cst);
    Waiter* next = nullptr;
    for (Waiter* w = waiters; w; w = next) {
      next = w->next;
      unsigned state;
      {
        mutex_lock lock(w->mu);
        state = w->state;
        w->state = Waiter::kSignaled;
      }
      // Note: w can be destroyed once we release the mutex,
      // so avoid notifying if it wasn't waiting.
      if (state == Waiter::kWaiting) w->cv.notify_one();
    }
  }

 private:
  enum {
    kWaiter = 1,
    kEpoch = 2,
  };
  mutex mutex_;
  std::atomic<unsigned> state_;
  std::vector<Waiter*> waiters_;
  TF_DISALLOW_COPY_AND_ASSIGN(EventCount);
};

}  // namespace thread
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_CORE_EVENTCOUNT_H_
