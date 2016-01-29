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

#ifndef TENSORFLOW_LIB_CORE_THREADPOOL_NONBLOCKING_H_
#define TENSORFLOW_LIB_CORE_THREADPOOL_NONBLOCKING_H_

#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include "tensorflow/core/lib/core/eventcount.h"
#include "tensorflow/core/lib/core/runqueue.h"
#include "tensorflow/core/lib/core/threadpool_impl.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace thread {

class ThreadPoolNonBlocking : public ThreadPool::Impl {
 public:
  typedef std::function<void()> Work;
  typedef RunQueue<Work, 1024> Queue;

  ThreadPoolNonBlocking(Env* env, const ThreadOptions& thread_options,
                        const string& name, int num_threads)
      : blocked_(), spinning_(), done_() {
    CHECK_GE(num_threads, 1);
    for (int i = 0; i < num_threads; i++) queues_.emplace_back(new Queue());
    for (int i = 0; i < num_threads; i++)
      threads_.emplace_back(env->StartThread(thread_options, "tf_" + name,
                                             [this, i]() { WorkerLoop(i); }));
  }

  ~ThreadPoolNonBlocking() {
    done_.store(true, std::memory_order_relaxed);
    // Now if all threads block without work, they will start exiting.
    // But note that threads can continue to work arbitrary long,
    // block, submit new work, unblock and otherwise live full life.
    ec_.Notify(&spinning_, true);

    // Join threads explicitly just to avoid destruction order issues.
    for (size_t i = 0; i < threads_.size(); i++) threads_[i].reset();
    CHECK_EQ(spinning_.load(), 0);
  }

  void Schedule(Work w) {
    PerThread* pt = GetPerThread();
    if (pt->pool == this) {
      // Worker thread of this pool, push onto the thread's queue.
      Queue* q = queues_[pt->index].get();
      w = q->PushFront(std::move(w));
    } else {
      // A free-standing thread (or worker of another pool), push onto a random
      // queue preserving affinity. Temporal affinity gives locality of both
      // push operations (hopefully acquire a hot in cache mutex) and work
      // execution locality (the same worker executes consecutive tasks).
      Queue* q = queues_[pt->affinity % queues_.size()].get();
      w = q->PushBack(std::move(w));
      if (w) pt->affinity = Rand(&pt->rand);
    }
    if (w)
      w();  // Push failed, execute directly.
    else if (spinning_ == 0)
      ec_.Notify(&spinning_, false);
  }

 private:
  struct PerThread {
    bool inited;
    ThreadPoolNonBlocking* pool;  // Parent pool, or null for normal threads.
    unsigned index;               // Worker thread index in pool.
    unsigned affinity;  // Temporal push affinity for free-standing threads.
    unsigned rand;      // Random generator state.
  };
  static thread_local PerThread per_thread_;

  std::vector<std::unique_ptr<Thread>> threads_;
  std::vector<std::unique_ptr<Queue>> queues_;
  std::atomic<unsigned> blocked_;
  std::atomic<unsigned> spinning_;
  std::atomic<bool> done_;
  EventCount ec_;

  // Main worker thread loop.
  void WorkerLoop(unsigned index) {
    PerThread* pt = GetPerThread();
    pt->pool = this;
    pt->index = index;
    Queue* q = queues_[index].get();
    EventCount::Waiter waiter;
    std::vector<Work> stolen;
    // spinning tracks whether this thread is accounted in spinning_ counter.
    bool spinning = false;
    for (;;) {
      Work w;
      if (!stolen.empty()) {
        w = std::move(stolen.back());
        stolen.pop_back();
      }
      if (!w) w = q->PopFront();
      if (!w) {
        if (Steal(&stolen)) {
          w = std::move(stolen.back());
          stolen.pop_back();
          while (stolen.size()) {
            Work w1 = q->PushFront(std::move(stolen.back()));
            stolen.pop_back();
            if (w1) {
              // There is not much we can do in this case. Just execute the
              // remaining directly.
              stolen.push_back(std::move(w1));
              break;
            }
          }
        }
      }
      if (w) {
        if (spinning) {
          // Several calls to Schedule can submit several work items and wake
          // only one worker thread. So if this is the last spinning thread
          // (that now becomes non-spinning) we need to wake up another
          // spinning thread.
          spinning = false;
          unsigned s = spinning_.fetch_sub(1);
          CHECK(s != 0);
          if (s == 1) ec_.Notify(&spinning_, false);
        }
        w();
        continue;
      }
      if (!spinning) {
        spinning = true;
        spinning_++;
      }
      bool nowork = true;
      for (int i = 0; i < 10; i++) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        if (!OutOfWork()) {
          nowork = false;
          break;
        }
      }
      if (!nowork) continue;
      spinning = false;
      CHECK(spinning_.fetch_sub(1));
      if (!WaitForWork(&waiter, &spinning)) {
        if (spinning) CHECK(spinning_.fetch_sub(1));
        return;
      }
    }
  }

  // Steal tries to steal work from other worker threads in best-effort manner.
  // TODO(dvyukov): consider stealing half of elements from a victim queue.
  // It is typical to steal just one element, but that assumes that work is
  // recursively subdivided in halves so that the stolen element is exactly half
  // of work. If work elements are equally-sized, then is makes sense to steal
  // half of elements at once and then work locally for a while.
  bool Steal(std::vector<Work>* stolen) {
    if (queues_.size() == 1) return false;
    PerThread* pt = GetPerThread();
    unsigned lastq = pt->index;
    for (unsigned i = queues_.size() * 2; i > 0; i--) {
      unsigned victim = Rand(&pt->rand) % queues_.size();
      if (victim == lastq && queues_.size() > 2) {
        i++;
        continue;
      }
      if (queues_[victim]->PopBack(stolen)) return true;
      lastq = victim;
    }
    // Just to make sure that we did not miss anything.
    for (unsigned i = queues_.size(); i > 0; i--) {
      if (queues_[i - 1]->PopBack(stolen)) return true;
    }
    return false;
  }

  // WaitForWork blocks until new work is available, or if it is time to exit.
  bool WaitForWork(EventCount::Waiter* waiter, bool* spinning) {
    // We already did best-effort emptiness check in Steal, so prepare blocking.
    ec_.Prewait(waiter);
    // Now do reliable emptiness check.
    if (!OutOfWork()) return true;
    // Number of blocked threads is used as termination condition.
    // If we are shutting down and all worker threads blocked without work,
    // that's we are done.
    blocked_++;
    if (done_ && blocked_ == threads_.size()) {
      // Almost done, but need to re-check queues.
      // Consider that all queues are empty and all worker threads are preempted
      // right after incrementing blocked_ above. Now a free-standing thread
      // submits work and calls destructor (which sets done_). If we don't
      // re-check queues, we will exit leaving the work unexecuted.
      if (!OutOfWork()) {
        // Note: we must not pop from queues before we decrement blocked_,
        // otherwise the following scenario is possible. Consider that instead
        // of checking for emptiness we popped the only element from queues.
        // Now other worker threads can start exiting, which is bad if the
        // work item submits other work. So we just check emptiness here,
        // which ensures that all worker threads exit at the same time.
        blocked_--;
        return true;
      }
      // Reached stable termination state.
      ec_.Notify(&spinning_, true);
      return false;
    }
    if (ec_.Wait(waiter)) *spinning = true;
    blocked_--;
    return true;
  }

  bool OutOfWork() {
    for (unsigned i = 0; i < queues_.size(); i++)
      if (!queues_[i]->Empty()) return false;
    return true;
  }

  PerThread* GetPerThread() {
    PerThread* pt = &per_thread_;
    if (pt->inited) return pt;
    pt->inited = true;
    pt->rand = std::hash<std::thread::id>()(std::this_thread::get_id());
    pt->affinity = Rand(&pt->rand);
    return pt;
  }

  static unsigned Rand(unsigned* state) {
    return *state = *state * 1103515245 + 12345;
  }
};

thread_local ThreadPoolNonBlocking::PerThread
    ThreadPoolNonBlocking::per_thread_;

ThreadPool::Impl* CreateNonBlockingThreadPool(
    Env* env, const ThreadOptions& thread_options, const string& name,
    int num_threads) {
  return new ThreadPoolNonBlocking(env, thread_options, name, num_threads);
}

}  // namespace thread
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_CORE_THREADPOOL_NONBLOCKING_H_
