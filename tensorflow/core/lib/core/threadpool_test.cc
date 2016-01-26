/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/nonblocking_threadpool.h"

#include <atomic>

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace thread {

static const int kNumThreads = 30;

TEST(ThreadPool, Empty) {
  for (int num_threads = 1; num_threads < kNumThreads; num_threads++) {
    ThreadPool pool(Env::Default(), "test", num_threads);
  }
}

TEST(ThreadPool, DoWork) {
  for (int num_threads = 1; num_threads < kNumThreads; num_threads++) {
    const int kWorkItems = 15;
    int work[kWorkItems];
    for (int i = 0; i < kWorkItems; i++) {
      work[i] = 0;
    }
    {
      ThreadPool pool(Env::Default(), "test", num_threads);
      for (int i = 0; i < kWorkItems; i++) {
        pool.Schedule([&work, i]() {
          ASSERT_FALSE(work[i]);
          work[i] += 1;
        });
      }
    }
    for (int i = 0; i < kWorkItems; i++) {
      ASSERT_EQ(work[i], 1);
    }
  }
}

// For the purposes of the tests, we make integers pass
// for pointers to avoid having to actually run jobs.
// The value zero is reserved for "no value was available".
struct Int64ThreadQueue {
  NonBlockingThreadPool::ThreadQueue q;

  int64_t Push(int64_t n) {
    CHECK(n != 0);
    return reinterpret_cast<int64_t>(
        q.Push(reinterpret_cast<NonBlockingThreadPool::ThreadJob*>(n))
            .release());
  };

  int64_t Steal() { return reinterpret_cast<int64_t>(q.Steal().release()); };

  int64_t Pop(int* path = 0) {
    return reinterpret_cast<int64_t>(q.Pop(path).release());
  };

  size_t num_queued() { return q.num_queued_; }
};

// Check basic functionality and invariants in one thread
void RunBasicTest(Int64ThreadQueue* q) {
  // Pop fails with no elements
  ASSERT_EQ(q->num_queued(), 0);
  ASSERT_EQ(q->Pop(), 0);

  // Pop works for one element
  ASSERT_EQ(q->num_queued(), 0);
  q->Push(1);
  ASSERT_EQ(q->num_queued(), 1);
  ASSERT_EQ(q->Pop(), 1);
  ASSERT_EQ(q->Pop(), 0);
  ASSERT_EQ(q->Steal(), 0);

  // Steal works for one element
  ASSERT_EQ(q->num_queued(), 0);
  q->Push(1);
  ASSERT_EQ(q->num_queued(), 1);
  ASSERT_EQ(q->Steal(), 1);
  ASSERT_EQ(q->Pop(), 0);
  ASSERT_EQ(q->Steal(), 0);

  // Steal removes earliest element
  ASSERT_EQ(q->num_queued(), 0);
  q->Push(1);
  q->Push(2);
  ASSERT_EQ(q->Steal(), 1);
  ASSERT_EQ(q->Pop(), 2);
  ASSERT_EQ(q->Pop(), 0);
  ASSERT_EQ(q->Steal(), 0);

  // Pop removes latest element
  ASSERT_EQ(q->num_queued(), 0);
  q->Push(1);
  q->Push(2);
  ASSERT_EQ(q->Pop(), 2);
  ASSERT_EQ(q->Steal(), 1);
  ASSERT_EQ(q->Pop(), 0);
  ASSERT_EQ(q->Steal(), 0);
};

// Check basic functionality and invariants in one thread
TEST(NonBlockingThreadPoolThreadQueue, Basics) {
  Int64ThreadQueue q;
  RunBasicTest(&q);
}

// Check basic functionality and invariants in one thread with wraparound
TEST(NonBlockingThreadPoolThreadQueue, BasicsWithWraparoundINT_MAX) {
  Int64ThreadQueue q;
  q.q.top_ = q.q.bottom_ = INT_MAX;
  RunBasicTest(&q);
}

// Check basic functionality and invariants in one thread with wraparound
TEST(NonBlockingThreadPoolThreadQueue, BasicsWithWraparoundINT_MAXMinusOne) {
  Int64ThreadQueue q;
  q.q.top_ = q.q.bottom_ = INT_MAX - 1;
  RunBasicTest(&q);
}

// Check basic functionality and invariants in one thread with wraparound
TEST(NonBlockingThreadPoolThreadQueue, BasicsWithWraparoundINT_MIN) {
  Int64ThreadQueue q;
  q.q.top_ = q.q.bottom_ = INT_MIN;
  RunBasicTest(&q);
}

// Check basic functionality and invariants in one thread with wraparound
TEST(NonBlockingThreadPoolThreadQueue, BasicsWithWraparoundINT_MINPlusOne) {
  Int64ThreadQueue q;
  q.q.top_ = q.q.bottom_ = INT_MIN + 1;
  RunBasicTest(&q);
}

// Test driver for one element races.  This is testing the low-level
// consistency of the lockless deque used for job queuing.
void TestRaceForOneElement(int num_stealing_threads, bool pop_element_in_race) {
  // In this test, we push one single element and set up a race
  // to pop or steal it.
  // We test the invariants that:
  // - exactly one of the threads wins the race
  // - the data structure is consistent afterwards
  // - the element returned is the one pushed

  constexpr int kNumTrials = 50000;

  Int64ThreadQueue q;

  std::atomic<int> current_epoch(0);

  struct StealThread {
    std::atomic<int64_t> stolen_element;
    std::atomic<int> acknowledged_epoch;
    Int64ThreadQueue* q;
    std::atomic<int>* current_epoch;
    std::unique_ptr<Thread> thread;
    char padding[128];  // avoid false sharing

    StealThread()
        : acknowledged_epoch(-1), q(nullptr), current_epoch(nullptr) {}

    // Required so we can put it in a vector
    StealThread(StealThread&& other)
        : q(other.q),
          current_epoch(other.current_epoch),
          thread(std::move(other.thread)) {}

    void Start(Int64ThreadQueue* q, std::atomic<int>* current_epoch) {
      this->q = q;
      this->current_epoch = current_epoch;

      thread.reset(Env::Default()->StartThread(ThreadOptions(), "test",
                                               [&]() { this->Run(); }));
    }

    void Run() {
      int known_epoch = 0;

      while (current_epoch->load() != -1) {
        // Busy wait until we're in a new epoch.  This is basically
        // a barrier operation.
        while (known_epoch == current_epoch->load())
          ;

        // We're in another epoch
        known_epoch = current_epoch->load();

        // Try to steal one element, and report back the result
        stolen_element = q->Steal();

        // Acknowledge we're done with this epoch
        acknowledged_epoch = known_epoch;
      }
    }

    void AwaitAcknowledgement() {
      while (acknowledged_epoch != current_epoch->load())
        ;
    }
  };

  // Steal threads are run outside of the trial loop to avoid
  // starting threads on every new trial.
  std::vector<StealThread> steal_threads(num_stealing_threads);
  for (auto& t : steal_threads) {
    t.Start(&q, &current_epoch);
  }

  for (int i = 0; i < kNumTrials; ++i) {
    // Push an element onto the queue
    q.Push(i + 1);

    // Tell the steal threads that we're in a new epoch,
    // so they can try to steal it
    ++current_epoch;

    // Try to pop it ourselves, if we're doing a test where we participate
    // in the race.  The path variable can be used to diagnose test failures;
    // the path of the run before or during the failure is likely the place
    // that caused the error.
    int path = 0;
    int64_t popped_element = 0;
    if (pop_element_in_race) {
      popped_element = q.Pop(&path);
    }

    // Wait for the steal threads to acknowledge they've finished the epoch
    for (auto& t : steal_threads) {
      t.AwaitAcknowledgement();
    }

    if (false) {
      LOG(INFO) << "element: stolen "
                << (steal_threads.empty()
                        ? 0
                        : steal_threads[0].stolen_element.load()) << " popped "
                << popped_element << " nqueued " << q.num_queued() << " path "
                << path;
    }

    // Now check the elements.  We should have exactly one winner,
    // which has popped the correct element.

    bool found_winner = false;

    if (popped_element != 0) {
      ASSERT_EQ(popped_element, i + 1);
      found_winner = true;
    }

    for (auto& t : steal_threads) {
      if (t.stolen_element == 0) {
        continue;
      }
      if (found_winner) {
        ASSERT_EQ(false && "More than one winner", true);
      }
      ASSERT_EQ(t.stolen_element, i + 1);
      found_winner = true;
    }

    ASSERT_EQ(found_winner, true);

    ASSERT_EQ(q.num_queued(), 0);
  }

  current_epoch = -1;
}

// Make sure that elements can be pushed then popped
TEST(NonBlockingThreadPoolThreadQueue, PopOneElement) {
  TestRaceForOneElement(0 /* steal thread */, true /* pop elements */);
}

// Make sure that elements can be pushed then stolen
TEST(NonBlockingThreadPoolThreadQueue, StealOneElement) {
  TestRaceForOneElement(1 /* steal thread */, false /* pop elements */);
}

// Make sure that in a race between one popping thread and one stealing
// thread, exactly one of them wins.
TEST(NonBlockingThreadPoolThreadQueue, PopAndOneStealThreadRaceForLastElement) {
  TestRaceForOneElement(1 /* steal thread */, true /* pop elements */);
}

// Make sure that in a race between two stealing threads, exactly one
// wins.  Two threads gives the highest likelyhood of catching a situation
// where none of them win.
TEST(NonBlockingThreadPoolThreadQueue, TwoStealThreadsRaceForLastElement) {
  TestRaceForOneElement(2 /* steal threads */, false /* pop elements */);
}

// Make sure that multiple stealing threads competing against each other work.
TEST(NonBlockingThreadPoolThreadQueue, ManyStealThreadsRaceForLastElement) {
  TestRaceForOneElement(8 /* steal threads */, false /* pop elements */);
}

// Many stealing threads competing with a pop thread
TEST(NonBlockingThreadPoolThreadQueue,
     PopAndManyStealThreadsRaceForLastElement) {
  TestRaceForOneElement(8 /* steal threads */, true /* pop elements */);
}

// A more involved test, that includes testing the queue when it's filled
// up.  We make sure that we can steal and pop all elements simultaneously
// over multiple threads.  Parameter tells us where we initialize top and
// bottom pointers so that we can test wraparound.
void TestPushPopSteal(int init_top_and_bottom = 0) {
  // One thread; push and pop with simultaneous stealing; ensure balanced
  constexpr int kNumIters = 20;
  constexpr int kNumStealThreads = 8;
  for (int i = 0; i < kNumIters; ++i) {
    int num_to_push_pop = 100000;
    // LOG(INFO) << "test iteration " << i;

    Int64ThreadQueue q;
    q.q.top_ = q.q.bottom_ = init_top_and_bottom;

    std::vector<std::unique_ptr<Thread> > threads;
    std::atomic<int> num_to_finish(num_to_push_pop);

    std::vector<int> item_is_done(num_to_push_pop, 0);

    auto MarkItemAsDone = [&](int64_t item) {
      if (item) {
        item -= 1;  // remove offset added on push
        ASSERT_EQ(item_is_done.at(item), 0);
        item_is_done.at(item) += 1;
        ASSERT_EQ(item_is_done.at(item), 1);
        --num_to_finish;
      };
    };

    auto run_steal_thread = [&]() {
      while (num_to_finish > 0) {
        MarkItemAsDone(q.Steal());
      }
    };

    for (int j = 0; j < kNumStealThreads; ++j) {
      threads.emplace_back(Env::Default()->StartThread(ThreadOptions(), "test",
                                                       run_steal_thread));
    }

    for (int j = 0; j < num_to_push_pop; /* no inc */) {
      int64_t overflow = q.Push(j + 1);

      // Attempt a pop on queue overflow or on every 8th push
      if (j % 8 == 0 || overflow) {
        int64_t item = q.Pop();
        if (item) {
          MarkItemAsDone(item);
        }
      }
      if (!overflow) j += 1;
    }

    while (num_to_finish > 0) {
      int64_t item = q.Pop();
      if (!item) {
        break;
      }
      MarkItemAsDone(item);
    }

    threads.clear();

    ASSERT_EQ(num_to_finish, 0);

    for (int count : item_is_done) {
      ASSERT_EQ(count, 1);
    }
  }
}

TEST(NonBlockingThreadPoolThreadQueue, PushPopSteal) {
  TestPushPopSteal(0 /* top and bottom of empty queue */);
}

TEST(NonBlockingThreadPoolThreadQueue, PushPopStealWithWraparound) {
  TestPushPopSteal(INT_MAX - 10 /* top and bottom of empty queue */);
}

extern const char* THREAD_POOL_IMPL_NAME;

static void BM_Sequential(int iters, const char* impl) {
  THREAD_POOL_IMPL_NAME = impl;
  ThreadPool pool(Env::Default(), "test", kNumThreads);
  // Decrement count sequentially until 0.
  int count = iters;
  mutex done_lock;
  condition_variable done;
  bool done_flag = false;
  std::function<void()> work =
      [&pool, &count, &done_lock, &done, &done_flag, &work]() {
        if (count--) {
          pool.Schedule(work);
        } else {
          mutex_lock l(done_lock);
          done_flag = true;
          done.notify_all();
        }
      };
  work();
  mutex_lock l(done_lock);
  while (!done_flag) {
    done.wait(l);
  }
}
static void BM_Parallel(int iters, const char* impl) {
  THREAD_POOL_IMPL_NAME = impl;
  ThreadPool pool(Env::Default(), "test", kNumThreads);
  // Decrement count concurrently until 0.
  std::atomic_int_fast32_t count(iters);
  mutex done_lock;
  condition_variable done;
  bool done_flag = false;
  for (int i = 0; i < iters; ++i) {
    pool.Schedule([&count, &done_lock, &done, &done_flag]() {
      if (count.fetch_sub(1) == 1) {
        mutex_lock l(done_lock);
        done_flag = true;
        done.notify_all();
      }
    });
  }
  mutex_lock l(done_lock);
  while (!done_flag) {
    done.wait(l);
  }
}
static void BM_SequentialLockFree(int iters) {
  BM_Sequential(iters, "lock_free");
}
BENCHMARK(BM_SequentialLockFree);
static void BM_ParallelLockFree(int iters) { BM_Parallel(iters, "lock_free"); }
BENCHMARK(BM_ParallelLockFree);
static void BM_SequentialDefault(int iters) {
  BM_Sequential(iters, "mutex_based");
}
BENCHMARK(BM_SequentialDefault);
static void BM_ParallelDefault(int iters) { BM_Parallel(iters, "mutex_based"); }
BENCHMARK(BM_ParallelDefault);

}  // namespace thread
}  // namespace tensorflow
