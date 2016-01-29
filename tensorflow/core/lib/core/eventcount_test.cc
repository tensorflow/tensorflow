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

#include "tensorflow/core/lib/core/eventcount.h"

#include <atomic>
#include <thread>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace thread {

TEST(EventCount, Basic) {
  EventCount ec;
  EventCount::Waiter w;
  std::atomic<unsigned> signaled(0);
  ec.Notify(&signaled, false);
  ec.Prewait(&w);
  ec.Notify(&signaled, true);
  EXPECT_FALSE(ec.Wait(&w));
  EXPECT_EQ(signaled.load(), 0);
}

// Fake bounded counter-based queue.
struct TestQueue {
  std::atomic<int> val_;
  static const int kQueueSize = 10;

  TestQueue() : val_() {}

  ~TestQueue() { EXPECT_EQ(val_.load(), 0); }

  bool Push() {
    int val = val_.load(std::memory_order_relaxed);
    for (;;) {
      EXPECT_GE(val, 0);
      EXPECT_LE(val, kQueueSize);
      if (val == kQueueSize) return false;
      if (val_.compare_exchange_weak(val, val + 1, std::memory_order_relaxed))
        return true;
    }
  }

  bool Pop() {
    int val = val_.load(std::memory_order_relaxed);
    for (;;) {
      EXPECT_GE(val, 0);
      EXPECT_LE(val, kQueueSize);
      if (val == 0) return false;
      if (val_.compare_exchange_weak(val, val - 1, std::memory_order_relaxed))
        return true;
    }
  }

  bool Empty() { return val_.load(std::memory_order_relaxed) == 0; }
};

const int TestQueue::kQueueSize;

// A number of producers send messages to a set of consumers using a set of
// fake queues. Ensure that it does not crash, consumers don't deadlock and
// number of blocked and unblocked threads match.
TEST(EventCount, Stress) {
  const int kThreads = std::thread::hardware_concurrency();
  const int kEvents = 1 << 16;
  const int kQueues = 10;

  EventCount ec;
  std::atomic<unsigned> signaled(0);
  TestQueue queues[kQueues];

  std::vector<std::unique_ptr<Thread>> producers;
  for (int i = 0; i < kThreads; i++) {
    producers.emplace_back(Env::Default()->StartThread(
        ThreadOptions(), "ec_test", [&ec, &signaled, &queues]() {
          unsigned rnd =
              std::hash<std::thread::id>()(std::this_thread::get_id());
          for (int i = 0; i < kEvents; i++) {
            unsigned idx = rand_r(&rnd) % kQueues;
            if (queues[idx].Push()) {
              ec.Notify(&signaled, false);
              continue;
            }
            std::this_thread::yield();
            i--;
          }
        }));
  }

  std::vector<std::unique_ptr<Thread>> consumers;
  for (int i = 0; i < kThreads; i++) {
    consumers.emplace_back(Env::Default()->StartThread(
        ThreadOptions(), "ec_test", [&ec, &signaled, &queues]() {
          EventCount::Waiter w;
          unsigned rnd =
              std::hash<std::thread::id>()(std::this_thread::get_id());
          for (int i = 0; i < kEvents; i++) {
            unsigned idx = rand_r(&rnd) % kQueues;
            if (queues[idx].Pop()) continue;
            i--;
            ec.Prewait(&w);
            bool empty = true;
            for (int q = 0; q < kQueues; q++) {
              if (!queues[q].Empty()) {
                empty = false;
                break;
              }
            }
            if (!empty) continue;
            if (ec.Wait(&w)) {
              unsigned s = signaled.fetch_sub(1);
              EXPECT_GE(s, 1);
            }
          }
        }));
  }

  for (int i = 0; i < kThreads; i++) {
    producers[i].reset();
    consumers[i].reset();
  }
  EXPECT_EQ(signaled.load(), 0);
}

}  // namespace thread
}  // namespace tensorflow
