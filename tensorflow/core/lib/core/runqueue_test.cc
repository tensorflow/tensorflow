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

#include "tensorflow/core/lib/core/runqueue.h"

#include <thread>
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace thread {

TEST(RunQueue, Basic) {
  RunQueue<int, 4> q;
  // Check empty state.
  EXPECT_TRUE(q.Empty());
  EXPECT_EQ(0, q.Size());
  EXPECT_EQ(0, q.PopFront());
  std::vector<int> stolen;
  EXPECT_EQ(0, q.PopBack(&stolen));
  EXPECT_EQ(0, stolen.size());
  // Push one front, pop one front.
  EXPECT_EQ(0, q.PushFront(1));
  EXPECT_EQ(1, q.Size());
  EXPECT_EQ(1, q.PopFront());
  EXPECT_EQ(0, q.Size());
  // Push front to overflow.
  EXPECT_EQ(0, q.PushFront(2));
  EXPECT_EQ(1, q.Size());
  EXPECT_EQ(0, q.PushFront(3));
  EXPECT_EQ(2, q.Size());
  EXPECT_EQ(0, q.PushFront(4));
  EXPECT_EQ(3, q.Size());
  EXPECT_EQ(0, q.PushFront(5));
  EXPECT_EQ(4, q.Size());
  EXPECT_EQ(6, q.PushFront(6));
  EXPECT_EQ(4, q.Size());
  EXPECT_EQ(5, q.PopFront());
  EXPECT_EQ(3, q.Size());
  EXPECT_EQ(4, q.PopFront());
  EXPECT_EQ(2, q.Size());
  EXPECT_EQ(3, q.PopFront());
  EXPECT_EQ(1, q.Size());
  EXPECT_EQ(2, q.PopFront());
  EXPECT_EQ(0, q.Size());
  EXPECT_EQ(0, q.PopFront());
  // Push one back, pop one back.
  EXPECT_EQ(0, q.PushBack(7));
  EXPECT_EQ(1, q.Size());
  EXPECT_EQ(1, q.PopBack(&stolen));
  EXPECT_EQ(1, stolen.size());
  EXPECT_EQ(7, stolen[0]);
  EXPECT_EQ(0, q.Size());
  stolen.clear();
  // Push back to overflow.
  EXPECT_EQ(0, q.PushBack(8));
  EXPECT_EQ(1, q.Size());
  EXPECT_EQ(0, q.PushBack(9));
  EXPECT_EQ(2, q.Size());
  EXPECT_EQ(0, q.PushBack(10));
  EXPECT_EQ(3, q.Size());
  EXPECT_EQ(0, q.PushBack(11));
  EXPECT_EQ(4, q.Size());
  EXPECT_EQ(12, q.PushBack(12));
  EXPECT_EQ(4, q.Size());
  // Pop back in halves.
  EXPECT_EQ(2, q.PopBack(&stolen));
  EXPECT_EQ(2, stolen.size());
  EXPECT_EQ(10, stolen[0]);
  EXPECT_EQ(11, stolen[1]);
  EXPECT_EQ(2, q.Size());
  stolen.clear();
  EXPECT_EQ(1, q.PopBack(&stolen));
  EXPECT_EQ(1, stolen.size());
  EXPECT_EQ(9, stolen[0]);
  EXPECT_EQ(1, q.Size());
  stolen.clear();
  EXPECT_EQ(1, q.PopBack(&stolen));
  EXPECT_EQ(1, stolen.size());
  EXPECT_EQ(8, stolen[0]);
  stolen.clear();
  EXPECT_EQ(0, q.PopBack(&stolen));
  EXPECT_EQ(0, stolen.size());
  // Empty again.
  EXPECT_TRUE(q.Empty());
  EXPECT_EQ(0, q.Size());
}

// Empty tests that the queue is not claimed to be empty when is is in fact not.
// Emptiness property is crucial part of thread pool blocking scheme,
// so we go to great effort to ensure this property. We create a queue with
// 1 element and then push 1 element (either front or back at random) and pop
// 1 element (either front or back at random). So queue always contains at least
// 1 element, but otherwise changes chaotically. Another thread constantly tests
// that the queue is not claimed to be empty.
TEST(RunQueue, Empty) {
  RunQueue<int, 4> q;
  q.PushFront(1);
  std::atomic<bool> done(false);
  std::unique_ptr<Thread> mutator(
      Env::Default()->StartThread(ThreadOptions(), "queue_test", [&q, &done]() {
        unsigned rnd = 0;
        std::vector<int> stolen;
        for (int i = 0; i < 1 << 18; i++) {
          if (rand_r(&rnd) % 2)
            EXPECT_EQ(0, q.PushFront(1));
          else
            EXPECT_EQ(0, q.PushBack(1));
          if (rand_r(&rnd) % 2)
            EXPECT_EQ(1, q.PopFront());
          else {
            for (;;) {
              if (q.PopBack(&stolen) == 1) {
                stolen.clear();
                break;
              }
              EXPECT_EQ(0, stolen.size());
            }
          }
        }
        done = true;
      }));
  while (!done) {
    EXPECT_FALSE(q.Empty());
    int size = q.Size();
    EXPECT_GE(size, 1);
    EXPECT_LE(size, 2);
  }
  EXPECT_EQ(1, q.PopFront());
}

// Stress is a chaotic random test.
// One thread (owner) calls PushFront/PopFront, other threads call PushBack/
// PopBack. Ensure that we don't crash, deadlock, and all sanity checks pass.
TEST(RunQueue, Stress) {
  const int kEvents = 1 << 18;
  RunQueue<int, 8> q;
  std::atomic<int> total(0);
  std::vector<std::unique_ptr<Thread>> threads;
  threads.emplace_back(Env::Default()->StartThread(
      ThreadOptions(), "queue_test", [&q, &total]() {
        int sum = 0;
        int pushed = 1;
        int popped = 1;
        while (pushed < kEvents || popped < kEvents) {
          if (pushed < kEvents) {
            if (q.PushFront(pushed) == 0) {
              sum += pushed;
              pushed++;
            }
          }
          if (popped < kEvents) {
            int v = q.PopFront();
            if (v != 0) {
              sum -= v;
              popped++;
            }
          }
        }
        total += sum;
      }));
  for (int i = 0; i < 2; i++) {
    threads.emplace_back(Env::Default()->StartThread(
        ThreadOptions(), "queue_test", [&q, &total]() {
          int sum = 0;
          for (int i = 1; i < kEvents; i++) {
            if (q.PushBack(i) == 0) {
              sum += i;
              continue;
            }
            std::this_thread::yield();
            i--;
          }
          total += sum;
        }));
    threads.emplace_back(Env::Default()->StartThread(
        ThreadOptions(), "queue_test", [&q, &total]() {
          int sum = 0;
          std::vector<int> stolen;
          for (int i = 1; i < kEvents;) {
            if (q.PopBack(&stolen) == 0) {
              std::this_thread::yield();
              continue;
            }
            while (stolen.size() && i < kEvents) {
              int v = stolen.back();
              stolen.pop_back();
              EXPECT_NE(v, 0);
              sum += v;
              i++;
            }
          }
          while (stolen.size()) {
            int v = stolen.back();
            stolen.pop_back();
            EXPECT_NE(v, 0);
            while ((v = q.PushBack(v)) != 0) std::this_thread::yield();
          }
          total -= sum;
        }));
  }
  for (size_t i = 0; i < threads.size(); i++) threads[i].reset();
  CHECK(q.Empty());
  CHECK(total.load() == 0);
}

}  // namespace thread
}  // namespace tensorflow
