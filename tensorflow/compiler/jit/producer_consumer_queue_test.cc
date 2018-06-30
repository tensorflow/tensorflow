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

#include "tensorflow/compiler/jit/producer_consumer_queue.h"

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

typedef ProducerConsumerQueue<int> IntQueue;

// Insert integers between low inclusive and high exclusive into q.
void PushRange(IntQueue *q, int low, int high) {
  while (low != high) {
    q->Put(low);
    VLOG(2) << "Pushing " << low;
    ++low;
  }
}

// Push the numbers between 0 and 999 inclusive from several threads in the
// pool.
void PushRanges(IntQueue *queue, thread::ThreadPool *pool) {
  VLOG(1) << "Adding 20-36";
  pool->Schedule([queue] { PushRange(queue, 20, 36); });
  VLOG(1) << "Adding 7-20";
  pool->Schedule([queue] { PushRange(queue, 7, 20); });
  VLOG(1) << "Adding 36-501";
  pool->Schedule([queue] { PushRange(queue, 36, 501); });
  VLOG(1) << "Adding 501-1000";
  pool->Schedule([queue] { PushRange(queue, 501, 1000); });
  VLOG(1) << "Adding 0-5";
  pool->Schedule([queue] { PushRange(queue, 0, 5); });
  VLOG(1) << "Adding 5-7";
  pool->Schedule([queue] { PushRange(queue, 5, 7); });
}

// Pop elements from queue using Get().  Make sure that exactly <high> elements
// were present and their values are all integers between 0 and high-1
// inclusive.
void GetRange(IntQueue *queue, int high) {
  VLOG(1) << "Testing Wait";
  std::vector<int> results;
  for (int i = 0; i != high; ++i) {
    int r = queue->Get();
    VLOG(2) << "Waited and got " << r;
    results.push_back(r);
  }
  CHECK_EQ(queue->count(), 0);
  std::sort(results.begin(), results.end());
  for (int i = 0; i != high; ++i) {
    CHECK(results[i] == i);
  }
}

// Pop elements from queue using TryGet().  Make sure that exactly <high>
// elements were present and their values are all integers between 0 and high-1
// inclusive.
void TryGetRange(IntQueue *queue, int high) {
  std::vector<int> results;
  // Give up if we don't get all the elements back from the queue
  // in 10 seconds.
  int timeout = 10;
  int r;
  for (int i = 0; i != high; ++i) {
    while (!queue->TryGet(&r)) {
      if (!timeout--) {
        LOG(FATAL) << "Can't find all elements in the queue";
      }
      VLOG(1) << "Sleeping for a second...";
      sleep(1);
    }
    VLOG(2) << "Popped " << r;
    results.push_back(r);
  }
  CHECK_EQ(queue->count(), 0);
  CHECK(!queue->TryGet(&r));
  std::sort(results.begin(), results.end());
  for (int i = 0; i != high; ++i) {
    CHECK_EQ(i, results[i]);
  }
}

const int kNumThreads = 15;

TEST(ProducerConsumerQueue, GetRange) {
  IntQueue queue;
  {
    thread::ThreadPool pool(Env::Default(), "test", kNumThreads);
    PushRanges(&queue, &pool);
  }
  GetRange(&queue, 1000);
}

TEST(ProducerConsumerQueue, TryGetRange) {
  IntQueue queue;
  {
    thread::ThreadPool pool(Env::Default(), "test", kNumThreads);
    PushRanges(&queue, &pool);
  }
  TryGetRange(&queue, 1000);
}

TEST(ProducerConsumerQueue, ParallelGetRange) {
  IntQueue queue;
  {
    thread::ThreadPool pool(Env::Default(), "test", kNumThreads);
    pool.Schedule([&queue] { GetRange(&queue, 1000); });
    PushRanges(&queue, &pool);
  }
}

TEST(ProducerConsumerQueue, ParallelTryGetRange) {
  IntQueue queue;
  {
    thread::ThreadPool pool(Env::Default(), "test", kNumThreads);
    pool.Schedule([&queue] { TryGetRange(&queue, 1000); });
    PushRanges(&queue, &pool);
  }
}

}  // namespace
}  // namespace tensorflow
