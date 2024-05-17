/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/incremental_barrier.h"

#include <atomic>

#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "benchmark/benchmark.h"  // from @com_google_benchmark
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {
namespace {

// A thread-safe counter class.
class Counter {
 public:
  void Increment() TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    ++count_;
  }

  int GetCount() TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    return count_;
  }

 private:
  mutex mu_;
  int count_ = 0;
};

TEST(IncrementalBarrierTest, RunInstantlyWhenZeroClosure) {
  Counter counter;
  EXPECT_EQ(counter.GetCount(), 0);
  {
    IncrementalBarrier::DoneCallback done_callback =
        absl::bind_front(&Counter::Increment, &counter);
    IncrementalBarrier barrier(done_callback);
    EXPECT_EQ(counter.GetCount(), 0);
  }
  EXPECT_EQ(counter.GetCount(), 1);
}

TEST(IncrementalBarrierTest, RunAfterNumClosuresOneNowTwoLater) {
  Counter counter;

  IncrementalBarrier::BarrierCallback bc1, bc2;
  {
    IncrementalBarrier::DoneCallback done_callback =
        absl::bind_front(&Counter::Increment, &counter);
    IncrementalBarrier barrier(done_callback);

    CHECK_EQ(counter.GetCount(), 0);

    bc1 = barrier.Inc();
    bc2 = barrier.Inc();

    IncrementalBarrier::BarrierCallback bc3 = barrier.Inc();
    bc3();

    CHECK_EQ(counter.GetCount(), 0);
  }

  CHECK_EQ(counter.GetCount(), 0);
  bc1();
  CHECK_EQ(counter.GetCount(), 0);
  bc2();
  CHECK_EQ(counter.GetCount(), 1);
}

TEST(IncrementalBarrierTest, RunAfterNumClosuresConcurrency) {
  const int num_closure = 100, num_thread = 2;
  std::atomic<int> schedule_count{0};
  Counter counter;

  {
    IncrementalBarrier::DoneCallback done_callback =
        absl::bind_front(&Counter::Increment, &counter);
    IncrementalBarrier barrier(done_callback);

    CHECK_EQ(counter.GetCount(), 0);

    tensorflow::thread::ThreadPool pool(tensorflow::Env::Default(),
                                        "BarrierClosure", num_thread);
    for (int i = 0; i < num_closure; ++i) {
      pool.Schedule([&barrier, &schedule_count]() {
        schedule_count.fetch_add(1);
        IncrementalBarrier::BarrierCallback bc = barrier.Inc();

        Env::Default()->SleepForMicroseconds(100);
        bc();
      });
    }

    CHECK_EQ(counter.GetCount(), 0);
  }

  CHECK_EQ(schedule_count.load(std::memory_order_relaxed), 100);
  CHECK_EQ(counter.GetCount(), 1);
}

#if defined(PLATFORM_GOOGLE)
void BM_FunctionInc(benchmark::State& state) {
  IncrementalBarrier barrier([] {});
  for (auto _ : state) {
    barrier.Inc()();
  }
}

BENCHMARK(BM_FunctionInc);
#endif  // PLATFORM_GOOGLE

}  // namespace
}  // namespace tensorflow
