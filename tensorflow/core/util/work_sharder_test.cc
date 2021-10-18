/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/work_sharder.h"

#include <atomic>
#include <vector>
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

void RunSharding(int64_t num_workers, int64_t total, int64_t cost_per_unit,
                 int64_t per_thread_max_parallelism,
                 thread::ThreadPool* threads) {
  mutex mu;
  int64_t num_shards = 0;
  int64_t num_done_work = 0;
  std::vector<bool> work(total, false);
  Shard(num_workers, threads, total, cost_per_unit,
        [=, &mu, &num_shards, &num_done_work, &work](int64_t start,
                                                     int64_t limit) {
          VLOG(1) << "Shard [" << start << "," << limit << ")";
          EXPECT_GE(start, 0);
          EXPECT_LE(limit, total);
          mutex_lock l(mu);
          ++num_shards;
          for (; start < limit; ++start) {
            EXPECT_FALSE(work[start]);  // No duplicate
            ++num_done_work;
            work[start] = true;
          }
        });
  LOG(INFO) << num_workers << " " << total << " " << cost_per_unit << " "
            << num_shards;
  EXPECT_EQ(num_done_work, total);
  if (std::min(num_workers, per_thread_max_parallelism) <
      threads->NumThreads()) {
    // If the intention is to limit the parallelism explicitly, we'd
    // better honor it. Ideally, even if per_thread_max_parallelism >
    // num_workers, we should expect that Shard() implementation do
    // not over-shard. Unfortunately, ThreadPoolDevice::parallelFor
    // tends to over-shard.
    EXPECT_LE(num_shards, 1 + per_thread_max_parallelism);
  }
}

TEST(Shard, Basic) {
  thread::ThreadPool threads(Env::Default(), "test", 16);
  for (auto workers : {0, 1, 2, 3, 5, 7, 10, 11, 15, 100, 1000}) {
    for (auto total : {0, 1, 7, 10, 64, 100, 256, 1000, 9999}) {
      for (auto cost_per_unit : {0, 1, 11, 102, 1003, 10005, 1000007}) {
        for (auto maxp : {1, 2, 4, 8, 100}) {
          ScopedPerThreadMaxParallelism s(maxp);
          RunSharding(workers, total, cost_per_unit, maxp, &threads);
        }
      }
    }
  }
}

TEST(Shard, OverflowTest) {
  thread::ThreadPool threads(Env::Default(), "test", 3);
  for (auto workers : {1, 2, 3}) {
    const int64_t total_elements = 1LL << 32;
    const int64_t cost_per_unit = 10;
    std::atomic<int64_t> num_elements(0);
    Shard(workers, &threads, total_elements, cost_per_unit,
          [&num_elements](int64_t start, int64_t limit) {
            num_elements += limit - start;
          });
    EXPECT_EQ(num_elements.load(), total_elements);
  }
}

void BM_Sharding(::testing::benchmark::State& state) {
  const int arg = state.range(0);

  thread::ThreadPool threads(Env::Default(), "test", 16);
  const int64_t total = 1LL << 30;
  auto lambda = [](int64_t start, int64_t limit) {};
  auto work = std::cref(lambda);
  for (auto s : state) {
    Shard(arg - 1, &threads, total, 1, work);
  }
}
BENCHMARK(BM_Sharding)->Range(1, 128);

}  // namespace
}  // namespace tensorflow
