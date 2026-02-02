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

void RunSharding(int64 num_workers, int64 total, int64 cost_per_unit,
                 thread::ThreadPool* threads) {
  mutex mu;
  int64 num_shards = 0;
  int64 num_done_work = 0;
  std::vector<bool> work(total, false);
  Shard(num_workers, threads, total, cost_per_unit,
        [=, &mu, &num_shards, &num_done_work, &work](int64 start, int64 limit) {
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
  EXPECT_EQ(num_done_work, total);
  LOG(INFO) << num_workers << " " << total << " " << cost_per_unit << " "
            << num_shards;
}

TEST(Shard, Basic) {
  thread::ThreadPool threads(Env::Default(), "test", 16);
  for (auto workers : {0, 1, 2, 3, 5, 7, 10, 11, 15, 100, 1000}) {
    for (auto total : {0, 1, 7, 10, 64, 100, 256, 1000, 9999}) {
      for (auto cost_per_unit : {0, 1, 11, 102, 1003, 10005, 1000007}) {
        RunSharding(workers, total, cost_per_unit, &threads);
      }
    }
  }
}

TEST(Shard, OverflowTest) {
  thread::ThreadPool threads(Env::Default(), "test", 3);
  for (auto workers : {1, 2, 3}) {
    const int64 total_elements = 1LL << 32;
    const int64 cost_per_unit = 10;
    std::atomic<int64> num_elements(0);
    Shard(workers, &threads, total_elements, cost_per_unit,
          [&num_elements](int64 start, int64 limit) {
            num_elements += limit - start;
          });
    EXPECT_EQ(num_elements.load(), total_elements);
  }
}

void BM_Sharding(int iters, int arg) {
  thread::ThreadPool threads(Env::Default(), "test", 16);
  const int64 total = 1LL << 30;
  auto lambda = [](int64 start, int64 limit) {};
  auto work = std::cref(lambda);
  for (; iters > 0; iters -= arg) {
    Shard(arg - 1, &threads, total, 1, work);
  }
}
BENCHMARK(BM_Sharding)->Range(1, 128);

}  // namespace
}  // namespace tensorflow
