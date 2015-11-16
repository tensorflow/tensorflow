#include "tensorflow/core/util/work_sharder.h"

#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

void RunSharding(int64 num_workers, int64 total, int64 cost_per_unit) {
  thread::ThreadPool threads(Env::Default(), "test", 16);
  mutex mu;
  int64 num_shards = 0;
  int64 num_done_work = 0;
  std::vector<bool> work(total, false);
  Shard(num_workers, &threads, total, cost_per_unit,
        [&mu, &num_shards, &num_done_work, &work](int start, int limit) {
          VLOG(1) << "Shard [" << start << "," << limit << ")";
          mutex_lock l(mu);
          ++num_shards;
          for (; start < limit; ++start) {
            EXPECT_FALSE(work[start]);  // No duplicate
            ++num_done_work;
            work[start] = true;
          }
        });
  EXPECT_LE(num_shards, num_workers + 1);
  EXPECT_EQ(num_done_work, total);
  LOG(INFO) << num_workers << " " << total << " " << cost_per_unit << " "
            << num_shards;
}

TEST(Shard, Basic) {
  for (auto workers : {0, 1, 2, 3, 5, 7, 10, 11, 15, 100, 1000}) {
    for (auto total : {0, 1, 7, 10, 64, 100, 256, 1000, 9999}) {
      for (auto cost_per_unit : {0, 1, 11, 102, 1003, 10005, 1000007}) {
        RunSharding(workers, total, cost_per_unit);
      }
    }
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
