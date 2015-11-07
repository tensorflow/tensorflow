#include "tensorflow/core/util/work_sharder.h"

#include <vector>
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

void Shard(int num_workers, thread::ThreadPool* workers, int64 total,
           int64 cost_per_unit, std::function<void(int64, int64)> work) {
  CHECK_GE(total, 0);
  if (total == 0) {
    return;
  }
  if (num_workers <= 1) {
    // Just inline the whole work since we only have 1 thread (core).
    work(0, total);
    return;
  }
  cost_per_unit = std::max(1LL, cost_per_unit);
  // We shard [0, total) into "num_shards" shards.
  //   1 <= num_shards <= num worker threads
  //
  // If total * cost_per_unit is small, it is not worth shard too
  // much. Let us assume each cost unit is 1ns, kMinCostPerShard=10000
  // is 10us.
  static const int64 kMinCostPerShard = 10000;
  const int num_shards = std::max(
      1, std::min<int>(num_workers, total * cost_per_unit / kMinCostPerShard));
  // Each shard contains up to "block_size" units. [0, total) is sharded
  // into:
  //   [0, block_size), [block_size, 2*block_size), ...
  // The 1st shard is done by the caller thread and the other shards
  // are dispatched to the worker threads. The last shard may be smaller than
  // block_size.
  const int64 block_size = (total + num_shards - 1) / num_shards;
  CHECK_GT(block_size, 0);  // total > 0 guarantees this.
  if (block_size >= total) {
    work(0, total);
    return;
  }
  const int num_shards_used = (total + block_size - 1) / block_size;
  BlockingCounter counter(num_shards_used - 1);
  for (int64 start = block_size; start < total; start += block_size) {
    auto limit = std::min(start + block_size, total);
    workers->Schedule([&work, &counter, start, limit]() {
      work(start, limit);        // Compute the shard.
      counter.DecrementCount();  // The shard is done.
    });
  }

  // Inline execute the 1st shard.
  work(0, std::min(block_size, total));
  counter.Wait();
}

}  // end namespace tensorflow
