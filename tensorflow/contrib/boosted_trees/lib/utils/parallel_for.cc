// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include "tensorflow/contrib/boosted_trees/lib/utils/parallel_for.h"
#include "tensorflow/core/lib/core/blocking_counter.h"

namespace tensorflow {
namespace boosted_trees {
namespace utils {

void ParallelFor(int64 batch_size, int64 desired_parallelism,
                 thread::ThreadPool* thread_pool,
                 std::function<void(int64, int64)> do_work) {
  // Parallelize work over the batch.
  if (desired_parallelism <= 0) {
    do_work(0, batch_size);
    return;
  }
  const int num_shards = std::max<int>(
      1, std::min(static_cast<int64>(desired_parallelism), batch_size));
  const int64 block_size = (batch_size + num_shards - 1) / num_shards;
  CHECK_GT(block_size, 0);
  const int num_shards_used = (batch_size + block_size - 1) / block_size;
  BlockingCounter counter(num_shards_used - 1);
  for (int64 start = block_size; start < batch_size; start += block_size) {
    auto end = std::min(start + block_size, batch_size);
    thread_pool->Schedule([&do_work, &counter, start, end]() {
      do_work(start, end);
      counter.DecrementCount();
    });
  }

  // Execute first shard on main thread.
  do_work(0, std::min(block_size, batch_size));
  counter.Wait();
}

}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow
