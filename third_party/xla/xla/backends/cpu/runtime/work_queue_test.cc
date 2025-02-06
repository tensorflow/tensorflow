/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/work_queue.h"

#include <atomic>
#include <cstddef>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/time/time.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/platform/threadpool.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

TEST(WorkQueueTest, WorkQueueSimple) {
  WorkQueue queue(20, 10);

  EXPECT_EQ(queue.Pop(0), std::make_optional(0));
  EXPECT_EQ(queue.Pop(0), std::make_optional(1));
  EXPECT_EQ(queue.Pop(0), std::nullopt);

  EXPECT_EQ(queue.Pop(1), std::make_optional(2));
}

TEST(WorkQueueTest, WorkQueueEmptyPartitions) {
  WorkQueue queue(1, 10);

  EXPECT_EQ(queue.Pop(0), std::make_optional(0));
  EXPECT_EQ(queue.Pop(0), std::nullopt);

  for (size_t i = 1; i < 10; ++i) {
    EXPECT_EQ(queue.Pop(i), std::nullopt);
  }
}

TEST(WorkQueueTest, WorkQueue) {
  for (size_t size : {1, 2, 4, 8, 16, 32, 64}) {
    for (size_t num_partitions : {1, 2, 3, 4, 5, 6, 7, 8}) {
      WorkQueue queue(size, num_partitions);

      std::vector<size_t> expected_tasks(size);
      absl::c_iota(expected_tasks, 0);

      std::vector<size_t> tasks;
      for (size_t i = 0; i < num_partitions; ++i) {
        while (std::optional<size_t> task = queue.Pop(i)) {
          tasks.push_back(*task);
        }
      }

      EXPECT_EQ(tasks, expected_tasks);
    }
  }
}

TEST(WorkQueueTest, Worker) {
  for (size_t size : {1, 2, 4, 8, 16, 32, 64}) {
    for (size_t num_partitions : {1, 2, 3, 4, 5, 6, 7, 8}) {
      // We check that no matter what is the initial partition, the worker
      // processes all tasks in the queue.
      for (size_t i = 0; i < num_partitions; ++i) {
        WorkQueue queue(size, num_partitions);
        Worker worker(i, &queue);

        std::vector<size_t> expected_tasks(size);
        absl::c_iota(expected_tasks, 0);

        std::vector<size_t> tasks;
        while (std::optional<size_t> task = worker.Pop()) {
          tasks.push_back(*task);
        }

        absl::c_sort(tasks);  // we pop tasks out of order
        EXPECT_EQ(tasks, expected_tasks);
      }
    }
  }
}

TEST(WorkQueueTest, WorkerConcurrency) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);

  size_t size = 1024;
  size_t num_partitions = 128;

  WorkQueue queue(size, num_partitions);

  // Check that we pop exactly `size` tasks.
  std::atomic<size_t> num_tasks(0);

  absl::BlockingCounter counter(num_partitions);
  for (size_t i = 0; i < num_partitions; ++i) {
    threads.Schedule([&, i] {
      Worker worker(i, &queue);
      while (std::optional<size_t> task = worker.Pop()) {
        ++num_tasks;
      }
      counter.DecrementCount();
    });
  }

  counter.Wait();
  EXPECT_EQ(num_tasks.load(), size);
}

TEST(WorkQueueTest, WorkerParallelize) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(), 8);

  std::vector<size_t> data(1024, 0);

  auto event = Worker::Parallelize(
      &device, 128, 1024, [&](size_t task_index) { ++data[task_index]; });
  tsl::BlockUntilReady(event);

  std::vector<size_t> expected(1024, 1);
  EXPECT_EQ(data, expected);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks.
//===----------------------------------------------------------------------===//

static void BM_PopTask(benchmark::State& state) {
  std::optional<WorkQueue> queue;
  std::optional<Worker> worker;

  size_t n = 0;
  for (auto _ : state) {
    if (n++ % (1024 * 10) == 0) {
      queue.emplace(/*num_tasks=*/1024 * 10, /*num_partitions=*/10);
      worker.emplace(0, &*queue);
    }
    worker->Pop();
  }
}

BENCHMARK(BM_PopTask);

}  // namespace
}  // namespace xla::cpu
