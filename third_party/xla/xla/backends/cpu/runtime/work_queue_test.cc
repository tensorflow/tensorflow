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
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/synchronization/blocking_counter.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/platform/threadpool.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

TEST(WorkQueueTest, WorkQueuePartitions) {
  auto task_range = [](size_t begin, size_t end) {
    return std::make_pair(begin, end);
  };

  {
    WorkQueue queue(/*num_tasks=*/2, /*num_partitions=*/4);
    EXPECT_EQ(queue.partition_range(0), task_range(0, 1));
    EXPECT_EQ(queue.partition_range(1), task_range(1, 2));
    EXPECT_EQ(queue.partition_range(2), task_range(2, 2));
    EXPECT_EQ(queue.partition_range(3), task_range(2, 2));
  }

  {
    WorkQueue queue(/*num_tasks=*/4, /*num_partitions=*/4);
    EXPECT_EQ(queue.partition_range(0), task_range(0, 1));
    EXPECT_EQ(queue.partition_range(1), task_range(1, 2));
    EXPECT_EQ(queue.partition_range(2), task_range(2, 3));
    EXPECT_EQ(queue.partition_range(3), task_range(3, 4));
  }

  {
    WorkQueue queue(/*num_tasks=*/5, /*num_partitions=*/4);
    EXPECT_EQ(queue.partition_range(0), task_range(0, 2));
    EXPECT_EQ(queue.partition_range(1), task_range(2, 3));
    EXPECT_EQ(queue.partition_range(2), task_range(3, 4));
    EXPECT_EQ(queue.partition_range(3), task_range(4, 5));
  }

  {
    WorkQueue queue(/*num_tasks=*/9, /*num_partitions=*/4);
    EXPECT_EQ(queue.partition_range(0), task_range(0, 3));
    EXPECT_EQ(queue.partition_range(1), task_range(3, 5));
    EXPECT_EQ(queue.partition_range(2), task_range(5, 7));
    EXPECT_EQ(queue.partition_range(3), task_range(7, 9));
  }

  {
    WorkQueue queue(/*num_tasks=*/14, /*num_partitions=*/4);
    EXPECT_EQ(queue.partition_range(0), task_range(0, 4));
    EXPECT_EQ(queue.partition_range(1), task_range(4, 8));
    EXPECT_EQ(queue.partition_range(2), task_range(8, 11));
    EXPECT_EQ(queue.partition_range(3), task_range(11, 14));
  }
}

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

TEST(WorkQueueTest, WorkerParallelizeDeadlockProof) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(), 8);

  std::vector<size_t> data(10 * 1024, 0);
  absl::BlockingCounter counter(10);

  // Dispatch and wait for parallel loops completion in the same thread pool
  // where they execute, to test that this work scheduling pattern doesn't lead
  // to deadlocks.
  for (size_t i = 0; i < 10; ++i) {
    threads.Schedule([&, i] {
      auto event = Worker::Parallelize(
          &device, 32, 1024,
          [&](size_t task_index) { ++data[i * 1024 + task_index]; });
      tsl::BlockUntilReady(event);
      counter.DecrementCount();
    });
  }

  counter.Wait();

  std::vector<size_t> expected(10 * 1024, 1);
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

static void BM_PopTaskMultiThreaded(benchmark::State& state) {
  size_t num_threads = state.range(0);
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "benchmark",
                                  num_threads);

  for (auto _ : state) {
    absl::BlockingCounter counter(num_threads);
    WorkQueue queue(/*num_tasks=*/1024 * 10, /*num_partitions=*/num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
      threads.Schedule([i, &queue, &counter] {
        Worker worker(i, &queue);
        while (std::optional<size_t> task = worker.Pop()) {
        }
        counter.DecrementCount();
      });
    }

    counter.Wait();
  }

  state.SetItemsProcessed(state.iterations() * 1024 * 10);
}

BENCHMARK(BM_PopTaskMultiThreaded)
    ->MeasureProcessCPUTime()
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64);

}  // namespace
}  // namespace xla::cpu
