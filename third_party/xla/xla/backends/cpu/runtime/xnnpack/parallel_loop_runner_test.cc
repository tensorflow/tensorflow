/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/xnnpack/parallel_loop_runner.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/platform/threadpool.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

TEST(ParallelLoopRunnerTest, WorkQueueSimple) {
  ParallelLoopRunner::WorkQueue queue(20, 10);

  EXPECT_EQ(queue.Pop(0), std::make_optional(0));
  EXPECT_EQ(queue.Pop(0), std::make_optional(1));
  EXPECT_EQ(queue.Pop(0), std::nullopt);

  EXPECT_EQ(queue.Pop(1), std::make_optional(2));
}

TEST(ParallelLoopRunnerTest, WorkQueueEmptyPartitions) {
  ParallelLoopRunner::WorkQueue queue(1, 10);

  EXPECT_EQ(queue.Pop(0), std::make_optional(0));
  EXPECT_EQ(queue.Pop(0), std::nullopt);

  for (size_t i = 1; i < 10; ++i) {
    EXPECT_EQ(queue.Pop(i), std::nullopt);
  }
}

TEST(ParallelLoopRunnerTest, WorkQueue) {
  for (size_t size : {1, 2, 4, 8, 16, 32, 64}) {
    for (size_t num_partitions : {1, 2, 3, 4, 5, 6, 7, 8}) {
      ParallelLoopRunner::WorkQueue queue(size, num_partitions);

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

TEST(ParallelLoopRunnerTest, Worker) {
  for (size_t size : {1, 2, 4, 8, 16, 32, 64}) {
    for (size_t num_partitions : {1, 2, 3, 4, 5, 6, 7, 8}) {
      // We check that no matter what is the initial partition, the worker
      // processes all tasks in the queue.
      for (size_t i = 0; i < num_partitions; ++i) {
        ParallelLoopRunner::WorkQueue queue(size, num_partitions);
        ParallelLoopRunner::Worker worker(i, &queue);

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

TEST(ParallelLoopRunnerTest, WorkerConcurrency) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);

  size_t size = 1024;
  size_t num_partitions = 128;

  ParallelLoopRunner::WorkQueue queue(size, num_partitions);

  // Check that we pop exactly `size` tasks.
  std::atomic<size_t> num_tasks(0);

  absl::BlockingCounter counter(num_partitions);
  for (size_t i = 0; i < num_partitions; ++i) {
    threads.Schedule([&, i] {
      ParallelLoopRunner::Worker worker(i, &queue);
      while (std::optional<size_t> task = worker.Pop()) {
        ++num_tasks;
      }
      counter.DecrementCount();
    });
  }

  counter.Wait();
  EXPECT_EQ(num_tasks.load(), size);
}

TEST(ParallelLoopRunnerTest, Parallelize1D) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());
  ParallelLoopRunner runner(&device);

  constexpr int32_t d0 = 128;

  auto* data = new int32_t[d0]();
  auto cleanup = absl::Cleanup([&]() { delete[] data; });

  auto increment = [&](size_t offset) { data[offset] += 1; };

  runner.Parallelize(d0, increment);
  runner.Parallelize(d0, increment);
  runner.Parallelize(d0, increment);
  runner.Parallelize(d0, increment);
  runner.Parallelize(d0, increment);

  tsl::BlockUntilReady(ParallelLoopRunner::TakeDoneEvent(std::move(runner)));
  ASSERT_TRUE(absl::c_all_of(absl::MakeSpan(&data[0], d0),
                             [](int32_t value) { return value == 5; }));
}

TEST(ParallelLoopRunnerTest, Parallelize1DTile1D) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());
  ParallelLoopRunner runner(&device);

  constexpr int32_t d0 = 128;

  auto* data = new int32_t[d0]();
  auto cleanup = absl::Cleanup([&]() { delete[] data; });

  auto increment = [&](size_t offset, size_t extent) {
    for (size_t i = offset; i < offset + extent; ++i) {
      data[i] += 1;
    }
  };

  runner.Parallelize(d0, 1, increment);
  runner.Parallelize(d0, 2, increment);
  runner.Parallelize(d0, 3, increment);
  runner.Parallelize(d0, 4, increment);
  runner.Parallelize(d0, 5, increment);

  tsl::BlockUntilReady(ParallelLoopRunner::TakeDoneEvent(std::move(runner)));
  ASSERT_TRUE(absl::c_all_of(absl::MakeSpan(&data[0], d0),
                             [](int32_t value) { return value == 5; }));
}

TEST(ParallelLoopRunnerTest, Parallelize2DTile1D) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());
  ParallelLoopRunner runner(&device);

  constexpr int32_t d0 = 4;
  constexpr int32_t d1 = 39;

  auto* data = new int32_t[d0][d1]();
  auto cleanup = absl::Cleanup([&]() { delete[] data; });

  auto increment = [&](size_t i, size_t offset_j, size_t extent_j) {
    for (size_t j = offset_j; j < offset_j + extent_j; ++j) {
      data[i][j] += 1;
    }
  };

  runner.Parallelize(d0, d1, 1, increment);
  runner.Parallelize(d0, d1, 2, increment);
  runner.Parallelize(d0, d1, 3, increment);
  runner.Parallelize(d0, d1, 4, increment);
  runner.Parallelize(d0, d1, 5, increment);

  tsl::BlockUntilReady(ParallelLoopRunner::TakeDoneEvent(std::move(runner)));
  ASSERT_TRUE(absl::c_all_of(absl::MakeSpan(&data[0][0], d0 * d1),
                             [](int32_t value) { return value == 5; }));
}

TEST(ParallelLoopRunnerTest, Parallelize3DTile2D) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());
  ParallelLoopRunner runner(&device);

  constexpr int32_t d0 = 4;
  constexpr int32_t d1 = 39;
  constexpr int32_t d2 = 63;

  auto* data = new int32_t[d0][d1][d2]();
  auto cleanup = absl::Cleanup([&]() { delete[] data; });

  auto increment = [&](size_t i, size_t offset_j, size_t offset_k,
                       size_t extent_j, size_t extent_k) {
    for (size_t j = offset_j; j < offset_j + extent_j; ++j) {
      for (size_t k = offset_k; k < offset_k + extent_k; ++k) {
        data[i][j][k] += 1;
      }
    }
  };

  runner.Parallelize(d0, d1, d2, 1, 5, increment);
  runner.Parallelize(d0, d1, d2, 2, 4, increment);
  runner.Parallelize(d0, d1, d2, 3, 4, increment);
  runner.Parallelize(d0, d1, d2, 4, 3, increment);
  runner.Parallelize(d0, d1, d2, 5, 1, increment);

  tsl::BlockUntilReady(ParallelLoopRunner::TakeDoneEvent(std::move(runner)));
  ASSERT_TRUE(absl::c_all_of(absl::MakeSpan(&data[0][0][0], d0 * d1 * d2),
                             [](int32_t value) { return value == 5; }));
}

//===----------------------------------------------------------------------===//
// Performance benchmarks.
//===----------------------------------------------------------------------===//

static void BM_SingleTask1DLoop(benchmark::State& state) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());
  ParallelLoopRunner runner(&device);

  for (auto _ : state) {
    runner.Parallelize(1, 1, [](size_t, size_t) {});
    tsl::BlockUntilReady(runner.done_event());
  }
}

BENCHMARK(BM_SingleTask1DLoop);

static void BM_Parallelize2DTile1D(benchmark::State& state) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());
  ParallelLoopRunner runner(&device);

  size_t range = 4;
  size_t tile = 1;

  for (auto _ : state) {
    runner.Parallelize(range, range, tile, [](size_t, size_t, size_t) {});
    tsl::BlockUntilReady(runner.done_event());
  }
}

BENCHMARK(BM_Parallelize2DTile1D);

static void BM_Parallelize3DTile2D(benchmark::State& state) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());
  ParallelLoopRunner runner(&device);

  size_t range = 4;
  size_t tile = 1;

  for (auto _ : state) {
    runner.Parallelize(range, range, range, tile, tile,
                       [](size_t, size_t, size_t, size_t, size_t) {});
    tsl::BlockUntilReady(runner.done_event());
  }
}

BENCHMARK(BM_Parallelize3DTile2D);

}  // namespace
}  // namespace xla::cpu
