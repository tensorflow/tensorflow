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

#include "xla/backends/cpu/runtime/ynnpack/slinky_threadpool.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/strings/str_format.h"
#include "slinky/base/ref_count.h"
#include "slinky/base/thread_pool.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::cpu {

TEST(SlinkyThreadPoolTest, InlineScheduling) {
  SlinkyThreadPool thread_pool(
      static_cast<Eigen::ThreadPoolInterface*>(nullptr));

  static constexpr size_t size = 10000;

  std::vector<int32_t> data(size, 0);
  auto inc = [&](size_t i) { data[i]++; };

  thread_pool.parallel_for(size, inc);

  std::vector<int32_t> expected(size, 1);
  EXPECT_EQ(data, expected);
}

TEST(SlinkyThreadPoolTest, SingleLoop) {
  tsl::thread::ThreadPool test_thread_pool(tsl::Env::Default(), "test", 4);
  SlinkyThreadPool thread_pool(test_thread_pool.AsEigenThreadPool());

  static constexpr size_t size = 10000;

  std::vector<int32_t> data(size, 0);
  auto inc = [&](size_t i) { data[i]++; };

  thread_pool.parallel_for(size, inc);

  std::vector<int32_t> expected(size, 1);
  EXPECT_EQ(data, expected);
}

TEST(SlinkyThreadPoolTest, LoopChain) {
  tsl::thread::ThreadPool test_thread_pool(tsl::Env::Default(), "test", 4);
  SlinkyThreadPool thread_pool(test_thread_pool.AsEigenThreadPool());

  static constexpr size_t size = 10000;

  std::vector<int32_t> data(size, 0);
  auto inc = [&](size_t i) { data[i]++; };

  thread_pool.parallel_for(size, inc);
  thread_pool.parallel_for(size, inc);
  thread_pool.parallel_for(size, inc);
  thread_pool.parallel_for(size, inc);
  thread_pool.parallel_for(size, inc);

  std::vector<int32_t> expected(size, 5);
  EXPECT_EQ(data, expected);
}

TEST(SlinkyThreadPoolTest, NestedLoops) {
  tsl::thread::ThreadPool test_thread_pool(tsl::Env::Default(), "test", 4);
  SlinkyThreadPool thread_pool(test_thread_pool.AsEigenThreadPool());

  static constexpr size_t size = 100;

  std::array<std::atomic<int32_t>, size> data = {{0}};
  auto inc = [&](size_t i) { data[i]++; };

  thread_pool.parallel_for(
      size, [&](size_t i) { thread_pool.parallel_for(size, inc); });

  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(data[i], size);
  }
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below.
//===----------------------------------------------------------------------===//

static void BM_ParallelFor(benchmark::State& state) {
  int64_t num_threads = state.range(0);
  int64_t num_threadpools = state.range(1);

  tsl::thread::ThreadPool threads(tsl::Env::Default(), "bench", num_threads);
  std::vector<SlinkyThreadPool> thread_pools;
  for (size_t i = 0; i < num_threadpools; ++i) {
    thread_pools.emplace_back(threads.AsEigenThreadPool());
  }

  static constexpr size_t kNumLoops = 100;
  static constexpr size_t kLoopSize = 100;

  for (auto _ : state) {
    std::vector<slinky::ref_count<slinky::thread_pool::task>> tasks;

    for (size_t i = 0; i < kNumLoops; ++i) {
      SlinkyThreadPool& thread_pool = thread_pools[i % num_threadpools];
      tasks.push_back(thread_pool.enqueue(
          kLoopSize, [](int64_t) {}, std::numeric_limits<int32_t>::max()));
    }

    for (size_t i = 0; i < kNumLoops; ++i) {
      SlinkyThreadPool& thread_pool = thread_pools[i % num_threadpools];
      thread_pool.wait_for(&*tasks[i]);
    }
  }

  state.SetItemsProcessed(kLoopSize * kNumLoops * state.iterations());
  state.SetLabel(absl::StrFormat("#threads=%d, #threadpools=%d", num_threads,
                                 num_threadpools));
}

BENCHMARK(BM_ParallelFor)
    ->MeasureProcessCPUTime()
    ->ArgPair(8, 1)
    ->ArgPair(8, 2)
    ->ArgPair(8, 4)
    ->ArgPair(8, 8)
    ->ArgPair(8, 16)
    ->ArgPair(16, 1)
    ->ArgPair(16, 2)
    ->ArgPair(16, 4)
    ->ArgPair(16, 8)
    ->ArgPair(16, 16)
    ->ArgPair(32, 1)
    ->ArgPair(32, 2)
    ->ArgPair(32, 4)
    ->ArgPair(32, 8)
    ->ArgPair(32, 16);

// Measures single-task latency: enqueue one parallel loop and wait. Dominant
// cost here is fixed overhead: task allocation, mutex ops, thread wake-ups.
static void BM_SingleTaskLatency(benchmark::State& state) {
  int64_t num_threads = state.range(0);
  int64_t loop_size = state.range(1);

  tsl::thread::ThreadPool threads(tsl::Env::Default(), "bench", num_threads);
  SlinkyThreadPool thread_pool(threads.AsEigenThreadPool());

  for (auto _ : state) {
    auto task = thread_pool.enqueue(
        loop_size, [](int64_t) {}, std::numeric_limits<int32_t>::max());
    thread_pool.wait_for(&*task);
  }

  state.SetItemsProcessed(loop_size * state.iterations());
  state.SetLabel(
      absl::StrFormat("#threads=%d, loop=%d", num_threads, loop_size));
}

BENCHMARK(BM_SingleTaskLatency)
    ->MeasureProcessCPUTime()
    ->ArgPair(8, 1)
    ->ArgPair(8, 8)
    ->ArgPair(8, 64)
    ->ArgPair(8, 1024)
    ->ArgPair(16, 1)
    ->ArgPair(16, 8)
    ->ArgPair(16, 64)
    ->ArgPair(16, 1024);

// Stresses nested parallel_for; inner loops are enqueued from Eigen worker
// threads. Exercises the recursion-prevention path and nested scheduling.
static void BM_NestedParallelFor(benchmark::State& state) {
  int64_t num_threads = state.range(0);
  int64_t outer = state.range(1);
  int64_t inner = state.range(2);

  tsl::thread::ThreadPool threads(tsl::Env::Default(), "bench", num_threads);
  SlinkyThreadPool thread_pool(threads.AsEigenThreadPool());

  for (auto _ : state) {
    thread_pool.parallel_for(outer, [&](size_t i) {
      thread_pool.parallel_for(inner,
                               [](size_t j) { benchmark::DoNotOptimize(j); });
    });
  }

  state.SetItemsProcessed(outer * inner * state.iterations());
  state.SetLabel(absl::StrFormat("#threads=%d, outer=%d, inner=%d", num_threads,
                                 outer, inner));
}

BENCHMARK(BM_NestedParallelFor)
    ->MeasureProcessCPUTime()
    ->Args({8, 16, 16})
    ->Args({8, 32, 32})
    ->Args({16, 16, 16})
    ->Args({16, 32, 32});

// Measures enqueue-only cost to isolate task creation + scheduling from the
// joint cost of execution. We wait on tasks in a second loop, which typically
// finds them already done when num_threads >= 1.
static void BM_EnqueueBatch(benchmark::State& state) {
  int64_t num_threads = state.range(0);
  int64_t batch = state.range(1);

  tsl::thread::ThreadPool threads(tsl::Env::Default(), "bench", num_threads);
  SlinkyThreadPool thread_pool(threads.AsEigenThreadPool());

  std::vector<slinky::ref_count<slinky::thread_pool::task>> tasks;
  tasks.reserve(batch);

  for (auto _ : state) {
    tasks.clear();
    for (int64_t i = 0; i < batch; ++i) {
      tasks.push_back(thread_pool.enqueue(
          1, [](int64_t) {}, std::numeric_limits<int32_t>::max()));
    }
    for (int64_t i = 0; i < batch; ++i) {
      thread_pool.wait_for(&*tasks[i]);
    }
  }

  state.SetItemsProcessed(batch * state.iterations());
  state.SetLabel(absl::StrFormat("#threads=%d, batch=%d", num_threads, batch));
}

BENCHMARK(BM_EnqueueBatch)
    ->MeasureProcessCPUTime()
    ->ArgPair(8, 10)
    ->ArgPair(8, 100)
    ->ArgPair(16, 10)
    ->ArgPair(16, 100);

}  // namespace xla::cpu
