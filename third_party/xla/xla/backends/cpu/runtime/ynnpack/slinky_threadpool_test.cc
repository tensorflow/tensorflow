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

}  // namespace xla::cpu
