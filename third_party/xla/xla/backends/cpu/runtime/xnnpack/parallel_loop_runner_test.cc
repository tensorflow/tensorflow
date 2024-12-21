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

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
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
