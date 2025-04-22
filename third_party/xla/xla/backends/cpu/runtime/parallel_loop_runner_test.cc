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

#include "xla/backends/cpu/runtime/parallel_loop_runner.h"

#include <cstddef>
#include <cstdint>
#include <tuple>
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

using RangeDim = ParallelLoopRunner::RangeDim;
using RangeIndex = ParallelLoopRunner::RangeIndex;

using TileDim = ParallelLoopRunner::TileDim;
using TileIndex = ParallelLoopRunner::TileIndex;

TEST(ParallelLoopRunnerTest, NumTasks) {
  EXPECT_EQ(ParallelLoopRunner::NumTasks(RangeDim{2}), 2);
  EXPECT_EQ(ParallelLoopRunner::NumTasks(RangeDim{2}, RangeDim{3}), 2 * 3);
  EXPECT_EQ(ParallelLoopRunner::NumTasks(RangeDim{2}, TileDim{10, 4}), 2 * 3);
}

TEST(ParallelLoopRunnerTest, Delinearize) {
  EXPECT_EQ(ParallelLoopRunner::Delinearize(0, RangeDim{2}, RangeDim{3}),
            std::make_tuple(RangeIndex{0}, RangeIndex{0}));
  EXPECT_EQ(ParallelLoopRunner::Delinearize(1, RangeDim{2}, RangeDim{3}),
            std::make_tuple(RangeIndex{0}, RangeIndex{1}));
  EXPECT_EQ(ParallelLoopRunner::Delinearize(2, RangeDim{2}, RangeDim{3}),
            std::make_tuple(RangeIndex{0}, RangeIndex{2}));
  EXPECT_EQ(ParallelLoopRunner::Delinearize(3, RangeDim{2}, RangeDim{3}),
            std::make_tuple(RangeIndex{1}, RangeIndex{0}));

  EXPECT_EQ(ParallelLoopRunner::Delinearize(0, RangeDim{2}, TileDim{10, 4}),
            std::make_tuple(RangeIndex{0}, TileIndex{0, 4}));
  EXPECT_EQ(ParallelLoopRunner::Delinearize(1, RangeDim{2}, TileDim{10, 4}),
            std::make_tuple(RangeIndex{0}, TileIndex{4, 4}));
  EXPECT_EQ(ParallelLoopRunner::Delinearize(2, RangeDim{2}, TileDim{10, 4}),
            std::make_tuple(RangeIndex{0}, TileIndex{8, 2}));
  EXPECT_EQ(ParallelLoopRunner::Delinearize(3, RangeDim{2}, TileDim{10, 4}),
            std::make_tuple(RangeIndex{1}, TileIndex{0, 4}));
}

TEST(ParallelLoopRunnerTest, DynamicDimensions) {
  EXPECT_EQ(ParallelLoopRunner::DynamicDimensions(4, TileDim{128, 4}),
            std::make_tuple(TileDim{128, 128}));
  EXPECT_EQ(ParallelLoopRunner::DynamicDimensions(4, TileDim{1024, 4}),
            std::make_tuple(TileDim{1024, 256}));
  EXPECT_EQ(ParallelLoopRunner::DynamicDimensions(4, TileDim{1024, 512}),
            std::make_tuple(TileDim{1024, 512}));
  EXPECT_EQ(ParallelLoopRunner::DynamicDimensions(4, TileDim{1024, 400}),
            std::make_tuple(TileDim{1024, 400}));

  EXPECT_EQ(
      ParallelLoopRunner::DynamicDimensions(4, RangeDim{2}, TileDim{1024, 128}),
      std::make_tuple(RangeDim{2}, TileDim{1024, 512}));

  EXPECT_EQ(ParallelLoopRunner::DynamicDimensions(16, TileDim{1024, 128},
                                                  TileDim{1024, 128}),
            std::make_tuple(TileDim{1024, 128}, TileDim{1024, 512}));

  EXPECT_EQ(ParallelLoopRunner::DynamicDimensions(32, TileDim{1024, 128},
                                                  TileDim{1024, 128}),
            std::make_tuple(TileDim{1024, 128}, TileDim{1024, 256}));

  EXPECT_EQ(ParallelLoopRunner::DynamicDimensions(
                32, RangeDim{1}, TileDim{512, 16}, TileDim{12, 5}),
            std::make_tuple(RangeDim{1}, TileDim{512, 128}, TileDim{12, 12}));
}

TEST(ParallelLoopRunnerTest, Parallelize1D) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());
  ParallelLoopRunner runner(&device);

  constexpr RangeDim d0 = {128};

  auto* data = new int32_t[d0.range]();
  auto cleanup = absl::Cleanup([&]() { delete[] data; });

  auto increment = [&](RangeIndex i) { data[i.offset] += 1; };

  runner.Parallelize(d0, increment);
  runner.Parallelize(d0, increment);
  runner.Parallelize(d0, increment);
  runner.Parallelize(d0, increment);
  runner.Parallelize(d0, increment);

  tsl::BlockUntilReady(ParallelLoopRunner::TakeDoneEvent(std::move(runner)));
  ASSERT_TRUE(absl::c_all_of(absl::MakeSpan(&data[0], d0.range),
                             [](int32_t value) { return value == 5; }));
}

TEST(ParallelLoopRunnerTest, Parallelize2D) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());
  ParallelLoopRunner runner(&device);

  constexpr RangeDim d0 = {8};
  constexpr RangeDim d1 = {9};

  auto* data = new int32_t[d0.range][d1.range]();
  auto cleanup = absl::Cleanup([&]() { delete[] data; });

  auto increment = [&](RangeIndex i, RangeIndex j) {
    data[i.offset][j.offset] += 1;
  };

  runner.Parallelize(d0, d1, increment);
  runner.Parallelize(d0, d1, increment);
  runner.Parallelize(d0, d1, increment);
  runner.Parallelize(d0, d1, increment);
  runner.Parallelize(d0, d1, increment);

  tsl::BlockUntilReady(ParallelLoopRunner::TakeDoneEvent(std::move(runner)));
  ASSERT_TRUE(absl::c_all_of(absl::MakeSpan(&data[0][0], d0.range * d1.range),
                             [](int32_t value) { return value == 5; }));
}

TEST(ParallelLoopRunnerTest, Parallelize1DTile1D) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());
  ParallelLoopRunner runner(&device);

  constexpr int32_t r0 = 128;

  auto* data = new int32_t[r0]();
  auto cleanup = absl::Cleanup([&]() { delete[] data; });

  auto increment = [&](TileIndex i) {
    for (size_t i0 = i.offset; i0 < i.offset + i.count; ++i0) {
      data[i0] += 1;
    }
  };

  runner.Parallelize(TileDim{r0, 1}, increment);
  runner.Parallelize(TileDim{r0, 2}, increment);
  runner.Parallelize(TileDim{r0, 3}, increment);
  runner.Parallelize(TileDim{r0, 4}, increment);
  runner.Parallelize(TileDim{r0, 5}, increment);

  tsl::BlockUntilReady(ParallelLoopRunner::TakeDoneEvent(std::move(runner)));
  ASSERT_TRUE(absl::c_all_of(absl::MakeSpan(&data[0], r0),
                             [](int32_t value) { return value == 5; }));
}

TEST(ParallelLoopRunnerTest, Parallelize2DTile1D) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());
  ParallelLoopRunner runner(&device);

  constexpr int32_t r0 = 4;
  constexpr int32_t r1 = 39;

  auto* data = new int32_t[r0][r1]();
  auto cleanup = absl::Cleanup([&]() { delete[] data; });

  auto increment = [&](RangeIndex i, TileIndex j) {
    for (size_t j0 = j.offset; j0 < j.offset + j.count; ++j0) {
      data[i.offset][j0] += 1;
    }
  };

  runner.Parallelize(RangeDim{r0}, TileDim{r1, 1}, increment);
  runner.Parallelize(RangeDim{r0}, TileDim{r1, 2}, increment);
  runner.Parallelize(RangeDim{r0}, TileDim{r1, 3}, increment);
  runner.Parallelize(RangeDim{r0}, TileDim{r1, 4}, increment);
  runner.Parallelize(RangeDim{r0}, TileDim{r1, 5}, increment);

  tsl::BlockUntilReady(ParallelLoopRunner::TakeDoneEvent(std::move(runner)));
  ASSERT_TRUE(absl::c_all_of(absl::MakeSpan(&data[0][0], r0 * r1),
                             [](int32_t value) { return value == 5; }));
}

TEST(ParallelLoopRunnerTest, Parallelize2DTile2D) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());
  ParallelLoopRunner runner(&device);

  constexpr int32_t r0 = 4;
  constexpr int32_t r1 = 39;

  auto* data = new int32_t[r0][r1]();
  auto cleanup = absl::Cleanup([&]() { delete[] data; });

  auto increment = [&](TileIndex i, TileIndex j) {
    for (size_t i0 = i.offset; i0 < i.offset + i.count; ++i0) {
      for (size_t j0 = j.offset; j0 < j.offset + j.count; ++j0) {
        data[i0][j0] += 1;
      }
    }
  };

  runner.Parallelize(TileDim{r0, 5}, TileDim{r1, 1}, increment);
  runner.Parallelize(TileDim{r0, 4}, TileDim{r1, 2}, increment);
  runner.Parallelize(TileDim{r0, 3}, TileDim{r1, 3}, increment);
  runner.Parallelize(TileDim{r0, 2}, TileDim{r1, 4}, increment);
  runner.Parallelize(TileDim{r0, 1}, TileDim{r1, 5}, increment);

  tsl::BlockUntilReady(ParallelLoopRunner::TakeDoneEvent(std::move(runner)));
  ASSERT_TRUE(absl::c_all_of(absl::MakeSpan(&data[0][0], r0 * r1),
                             [](int32_t value) { return value == 5; }));
}

TEST(ParallelLoopRunnerTest, Parallelize3DTile2D) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());
  ParallelLoopRunner runner(&device);

  constexpr int32_t r0 = 4;
  constexpr int32_t r1 = 39;
  constexpr int32_t r2 = 63;

  auto* data = new int32_t[r0][r1][r2]();
  auto cleanup = absl::Cleanup([&]() { delete[] data; });

  auto increment = [&](RangeIndex i, TileIndex j, TileIndex k) {
    for (size_t j0 = j.offset; j0 < j.offset + j.count; ++j0) {
      for (size_t k0 = k.offset; k0 < k.offset + k.count; ++k0) {
        data[i.offset][j0][k0] += 1;
      }
    }
  };

  runner.Parallelize(RangeDim{r0}, TileDim{r1, 5}, TileDim{r2, 1}, increment);
  runner.Parallelize(RangeDim{r0}, TileDim{r1, 4}, TileDim{r2, 2}, increment);
  runner.Parallelize(RangeDim{r0}, TileDim{r1, 3}, TileDim{r2, 3}, increment);
  runner.Parallelize(RangeDim{r0}, TileDim{r1, 2}, TileDim{r2, 4}, increment);
  runner.Parallelize(RangeDim{r0}, TileDim{r1, 1}, TileDim{r2, 5}, increment);

  tsl::BlockUntilReady(ParallelLoopRunner::TakeDoneEvent(std::move(runner)));
  ASSERT_TRUE(absl::c_all_of(absl::MakeSpan(&data[0][0][0], r0 * r1 * r2),
                             [](int32_t value) { return value == 5; }));
}

//===----------------------------------------------------------------------===//
// Performance benchmarks.
//===----------------------------------------------------------------------===//

static void BM_NumTasks4DTile4D(benchmark::State& state) {
  auto t0 = TileDim{10, 4};
  auto t1 = TileDim{20, 5};
  auto t2 = TileDim{30, 6};
  auto t3 = TileDim{40, 70};

  for (auto _ : state) {
    benchmark::DoNotOptimize(t0);
    benchmark::DoNotOptimize(t1);
    benchmark::DoNotOptimize(t2);
    benchmark::DoNotOptimize(t3);
    benchmark::DoNotOptimize(ParallelLoopRunner::NumTasks(t0, t1, t2, t3));
  }
}

BENCHMARK(BM_NumTasks4DTile4D);

static void BM_Delinearize2D(benchmark::State& state) {
  auto t0 = RangeDim{10};
  auto t1 = RangeDim{20};

  for (auto _ : state) {
    benchmark::DoNotOptimize(t0);
    benchmark::DoNotOptimize(t1);
    benchmark::DoNotOptimize(ParallelLoopRunner::Delinearize(123, t0, t1));
  }
}

BENCHMARK(BM_Delinearize2D);

static void BM_Delinearize2DTile1D(benchmark::State& state) {
  auto t0 = RangeDim{10};
  auto t1 = TileDim{10, 4};

  for (auto _ : state) {
    benchmark::DoNotOptimize(t0);
    benchmark::DoNotOptimize(t1);
    benchmark::DoNotOptimize(ParallelLoopRunner::Delinearize(123, t0, t1));
  }
}

BENCHMARK(BM_Delinearize2DTile1D);

static void BM_Delinearize2DTile2D(benchmark::State& state) {
  auto t0 = TileDim{10, 4};
  auto t1 = TileDim{20, 5};

  for (auto _ : state) {
    benchmark::DoNotOptimize(t0);
    benchmark::DoNotOptimize(t1);
    benchmark::DoNotOptimize(ParallelLoopRunner::Delinearize(123, t0, t1));
  }
}

BENCHMARK(BM_Delinearize2DTile2D);

static void BM_SingleTask1DLoop(benchmark::State& state) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());
  ParallelLoopRunner runner(&device);

  for (auto _ : state) {
    runner.Parallelize(TileDim{1, 1}, [](TileIndex) {});
    tsl::BlockUntilReady(runner.done_event());
  }
}

BENCHMARK(BM_SingleTask1DLoop);

static void BM_Parallelize2DTile1D(benchmark::State& state) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());

  ParallelLoopRunner runner(&device);

  RangeDim d0 = {4};
  TileDim d1 = {4, 1};

  for (auto _ : state) {
    runner.Parallelize(d0, d1, [](RangeIndex, TileIndex) {});
    tsl::BlockUntilReady(runner.done_event());
  }
}

BENCHMARK(BM_Parallelize2DTile1D)->Arg(0)->Arg(100)->Arg(10000);

static void BM_Parallelize3DTile2D(benchmark::State& state) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());

  ParallelLoopRunner runner(&device);

  RangeDim d0 = {4};
  TileDim d1 = {4, 1};
  TileDim d2 = {4, 1};

  for (auto _ : state) {
    runner.Parallelize(d0, d1, d2, [](RangeIndex, TileIndex, TileIndex) {});
    tsl::BlockUntilReady(runner.done_event());
  }
}

BENCHMARK(BM_Parallelize3DTile2D);

}  // namespace
}  // namespace xla::cpu
