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
#include <vector>

#include "absl/algorithm/container.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/env.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"
#include "tsl/platform/threadpool.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

TEST(ParallelLoopRunnerTest, BackToBack1DLoops) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());
  ParallelLoopRunner runner(&device);

  std::vector<int32_t> data(1024);
  auto inc_range = [&](size_t offset, size_t extent) {
    for (size_t i = offset; i < offset + extent; ++i) {
      data[i] += 1;
    }
  };

  runner.Parallelize(1024, 1, inc_range);
  runner.Parallelize(1024, 2, inc_range);
  runner.Parallelize(1024, 3, inc_range);
  runner.Parallelize(1024, 4, inc_range);
  runner.Parallelize(1024, 5, inc_range);

  tsl::BlockUntilReady(ParallelLoopRunner::TakeDoneEvent(std::move(runner)));
  ASSERT_TRUE(absl::c_all_of(data, [](int32_t value) { return value == 5; }));
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

}  // namespace
}  // namespace xla::cpu
