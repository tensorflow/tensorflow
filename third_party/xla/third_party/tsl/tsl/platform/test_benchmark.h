/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// Simple benchmarking facility.
#ifndef TENSORFLOW_TSL_PLATFORM_TEST_BENCHMARK_H_
#define TENSORFLOW_TSL_PLATFORM_TEST_BENCHMARK_H_

#include "benchmark/benchmark.h"  // from @com_google_benchmark  // IWYU pragma: export
#include "tsl/platform/platform.h"

// FIXME(vyng): Remove this.
// Background: During the benchmark-migration projects, all benchmarks were made
// to use "testing::benchmark::" prefix because that is what the internal
// Google benchmark library use.
namespace testing {
namespace benchmark {
using ::benchmark::State;  // NOLINT
}  // namespace benchmark
}  // namespace testing

namespace tsl {
namespace testing {

inline void RunBenchmarks() { benchmark::RunSpecifiedBenchmarks(); }
inline void InitializeBenchmarks(int* argc, char** argv) {
  benchmark::Initialize(argc, argv);
}

template <class T>
void DoNotOptimize(const T& var) {
  ::benchmark::DoNotOptimize(var);
}
}  // namespace testing
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_TEST_BENCHMARK_H_
