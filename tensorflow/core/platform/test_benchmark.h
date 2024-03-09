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
#ifndef TENSORFLOW_CORE_PLATFORM_TEST_BENCHMARK_H_
#define TENSORFLOW_CORE_PLATFORM_TEST_BENCHMARK_H_

#include "tsl/platform/test_benchmark.h"

namespace tensorflow {
namespace testing {
using tsl::testing::DoNotOptimize;         // NOLINT
using tsl::testing::InitializeBenchmarks;  // NOLINT
using tsl::testing::RunBenchmarks;         // NOLINT
}  // namespace testing
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_TEST_BENCHMARK_H_
