/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/default/test_benchmark.h"

// Test the new interface: BM_benchmark(benchmark::State& state)
namespace tensorflow {
namespace testing {
namespace {

void BM_TestIterState(::testing::benchmark::State& state) {
  int i = 0;
  for (auto s : state) {
    ++i;
    DoNotOptimize(i);
  }
}

BENCHMARK(BM_TestIterState);

}  // namespace
}  // namespace testing
}  // namespace tensorflow

int main() {
  ::testing::benchmark::RunSpecifiedBenchmarks();
  return 0;
}
