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

bool no_arg_was_run_ = false;

namespace {

void BM_TestIterState(::testing::benchmark::State& state) {
  int i = 0;
  for (auto s : state) {
    ++i;
    DoNotOptimize(i);
  }
  no_arg_was_run_ = true;
}

BENCHMARK(BM_TestIterState);

const int kArgOne = 543;
const int kArgTwo = 345;

void BM_OneArg(::testing::benchmark::State& state) {
  const int arg1 = state.range(0);
  CHECK(arg1 == kArgOne);

  int i = 0;
  for (auto s : state) {
    ++i;
    DoNotOptimize(i);
  }
}

BENCHMARK(BM_OneArg)->Arg(kArgOne);

// void BM_OneArg_Use_Two(::testing::benchmark::State& state) {
//   // FIXME: This will trigger a failed CHECK.
//   // I don't know how to express the death-test in this framework.
//   const int arg2 = state.range(1);

//   int i = 0;
//   for (auto s : state) {
//     ++i;
//     DoNotOptimize(i);
//   }
// }

// BENCHMARK(BM_OneArg_Use_Two)->Arg(kArgOne);

void BM_TwoArgs(::testing::benchmark::State& state) {
  const int arg1 = state.range(0);
  const int arg2 = state.range(1);
  CHECK(arg1 == kArgOne);
  CHECK(arg2 == kArgTwo);

  int i = 0;
  for (auto s : state) {
    ++i;
    DoNotOptimize(i);
  }
}

BENCHMARK(BM_TwoArgs)->ArgPair(kArgOne, kArgTwo);

}  // namespace
}  // namespace testing
}  // namespace tensorflow

int main() {
  ::testing::benchmark::RunSpecifiedBenchmarks();
  CHECK(tensorflow::testing::no_arg_was_run_);
  return 0;
}
