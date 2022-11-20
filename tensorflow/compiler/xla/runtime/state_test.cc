/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/runtime/state.h"

#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/platform/test_benchmark.h"

namespace xla {
namespace runtime {

TEST(StateVectorTest, GetOrCreate) {
  int32_t cnt = 0;
  auto create = [&] { return cnt++; };

  StateVector<int32_t> state;

  StateVector<int32_t>::Snapshot empty_snapshot = state.snapshot();
  EXPECT_EQ(**empty_snapshot.GetOrCreate(0, create), 0);
  EXPECT_EQ(**empty_snapshot.GetOrCreate(0, create), 0);
  EXPECT_EQ(**empty_snapshot.GetOrCreate(1, create), 1);

  StateVector<int32_t>::Snapshot snapshot = state.snapshot();
  EXPECT_EQ(**snapshot.GetOrCreate(0, create), 0);
  EXPECT_EQ(**snapshot.GetOrCreate(1, create), 1);
  EXPECT_EQ(**snapshot.GetOrCreate(9, create), 2);

  State<int32_t> st0 = snapshot.state(0);
  State<int32_t> st1 = snapshot.state(1);
  EXPECT_EQ(**st0.GetOrCreate(create), 0);
  EXPECT_EQ(**st1.GetOrCreate(create), 1);

  EXPECT_EQ(cnt, 3);
}

TEST(StateVectorTest, GetOrCreateAtRandomOrder) {
  int32_t cnt = 0;
  auto create = [&] { return cnt++; };

  StateVector<int32_t> state;

  StateVector<int32_t>::Snapshot empty_snapshot = state.snapshot();
  EXPECT_EQ(**empty_snapshot.GetOrCreate(99, create), 0);
  EXPECT_EQ(**empty_snapshot.GetOrCreate(22, create), 1);
  EXPECT_EQ(**empty_snapshot.GetOrCreate(33, create), 2);

  StateVector<int32_t>::Snapshot snapshot = state.snapshot();
  EXPECT_EQ(**snapshot.GetOrCreate(99, create), 0);
  EXPECT_EQ(**snapshot.GetOrCreate(22, create), 1);
  EXPECT_EQ(**snapshot.GetOrCreate(33, create), 2);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks.
//===----------------------------------------------------------------------===//

static void BM_TakeSnapshot(benchmark::State& state) {
  StateVector<int32_t> ints;

  for (auto _ : state) {
    StateVector<int32_t>::Snapshot snapshot = ints.snapshot();
    benchmark::DoNotOptimize(snapshot);
  }
}

static void BM_GetFromStateVectorVector(benchmark::State& state) {
  StateVector<int32_t> ints;
  StateVector<int32_t>::Snapshot snapshot = ints.snapshot();

  for (auto _ : state) {
    auto value = snapshot.GetOrCreate(0, [] { return 0; });
    assert(value.ok() && "unexpected error");
    benchmark::DoNotOptimize(value);
  }
}

static void BM_GetFromSnapshot(benchmark::State& state) {
  StateVector<int32_t> ints;
  StateVector<int32_t>::Snapshot empty_snapshot = ints.snapshot();
  empty_snapshot.GetOrCreate(0, [] { return 0; }).IgnoreError();

  StateVector<int32_t>::Snapshot snapshot = ints.snapshot();

  for (auto _ : state) {
    auto value = snapshot.GetOrCreate(0, [] { return 0; });
    assert(value.ok() && "unexpected error");
    benchmark::DoNotOptimize(value);
  }
}

BENCHMARK(BM_TakeSnapshot);
BENCHMARK(BM_GetFromStateVectorVector);
BENCHMARK(BM_GetFromSnapshot);

}  // namespace runtime
}  // namespace xla
