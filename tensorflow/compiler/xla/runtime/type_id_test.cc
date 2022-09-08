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

#include "tensorflow/compiler/xla/runtime/type_id.h"

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace xla {
namespace runtime {

struct IdSetA {};
struct IdSetB {};
struct IdSetC {};

using DenseIdA = DenseTypeId<IdSetA>;
using DenseIdB = DenseTypeId<IdSetB>;
using DenseIdC = DenseTypeId<IdSetC>;

TEST(DenseTypeIdTest, GetId) {
  // Generate unique type ids in the set A.
  EXPECT_EQ(DenseIdA::get<int32_t>(), 0);
  EXPECT_EQ(DenseIdA::get<int64_t>(), 1);

  // Check that unique type ids in the set B are independent.
  EXPECT_EQ(DenseIdB::get<int64_t>(), 0);
  EXPECT_EQ(DenseIdB::get<int32_t>(), 1);

  // Check that we get back the same type id.
  EXPECT_EQ(DenseIdA::get<int32_t>(), 0);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks are below.
//===----------------------------------------------------------------------===//

static void BM_GetDenseTypeIdA(benchmark::State& state) {
  for (auto _ : state) {
    auto id = DenseIdA::get<int32_t>();
    benchmark::DoNotOptimize(id);
  }
}

static void BM_GetDenseTypeIdC(benchmark::State& state) {
  for (auto _ : state) {
    auto id = DenseIdC::get<int32_t>();
    benchmark::DoNotOptimize(id);
  }
}

BENCHMARK(BM_GetDenseTypeIdA);
BENCHMARK(BM_GetDenseTypeIdC);

}  // namespace runtime
}  // namespace xla

XLA_RUNTIME_DECLARE_EXPLICIT_DENSE_TYPE_ID(xla::runtime::IdSetC, int32_t);
XLA_RUNTIME_DEFINE_EXPLICIT_DENSE_TYPE_ID(xla::runtime::IdSetC, int32_t);
