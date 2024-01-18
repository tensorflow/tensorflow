/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/runtime/map_by_type.h"

#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"

namespace xla {
namespace runtime {

struct IdSetA {};
struct IdSetB {};

TEST(PtrMapByTypeTest, Basic) {
  PtrMapByType<IdSetA> map;
  EXPECT_FALSE(map.contains<int32_t>());

  int32_t i32 = 1;
  map.insert(&i32);

  EXPECT_TRUE(map.contains<int32_t>());
  EXPECT_FALSE(map.contains<const int>());
  EXPECT_EQ(*map.get<int32_t>(), 1);

  EXPECT_EQ(map.getIfExists<int64_t>(), nullptr);
  EXPECT_EQ(*map.getIfExists<int32_t>(), 1);
  EXPECT_EQ(map.getIfExists<int32_t>(), &i32);

  const int32_t ci32 = 2;
  map.insert(&ci32);

  EXPECT_TRUE(map.contains<const int32_t>());
  EXPECT_EQ(*map.get<const int32_t>(), 2);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks are below.
//===----------------------------------------------------------------------===//

static void BM_InsertAndGetPtrs(benchmark::State& state) {
  int32_t i32 = 1;
  int64_t i64 = 1;
  float f32 = 1.0;
  double f64 = 1.0;

  for (auto _ : state) {
    PtrMapByType<IdSetA> map;
    map.insert_all(&i32, &i64, &f32, &f64);
    benchmark::DoNotOptimize(map);
    benchmark::DoNotOptimize(map.getIfExists<int32_t>());
    benchmark::DoNotOptimize(map.getIfExists<int64_t>());
    benchmark::DoNotOptimize(map.getIfExists<float>());
    benchmark::DoNotOptimize(map.getIfExists<double>());
  }
}

static void BM_InsertAndGetOptPtrs(benchmark::State& state) {
  int32_t i32 = 1;
  int64_t i64 = 1;
  float f32 = 1.0;
  double f64 = 1.0;

  for (auto _ : state) {
    PtrMapByType<IdSetB> map;
    map.insert_all(&i32, &i64, &f32, &f64);
    benchmark::DoNotOptimize(map);
    benchmark::DoNotOptimize(map.getIfExists<int32_t>());
    benchmark::DoNotOptimize(map.getIfExists<int64_t>());
    benchmark::DoNotOptimize(map.getIfExists<float>());
    benchmark::DoNotOptimize(map.getIfExists<double>());
  }
}

BENCHMARK(BM_InsertAndGetPtrs);
BENCHMARK(BM_InsertAndGetOptPtrs);

}  // namespace runtime
}  // namespace xla

XLA_RUNTIME_DECLARE_EXPLICIT_DENSE_TYPE_ID(xla::runtime::IdSetB, int32_t);
XLA_RUNTIME_DECLARE_EXPLICIT_DENSE_TYPE_ID(xla::runtime::IdSetB, int64_t);
XLA_RUNTIME_DECLARE_EXPLICIT_DENSE_TYPE_ID(xla::runtime::IdSetB, float);
XLA_RUNTIME_DECLARE_EXPLICIT_DENSE_TYPE_ID(xla::runtime::IdSetB, double);

XLA_RUNTIME_DEFINE_EXPLICIT_DENSE_TYPE_ID(xla::runtime::IdSetB, int32_t);
XLA_RUNTIME_DEFINE_EXPLICIT_DENSE_TYPE_ID(xla::runtime::IdSetB, int64_t);
XLA_RUNTIME_DEFINE_EXPLICIT_DENSE_TYPE_ID(xla::runtime::IdSetB, float);
XLA_RUNTIME_DEFINE_EXPLICIT_DENSE_TYPE_ID(xla::runtime::IdSetB, double);
