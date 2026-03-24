/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/shape_pool.h"

#include <memory>

#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

TEST(ShapePoolTest, GetCanonicalShape) {
  ShapePool pool;

  {  // Use nested scope to allow garbage collection below.
    Shape s0 = ShapeUtil::MakeShape(F32, {1, 2});
    Shape s1 = ShapeUtil::MakeShape(F32, {2, 1});

    auto cs0_0 = pool.GetCanonicalShape(s0);
    auto cs0_1 = pool.GetCanonicalShape(s0);
    ASSERT_EQ(cs0_0, cs0_1);

    auto cs1_0 = pool.GetCanonicalShape(s1);
    auto cs1_1 = pool.GetCanonicalShape(s1);
    ASSERT_NE(cs0_0, cs1_0);
    ASSERT_EQ(cs1_0, cs1_1);
  }

  ASSERT_EQ(pool.GarbageCollect(), 2);
}

TEST(ShapePoolTest, GetCanonicalShapeFromSharedPtr) {
  ShapePool pool;

  {  // Use nested scope to allow garbage collection below.
    auto s0 = std::make_shared<Shape>(ShapeUtil::MakeShape(F32, {1, 2}));
    auto s1 = std::make_shared<Shape>(ShapeUtil::MakeShape(F32, {2, 1}));

    auto cs0_0 = pool.GetCanonicalShape(s0);
    auto cs0_1 = pool.GetCanonicalShape(s0);
    ASSERT_EQ(cs0_0, cs0_1);

    auto cs1_0 = pool.GetCanonicalShape(s1);
    auto cs1_1 = pool.GetCanonicalShape(s1);
    ASSERT_NE(cs0_0, cs1_0);
    ASSERT_EQ(cs1_0, cs1_1);
  }

  ASSERT_EQ(pool.GarbageCollect(), 2);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below.
//===----------------------------------------------------------------------===//

static void BM_GetCanonicalShape(benchmark::State& state) {
  ShapePool pool;

  auto s = std::make_shared<Shape>(ShapeUtil::MakeShape(F32, {1, 2}));

  for (auto _ : state) {
    auto cs = pool.GetCanonicalShape(s);
    benchmark::DoNotOptimize(cs);
  }

  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_GetCanonicalShape);

}  // namespace
}  // namespace xla
