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

#include <cstdint>
#include <random>
#include <string_view>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/shape_util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::cpu {

static void BM_BatchedDotF32(benchmark::State& state) {
  int64_t d0 = state.range(0);
  int64_t d1 = state.range(1);

  std::string_view hlo = R"(
    HloModule dot_f32_b$d0_d$d1

    ENTRY e {
      p0 = f32[$d0,$d1,$d1] parameter(0)
      p1 = f32[$d0,$d1,$d1] parameter(1)
      ROOT dot = f32[$d0,$d1,$d1] dot(p0, p1),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={1}
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(F32, {d0, d1, d1});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);
  auto p1 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0, &p1};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args,
                      {{"$d0", absl::StrCat(d0)}, {"$d1", absl::StrCat(d1)}}));
}

BENCHMARK(BM_BatchedDotF32)
    ->MeasureProcessCPUTime()
    ->ArgPair(1, 2)
    ->ArgPair(1, 32)
    ->ArgPair(1, 64)
    ->ArgPair(1, 128)
    ->ArgPair(1, 256)
    ->ArgPair(1, 512)
    ->ArgPair(2, 2)
    ->ArgPair(2, 32)
    ->ArgPair(2, 64)
    ->ArgPair(2, 128)
    ->ArgPair(2, 256)
    ->ArgPair(2, 512)
    ->ArgPair(4, 2)
    ->ArgPair(4, 32)
    ->ArgPair(4, 64)
    ->ArgPair(4, 128)
    ->ArgPair(4, 256)
    ->ArgPair(4, 512)
    ->ArgPair(8, 2)
    ->ArgPair(8, 32)
    ->ArgPair(8, 64)
    ->ArgPair(8, 128)
    ->ArgPair(8, 256)
    ->ArgPair(8, 512);

}  // namespace xla::cpu
