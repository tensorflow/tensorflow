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
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/backends/cpu/benchmarks/multi_benchmark_config.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

static void BM_ReduceAddF32(benchmark::State& state,
                            HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule reduce_add_f32_$d0

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    ENTRY e {
      p0 = f32[1,2,1,$d0,256] parameter(0)
      c0 = f32[] constant(0)
      ROOT reduce = f32[1,2] reduce(p0, c0), dimensions={2,3,4}, to_apply=add
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(F32, {1, 2, 1, d0, 256});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

static void BM_ReduceAddBF16(benchmark::State& state,
                             HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule reduce_add_bf16_$d0

    add {
      p0 = bf16[] parameter(0)
      p1 = bf16[] parameter(1)
      ROOT add = bf16[] add(p0, p1)
    }

    ENTRY e {
      p0 = bf16[1,2,1,$d0,256] parameter(0)
      c0 = bf16[] constant(0)
      ROOT reduce = bf16[1,2] reduce(p0, c0), dimensions={2,3,4}, to_apply=add
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(BF16, {1, 2, 1, d0, 256});
  auto p0 = *LiteralUtil::CreateRandomLiteral<BF16>(shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

#define BENCHMARK_SIZES(NAME)   \
  XLA_CPU_BENCHMARK(NAME)       \
      ->MeasureProcessCPUTime() \
      ->Arg(128)                \
      ->Arg(256)                \
      ->Arg(512)                \
      ->Arg(1024)               \
      ->Arg(8192)               \
      ->Arg(16384)

BENCHMARK_SIZES(BM_ReduceAddF32);
BENCHMARK_SIZES(BM_ReduceAddBF16);

}  // namespace xla::cpu
