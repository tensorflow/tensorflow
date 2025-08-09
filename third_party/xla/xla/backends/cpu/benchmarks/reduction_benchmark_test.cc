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

static void BM_ReduceWindowAddF32OuterAndInnerDim(benchmark::State& state,
                                                  HloBenchmarkOptions options) {
  int outer_dim = state.range(0);
  int inner_dim = state.range(1);

  constexpr absl::string_view hlo = R"(
  HloModule reduce_window_add_f32_outer_dim_$outer_dim_inner_dim_$inner_dim

  add {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    ROOT add = f32[] add(p0, p1)
  }

  ENTRY e {
    p0 = f32[1024,1024] parameter(0)
    c0 = f32[] constant(0)
    ROOT reduce = f32[$result_outer_dim,$result_inner_dim] 
      reduce-window(p0, c0), window={size=$outer_dimx$inner_dim stride=$outer_dimx$inner_dim}, to_apply=add
  }
)";

  CHECK_OK(
      RunHloBenchmark(state, hlo, {},
                      {
                          {"$outer_dim", absl::StrCat(outer_dim)},
                          {"$result_outer_dim", absl::StrCat(1024 / outer_dim)},
                          {"$inner_dim", absl::StrCat(inner_dim)},
                          {"$result_inner_dim", absl::StrCat(1024 / inner_dim)},
                      },
                      options));
}

static void BM_ReduceAddF32OverDimension(benchmark::State& state,
                                         HloBenchmarkOptions options) {
  int64_t reduce_dim = state.range(0);

  constexpr absl::string_view hlo = R"(
  HloModule reduce_add_f32_reduce_dim_$reduce_dim

  add {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    ROOT add = f32[] add(p0, p1)
  }

  ENTRY e {
    p0 = f32[1024,1024] parameter(0)
    c0 = f32[] constant(0)
    ROOT reduce = f32[1024] reduce(p0, c0), dimensions={$reduce_dim}, to_apply=add
  }
)";

  CHECK_OK(RunHloBenchmark(
      state, hlo, {}, {{"$reduce_dim", absl::StrCat(reduce_dim)}}, options));
}

static void BM_ReduceWindowAddF32SkippingData(benchmark::State& state,
                                              HloBenchmarkOptions options) {
  constexpr absl::string_view hlo = R"(
  HloModule reduce_window_add_f32_skipping_data

  add {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    ROOT add = f32[] add(p0, p1)
  }

  ENTRY e {
    p0 = f32[128,128] parameter(0)
    c0 = f32[] constant(0)
  ROOT reduce = f32[128,2] reduce-window(p0, c0), window={size=1x8 stride=1x64}, to_apply=add
  }
  )";

  CHECK_OK(RunHloBenchmark(state, hlo, {}, {{}}, options));
}

static void BM_ReduceWindowAddF32OverlappingWindows(
    benchmark::State& state, HloBenchmarkOptions options) {
  constexpr absl::string_view hlo = R"(
  HloModule reduce_window_add_f32_overlapping_windows

  add {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    ROOT add = f32[] add(p0, p1)
  }

  ENTRY e {
    p0 = f32[128,128] parameter(0)
    c0 = f32[] constant(0)
    ROOT reduce = f32[128,13] reduce-window(p0, c0), window={size=1x32 stride=1x8}, to_apply=add
  }
  )";

  CHECK_OK(RunHloBenchmark(state, hlo, {}, {{}}, options));
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

XLA_CPU_BENCHMARK(BM_ReduceAddF32OverDimension)
    ->ArgName("reduce_dim")
    ->Arg(0)
    ->Arg(1);

XLA_CPU_BENCHMARK(BM_ReduceWindowAddF32OuterAndInnerDim)
    ->MeasureProcessCPUTime()
    ->ArgNames({"outer_dim", "inner_dim"})
    ->Args({1, 32})
    ->Args({32, 1})
    ->Args({32, 2})
    ->Args({32, 4})
    ->Args({32, 8})
    ->Args({32, 16})
    ->Args({32, 32});

XLA_CPU_BENCHMARK(BM_ReduceWindowAddF32SkippingData)->MeasureProcessCPUTime();

XLA_CPU_BENCHMARK(BM_ReduceWindowAddF32OverlappingWindows)
    ->MeasureProcessCPUTime();

}  // namespace xla::cpu
