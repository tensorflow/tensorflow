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

static void BM_TransposeAndDot(benchmark::State& state,
                               HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule transpose_and_dot_$d0

    ENTRY e {
      p0 = f32[$d0,1000] parameter(0)
      p1 = f32[1000]{0} parameter(1)
      transpose = f32[1000,$d0] transpose(p0), dimensions={1,0}
      ROOT dot = f32[$d0]{0} dot(p1, transpose), lhs_contracting_dims={0},
                                                 rhs_contracting_dims={0}
    }
  )";

  std::minstd_rand0 engine;

  auto p0_shape = ShapeUtil::MakeShape(F32, {d0, 1000});
  auto p1_shape = ShapeUtil::MakeShape(F32, {1000});

  auto p0 =
      *LiteralUtil::CreateRandomLiteral<F32>(p0_shape, &engine, 1.0f, 0.1f);
  auto p1 =
      *LiteralUtil::CreateRandomLiteral<F32>(p1_shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0, &p1};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

static void BM_TransposedDot(benchmark::State& state,
                             HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule transposed_dot_$d0

    ENTRY e {
      p0 = f32[$d0,1000] parameter(0)
      p1 = f32[1000]{0} parameter(1)
      ROOT dot = f32[$d0]{0} dot(p1, p0), lhs_contracting_dims={0},
                                          rhs_contracting_dims={1}
    }
  )";

  std::minstd_rand0 engine;

  auto p0_shape = ShapeUtil::MakeShape(F32, {d0, 1000});
  auto p1_shape = ShapeUtil::MakeShape(F32, {1000});

  auto p0 =
      *LiteralUtil::CreateRandomLiteral<F32>(p0_shape, &engine, 1.0f, 0.1f);
  auto p1 =
      *LiteralUtil::CreateRandomLiteral<F32>(p1_shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0, &p1};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

#define REGISTER_BENCHMARK(NAME) \
  XLA_CPU_BENCHMARK(NAME)        \
      ->MeasureProcessCPUTime()  \
      ->Arg(128)                 \
      ->Arg(256)                 \
      ->Arg(512)                 \
      ->Arg(1024)                \
      ->Arg(4096);

REGISTER_BENCHMARK(BM_TransposeAndDot);
REGISTER_BENCHMARK(BM_TransposedDot);

}  // namespace xla::cpu
