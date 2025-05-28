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

static void BM_SelectAndScatterF32(benchmark::State& state,
                                   HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);
  int64_t d1 = (d0 - 1) / 2;

  absl::string_view hlo = R"(
    HloModule select_and_scatter_f32_$d0

    ge {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT compare = pred[] compare(a, b), direction=GE
    }

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    ENTRY e {
      p0 = f32[$d0,$d0] parameter(0)
      p1 = f32[$d1,$d1] parameter(1)
      p2 = f32[] parameter(2)
      ROOT sas = f32[$d0,$d0] select-and-scatter(p0, p1, p2),
           window={size=3x3 stride=2x2 pad=0_0x0_0}, select=ge, scatter=add
    }
  )";

  std::minstd_rand0 engine;

  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(
      ShapeUtil::MakeShape(F32, {d0, d0}), &engine, 1.0f, 0.1f);
  auto p1 = *LiteralUtil::CreateRandomLiteral<F32>(
      ShapeUtil::MakeShape(F32, {d1, d1}), &engine, 1.0f, 0.1f);
  auto p2 = LiteralUtil::CreateR0(1.0f);

  std::vector<const Literal*> args = {&p0, &p1, &p2};
  CHECK_OK(RunHloBenchmark(
      state, hlo, args, {{"$d0", absl::StrCat(d0)}, {"$d1", absl::StrCat(d1)}},
      options));
}

XLA_CPU_BENCHMARK(BM_SelectAndScatterF32)
    ->MeasureProcessCPUTime()
    ->Arg(128)
    ->Arg(256)
    ->Arg(512);

}  // namespace xla::cpu
