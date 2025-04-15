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

static void BM_ExpF32(benchmark::State& state, HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule exp_f32_$d0

    ENTRY e {
      input = f32[$d0] parameter(0)
      ROOT output = exponential(input)
    }
  )";

  std::minstd_rand0 engine;

  auto input_shape = ShapeUtil::MakeShape(F32, {d0});
  auto p0 =
      *LiteralUtil::CreateRandomLiteral<F32>(input_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

static void BM_ExpF16(benchmark::State& state) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule exp_f16_$d0

    ENTRY e {
      input = f16[$d0] parameter(0)
      ROOT output = exponential(input)
    }
  )";

  std::minstd_rand0 engine;

  auto input_shape = ShapeUtil::MakeShape(F16, {d0});
  auto p0 =
      *LiteralUtil::CreateRandomLiteral<F16>(input_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&p0};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}));
}

#define REGISTER_EXP_BENCHMARK(NAME) \
  XLA_CPU_BENCHMARK(NAME)            \
      ->MeasureProcessCPUTime()      \
      ->Arg(128)                     \
      ->Arg(256)                     \
      ->Arg(512)                     \
      ->Arg(1024)                    \
      ->Arg(4096);

REGISTER_EXP_BENCHMARK(BM_ExpF32);
// TODO(b/406431945): add AOT for f16 exp
BENCHMARK(BM_ExpF16)
    ->MeasureProcessCPUTime()
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(4096);

}  // namespace xla::cpu
