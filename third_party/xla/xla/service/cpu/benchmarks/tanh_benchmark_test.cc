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
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::cpu {

static void BM_TanhF32(benchmark::State& state) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule tanh_f32_$d0

    ENTRY e {
      input = f32[$d0] parameter(0)
      ROOT output = tanh(input)
    }
  )";

  std::minstd_rand0 engine;

  auto input_shape = ShapeUtil::MakeShape(F32, {d0});
  auto p0 =
      *LiteralUtil::CreateRandomLiteral<F32>(input_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&p0};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}));
}

static void BM_TanhF64(benchmark::State& state) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule tanh_f64_$d0

    ENTRY e {
      input = f64[$d0] parameter(0)
      ROOT output = tanh(input)
    }
  )";

  std::minstd_rand0 engine;

  auto input_shape = ShapeUtil::MakeShape(F64, {d0});
  auto p0 =
      *LiteralUtil::CreateRandomLiteral<F64>(input_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&p0};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}));
}

#define REGISTER_TANH_BENCHMARK(NAME) \
  BENCHMARK(NAME)                     \
      ->MeasureProcessCPUTime()       \
      ->Arg(128)                      \
      ->Arg(256)                      \
      ->Arg(512)                      \
      ->Arg(1024)                     \
      ->Arg(4096);

REGISTER_TANH_BENCHMARK(BM_TanhF32);
REGISTER_TANH_BENCHMARK(BM_TanhF64);

}  // namespace xla::cpu
