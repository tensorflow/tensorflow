/* Copyright 2026 The OpenXLA Authors.

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
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/backends/cpu/benchmarks/multi_benchmark_config.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

static void BM_SinF64(benchmark::State& state, HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);
  absl::string_view hlo = R"(
    HloModule sin_f64_$d0
    ENTRY e {
      input = f64[$d0] parameter(0)
      ROOT output = sin(input)
    }
  )";
  std::minstd_rand0 engine;
  auto input_shape = ShapeUtil::MakeShape(F64, {d0});
  absl::StatusOr<Literal> p0_status =
      LiteralUtil::CreateRandomLiteral<F64>(input_shape, &engine, 1.0, 0.1);
  CHECK_OK(p0_status.status());
  Literal p0 = std::move(p0_status).value();
  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
  state.SetItemsProcessed(state.iterations() * d0);
  state.SetBytesProcessed(state.iterations() *
                          ShapeUtil::ByteSizeOf(input_shape));
}

static void BM_CosF64(benchmark::State& state, HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);
  absl::string_view hlo = R"(
    HloModule cos_f64_$d0
    ENTRY e {
      input = f64[$d0] parameter(0)
      ROOT output = cos(input)
    }
  )";
  std::minstd_rand0 engine;
  auto input_shape = ShapeUtil::MakeShape(F64, {d0});
  absl::StatusOr<Literal> p0_status =
      LiteralUtil::CreateRandomLiteral<F64>(input_shape, &engine, 1.0, 0.1);
  CHECK_OK(p0_status.status());
  Literal p0 = std::move(p0_status).value();
  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
  state.SetItemsProcessed(state.iterations() * d0);
  state.SetBytesProcessed(state.iterations() *
                          ShapeUtil::ByteSizeOf(input_shape));
}

#define REGISTER_SINCOS_BENCHMARK(NAME) \
  XLA_CPU_BENCHMARK(NAME)               \
      ->MeasureProcessCPUTime()         \
      ->Arg(128)                        \
      ->Arg(256)                        \
      ->Arg(512)                        \
      ->Arg(1024)                       \
      ->Arg(8192)                       \
      ->Arg(32768);

REGISTER_SINCOS_BENCHMARK(BM_SinF64);
REGISTER_SINCOS_BENCHMARK(BM_CosF64);

}  // namespace xla::cpu
