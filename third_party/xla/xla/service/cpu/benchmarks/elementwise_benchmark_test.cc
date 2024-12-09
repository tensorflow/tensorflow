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

static void BM_AddF32(benchmark::State& state) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule add_f32_$d0

    ENTRY e {
      p0 = f32[1,2,1,$d0,256] parameter(0)
      p1 = f32[1,2,1,$d0,256] parameter(1)
      ROOT add = f32[1,2,1,$d0,256] add(p0, p1)
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(F32, {1, 2, 1, d0, 256});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);
  auto p1 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0, &p1};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}));
}

static void BM_AddBF16(benchmark::State& state) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule add_bf16_$d0

    ENTRY e {
      p0 = bf16[1,2,1,$d0,256] parameter(0)
      p1 = bf16[1,2,1,$d0,256] parameter(1)
      ROOT add = bf16[1,2,1,$d0,256] add(p0, p1)
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(BF16, {1, 2, 1, d0, 256});
  auto p0 = *LiteralUtil::CreateRandomLiteral<BF16>(shape, &engine, 1.0f, 0.1f);
  auto p1 = *LiteralUtil::CreateRandomLiteral<BF16>(shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0, &p1};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}));
}

static void BM_ConvertF32ToBF16(benchmark::State& state) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule convert_f32_to_bf16_$d0

    ENTRY e {
      p0 = f32[1,2,1,$d0,256] parameter(0)
      ROOT convert = bf16[1,2,1,$d0,256] convert(p0)
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(F32, {1, 2, 1, d0, 256});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}));
}

#define BENCHMARK_SIZES(NAME)   \
  BENCHMARK(NAME)               \
      ->MeasureProcessCPUTime() \
      ->Arg(128)                \
      ->Arg(256)                \
      ->Arg(512)                \
      ->Arg(1024)               \
      ->Arg(8192)               \
      ->Arg(16384)              \
      ->Arg(32768)

BENCHMARK_SIZES(BM_AddF32);
BENCHMARK_SIZES(BM_AddBF16);
BENCHMARK_SIZES(BM_ConvertF32ToBF16);

}  // namespace xla::cpu
