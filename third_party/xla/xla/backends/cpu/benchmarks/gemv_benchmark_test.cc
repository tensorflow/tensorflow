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

#include <cstddef>
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

static void BM_GemvF32(benchmark::State& state, HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);
  int64_t d1 = state.range(1);

  absl::string_view hlo = R"(
    HloModule gemv_f32_$d0_$d0x$d1

    ENTRY e {
      a = f32[$d0] parameter(0)
      b = f32[$d0,$d1] parameter(1)
      ROOT dot = dot(a, b), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    }
  )";

  std::minstd_rand0 engine;

  auto a_shape = ShapeUtil::MakeShape(F32, {1, d0});
  auto b_shape = ShapeUtil::MakeShape(F32, {d0, d1});

  auto a = *LiteralUtil::CreateRandomLiteral<F32>(a_shape, &engine, 1.0f, 0.1f);
  auto b = *LiteralUtil::CreateRandomLiteral<F32>(b_shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&a, &b};
  CHECK_OK(RunHloBenchmark(state, hlo, args,
                           {
                               {"$d0", absl::StrCat(d0)},
                               {"$d1", absl::StrCat(d1)},
                           },
                           options));
}

template <size_t batch_size>
static void BM_BatchedGemvF32(benchmark::State& state,
                              HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);
  int64_t d1 = state.range(1);

  absl::string_view hlo = R"(
    HloModule gemv_f32_$d0_$d0x$d1

    ENTRY e {
      a = f32[$batch,$d0] parameter(0)
      b = f32[$batch,$d0,$d1] parameter(1)
      ROOT dot = dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={1},
                            lhs_batch_dims={0}, rhs_batch_dims={0}
    }
  )";

  std::minstd_rand0 engine;

  auto a_shape = ShapeUtil::MakeShape(F32, {batch_size, d0});
  auto b_shape = ShapeUtil::MakeShape(F32, {batch_size, d0, d1});

  auto a = *LiteralUtil::CreateRandomLiteral<F32>(a_shape, &engine, 1.0f, 0.1f);
  auto b = *LiteralUtil::CreateRandomLiteral<F32>(b_shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&a, &b};
  CHECK_OK(RunHloBenchmark(state, hlo, args,
                           {
                               {"$batch", absl::StrCat(batch_size)},
                               {"$d0", absl::StrCat(d0)},
                               {"$d1", absl::StrCat(d1)},
                           },
                           options));
}

XLA_CPU_BENCHMARK(BM_GemvF32)
    ->MeasureProcessCPUTime()
    ->ArgPair(32, 128)
    ->ArgPair(128, 128)
    ->ArgPair(256, 128)
    ->ArgPair(512, 128)
    ->ArgPair(32, 512)
    ->ArgPair(128, 512)
    ->ArgPair(256, 512)
    ->ArgPair(512, 512)
    ->ArgPair(32, 1024)
    ->ArgPair(128, 1024)
    ->ArgPair(256, 1024)
    ->ArgPair(512, 1024);

XLA_CPU_BENCHMARK(BM_BatchedGemvF32<8>)
    ->MeasureProcessCPUTime()
    ->ArgPair(32, 128)
    ->ArgPair(128, 128)
    ->ArgPair(256, 128)
    ->ArgPair(512, 128)
    ->ArgPair(32, 512)
    ->ArgPair(128, 512)
    ->ArgPair(256, 512)
    ->ArgPair(512, 512)
    ->ArgPair(32, 1024)
    ->ArgPair(128, 1024)
    ->ArgPair(256, 1024)
    ->ArgPair(512, 1024);

}  // namespace xla::cpu
