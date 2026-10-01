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
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

template <PrimitiveType type>
static void BM_TransposeAndCopy(benchmark::State& state,
                                const HloBenchmarkOptions& options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule transpose_and_copy_$type_$d0

    ENTRY e {
      p0 = $type[$d0,1000] parameter(0)
      transpose = $type[1000,$d0] transpose(p0), dimensions={1,0}
      ROOT copy = $type[1000,$d0] copy(transpose)
    }
  )";

  std::minstd_rand0 engine;

  auto input_shape = ShapeUtil::MakeShape(type, {d0, 1000});
  auto p0 =
      *LiteralUtil::CreateRandomLiteral<type>(input_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&p0};
  CHECK_OK(RunHloBenchmark(
      state, hlo, args,
      {{"$d0", absl::StrCat(d0)},
       {"$type", primitive_util::LowercasePrimitiveTypeName(type)}},
      options));
}

// It is useful to also have a benchmark where the minor dimension is a power of
// two as it suffers from cache aliasing which then shows different performance
// characteristics.
template <PrimitiveType type>
static void BM_TransposeAndCopySquare(benchmark::State& state,
                                      const HloBenchmarkOptions& options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule transpose_and_copy_square_$type_$d0

    ENTRY e {
      p0 = $type[$d0,$d0] parameter(0)
      transpose = $type[$d0,$d0] transpose(p0), dimensions={1,0}
      ROOT copy = $type[$d0,$d0] copy(transpose)
    }
  )";

  std::minstd_rand0 engine;

  auto input_shape = ShapeUtil::MakeShape(type, {d0, d0});
  auto p0 =
      *LiteralUtil::CreateRandomLiteral<type>(input_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&p0};
  CHECK_OK(RunHloBenchmark(
      state, hlo, args,
      {{"$d0", absl::StrCat(d0)},
       {"$type", primitive_util::LowercasePrimitiveTypeName(type)}},
      options));
}

#define REGISTER_BENCHMARK(NAME) \
  XLA_CPU_BENCHMARK(NAME)        \
      ->MeasureProcessCPUTime()  \
      ->Arg(128)                 \
      ->Arg(256)                 \
      ->Arg(512)                 \
      ->Arg(1024)                \
      ->Arg(4096);

#define BENCHMARK_TRANSPOSE_AND_COPY(TYPE)                                    \
  static void BM_TransposeAndCopy##TYPE(benchmark::State& state,              \
                                        const HloBenchmarkOptions& options) { \
    BM_TransposeAndCopy<TYPE>(state, options);                                \
  }                                                                           \
  static void BM_TransposeAndCopySquare##TYPE(                                \
      benchmark::State& state, const HloBenchmarkOptions& options) {          \
    BM_TransposeAndCopySquare<TYPE>(state, options);                          \
  }                                                                           \
  REGISTER_BENCHMARK(BM_TransposeAndCopy##TYPE);                              \
  REGISTER_BENCHMARK(BM_TransposeAndCopySquare##TYPE);

BENCHMARK_TRANSPOSE_AND_COPY(BF16);
BENCHMARK_TRANSPOSE_AND_COPY(F16);
BENCHMARK_TRANSPOSE_AND_COPY(F32);
BENCHMARK_TRANSPOSE_AND_COPY(F64);

}  // namespace xla::cpu
