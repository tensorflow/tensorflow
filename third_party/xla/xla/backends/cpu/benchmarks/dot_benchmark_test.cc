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

static void BM_BatchedDot(benchmark::State& state,
                          HloBenchmarkOptions options) {
  PrimitiveType dtype = static_cast<PrimitiveType>(state.range(0));
  PrimitiveType out_dtype = F32;
  int64_t d0 = state.range(1);
  int64_t d1 = state.range(2);

  absl::string_view hlo = R"(
    HloModule dot_$dtype_b$d0_d$d1

    ENTRY e {
      p0 = $dtype[$d0,$d1,$d1] parameter(0)
      p1 = $dtype[$d0,$d1,$d1] parameter(1)
      ROOT dot = $out_dtype[$d0,$d1,$d1] dot(p0, p1),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={1}
    }
  )";

  Literal p0, p1;
  double mean = 1.0f;
  double stddev = 0.1f;
  std::minstd_rand0 engine;
  auto shape = ShapeUtil::MakeShape(dtype, {d0, d1, d1});
  if (dtype == F32) {
    p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, mean, stddev);
    p1 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, mean, stddev);
  } else if (dtype == BF16) {
    p0 = *LiteralUtil::CreateRandomLiteral<BF16>(shape, &engine, mean, stddev);
    p1 = *LiteralUtil::CreateRandomLiteral<BF16>(shape, &engine, mean, stddev);
  } else {
    LOG(FATAL) << "Add dtype to the if-else block before use: " << dtype;
  }

  std::vector<const Literal*> args = {&p0, &p1};
  CHECK_OK(RunHloBenchmark(
      state, hlo, args,
      {{"$dtype", primitive_util::LowercasePrimitiveTypeName(dtype)},
       {"$out_dtype", primitive_util::LowercasePrimitiveTypeName(out_dtype)},
       {"$d0", absl::StrCat(d0)},
       {"$d1", absl::StrCat(d1)}},
      options));
}

#define BENCHMARK_BATCHED_DOT(dtype) \
  XLA_CPU_BENCHMARK(BM_BatchedDot)   \
      ->MeasureProcessCPUTime()      \
      ->Args({dtype, 1, 2})          \
      ->Args({dtype, 1, 32})         \
      ->Args({dtype, 1, 64})         \
      ->Args({dtype, 1, 128})        \
      ->Args({dtype, 1, 256})        \
      ->Args({dtype, 1, 512})        \
      ->Args({dtype, 2, 2})          \
      ->Args({dtype, 2, 32})         \
      ->Args({dtype, 2, 64})         \
      ->Args({dtype, 2, 128})        \
      ->Args({dtype, 2, 256})        \
      ->Args({dtype, 2, 512})        \
      ->Args({dtype, 4, 2})          \
      ->Args({dtype, 4, 32})         \
      ->Args({dtype, 4, 64})         \
      ->Args({dtype, 4, 128})        \
      ->Args({dtype, 4, 256})        \
      ->Args({dtype, 4, 512})        \
      ->Args({dtype, 8, 2})          \
      ->Args({dtype, 8, 32})         \
      ->Args({dtype, 8, 64})         \
      ->Args({dtype, 8, 128})        \
      ->Args({dtype, 8, 256})        \
      ->Args({dtype, 8, 512})

BENCHMARK_BATCHED_DOT(F32);   // Shown as "11" in the benchmark name.
BENCHMARK_BATCHED_DOT(BF16);  // Shown as "16" in the benchmark name.

}  // namespace xla::cpu
