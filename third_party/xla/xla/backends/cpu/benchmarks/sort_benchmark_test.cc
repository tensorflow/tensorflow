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

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/backends/cpu/benchmarks/multi_benchmark_config.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

static void BM_PermutationSort_FP32(benchmark::State& state,
                                    HloBenchmarkOptions options) {
  int64_t batch = state.range(0);
  int64_t dim = state.range(1);

  absl::string_view hlo = R"(
    HloModule text_permutation_sort_benchmark_fp32

    compare {
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      max = u32[] constant(2147483647)
      zero = s32[] constant(0)
      lhs.signed = s32[] bitcast-convert(p.0.lhs)
      lhs.unsigned = u32[] bitcast-convert(p.0.lhs)
      lhs.flipped = u32[] subtract(max, lhs.unsigned)
      lhs.flipped.signed = s32[] bitcast-convert(lhs.flipped)
      lhs.is_negative = pred[] compare(lhs.flipped.signed, zero), direction=LT
      lhs.converted = s32[] select(lhs.is_negative, lhs.flipped.signed, lhs.signed)
      rhs.signed = s32[] bitcast-convert(p.0.rhs)
      rhs.unsigned = u32[] bitcast-convert(p.0.rhs)
      rhs.flipped = u32[] subtract(max, rhs.unsigned)
      rhs.flipped.signed = s32[] bitcast-convert(rhs.flipped)
      rhs.is_negative = pred[] compare(rhs.flipped.signed, zero), direction=LT
      rhs.converted = s32[] select(rhs.is_negative, rhs.flipped.signed, rhs.signed)
      ROOT lt = pred[] compare(lhs.converted, rhs.converted), direction=LT
    }

    compare2 {
      p.0.lhs = s32[] parameter(0)
      p.0.rhs = s32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
    }

    ENTRY sort_computation {
      keys = f32[$batch,$dim]{1,0} parameter(0)
      values = s32[$batch,$dim]{1,0} iota(), iota_dimension=1
      sort = (f32[$batch,$dim]{1,0}, s32[$batch,$dim]{1,0}) sort(keys, values), dimensions={1}, to_apply=compare
      gte = s32[$batch,$dim]{1,0} get-tuple-element(sort), index=1
      ROOT sort2 = (s32[$batch,$dim]{1,0}, s32[$batch,$dim]{1,0}) sort(gte, values), dimensions={1}, to_apply=compare2
    }
  )";

  std::minstd_rand0 engine(/*seed=*/0xCAFEFEED);
  auto keys_or = LiteralUtil::CreateRandomLiteral<F32>(
      ShapeUtil::MakeShape(F32, {batch, dim}), &engine, 1.0f, 0.1f);
  CHECK_OK(keys_or.status());
  Literal& keys = *keys_or;

  CHECK_OK(RunHloBenchmark(
      state, hlo, {&keys},
      {{"$batch", absl::StrCat(batch)}, {"$dim", absl::StrCat(dim)}}, options));
}

#define BENCHMARK_PERMUTATION_SORT(name) \
  XLA_CPU_BENCHMARK(name)                \
      ->MeasureProcessCPUTime()          \
      ->ArgNames({"batch", "dim"})       \
      ->Args({64, 8732})                 \
      ->Args({1, 8732})                  \
      ->Args({64, 1024})

BENCHMARK_PERMUTATION_SORT(BM_PermutationSort_FP32);

}  // namespace xla::cpu
