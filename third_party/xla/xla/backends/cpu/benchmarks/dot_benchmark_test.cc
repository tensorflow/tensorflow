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

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/benchmarks/aot_benchmark_helper.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

static void BM_BatchedDot(benchmark::State& state) {
  PrimitiveType dtype = static_cast<PrimitiveType>(state.range(0));
  int64_t d0 = state.range(1);
  int64_t d1 = state.range(2);
  bool is_aot = static_cast<bool>(state.range(3));

  absl::string_view hlo = R"(
    HloModule dot_$dtype_b$d0_d$d1

    ENTRY e {
      p0 = $dtype[$d0,$d1,$d1] parameter(0)
      p1 = $dtype[$d0,$d1,$d1] parameter(1)
      ROOT dot = $dtype[$d0,$d1,$d1] dot(p0, p1),
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

  HloBenchmarkOptions benchmark_options;
  benchmark_options.aot_options = is_aot ? GetAotCompilationOptions() : nullptr;

  CHECK_OK(RunHloBenchmark(
      state, hlo, args,
      {{"$dtype", primitive_util::LowercasePrimitiveTypeName(dtype)},
       {"$d0", absl::StrCat(d0)},
       {"$d1", absl::StrCat(d1)}},
      benchmark_options));
}

void GenerateBatchedDotArgs(benchmark::internal::Benchmark* benchmark) {
  benchmark->MeasureProcessCPUTime();
  benchmark->ArgNames({"dtype", "d0", "d1", "is_aot"});
  const std::vector<PrimitiveType> dtypes = {F32, BF16};
  const std::vector<bool> is_aot_values = {false, true};
  std::vector<std::array<int64_t, 2>> args_values = {
      {1, 2}, {1, 32}, {1, 64}, {1, 128}, {1, 256}, {1, 512},
      {2, 2}, {2, 32}, {2, 64}, {2, 128}, {2, 256}, {2, 512},
      {4, 2}, {4, 32}, {4, 64}, {4, 128}, {4, 256}, {4, 512},
      {8, 2}, {8, 32}, {8, 64}, {8, 128}, {8, 256}, {8, 512}};

  for (const auto& dtype : dtypes) {
    for (const auto& arg_value : args_values) {
      for (const auto& is_aot : is_aot_values) {
        std::vector<int64_t> all_arg_values = {dtype, arg_value[0],
                                               arg_value[1], is_aot};
        benchmark->Args(all_arg_values);
      }
    }
  }
}

BENCHMARK(BM_BatchedDot)->Apply(GenerateBatchedDotArgs);

}  // namespace xla::cpu
