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
#include "xla/backends/cpu/benchmarks/aot_benchmark_helper.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

static void BM_TransposeAndCopy(benchmark::State& state) {
  int64_t d0 = state.range(0);
  bool is_aot = static_cast<bool>(state.range(1));

  absl::string_view hlo = R"(
    HloModule transpose_and_copy_$d0

    ENTRY e {
      p0 = f32[$d0,1000] parameter(0)
      transpose = f32[1000,$d0] transpose(p0), dimensions={1,0}
      ROOT copy = f32[1000,$d0] copy(transpose)
    }
  )";

  std::minstd_rand0 engine;

  auto input_shape = ShapeUtil::MakeShape(F32, {d0, 1000});
  auto p0 =
      *LiteralUtil::CreateRandomLiteral<F32>(input_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&p0};

  HloBenchmarkOptions benchmark_options;
  benchmark_options.aot_options = is_aot ? GetAotCompilationOptions() : nullptr;

  CHECK_OK(RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}},
                           benchmark_options));
}

void GenerateTransposeAndCopyArgs(benchmark::internal::Benchmark* benchmark) {
  benchmark->MeasureProcessCPUTime();
  benchmark->ArgNames({"d0", "is_aot"});
  const std::vector<int64_t> args_values = {128, 256, 512, 1024, 4096};
  const std::vector<bool> is_aot_values = {false, true};

  for (const auto& arg_value : args_values) {
    for (const auto& is_aot : is_aot_values) {
      benchmark->Args({arg_value, is_aot});
    }
  }
}

BENCHMARK(BM_TransposeAndCopy)->Apply(GenerateTransposeAndCopyArgs);

}  // namespace xla::cpu
