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

static void BM_DynamicUpdateSliceF32(benchmark::State& state) {
  int64_t d0 = state.range(0);
  bool is_aot = static_cast<bool>(state.range(1));

  absl::string_view hlo = R"(
    HloModule dynamic_update_slice_f32_$d0

    ENTRY e {
      p0 = f32[$d0,256] parameter(0)
      p1 = f32[32,32] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT update = f32[$d0,256] dynamic-update-slice(p0, p1, p2, p3)
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(F32, {d0, 256});
  auto slice = ShapeUtil::MakeShape(F32, {32, 32});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);
  auto p1 = *LiteralUtil::CreateRandomLiteral<F32>(slice, &engine, 1.0f, 0.1f);
  auto p2 = LiteralUtil::CreateR0<int32_t>(0);
  auto p3 = LiteralUtil::CreateR0<int32_t>(0);

  std::vector<const Literal*> args = {&p0, &p1, &p2, &p3};

  HloBenchmarkOptions benchmark_options;
  benchmark_options.aot_options = is_aot ? GetAotCompilationOptions() : nullptr;

  CHECK_OK(RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}},
                           benchmark_options));
}

void GenerateDynamicUpdateSliceArgs(benchmark::internal::Benchmark* benchmark) {
  benchmark->MeasureProcessCPUTime();
  benchmark->ArgNames({"d0"});
  const std::vector<int64_t> args_values = {128, 256, 512, 1024, 8192, 16384};
  const std::vector<bool> is_aot_values = {false, true};

  for (int64_t d0 : args_values) {
    for (bool is_aot : is_aot_values) {
      benchmark->Args({d0, is_aot});
    }
  }
}

BENCHMARK(BM_DynamicUpdateSliceF32)->Apply(GenerateDynamicUpdateSliceArgs);

}  // namespace xla::cpu
