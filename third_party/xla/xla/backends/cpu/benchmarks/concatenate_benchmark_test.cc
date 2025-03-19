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
#include <memory>
#include <random>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/benchmarks/aot_benchmark_helper.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

static void BM_ConcatenateTwoR3F32(benchmark::State& state) {
  bool disable_parallel_backend = !static_cast<bool>(state.range(0));
  int64_t dims[3] = {state.range(1), state.range(2), state.range(3)};
  Shape shape = ShapeUtil::MakeShape(F32, dims);
  int64_t axis = state.range(4);
  const bool is_aot = static_cast<bool>(state.range(5));

  absl::string_view hlo = R"(
    HloModule concatenate_r3f32_$shape_repr

    ENTRY test {
      p0 = f32[$shape] parameter(0)
      p1 = f32[$shape] parameter(1)
      ROOT cat = f32[$out_shape] concatenate(p0, p1), dimensions={$axis}
    }
  )";

  // Expected dimension after concatenation
  int64_t out_dims[3] = {dims[0], dims[1], dims[2]};
  out_dims[axis] *= 2;

  std::minstd_rand0 engine;

  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);
  auto p1 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);

  HloBenchmarkOptions benchmark_options;
  benchmark_options.disable_parallel_task_assigner = disable_parallel_backend;
  benchmark_options.aot_options = is_aot ? GetAotCompilationOptions() : nullptr;

  std::vector<const Literal*> args = {&p0, &p1};
  CHECK_OK(RunHloBenchmark(state, hlo, args,
                           {{"$shape_repr", absl::StrJoin(dims, "x")},
                            {"$shape", absl::StrJoin(dims, ",")},
                            {"$out_shape", absl::StrJoin(out_dims, ",")},
                            {"$axis", absl::StrCat(axis)}},
                           benchmark_options));
}

void GenerateArgs(benchmark::internal::Benchmark* benchmark) {
  benchmark->ArgNames(
      {"parallel", "batch", "width", "height", "axis", "is_aot"});
  const std::vector<bool> parralel_values = {false, true};
  const std::vector<std::array<int64_t, 3>> batch_height_width_values = {
      {256, 128, 64}, {64, 256, 128}, {128, 64, 256}};
  const std::vector<int64_t> axis_values = {0, 1, 2};
  const std::vector<bool> is_aot_values = {false, true};

  for (bool parallel : parralel_values) {
    for (const auto& [batch, height, width] : batch_height_width_values) {
      for (int64_t axis : axis_values) {
        for (bool is_aot : is_aot_values) {
          benchmark->Args({parallel, batch, height, width, axis, is_aot});
        }
      }
    }
  }

  benchmark->MeasureProcessCPUTime();
}

BENCHMARK(BM_ConcatenateTwoR3F32)->Apply(GenerateArgs);

}  // namespace xla::cpu
