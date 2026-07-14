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
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/backends/cpu/benchmarks/multi_benchmark_config.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

static void BM_ConcatenateTwoR3F32(benchmark::State& state,
                                   HloBenchmarkOptions options) {
  bool disable_parallel_backend = !static_cast<bool>(state.range(0));
  int64_t dims[3] = {state.range(1), state.range(2), state.range(3)};
  Shape shape = ShapeUtil::MakeShape(F32, dims);
  int64_t axis = state.range(4);

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

  options.disable_parallel_task_assigner = disable_parallel_backend;

  std::vector<const Literal*> args = {&p0, &p1};
  CHECK_OK(RunHloBenchmark(state, hlo, args,
                           {{"$shape_repr", absl::StrJoin(dims, "x")},
                            {"$shape", absl::StrJoin(dims, ",")},
                            {"$out_shape", absl::StrJoin(out_dims, ",")},
                            {"$axis", absl::StrCat(axis)}},
                           options));
}

XLA_CPU_BENCHMARK(BM_ConcatenateTwoR3F32)
    ->MeasureProcessCPUTime()
    ->ArgNames({"parallel", "batch", "width", "height", "axis"})
    // Fast Concat (memcpy, no parallelism)
    //   axis=0
    ->Args({false, 256, 128, 64, 0})
    ->Args({false, 64, 256, 128, 0})
    ->Args({false, 128, 64, 256, 0})
    //   axis=1
    ->Args({false, 256, 128, 64, 1})
    ->Args({false, 64, 256, 128, 1})
    ->Args({false, 128, 64, 256, 1})
    //   axis=2
    ->Args({false, 256, 128, 64, 2})
    ->Args({false, 64, 256, 128, 2})
    ->Args({false, 128, 64, 256, 2})
    // Parallel Concat
    //   axis=0
    ->Args({true, 256, 128, 64, 0})
    ->Args({true, 64, 256, 128, 0})
    ->Args({true, 128, 64, 256, 0})
    //   axis=1
    ->Args({true, 256, 128, 64, 1})
    ->Args({true, 64, 256, 128, 1})
    ->Args({true, 128, 64, 256, 1})
    //   axis=2
    ->Args({true, 256, 128, 64, 2})
    ->Args({true, 64, 256, 128, 2})
    ->Args({true, 128, 64, 256, 2});

}  // namespace xla::cpu
