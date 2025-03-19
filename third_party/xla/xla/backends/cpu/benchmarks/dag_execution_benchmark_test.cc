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

static void BM_DagExecution(benchmark::State& state) {
  int64_t d0 = state.range(0);
  bool is_aot = static_cast<bool>(state.range(1));

  // We use this benchmark to test how well XLA does the scheduling of the HLO
  // module to extract available parallelism, and how well ThunkExecutor
  // exploits that parallelism at run time.
  absl::string_view hlo = R"(
    HloModule fusion_f32_$d0

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    ENTRY e {
      p0 = f32[1,2,1,$d0,256] parameter(0)
      c0 = f32[] constant(0)

      c1 = f32[] constant(1)
      bcast1 = f32[1,2,1,$d0,256] broadcast(c1), dimensions={}
      add1 = f32[1,2,1,$d0,256] add(p0, bcast1)

      c2 = f32[] constant(2)
      bcast2 = f32[1,2,1,$d0,256] broadcast(c2), dimensions={}
      add2 = f32[1,2,1,$d0,256] add(p0, bcast2)

      c3 = f32[] constant(3)
      bcast3 = f32[1,2,1,$d0,256] broadcast(c3), dimensions={}
      add3 = f32[1,2,1,$d0,256] add(p0, bcast3)

      c4 = f32[] constant(4)
      bcast4 = f32[1,2,1,$d0,256] broadcast(c4), dimensions={}
      add4 = f32[1,2,1,$d0,256] add(p0, bcast4)

      c5 = f32[] constant(5)
      bcast5 = f32[1,2,1,$d0,256] broadcast(c5), dimensions={}
      add5 = f32[1,2,1,$d0,256] add(p0, bcast5)

      r1 = f32[1,2] reduce(add1, c0), dimensions={2,3,4}, to_apply=add
      r2 = f32[1,2] reduce(add2, c0), dimensions={2,3,4}, to_apply=add
      r3 = f32[1,2] reduce(add3, c0), dimensions={2,3,4}, to_apply=add
      r4 = f32[1,2] reduce(add4, c0), dimensions={2,3,4}, to_apply=add
      r5 = f32[1,2] reduce(add5, c0), dimensions={2,3,4}, to_apply=add

      out0 = f32[1,2] add(r1, r2)
      out1 = f32[1,2] add(r3, r4)
      out2 = f32[1,2] add(out0, out1)
      ROOT out3 = f32[1,2] add(out2, r5)
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(F32, {1, 2, 1, d0, 256});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0};

  HloBenchmarkOptions benchmark_options;
  benchmark_options.aot_options = is_aot ? GetAotCompilationOptions() : nullptr;

  CHECK_OK(RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}},
                           benchmark_options));
}

void GenerateArgDagExecutionArgs(benchmark::internal::Benchmark* benchmark) {
  benchmark->MeasureProcessCPUTime();
  benchmark->ArgNames({"d0", "is_aot"});
  const std::vector<bool> is_aot_values = {false, true};
  std::vector<int64_t> args_values = {128, 256, 512, 1024, 8192, 16384};

  for (const auto& arg_value : args_values) {
    for (const auto& is_aot : is_aot_values) {
      benchmark->Args({arg_value, is_aot});
    }
  }
}

BENCHMARK(BM_DagExecution)->Apply(GenerateArgDagExecutionArgs);

}  // namespace xla::cpu
