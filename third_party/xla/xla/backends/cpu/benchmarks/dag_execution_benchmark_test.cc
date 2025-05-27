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
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

static void BM_DagExecution(benchmark::State& state,
                            HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

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
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

XLA_CPU_BENCHMARK(BM_DagExecution)
    ->MeasureProcessCPUTime()
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(8192)
    ->Arg(16384);

}  // namespace xla::cpu
