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

static void BM_Optimizer0(benchmark::State& state,
                          HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule jit_update_fn_$d0

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    ENTRY e {
      p0 = f32[1,2,1,$d0,256] parameter(0)
      p1 = f32[1,2,1,$d0,256] parameter(1)
      p2 = f32[1,2,1,$d0,256] parameter(2)
      p3 = s32[] parameter(3)
      c0 = f32[] constant(0.9)
      bcast0 = f32[1,2,1,$d0,256] broadcast(c0), dimensions={}
      mul0 = f32[1,2,1,$d0,256] multiply(p1, bcast0)
      c1 = f32[] constant(1)
      c2 = f32[] constant(0.95)
      convert0 = f32[] convert(p3)
      power0 = f32[] power(c2, convert0)
      sub0 = f32[] subtract(c1, power0)
      mul1 = f32[] multiply(sub0, c2)
      c3 = s32[] constant(1)
      add0 = s32[] add(p3, c3)
      convert1 = f32[] convert(add0)
      power1 = f32[] power(c2, convert1)
      sub1 = f32[] subtract(c1, power1)
      div0 = f32[] divide(mul1, sub1)
      bcast1 = f32[1,2,1,$d0,256] broadcast(div0), dimensions={}
      mul2 = f32[1,2,1,$d0,256] multiply(bcast1, p2)
      sub2 = f32[] subtract(c1, div0)
      bcast2 = f32[1,2,1,$d0,256] broadcast(sub2), dimensions={}
      mul3 = f32[1,2,1,$d0,256] multiply(p0, p0)
      mul4 = f32[1,2,1,$d0,256] multiply(bcast2, mul3)
      add1 = f32[1,2,1,$d0,256] add(mul2, mul4)
      c4 = f32[] constant(1e-30)
      bcast3 = f32[1,2,1,$d0,256] broadcast(c4), dimensions={}
      add2 = f32[1,2,1,$d0,256] add(add1, bcast3)
      sqrt0 = f32[1,2,1,$d0,256] sqrt(add2)
      div1 = f32[1,2,1,$d0,256] divide(p0, sqrt0)
      mul5 = f32[1,2,1,$d0,256] multiply(div1, div1)
      c5 = f32[] constant(0)
      reduce0 = f32[1,2]{1,0} reduce(mul5, c5), dimensions={2,3,4}, to_apply=add
      reshape0 = f32[1,2,1,1,1] reshape(reduce0)
      c6 = f32[] constant(32768)
      bcast4 = f32[1,2,1,1,1] broadcast(c6), dimensions={}
      div2 = f32[1,2,1,1,1] divide(reshape0, bcast4)
      sqrt1 = f32[1,2,1,1,1] sqrt(div2)
      c7 = f32[] constant(1)
      bcast5 = f32[1,2,1,1,1] broadcast(c7), dimensions={}
      div3 = f32[1,2,1,1,1] divide(sqrt1, bcast5)
      max0 = f32[1,2,1,1,1] maximum(div3, bcast5)
      bcast6 = f32[1,2,1,1,1] broadcast(max0), dimensions={0,1,2,3,4}
      reshape1 = f32[1,2,1]{2,1,0} reshape(bcast6)
      bcast7 = f32[1,2,1,$d0,256] broadcast(reshape1), dimensions={0,1,2}
      div4 = f32[1,2,1,$d0,256] divide(div1, bcast7)
      c8 = f32[] constant(0.4358899)
      bcast8 = f32[1,2,1,$d0,256] broadcast(c8), dimensions={}
      mul6 = f32[1,2,1,$d0,256] multiply(div4, bcast8)
      add3 = f32[1,2,1,$d0,256] add(mul0, mul6)
      ROOT tuple = (f32[1,2,1,$d0,256], f32[1,2,1,$d0,256], f32[1,2,1,$d0,256])
                    tuple(add3, add3, add1)
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(F32, {1, 2, 1, d0, 256});
  auto scalar = ShapeUtil::MakeShape(S32, {});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);
  auto p1 = *LiteralUtil::CreateRandomLiteral<S32>(scalar, &engine, 1, 2);

  std::vector<const Literal*> args = {&p0, &p0, &p0, &p1};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

XLA_CPU_BENCHMARK(BM_Optimizer0)
    ->MeasureProcessCPUTime()
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(8192)
    ->Arg(16384);

}  // namespace xla::cpu
