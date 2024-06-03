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

#include <random>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/shape_util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::cpu {

static void BM_AddF32(benchmark::State& state) {
  std::string hlo_module = R"(
    HloModule m

    add {
      p0 = f32[1024] parameter(0)
      p1 = f32[1024] parameter(1)
      ROOT add = f32[1024] add(p0, p1)
    }

    ENTRY e {
      p0 = f32[1024] parameter(0)
      p1 = f32[1024] parameter(1)
      ROOT fusion = f32[1024] fusion(p0, p1), kind=kLoop, calls=add
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(F32, {1024});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);
  auto p1 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0, &p1};
  CHECK_OK(RunHloBenchmark(state, hlo_module, args));
}

BENCHMARK(BM_AddF32)->MeasureProcessCPUTime();

}  // namespace xla::cpu
