/* Copyright 2025 The OpenXLA Authors.

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

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/backends/cpu/benchmarks/multi_benchmark_config.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/stacktrace_handler.h"

namespace xla::cpu {
namespace {

static void BM_ModelWithAliasing(benchmark::State& state,
                                 HloBenchmarkOptions options) {
  int64_t num_executions = state.range(0);

  absl::string_view hlo = R"(
HloModule add_one_aliased, input_output_alias={ {0}: (0, {}, may-alias) }

ENTRY main.5 {
  x = f32[] parameter(0), sharding={replicated}
  constant = f32[] constant(1)
  add_result = f32[] add(x, constant)
  ROOT tuple_result = (f32[]) tuple(add_result)
}
)";

  options.num_executions = num_executions;

  CHECK_OK(RunHloBenchmark(state, hlo, {}, {}, options));
}

XLA_CPU_BENCHMARK(BM_ModelWithAliasing)
    ->ArgName("num_executions")
    ->Arg(1)
    ->Arg(8);

}  // namespace
}  // namespace xla::cpu

GTEST_API_ int main(int argc, char** argv) {
  tsl::testing::InstallStacktraceHandler();

  ::benchmark::Initialize(&argc, argv);
  testing::InitGoogleTest(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
