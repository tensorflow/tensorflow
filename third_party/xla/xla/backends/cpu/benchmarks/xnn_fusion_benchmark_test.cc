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
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
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

static absl::Status RunFusionBenchmark(benchmark::State& state,
                                       HloBenchmarkOptions options,
                                       absl::string_view hlo,
                                       bool is_xnn_fusion = false) {
  int64_t d0 = state.range(0);  // Tensor size.
  int64_t n = state.range(1);   // Number of add-multiply iterations.

  // Adding `n` iterations of `add` and `multiply`.
  std::string repeat;
  for (int i = 1; i <= n; ++i) {
    repeat += absl::Substitute(
        "\n  add$0 = f32[$1,$1] add(add$2, mul$2)"
        "\n  mul$0 = f32[$1,$1] multiply(add$0, add$0)",
        i, d0, i - 1);
  }

  // Generate inputs.
  std::minstd_rand0 engine;
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(
      ShapeUtil::MakeShape(F32, {d0, d0}), &engine, 1.0f, 0.1f);
  auto p1 = *LiteralUtil::CreateRandomLiteral<F32>(
      ShapeUtil::MakeShape(F32, {d0, d0}), &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&p0, &p1};

  if (is_xnn_fusion) {
    options.disable_parallel_task_assigner = true;
    options.aot_options = nullptr;
  }

  return RunHloBenchmark(state, hlo, args,
                         {{"$d0", absl::StrCat(d0)},
                          {"$n", absl::StrCat(n)},
                          {"$repeat_n_iterations", repeat}},
                         options);
}

static void BM_EltwiseF32(benchmark::State& state,
                          HloBenchmarkOptions options) {
  // Perform `n+1` iterations of `add` and `multiply`, then end with `subtract`.
  absl::string_view hlo = R"(
    HloModule eltwise_f32_$n

    ENTRY e {
      p0 = f32[$d0,$d0] parameter(0)
      p1 = f32[$d0,$d0] parameter(1)
      add0 = f32[$d0,$d0] add(p0, p1)
      mul0 = f32[$d0,$d0] multiply(add0, add0)
      $repeat_n_iterations
      ROOT sub = f32[$d0,$d0] subtract(mul$n, p0)
    }
  )";
  CHECK_OK(RunFusionBenchmark(state, std::move(options), hlo));
}

static void BM_XnnEltwiseF32(benchmark::State& state,
                             HloBenchmarkOptions options) {
  // Perform `n+1` iterations of `add` and `multiply`, then end with `subtract`.
  absl::string_view hlo = R"(
    HloModule eltwise_f32_$n

    xnn_fusion {
      p0 = f32[$d0,$d0] parameter(0)
      p1 = f32[$d0,$d0] parameter(1)
      add0 = f32[$d0,$d0] add(p0, p1)
      mul0 = f32[$d0,$d0] multiply(add0, add0)
      $repeat_n_iterations
      ROOT sub = f32[$d0,$d0] subtract(mul$n, p0)
    }

    ENTRY e {
      p0 = f32[$d0,$d0] parameter(0)
      p1 = f32[$d0,$d0] parameter(1)
      ROOT %result = f32[$d0,$d0] fusion(%p0, %p1), kind=kCustom,
        calls=xnn_fusion,
        backend_config={"fusion_config": {kind: "__xnn_fusion"}}
    }
  )";
  CHECK_OK(RunFusionBenchmark(state, std::move(options), hlo,
                              /*is_xnn_fusion=*/true));
}

static void BM_DotAndEltwiseF32(benchmark::State& state,
                                HloBenchmarkOptions options) {
  // Perform `dot` followed by `n+1` iterations of `add` and `multiply`, then
  // end with `subtract`.
  absl::string_view hlo = R"(
    HloModule dot_and_eltwise_f32_$n

    ENTRY e {
      p0 = f32[$d0,$d0] parameter(0)
      p1 = f32[$d0,$d0] parameter(1)
      dot0 = f32[$d0,$d0] dot(p0, p1), lhs_contracting_dims={1},
                                       rhs_contracting_dims={0}
      add0 = f32[$d0,$d0] add(dot0, p1)
      mul0 = f32[$d0,$d0] multiply(add0, add0)
      $repeat_n_iterations
      ROOT sub = f32[$d0,$d0] subtract(mul$n, p0)
    }
  )";
  CHECK_OK(RunFusionBenchmark(state, std::move(options), hlo));
}

static void BM_XnnDotAndEltwiseF32(benchmark::State& state,
                                   HloBenchmarkOptions options) {
  // Perform `dot` followed by `n+1` iterations of `add` and `multiply`, then
  // end with `subtract`.
  absl::string_view hlo = R"(
    HloModule dot_and_eltwise_f32_$n

    xnn_fusion {
      p0 = f32[$d0,$d0] parameter(0)
      p1 = f32[$d0,$d0] parameter(1)
      dot0 = f32[$d0,$d0] dot(p0, p1), lhs_contracting_dims={1},
                                       rhs_contracting_dims={0}
      add0 = f32[$d0,$d0] add(dot0, p1)
      mul0 = f32[$d0,$d0] multiply(add0, add0)
      $repeat_n_iterations
      ROOT sub = f32[$d0,$d0] subtract(mul$n, p0)
    }

    ENTRY e {
      p0 = f32[$d0,$d0] parameter(0)
      p1 = f32[$d0,$d0] parameter(1)
      ROOT %result = f32[$d0,$d0] fusion(%p0, %p1), kind=kCustom,
        calls=xnn_fusion,
        backend_config={"fusion_config": {kind: "__xnn_fusion"}}
    }
  )";
  CHECK_OK(RunFusionBenchmark(state, std::move(options), hlo,
                              /*is_xnn_fusion=*/true));
}

#define BENCHMARK_FUSION(name)  \
  XLA_CPU_BENCHMARK(name)       \
      ->MeasureProcessCPUTime() \
      ->Args({1024, 4})         \
      ->Args({1024, 8})         \
      ->Args({1024, 16})        \
      ->Args({1024, 32})

BENCHMARK_FUSION(BM_EltwiseF32);
BENCHMARK_FUSION(BM_XnnEltwiseF32);
BENCHMARK_FUSION(BM_DotAndEltwiseF32);
BENCHMARK_FUSION(BM_XnnDotAndEltwiseF32);

}  // namespace xla::cpu
