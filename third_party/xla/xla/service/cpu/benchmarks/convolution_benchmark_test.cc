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

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/shape_util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::cpu {

static void BM_Conv2DF32_3x3(benchmark::State& state) {
  int feature_size = state.range(0);
  int input_channels = state.range(1);
  int output_channels = state.range(2);

  std::string hlo_module = R"(
    HloModule TestModule

    ENTRY TestComputation {
      %p0 = $input_shape parameter(0)
      %p1 = $kernel_shape parameter(1)
      ROOT conv = convolution(p0, p1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
    }
  )";

  std::minstd_rand0 engine;

  auto input_shape = ShapeUtil::MakeShape(
      F32, {8, feature_size, feature_size, input_channels});
  auto kernel_shape =
      ShapeUtil::MakeShape(F32, {3, 3, input_channels, output_channels});
  auto input =
      *LiteralUtil::CreateRandomLiteral<F32>(input_shape, &engine, 1.0f, 0.1f);
  auto kernel =
      *LiteralUtil::CreateRandomLiteral<F32>(kernel_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&input, &kernel};

  CHECK_OK(RunHloBenchmark(state, hlo_module, args,
                           {{"$input_shape", input_shape.ToString()},
                            {"$kernel_shape", kernel_shape.ToString()}}));
}

static void BM_Conv2DF32_1x1(benchmark::State& state) {
  int feature_size = state.range(0);
  int input_channels = state.range(1);
  int output_channels = state.range(2);

  std::string hlo_module = R"(
    HloModule TestModule

    ENTRY TestComputation {
      %p0 = $input_shape parameter(0)
      %p1 = $kernel_shape parameter(1)
      ROOT conv = convolution(p0, p1), window={size=1x1 pad=0_0x0_0}, dim_labels=b01f_01io->b01f
    }
  )";

  std::minstd_rand0 engine;

  auto input_shape = ShapeUtil::MakeShape(
      F32, {8, feature_size, feature_size, input_channels});
  auto kernel_shape =
      ShapeUtil::MakeShape(F32, {1, 1, input_channels, output_channels});
  auto input =
      *LiteralUtil::CreateRandomLiteral<F32>(input_shape, &engine, 1.0f, 0.1f);
  auto kernel =
      *LiteralUtil::CreateRandomLiteral<F32>(kernel_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&input, &kernel};

  CHECK_OK(RunHloBenchmark(state, hlo_module, args,
                           {{"$input_shape", input_shape.ToString()},
                            {"$kernel_shape", kernel_shape.ToString()}}));
}

BENCHMARK(BM_Conv2DF32_3x3)
    ->MeasureProcessCPUTime()
    ->Args({5, 1, 32})
    ->Args({5, 4, 32})
    ->Args({128, 4, 8});

BENCHMARK(BM_Conv2DF32_1x1)
    ->MeasureProcessCPUTime()
    ->Args({5, 1, 32})
    ->Args({5, 4, 32})
    ->Args({128, 4, 8});

}  // namespace xla::cpu
