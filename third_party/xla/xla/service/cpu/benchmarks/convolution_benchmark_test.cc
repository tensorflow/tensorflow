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

#include "absl/strings/str_cat.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/shape_util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::cpu {
namespace {

bool IsOdd(int n) { return n % 2 == 1; }

template <PrimitiveType ElementType>
static void BM_Conv2D(benchmark::State& state) {
  int batch = state.range(0);
  int height = state.range(1);
  int width = state.range(2);
  int input_channels = state.range(3);
  int kernel_h = state.range(4);
  int kernel_w = state.range(5);
  int output_channels = state.range(6);

  // Padding values for 'SAME' padding. Only odd kernel sizes are supported.
  CHECK(IsOdd(kernel_h));
  CHECK(IsOdd(kernel_w));
  int padding_h = (kernel_h - 1) / 2;
  int padding_w = (kernel_w - 1) / 2;

  std::string hlo_module = R"(
    HloModule TestModule

    ENTRY TestComputation {
      %p0 = $input_shape parameter(0)
      %p1 = $kernel_shape parameter(1)
      ROOT conv = convolution(p0, p1), window={size=$window_size pad=$padding},
        dim_labels=b01f_01io->b01f
    }
  )";

  std::minstd_rand0 engine;

  // Input format is NHWC.
  auto input_shape =
      ShapeUtil::MakeShape(ElementType, {batch, height, width, input_channels});
  // Filter format is HWIO.
  auto kernel_shape = ShapeUtil::MakeShape(
      ElementType, {kernel_h, kernel_w, input_channels, output_channels});
  auto input = *LiteralUtil::CreateRandomLiteral<ElementType>(
      input_shape, &engine, 1.0f, 0.1f);
  auto kernel = *LiteralUtil::CreateRandomLiteral<ElementType>(
      kernel_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&input, &kernel};

  CHECK_OK(
      RunHloBenchmark(state, hlo_module, args,
                      {{"$input_shape", input_shape.ToString()},
                       {"$kernel_shape", kernel_shape.ToString()},
                       {"$window_size", absl::StrCat(kernel_h, "x", kernel_w)},
                       {"$padding", absl::StrCat(padding_h, "_", padding_h, "x",
                                                 padding_w, "_", padding_w)}}));
}

}  // namespace

// -------------------------------------------------------------------------- //
// Pixel CNN convolutions.
// -------------------------------------------------------------------------- //

// Shapes from XLA convolution tests
BENCHMARK(BM_Conv2D<PrimitiveType::F32>)
    ->MeasureProcessCPUTime()
    ->Args({8, 5, 5, 1, 1, 1, 32})
    ->Args({8, 5, 5, 4, 1, 1, 32})
    ->Args({8, 128, 128, 4, 1, 1, 8});

// Shapes from TF convolution benchmarks.
BENCHMARK(BM_Conv2D<PrimitiveType::F32>)
    ->MeasureProcessCPUTime()
    ->Args({8, 32, 32, 128, 1, 1, 1024})
    ->Args({16, 32, 32, 128, 1, 1, 1024})
    ->Args({32, 32, 32, 128, 1, 1, 1024});

// Shapes similar to Eigen spatial convolution benchmarks.
BENCHMARK(BM_Conv2D<PrimitiveType::F32>)
    ->MeasureProcessCPUTime()
    ->Args({32, 64, 64, 32, 1, 1, 64})
    ->Args({32, 256, 256, 4, 1, 1, 16})
    ->Args({32, 64, 64, 4, 1, 1, 16})
    ->Args({32, 32, 32, 96, 1, 1, 96});

// -------------------------------------------------------------------------- //
// 3x3 Convolution: SpatialConvolution
// -------------------------------------------------------------------------- //

// Shapes from XLA convolution tests
BENCHMARK(BM_Conv2D<PrimitiveType::F32>)
    ->MeasureProcessCPUTime()
    ->Args({8, 5, 5, 1, 3, 3, 32})
    ->Args({8, 5, 5, 4, 3, 3, 32})
    ->Args({8, 128, 128, 4, 3, 3, 8});

// Shapes from TF convolution benchmarks
BENCHMARK(BM_Conv2D<PrimitiveType::F32>)
    ->MeasureProcessCPUTime()
    ->Args({8, 32, 32, 128, 3, 3, 1024})
    ->Args({16, 32, 32, 128, 3, 3, 1024})
    ->Args({32, 32, 32, 128, 3, 3, 1024});

// Shapes similar to Eigen spatial convolution benchmarks.
BENCHMARK(BM_Conv2D<PrimitiveType::F32>)
    ->MeasureProcessCPUTime()
    ->Args({32, 64, 64, 32, 3, 3, 64})
    ->Args({32, 256, 256, 4, 3, 3, 16})
    ->Args({32, 64, 64, 4, 3, 3, 16})
    ->Args({32, 32, 32, 96, 3, 3, 96});

}  // namespace xla::cpu
