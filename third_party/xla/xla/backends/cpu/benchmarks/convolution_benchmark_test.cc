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
#include <random>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "xla/backends/cpu/benchmarks/aot_benchmark_helper.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

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
  bool is_aot = static_cast<bool>(state.range(7));

  // Padding values for 'SAME' padding. Only odd kernel sizes are supported.
  CHECK(IsOdd(kernel_h) && IsOdd(kernel_w));
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

  HloBenchmarkOptions benchmark_options;
  benchmark_options.aot_options = is_aot ? GetAotCompilationOptions() : nullptr;

  CHECK_OK(
      RunHloBenchmark(state, hlo_module, args,
                      {{"$input_shape", input_shape.ToString()},
                       {"$kernel_shape", kernel_shape.ToString()},
                       {"$window_size", absl::StrCat(kernel_h, "x", kernel_w)},
                       {"$padding", absl::StrCat(padding_h, "_", padding_h, "x",
                                                 padding_w, "_", padding_w)}},
                      benchmark_options));
}

static void BM_GroupedConv2D(benchmark::State& state) {
  int batch = state.range(0);
  int height = state.range(1);
  int width = state.range(2);
  int input_channels = state.range(3);
  int kernel_h = state.range(4);
  int kernel_w = state.range(5);
  int output_channels = state.range(6);
  int feature_group_count = state.range(7);
  bool is_aot = static_cast<bool>(state.range(8));

  // Derive filter channels from input channels and feature group count.
  int filter_channels = input_channels / feature_group_count;

  // Padding values for 'SAME' padding. Only odd kernel sizes are supported.
  CHECK(IsOdd(kernel_h) && IsOdd(kernel_w));
  int padding_h = (kernel_h - 1) / 2;
  int padding_w = (kernel_w - 1) / 2;

  std::string hlo_module = R"(
    HloModule TestModule

    ENTRY TestComputation {
      %p0 = $input_shape parameter(0)
      %p1 = $kernel_shape parameter(1)
      ROOT conv = convolution(p0, p1), window={size=$window_size pad=$padding},
        dim_labels=b01f_01io->b01f, feature_group_count=$feature_group_count
    }
  )";

  std::minstd_rand0 engine;

  // Input format is NHWC.
  auto input_shape =
      ShapeUtil::MakeShape(F32, {batch, height, width, input_channels});
  // Filter format is HWIO.
  auto kernel_shape = ShapeUtil::MakeShape(
      F32, {kernel_h, kernel_w, filter_channels, output_channels});

  auto input =
      *LiteralUtil::CreateRandomLiteral<F32>(input_shape, &engine, 1.0f, 0.1f);
  auto kernel =
      *LiteralUtil::CreateRandomLiteral<F32>(kernel_shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&input, &kernel};

  HloBenchmarkOptions benchmark_options;
  benchmark_options.aot_options = is_aot ? GetAotCompilationOptions() : nullptr;

  CHECK_OK(RunHloBenchmark(
      state, hlo_module, args,
      {{"$input_shape", input_shape.ToString()},
       {"$kernel_shape", kernel_shape.ToString()},
       {"$window_size", absl::StrCat(kernel_h, "x", kernel_w)},
       {"$padding", absl::StrCat(padding_h, "_", padding_h, "x", padding_w, "_",
                                 padding_w)},
       {"$feature_group_count", absl::StrCat(feature_group_count)}},
      benchmark_options));
}

// Regular strided 1D convolution. Shapes come from an actual use case.
static void BM_Conv1DStrided(benchmark::State& state) {
  int input_channels = state.range(0);
  int output_channels = state.range(1);
  bool is_aot = static_cast<bool>(state.range(2));

  std::string hlo_module = R"(
    HloModule jit_jconvf

    ENTRY main.6 {
      Arg_0.1 = $input_shape parameter(0)
      Arg_1.2 = $kernel_shape parameter(1)
      ROOT conv.3 = $output_shape convolution(Arg_0.1, Arg_1.2),
        window={size=256 stride=64 pad=96_96}, dim_labels=bf0_io0->bf0
    }
  )";

  std::minstd_rand0 engine;

  // NCW layout
  auto input_shape = ShapeUtil::MakeShape(F32, {16, input_channels, 25600});
  auto output_shape = ShapeUtil::MakeShape(F32, {16, output_channels, 400});
  // IOW layout
  auto kernel_shape =
      ShapeUtil::MakeShape(F32, {input_channels, output_channels, 256});

  auto input =
      *LiteralUtil::CreateRandomLiteral<F32>(input_shape, &engine, 1.0f, 0.1f);
  auto kernel =
      *LiteralUtil::CreateRandomLiteral<F32>(kernel_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&input, &kernel};

  HloBenchmarkOptions benchmark_options;
  benchmark_options.aot_options = is_aot ? GetAotCompilationOptions() : nullptr;

  CHECK_OK(RunHloBenchmark(state, hlo_module, args,
                           {{"$input_shape", input_shape.ToString()},
                            {"$kernel_shape", kernel_shape.ToString()},
                            {"$output_shape", output_shape.ToString()}},
                           benchmark_options));
}

// Transposed version (i.e. gradient) of BM_Conv1DStrided. In terms of shapes,
// this operation can be thought of as reverse of regular strided convolution,
// that's why input and output shapes are swapped (so we can directly compare
// performance of this function with BM_Conv1DStrided).
// Currently, the performance is few times worse than regular conv when they
// should be similar.
static void BM_Conv1DTransposedStrided(benchmark::State& state) {
  int input_channels = state.range(0);
  int output_channels = state.range(1);
  bool is_aot = static_cast<bool>(state.range(2));

  std::string hlo_module = R"(
    HloModule jit_jconvt

    ENTRY main.6 {
      Arg_0.1 = $input_shape parameter(0)
      Arg_1.2 = $kernel_shape parameter(1)
      ROOT conv.3 = $output_shape convolution(Arg_0.1, Arg_1.2),
        window={size=256 pad=159_159 lhs_dilate=64}, dim_labels=bf0_io0->bf0
    }
  )";

  std::minstd_rand0 engine;

  // NCW layout
  auto input_shape = ShapeUtil::MakeShape(F32, {16, input_channels, 400});
  auto output_shape = ShapeUtil::MakeShape(F32, {16, output_channels, 25600});
  // IOW layout
  auto kernel_shape =
      ShapeUtil::MakeShape(F32, {input_channels, output_channels, 256});

  auto input =
      *LiteralUtil::CreateRandomLiteral<F32>(input_shape, &engine, 1.0f, 0.1f);
  auto kernel =
      *LiteralUtil::CreateRandomLiteral<F32>(kernel_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&input, &kernel};

  HloBenchmarkOptions benchmark_options;
  benchmark_options.aot_options = is_aot ? GetAotCompilationOptions() : nullptr;

  CHECK_OK(RunHloBenchmark(state, hlo_module, args,
                           {{"$input_shape", input_shape.ToString()},
                            {"$kernel_shape", kernel_shape.ToString()},
                            {"$output_shape", output_shape.ToString()}},
                           benchmark_options));
}

// The same shapes as BM_Conv1DTransposedStrided, but with a different layout.
static void BM_Conv1DTransposedStridedNonDefaultLayout(
    benchmark::State& state) {
  int input_channels = state.range(0);
  int output_channels = state.range(1);
  bool is_aot = static_cast<bool>(state.range(2));

  std::string hlo_module = R"(
    HloModule jit_jconvt

    ENTRY main.6 {
      Arg_0.1 = $input_shape parameter(0)
      Arg_1.2 = $kernel_shape parameter(1)
      ROOT conv.3 = $output_shape convolution(Arg_0.1, Arg_1.2),
        window={size=256 pad=159_159 lhs_dilate=64}, dim_labels=b0f_0oi->b0f
    }
  )";

  std::minstd_rand0 engine;

  // NWC layout
  auto input_shape = ShapeUtil::MakeShape(F32, {16, 400, input_channels});
  auto output_shape = ShapeUtil::MakeShape(F32, {16, 25600, output_channels});
  // WOI layout
  auto kernel_shape =
      ShapeUtil::MakeShape(F32, {256, output_channels, input_channels});

  auto input =
      *LiteralUtil::CreateRandomLiteral<F32>(input_shape, &engine, 1.0f, 0.1f);
  auto kernel =
      *LiteralUtil::CreateRandomLiteral<F32>(kernel_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&input, &kernel};

  HloBenchmarkOptions benchmark_options;
  benchmark_options.aot_options = is_aot ? GetAotCompilationOptions() : nullptr;

  CHECK_OK(RunHloBenchmark(state, hlo_module, args,
                           {{"$input_shape", input_shape.ToString()},
                            {"$kernel_shape", kernel_shape.ToString()},
                            {"$output_shape", output_shape.ToString()}},
                           benchmark_options));
}

// Regular strided 2D convolution. Buffer sizes and convolution parameters are
// based on an actual 1D use case, but adapted to a 2D convolution.
static void BM_Conv2DStrided(benchmark::State& state) {
  bool is_aot = static_cast<bool>(state.range(0));

  std::string hlo_module = R"(
    HloModule jit_jconvf

    ENTRY main.6 {
      Arg_0.1 = f32[16,1,160,160]{3,2,1,0} parameter(0)
      Arg_1.2 = f32[1,129,16,16]{3,2,1,0} parameter(1)
      ROOT conv.3 = f32[16,129,20,20]{3,2,1,0} convolution( Arg_0.1, Arg_1.2),
        window={size=16x16 stride=8x8 pad=4_4x4_4}, dim_labels=bf01_io01->bf01
    }
  )";

  std::minstd_rand0 engine;

  // NCHW layout
  auto input_shape = ShapeUtil::MakeShape(F32, {16, 1, 160, 160});
  // IOHW layout
  auto kernel_shape = ShapeUtil::MakeShape(F32, {1, 129, 16, 16});

  auto input =
      *LiteralUtil::CreateRandomLiteral<F32>(input_shape, &engine, 1.0f, 0.1f);
  auto kernel =
      *LiteralUtil::CreateRandomLiteral<F32>(kernel_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&input, &kernel};

  HloBenchmarkOptions benchmark_options;
  benchmark_options.aot_options = is_aot ? GetAotCompilationOptions() : nullptr;

  CHECK_OK(RunHloBenchmark(state, hlo_module, args, {}, benchmark_options));
}

// Transposed version (i.e. gradient) of BM_Conv2DStrided. In terms of shapes,
// this operation can be thought of as reverse of regular strided convolution,
// that's why input and output shapes are swapped (so we can directly compare
// performance of this function with BM_Conv2DStrided).
// Currently, the performance is orders of magnitude worse than regular conv
// when they should be similar.
static void BM_Conv2DTransposedStrided(benchmark::State& state) {
  bool is_aot = static_cast<bool>(state.range(0));

  std::string hlo_module = R"(
    HloModule jit_jconvt

    ENTRY main.6 {
      Arg_0.1 = f32[16,129,20,20]{3,2,1,0} parameter(0)
      Arg_1.2 = f32[129,1,16,16]{3,2,1,0} parameter(1)
      ROOT conv.3 = f32[16,1,160,160]{3,2,1,0} convolution(Arg_0.1, Arg_1.2),
        window={size=16x16 pad=11_11x11_11 lhs_dilate=8x8},
        dim_labels=bf01_io01->bf01
    }
  )";

  std::minstd_rand0 engine;

  // NCHW layout
  auto input_shape = ShapeUtil::MakeShape(F32, {16, 129, 20, 20});
  // IOHW layout
  auto kernel_shape = ShapeUtil::MakeShape(F32, {129, 1, 16, 16});

  auto input =
      *LiteralUtil::CreateRandomLiteral<F32>(input_shape, &engine, 1.0f, 0.1f);
  auto kernel =
      *LiteralUtil::CreateRandomLiteral<F32>(kernel_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&input, &kernel};

  HloBenchmarkOptions benchmark_options;
  benchmark_options.aot_options = is_aot ? GetAotCompilationOptions() : nullptr;

  CHECK_OK(RunHloBenchmark(state, hlo_module, args, {}, benchmark_options));
}

// Regular (i.e. non-transposed) grouped and strided 2D convolution.
static void BM_GroupedConv2DStrided(benchmark::State& state) {
  int input_channels = state.range(0);
  int output_channels = state.range(1);
  int feature_group_count = state.range(2);
  bool is_aot = static_cast<bool>(state.range(3));

  // Derive filter channels from input channels and feature group count.
  int filter_channels = input_channels / feature_group_count;

  std::string hlo_module = R"(
    HloModule jit_jconvf

    ENTRY main.6 {
      Arg_0.1 = $input_shape parameter(0)
      Arg_1.2 = $kernel_shape parameter(1)
      ROOT conv.3 = convolution(Arg_0.1, Arg_1.2),
        window={size=16x16 stride=8x8 pad=4_4x4_4}, dim_labels=bf01_io01->bf01,
        feature_group_count=$feature_group_count
    }
  )";

  std::minstd_rand0 engine;

  // NCHW layout
  auto input_shape = ShapeUtil::MakeShape(F32, {2, input_channels, 80, 80});
  // IOHW layout
  auto kernel_shape =
      ShapeUtil::MakeShape(F32, {filter_channels, output_channels, 16, 16});

  auto input =
      *LiteralUtil::CreateRandomLiteral<F32>(input_shape, &engine, 1.0f, 0.1f);
  auto kernel =
      *LiteralUtil::CreateRandomLiteral<F32>(kernel_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&input, &kernel};

  HloBenchmarkOptions benchmark_options;
  benchmark_options.aot_options = is_aot ? GetAotCompilationOptions() : nullptr;

  CHECK_OK(RunHloBenchmark(
      state, hlo_module, args,
      {{"$input_shape", input_shape.ToString()},
       {"$kernel_shape", kernel_shape.ToString()},
       {"$feature_group_count", std::to_string(feature_group_count)}},
      benchmark_options));
}

// Transposed version (i.e. gradient) of BM_GroupedConv2DStrided. In terms of
// shapes, this operation can be thought of as reverse of regular strided
// convolution, that's why input and output shapes are swapped (so we can
// directly compare performance of this function with BM_GroupedConv2DStrided).
static void BM_GroupedConv2DTransposedStrided(benchmark::State& state) {
  int input_channels = state.range(0);
  int output_channels = state.range(1);
  int feature_group_count = state.range(2);
  bool is_aot = static_cast<bool>(state.range(3));

  // Derive filter channels from input channels and feature group count.
  int filter_channels = input_channels / feature_group_count;

  std::string hlo_module = R"(
    HloModule jit_jconvt

    ENTRY main.6 {
      Arg_0.1 = $input_shape parameter(0)
      Arg_1.2 = $kernel_shape parameter(1)
      ROOT conv.3 = convolution(Arg_0.1, Arg_1.2),
        window={size=16x16 pad=11_11x11_11 lhs_dilate=8x8},
        dim_labels=bf01_io01->bf01, feature_group_count=$feature_group_count
    }
  )";

  std::minstd_rand0 engine;

  // NCHW layout
  auto input_shape = ShapeUtil::MakeShape(F32, {2, input_channels, 10, 10});
  // IOHW layout
  auto kernel_shape =
      ShapeUtil::MakeShape(F32, {filter_channels, output_channels, 16, 16});

  auto input =
      *LiteralUtil::CreateRandomLiteral<F32>(input_shape, &engine, 1.0f, 0.1f);
  auto kernel =
      *LiteralUtil::CreateRandomLiteral<F32>(kernel_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&input, &kernel};

  HloBenchmarkOptions benchmark_options;
  benchmark_options.aot_options = is_aot ? GetAotCompilationOptions() : nullptr;

  CHECK_OK(RunHloBenchmark(
      state, hlo_module, args,
      {{"$input_shape", input_shape.ToString()},
       {"$kernel_shape", kernel_shape.ToString()},
       {"$feature_group_count", std::to_string(feature_group_count)}},
      benchmark_options));
}

// -------------------------------------------------------------------------- //
// Conv2D
// -------------------------------------------------------------------------- //

void GenerateArgsConv2DArgs(benchmark::internal::Benchmark* benchmark) {
  benchmark->MeasureProcessCPUTime();
  benchmark->ArgNames({
      "batch",
      "height",
      "width",
      "input_channels",
      "kernel_h",
      "kernel_w",
      "output_channels",
      "is_aot",
  });

  const std::vector<bool> is_aot_values = {false, true};

  std::vector<std::array<int64_t, 7>> args_values = {
      // --------------------------------------------------------------------------
      // //
      // Pixel CNN convolutions.
      // --------------------------------------------------------------------------
      // //
      // Shapes from XLA convolution tests
      {8, 5, 5, 1, 1, 1, 32},
      {8, 5, 5, 4, 1, 1, 32},
      {8, 128, 128, 4, 1, 1, 8},
      // Shapes from TF convolution benchmarks.
      {8, 32, 32, 128, 1, 1, 1024},
      {16, 32, 32, 128, 1, 1, 1024},
      {32, 32, 32, 128, 1, 1, 1024},
      // Shapes similar to Eigen spatial convolution benchmarks.
      {32, 64, 64, 32, 1, 1, 64},
      {32, 256, 256, 4, 1, 1, 16},
      {32, 64, 64, 4, 1, 1, 16},
      {32, 32, 32, 96, 1, 1, 96},
      // --------------------------------------------------------------------------
      // //
      // 3x3 Convolution: SpatialConvolution
      // --------------------------------------------------------------------------
      // //
      // Shapes from XLA convolution tests
      {8, 5, 5, 1, 3, 3, 32},
      {8, 5, 5, 4, 3, 3, 32},
      {8, 128, 128, 4, 3, 3, 8},
      // Shapes from TF convolution benchmarks
      {8, 32, 32, 128, 3, 3, 1024},
      {16, 32, 32, 128, 3, 3, 1024},
      {32, 32, 32, 128, 3, 3, 1024},
      // Shapes similar to Eigen spatial convolution benchmarks.
      {32, 64, 64, 32, 3, 3, 64},
      {32, 256, 256, 4, 3, 3, 16},
      {32, 64, 64, 4, 3, 3, 16},
      {32, 32, 32, 96, 3, 3, 96},
  };

  for (const auto& arg_value : args_values) {
    for (bool is_aot : is_aot_values) {
      std::vector<int64_t> arg_values_with_is_aot(arg_value.begin(),
                                                  arg_value.end());
      arg_values_with_is_aot.push_back(is_aot);
      benchmark->Args(arg_values_with_is_aot);
    }
  }
}

BENCHMARK(BM_Conv2D<F32>)->Apply(GenerateArgsConv2DArgs);

// -------------------------------------------------------------------------- //
// Grouped convolution
// -------------------------------------------------------------------------- //

void GenerateArgsGroupedConv2DArgs(benchmark::internal::Benchmark* benchmark) {
  benchmark->MeasureProcessCPUTime();
  benchmark->ArgNames({
      "batch",
      "height",
      "width",
      "input_channels",
      "kernel_h",
      "kernel_w",
      "output_channels",
      "feature_group_count",
      "is_aot",
  });

  const std::vector<bool> is_aot_values = {false, true};
  const std::vector<std::array<int64_t, 8>> args_values = {
      {1, 45, 45, 1024, 5, 5, 1024, 1024}};

  for (const auto& arg_value : args_values) {
    for (const auto& is_aot : is_aot_values) {
      std::vector<int64_t> arg_values_with_is_aot(arg_value.begin(),
                                                  arg_value.end());
      arg_values_with_is_aot.push_back(is_aot);
      benchmark->Args(arg_values_with_is_aot);
    }
  }
}

BENCHMARK(BM_GroupedConv2D)->Apply(GenerateArgsGroupedConv2DArgs);

// --------------------------------------------------------------------------
// // 1D and 2D strided convolutions
// --------------------------------------------------------------------------
// //

void GenerateArgsConv1DStridedArgs(benchmark::internal::Benchmark* benchmark) {
  benchmark->MeasureProcessCPUTime();
  benchmark->ArgNames({"input_channels", "output_channels", "is_aot"});
  const std::vector<bool> is_aot_values = {false, true};
  std::vector<std::array<int64_t, 2>> args_values = {
      {1, 129},
      {3, 129},
  };

  for (const auto& arg_value : args_values) {
    for (const auto& is_aot : is_aot_values) {
      std::vector<int64_t> arg_values_with_is_aot(arg_value.begin(),
                                                  arg_value.end());
      arg_values_with_is_aot.push_back(is_aot);
      benchmark->Args(arg_values_with_is_aot);
    }
  }
}

BENCHMARK(BM_Conv1DStrided)->Apply(GenerateArgsConv1DStridedArgs);

void GenerateArgsConv1DTransposedStridedArgs(
    benchmark::internal::Benchmark* benchmark) {
  benchmark->MeasureProcessCPUTime();
  benchmark->ArgNames({"input_channels", "output_channels", "is_aot"});
  const std::vector<bool> is_aot_values = {false, true};
  std::vector<std::array<int64_t, 2>> args_values = {
      {129, 1},
      {129, 3},
  };

  for (const auto& arg_value : args_values) {
    for (const auto& is_aot : is_aot_values) {
      std::vector<int64_t> arg_values_with_is_aot(arg_value.begin(),
                                                  arg_value.end());
      arg_values_with_is_aot.push_back(is_aot);
      benchmark->Args(arg_values_with_is_aot);
    }
  }
}

// BM_Conv1DTransposedStrided and BM_Conv1DTransposedStridedNonDefaultLayout use
// the same args
BENCHMARK(BM_Conv1DTransposedStrided)
    ->Apply(GenerateArgsConv1DTransposedStridedArgs);
BENCHMARK(BM_Conv1DTransposedStridedNonDefaultLayout)
    ->Apply(GenerateArgsConv1DTransposedStridedArgs);

void GenerateArgsConv2DStridedArgs(benchmark::internal::Benchmark* benchmark) {
  benchmark->MeasureProcessCPUTime();
  benchmark->ArgNames({"is_aot"});
  const std::vector<bool> is_aot_values = {false, true};

  for (const auto& is_aot : is_aot_values) {
    benchmark->Args({is_aot});
  }
}

BENCHMARK(BM_Conv2DStrided)->Apply(GenerateArgsConv2DStridedArgs);
BENCHMARK(BM_Conv2DTransposedStrided)->Apply(GenerateArgsConv2DStridedArgs);

// --------------------------------------------------------------------------
// // Grouped strided convolutions
// --------------------------------------------------------------------------
// //

void GenerateArgsGroupedConv2DStrided(
    benchmark::internal::Benchmark* benchmark) {
  benchmark->MeasureProcessCPUTime();
  benchmark->ArgNames(
      {"input_channels", "output_channels", "feature_group_count", "is_aot"});
  const std::vector<bool> is_aot_values = {false, true};
  std::vector<std::array<int64_t, 3>> args_values = {{128, 128, 128},
                                                     {128, 128, 16}};

  for (const auto& arg_value : args_values) {
    for (const auto& is_aot : is_aot_values) {
      std::vector<int64_t> arg_values_with_is_aot(arg_value.begin(),
                                                  arg_value.end());
      arg_values_with_is_aot.push_back(is_aot);
      benchmark->Args(arg_values_with_is_aot);
    }
  }
}

BENCHMARK(BM_GroupedConv2DStrided)->Apply(GenerateArgsGroupedConv2DStrided);
BENCHMARK(BM_GroupedConv2DTransposedStrided)
    ->Apply(GenerateArgsGroupedConv2DStrided);

}  // namespace
}  // namespace xla::cpu
