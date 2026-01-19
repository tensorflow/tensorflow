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
#include <string>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/backends/cpu/benchmarks/multi_benchmark_config.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

bool IsOdd(int n) { return n % 2 == 1; }

struct TypeConfig {
  PrimitiveType input;
  PrimitiveType kernel;
  PrimitiveType output;
};

static const std::vector<TypeConfig>& GetTypeConfigs() {
  static const absl::NoDestructor<std::vector<TypeConfig>> v(
      {{F32, F32, F32}, {BF16, BF16, F32}, {S8, S8, S32}});
  return *v;
}

Literal GetRandomLiteral(PrimitiveType type, const Shape& shape,
                         std::minstd_rand0& engine) {
  if (type == F32) {
    return *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);
  }
  if (type == BF16) {
    return *LiteralUtil::CreateRandomLiteral<BF16>(shape, &engine, 1.0f, 0.1f);
  }
  if (type == S8) {
    return *LiteralUtil::CreateRandomLiteral<S8>(shape, &engine, -10, 10);
  }
  LOG(FATAL) << "Unsupported type";
}

static void BM_Conv2D(benchmark::State& state,
                      const HloBenchmarkOptions& options) {
  int batch = state.range(0);
  int height = state.range(1);
  int width = state.range(2);
  int input_channels = state.range(3);
  int kernel_h = state.range(4);
  int kernel_w = state.range(5);
  int output_channels = state.range(6);
  int type_config_idx = state.range(7);

  const auto& types = GetTypeConfigs()[type_config_idx];

  // Padding values for 'SAME' padding. Only odd kernel sizes are supported.
  CHECK(IsOdd(kernel_h) && IsOdd(kernel_w));
  int padding_h = (kernel_h - 1) / 2;
  int padding_w = (kernel_w - 1) / 2;

  std::string hlo_module = R"(
    HloModule TestModule

    ENTRY TestComputation {
      %p0 = $input_shape parameter(0)
      %p1 = $kernel_shape parameter(1)
      ROOT conv = $output_shape convolution(p0, p1), window={size=$window_size pad=$padding},
        dim_labels=b01f_01io->b01f
    }
  )";

  std::minstd_rand0 engine;

  // Input format is NHWC.
  auto input_shape =
      ShapeUtil::MakeShape(types.input, {batch, height, width, input_channels});
  // Filter format is HWIO.
  auto kernel_shape = ShapeUtil::MakeShape(
      types.kernel, {kernel_h, kernel_w, input_channels, output_channels});
  auto output_shape = ShapeUtil::MakeShape(
      types.output, {batch, height, width, output_channels});

  auto input = GetRandomLiteral(types.input, input_shape, engine);
  auto kernel = GetRandomLiteral(types.kernel, kernel_shape, engine);
  std::vector<const Literal*> args = {&input, &kernel};

  CHECK_OK(
      RunHloBenchmark(state, hlo_module, args,
                      {{"$input_shape", input_shape.ToString()},
                       {"$kernel_shape", kernel_shape.ToString()},
                       {"$output_shape", output_shape.ToString()},
                       {"$window_size", absl::StrCat(kernel_h, "x", kernel_w)},
                       {"$padding", absl::StrCat(padding_h, "_", padding_h, "x",
                                                 padding_w, "_", padding_w)}},
                      options));
}

static void BM_GroupedConv2D(benchmark::State& state,
                             const HloBenchmarkOptions& options) {
  int batch = state.range(0);
  int height = state.range(1);
  int width = state.range(2);
  int input_channels = state.range(3);
  int kernel_h = state.range(4);
  int kernel_w = state.range(5);
  int output_channels = state.range(6);
  int feature_group_count = state.range(7);
  int type_config_idx = state.range(8);

  const auto& types = GetTypeConfigs()[type_config_idx];

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
      ROOT conv = $output_shape convolution(p0, p1), window={size=$window_size pad=$padding},
        dim_labels=b01f_01io->b01f, feature_group_count=$feature_group_count
    }
  )";

  std::minstd_rand0 engine;

  // Input format is NHWC.
  auto input_shape =
      ShapeUtil::MakeShape(types.input, {batch, height, width, input_channels});
  // Filter format is HWIO.
  auto kernel_shape = ShapeUtil::MakeShape(
      types.kernel, {kernel_h, kernel_w, filter_channels, output_channels});
  auto output_shape = ShapeUtil::MakeShape(
      types.output, {batch, height, width, output_channels});

  auto input = GetRandomLiteral(types.input, input_shape, engine);
  auto kernel = GetRandomLiteral(types.kernel, kernel_shape, engine);

  std::vector<const Literal*> args = {&input, &kernel};

  CHECK_OK(RunHloBenchmark(
      state, hlo_module, args,
      {{"$input_shape", input_shape.ToString()},
       {"$kernel_shape", kernel_shape.ToString()},
       {"$output_shape", output_shape.ToString()},
       {"$window_size", absl::StrCat(kernel_h, "x", kernel_w)},
       {"$padding", absl::StrCat(padding_h, "_", padding_h, "x", padding_w, "_",
                                 padding_w)},
       {"$feature_group_count", absl::StrCat(feature_group_count)}},
      options));
}

// Regular strided 1D convolution. Shapes come from an actual use case.
static void BM_Conv1DStrided(benchmark::State& state,
                             HloBenchmarkOptions options) {
  int input_channels = state.range(0);
  int output_channels = state.range(1);

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

  CHECK_OK(RunHloBenchmark(state, hlo_module, args,
                           {{"$input_shape", input_shape.ToString()},
                            {"$kernel_shape", kernel_shape.ToString()},
                            {"$output_shape", output_shape.ToString()}},
                           options));
}

// Transposed version (i.e. gradient) of BM_Conv1DStrided. In terms of shapes,
// this operation can be thought of as reverse of regular strided convolution,
// that's why input and output shapes are swapped (so we can directly compare
// performance of this function with BM_Conv1DStrided).
// Currently, the performance is few times worse than regular conv when they
// should be similar.
static void BM_Conv1DTransposedStrided(benchmark::State& state,
                                       HloBenchmarkOptions options) {
  int input_channels = state.range(0);
  int output_channels = state.range(1);

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

  CHECK_OK(RunHloBenchmark(state, hlo_module, args,
                           {{"$input_shape", input_shape.ToString()},
                            {"$kernel_shape", kernel_shape.ToString()},
                            {"$output_shape", output_shape.ToString()}},
                           options));
}

// The same shapes as BM_Conv1DTransposedStrided, but with a different layout.
static void BM_Conv1DTransposedStridedNonDefaultLayout(
    benchmark::State& state, HloBenchmarkOptions options) {
  int input_channels = state.range(0);
  int output_channels = state.range(1);
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

  CHECK_OK(RunHloBenchmark(state, hlo_module, args,
                           {{"$input_shape", input_shape.ToString()},
                            {"$kernel_shape", kernel_shape.ToString()},
                            {"$output_shape", output_shape.ToString()}},
                           options));
}

// Regular strided 2D convolution. Buffer sizes and convolution parameters are
// based on an actual 1D use case, but adapted to a 2D convolution.
static void BM_Conv2DStrided(benchmark::State& state,
                             HloBenchmarkOptions options) {
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

  CHECK_OK(RunHloBenchmark(state, hlo_module, args, {}, options));
}

// Transposed version (i.e. gradient) of BM_Conv2DStrided. In terms of shapes,
// this operation can be thought of as reverse of regular strided convolution,
// that's why input and output shapes are swapped (so we can directly compare
// performance of this function with BM_Conv2DStrided).
// Currently, the performance is orders of magnitude worse than regular conv
// when they should be similar.
static void BM_Conv2DTransposedStrided(benchmark::State& state,
                                       HloBenchmarkOptions options) {
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

  CHECK_OK(RunHloBenchmark(state, hlo_module, args, {}, options));
}

// Regular (i.e. non-transposed) grouped and strided 2D convolution.
static void BM_GroupedConv2DStrided(benchmark::State& state,
                                    HloBenchmarkOptions options) {
  int input_channels = state.range(0);
  int output_channels = state.range(1);
  int feature_group_count = state.range(2);

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

  CHECK_OK(RunHloBenchmark(
      state, hlo_module, args,
      {{"$input_shape", input_shape.ToString()},
       {"$kernel_shape", kernel_shape.ToString()},
       {"$feature_group_count", std::to_string(feature_group_count)}},
      options));
}

// Transposed version (i.e. gradient) of BM_GroupedConv2DStrided. In terms of
// shapes, this operation can be thought of as reverse of regular strided
// convolution, that's why input and output shapes are swapped (so we can
// directly compare performance of this function with BM_GroupedConv2DStrided).
static void BM_GroupedConv2DTransposedStrided(benchmark::State& state,
                                              HloBenchmarkOptions options) {
  int input_channels = state.range(0);
  int output_channels = state.range(1);
  int feature_group_count = state.range(2);

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

  CHECK_OK(RunHloBenchmark(
      state, hlo_module, args,
      {{"$input_shape", input_shape.ToString()},
       {"$kernel_shape", kernel_shape.ToString()},
       {"$feature_group_count", std::to_string(feature_group_count)}},
      options));
}

void AddConv2DArgs(::benchmark::internal::Benchmark* b, int type_config) {
  auto add_args = [&](const std::vector<int64_t>& shape_args) {
    std::vector<int64_t> args = shape_args;
    args.push_back(type_config);
    b->Args(args);
  };

  // --------------------------------------------------------------------------
  // // Pixel CNN convolutions.
  // --------------------------------------------------------------------------
  // // Shapes from XLA convolution tests

  // This test case hits b/473570788.
  if (type_config != 2) {
    add_args({8, 5, 5, 1, 1, 1, 32});
  }
  add_args({8, 5, 5, 4, 1, 1, 32});
  add_args({8, 128, 128, 4, 1, 1, 8});
  // Shapes from TF convolution benchmarks.
  add_args({8, 32, 32, 128, 1, 1, 1024});
  add_args({16, 32, 32, 128, 1, 1, 1024});
  add_args({32, 32, 32, 128, 1, 1, 1024});
  // Shapes similar to Eigen spatial convolution benchmarks.
  add_args({32, 64, 64, 32, 1, 1, 64});
  add_args({32, 256, 256, 4, 1, 1, 16});
  add_args({32, 64, 64, 4, 1, 1, 16});
  add_args({32, 32, 32, 96, 1, 1, 96});
  // --------------------------------------------------------------------------
  // // 3x3 Convolution: SpatialConvolution
  // --------------------------------------------------------------------------
  // // Shapes from XLA convolution tests
  add_args({8, 5, 5, 1, 3, 3, 32});
  add_args({8, 5, 5, 4, 3, 3, 32});
  add_args({8, 128, 128, 4, 3, 3, 8});
  // Shapes from TF convolution benchmarks
  add_args({8, 32, 32, 128, 3, 3, 1024});
  add_args({16, 32, 32, 128, 3, 3, 1024});
  add_args({32, 32, 32, 128, 3, 3, 1024});
  // Shapes similar to Eigen spatial convolution benchmarks.
  add_args({32, 64, 64, 32, 3, 3, 64});
  add_args({32, 256, 256, 4, 3, 3, 16});
  add_args({32, 64, 64, 4, 3, 3, 16});
  add_args({32, 32, 32, 96, 3, 3, 96});
}

void AddGroupedConv2DArgs(::benchmark::internal::Benchmark* b,
                          int type_config) {
  auto add_args = [&](const std::vector<int64_t>& shape_args) {
    std::vector<int64_t> args = shape_args;
    args.push_back(type_config);
    b->Args(args);
  };
  add_args({1, 45, 45, 1024, 5, 5, 1024, 1024});
}

void RegisterBenchmarks() {
  const auto& configs = GetTypeConfigs();
  for (int i = 0; i < configs.size(); ++i) {
    const auto& config = configs[i];
    std::string name = absl::StrFormat(
        "BM_Conv2D_%s_%s_%s",
        primitive_util::LowercasePrimitiveTypeName(config.input),
        primitive_util::LowercasePrimitiveTypeName(config.kernel),
        primitive_util::LowercasePrimitiveTypeName(config.output));
    auto* b =
        benchmark::RegisterBenchmark(name.c_str(), [](benchmark::State& state) {
          BM_Conv2D(state, HloBenchmarkOptions());
        });
    b->MeasureProcessCPUTime();
    AddConv2DArgs(b, i);
  }

  for (int i = 0; i < configs.size(); ++i) {
    const auto& config = configs[i];
    std::string name = absl::StrFormat(
        "BM_GroupedConv2D_%s_%s_%s",
        primitive_util::LowercasePrimitiveTypeName(config.input),
        primitive_util::LowercasePrimitiveTypeName(config.kernel),
        primitive_util::LowercasePrimitiveTypeName(config.output));
    auto* b =
        benchmark::RegisterBenchmark(name.c_str(), [](benchmark::State& state) {
          BM_GroupedConv2D(state, HloBenchmarkOptions());
        });
    b->MeasureProcessCPUTime();
    AddGroupedConv2DArgs(b, i);
  }
}

static int registration = [] {
  RegisterBenchmarks();
  return 0;
}();

// -------------------------------------------------------------------------- //
// 1D and 2D strided convolutions
// -------------------------------------------------------------------------- //

XLA_CPU_BENCHMARK(BM_Conv1DStrided)
    ->MeasureProcessCPUTime()
    ->Args({1, 129})
    ->Args({3, 129});
XLA_CPU_BENCHMARK(BM_Conv1DTransposedStrided)
    ->MeasureProcessCPUTime()
    ->MeasureProcessCPUTime()
    ->Args({129, 1})
    ->Args({129, 3});
XLA_CPU_BENCHMARK(BM_Conv1DTransposedStridedNonDefaultLayout)
    ->MeasureProcessCPUTime()
    ->Args({129, 1})
    ->Args({129, 3});

XLA_CPU_BENCHMARK(BM_Conv2DStrided)->MeasureProcessCPUTime();
XLA_CPU_BENCHMARK(BM_Conv2DTransposedStrided)->MeasureProcessCPUTime();

// -------------------------------------------------------------------------- //
// Grouped strided convolutions
// -------------------------------------------------------------------------- //

XLA_CPU_BENCHMARK(BM_GroupedConv2DStrided)
    ->MeasureProcessCPUTime()
    ->Args({128, 128, 128})
    ->Args({128, 128, 16});
XLA_CPU_BENCHMARK(BM_GroupedConv2DTransposedStrided)
    ->MeasureProcessCPUTime()
    ->Args({128, 128, 128})
    ->Args({128, 128, 16});

}  // namespace
}  // namespace xla::cpu
