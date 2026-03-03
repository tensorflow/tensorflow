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
#include "absl/strings/str_join.h"
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

static const int64_t kValidPadding = 0;
static const int64_t kSamePadding = 1;

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

static void ConvBenchmark(benchmark::State& state,
                          const HloBenchmarkOptions& options,
                          int num_spatial_dims, int64_t batch,
                          absl::Span<const int64_t> spatial_dims,
                          int64_t input_channels,
                          absl::Span<const int64_t> kernel_dims, int64_t stride,
                          int64_t output_channels, int64_t feature_group_count,
                          int padding_mode, int type_config_idx) {
  const auto& types = GetTypeConfigs()[type_config_idx];
  std::vector<int64_t> padding(num_spatial_dims);
  std::vector<int64_t> output_spatial_dims(num_spatial_dims);

  for (int i = 0; i < num_spatial_dims; ++i) {
    if (padding_mode == kSamePadding) {
      CHECK(IsOdd(kernel_dims[i]));
      padding[i] = (kernel_dims[i] - 1) / 2;
      output_spatial_dims[i] = (spatial_dims[i] + stride - 1) / stride;
    } else {
      padding[i] = 0;
      output_spatial_dims[i] = (spatial_dims[i] - kernel_dims[i]) / stride + 1;
    }
  }

  const int64_t filter_channels = input_channels / feature_group_count;

  std::string hlo_module = R"(
    HloModule TestModule

    ENTRY TestComputation {
      %p0 = $input_shape parameter(0)
      %p1 = $kernel_shape parameter(1)
      ROOT conv = $output_shape convolution(p0, p1),
        window={size=$window_size stride=$window_stride pad=$padding},
          dim_labels=$dim_labels, feature_group_count=$feature_group_count
    }
  )";

  std::minstd_rand0 engine;

  std::vector<int64_t> input_shape_dims = {batch};
  for (int64_t d : spatial_dims) {
    input_shape_dims.push_back(d);
  }
  input_shape_dims.push_back(input_channels);
  auto input_shape = ShapeUtil::MakeShape(types.input, input_shape_dims);

  std::vector<int64_t> kernel_shape_dims;
  for (int64_t d : kernel_dims) {
    kernel_shape_dims.push_back(d);
  }
  kernel_shape_dims.push_back(filter_channels);
  kernel_shape_dims.push_back(output_channels);
  auto kernel_shape = ShapeUtil::MakeShape(types.kernel, kernel_shape_dims);

  std::vector<int64_t> output_shape_dims = {batch};
  for (int64_t d : output_spatial_dims) {
    output_shape_dims.push_back(d);
  }
  output_shape_dims.push_back(output_channels);
  auto output_shape = ShapeUtil::MakeShape(types.output, output_shape_dims);

  auto input = GetRandomLiteral(types.input, input_shape, engine);
  auto kernel = GetRandomLiteral(types.kernel, kernel_shape, engine);
  std::vector<const Literal*> args = {&input, &kernel};

  std::string window_size = absl::StrJoin(kernel_dims, "x");
  std::string window_stride =
      absl::StrJoin(std::vector<int64_t>(num_spatial_dims, stride), "x");

  std::vector<std::string> padding_strs;
  for (int64_t p : padding) {
    padding_strs.push_back(absl::StrCat(p, "_", p));
  }
  std::string padding_str = absl::StrJoin(padding_strs, "x");

  std::string dim_labels;
  if (num_spatial_dims == 1) {
    dim_labels = "b0f_0io->b0f";
  } else {
    dim_labels = "b01f_01io->b01f";
  }

  CHECK_OK(RunHloBenchmark(
      state, hlo_module, args,
      {{"$input_shape", input_shape.ToString()},
       {"$kernel_shape", kernel_shape.ToString()},
       {"$output_shape", output_shape.ToString()},
       {"$window_size", window_size},
       {"$window_stride", window_stride},
       {"$padding", padding_str},
       {"$dim_labels", dim_labels},
       {"$feature_group_count", std::to_string(feature_group_count)}},
      options));
}

static void BM_Conv2D(benchmark::State& state,
                      const HloBenchmarkOptions& options, int type_config_idx) {
  ConvBenchmark(state, options, 2, /*batch=*/state.range(0),
                /*spatial_dims=*/{state.range(1), state.range(2)},
                /*input_channels=*/state.range(3),
                /*kernel_dims=*/{state.range(4), state.range(5)},
                /*stride=*/state.range(6),
                /*output_channels=*/state.range(7),
                /*feature_group_count=*/1,
                /*padding_mode=*/state.range(8),
                /*type_config_idx=*/type_config_idx);
}

static void BM_GroupedConv2D(benchmark::State& state,
                             const HloBenchmarkOptions& options,
                             int type_config_idx) {
  ConvBenchmark(state, options, 2, /*batch=*/state.range(0),
                /*spatial_dims=*/{state.range(1), state.range(2)},
                /*input_channels=*/state.range(3),
                /*kernel_dims=*/{state.range(4), state.range(5)},
                /*stride=*/state.range(6),
                /*output_channels=*/state.range(7),
                /*feature_group_count=*/state.range(8),
                /*padding_mode=*/kSamePadding,
                /*type_config_idx=*/type_config_idx);
}

static void BM_Conv1D(benchmark::State& state,
                      const HloBenchmarkOptions& options, int type_config_idx) {
  ConvBenchmark(state, options, 1, /*batch=*/state.range(0),
                /*spatial_dims=*/{state.range(1)},
                /*input_channels=*/state.range(2),
                /*kernel_dims=*/{state.range(3)},
                /*stride=*/state.range(4),
                /*output_channels=*/state.range(5),
                /*feature_group_count=*/1,
                /*padding_mode=*/state.range(6),
                /*type_config_idx=*/type_config_idx);
}

static void BM_GroupedConv1D(benchmark::State& state,
                             const HloBenchmarkOptions& options,
                             int type_config_idx) {
  ConvBenchmark(state, options, 1, /*batch=*/state.range(0),
                /*spatial_dims=*/{state.range(1)},
                /*input_channels=*/state.range(2),
                /*kernel_dims=*/{state.range(3)},
                /*stride=*/state.range(4),
                /*output_channels=*/state.range(5),
                /*feature_group_count=*/state.range(6),
                /*padding_mode=*/kSamePadding,
                /*type_config_idx=*/type_config_idx);
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
    b->Args(shape_args);
  };

  // --------------------------------------------------------------------------
  // // Pixel CNN convolutions.
  // --------------------------------------------------------------------------
  // // Shapes from XLA convolution tests

  // This test case hits b/473570788.
  if (type_config == 0) {
    add_args({8, 5, 5, 1, 1, 1, 1, 32, kSamePadding});
  }
  add_args({8, 5, 5, 4, 1, 1, 1, 32, kSamePadding});
  add_args({8, 128, 128, 4, 1, 1, 1, 8, kSamePadding});
  // Shapes from TF convolution benchmarks.
  add_args({8, 32, 32, 128, 1, 1, 1, 1024, kSamePadding});
  add_args({16, 32, 32, 128, 1, 1, 1, 1024, kSamePadding});
  add_args({32, 32, 32, 128, 1, 1, 1, 1024, kSamePadding});
  // Shapes similar to Eigen spatial convolution benchmarks.
  add_args({32, 64, 64, 32, 1, 1, 1, 64, kSamePadding});
  add_args({32, 256, 256, 4, 1, 1, 1, 16, kSamePadding});
  add_args({32, 64, 64, 4, 1, 1, 1, 16, kSamePadding});
  add_args({32, 32, 32, 96, 1, 1, 1, 96, kSamePadding});
  // --------------------------------------------------------------------------
  // // 3x3 Convolution: SpatialConvolution
  // --------------------------------------------------------------------------
  // // Shapes from XLA convolution tests
  add_args({8, 5, 5, 1, 3, 3, 1, 32, kSamePadding});
  add_args({8, 5, 5, 4, 3, 3, 1, 32, kSamePadding});
  add_args({8, 128, 128, 4, 3, 3, 1, 8, kSamePadding});
  // Shapes from TF convolution benchmarks
  add_args({8, 32, 32, 128, 3, 3, 1, 1024, kSamePadding});
  add_args({16, 32, 32, 128, 3, 3, 1, 1024, kSamePadding});
  add_args({32, 32, 32, 128, 3, 3, 1, 1024, kSamePadding});
  // Shapes similar to Eigen spatial convolution benchmarks.
  // Same padding.
  add_args({32, 64, 64, 32, 3, 3, 1, 64, kSamePadding});
  add_args({32, 256, 256, 4, 3, 3, 1, 16, kSamePadding});
  add_args({32, 64, 64, 4, 3, 3, 1, 16, kSamePadding});
  add_args({32, 32, 32, 96, 3, 3, 1, 96, kSamePadding});
  // Valid padding.
  add_args({32, 64, 64, 32, 3, 3, 1, 64, kValidPadding});
  add_args({32, 256, 256, 4, 3, 3, 1, 16, kValidPadding});
  add_args({32, 64, 64, 4, 3, 3, 1, 16, kValidPadding});
  add_args({32, 32, 32, 96, 3, 3, 1, 96, kValidPadding});
  // --------------------------------------------------------------------------
  // // ResNet-50 shapes with stride=2
  // --------------------------------------------------------------------------
  add_args({32, 56, 56, 256, 1, 1, 2, 128, kValidPadding});
  add_args({32, 56, 56, 64, 3, 3, 2, 64, kSamePadding});
}

void AddConv1DArgs(::benchmark::internal::Benchmark* b) {
  auto add_args = [&](const std::vector<int64_t>& shape_args) {
    b->Args(shape_args);
  };
  add_args({8, 128, 4, 1, 1, 32, kSamePadding});
  add_args({8, 128, 4, 3, 1, 32, kSamePadding});
  add_args({32, 256, 4, 3, 1, 16, kSamePadding});
  add_args({32, 256, 4, 3, 1, 16, kValidPadding});
}

void AddGroupedConv1DArgs(::benchmark::internal::Benchmark* b) {
  auto add_args = [&](const std::vector<int64_t>& shape_args) {
    b->Args(shape_args);
  };
  add_args({16, 112, 32, 3, 1, 32, 32});
  add_args({16, 112, 64, 3, 2, 64, 64});
}

void AddGroupedConv2DArgs(::benchmark::internal::Benchmark* b) {
  auto add_args = [&](const std::vector<int64_t>& shape_args) {
    b->Args(shape_args);
  };
  add_args({1, 45, 45, 1024, 5, 5, 1, 1024, 1024});

  // --------------------------------------------------------------------------
  // MobilenetV1 depthwise convolutions.
  // --------------------------------------------------------------------------
  add_args({16, 112, 112, 32, 3, 3, 1, 32, 32});
  add_args({16, 112, 112, 64, 3, 3, 2, 64, 64});
  add_args({16, 56, 56, 128, 3, 3, 1, 128, 128});
  add_args({16, 56, 56, 128, 3, 3, 2, 128, 128});
  add_args({16, 28, 28, 256, 3, 3, 1, 256, 256});
  add_args({16, 28, 28, 256, 3, 3, 2, 256, 256});
  add_args({16, 14, 14, 512, 3, 3, 1, 512, 512});
  add_args({16, 14, 14, 512, 3, 3, 2, 512, 512});
  add_args({16, 7, 7, 1024, 3, 3, 1, 1024, 1024});
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
    auto* b = benchmark::RegisterBenchmark(
        name.c_str(), [i](benchmark::State& state) {
          BM_Conv2D(state, HloBenchmarkOptions(), i);
        });
    b->MeasureProcessCPUTime();
    b->ArgNames({"b", "h", "w", "i", "kh", "kw", "s", "o", "pad"});
    AddConv2DArgs(b, i);
  }

  for (int i = 0; i < configs.size(); ++i) {
    const auto& config = configs[i];
    std::string name = absl::StrFormat(
        "BM_GroupedConv2D_%s_%s_%s",
        primitive_util::LowercasePrimitiveTypeName(config.input),
        primitive_util::LowercasePrimitiveTypeName(config.kernel),
        primitive_util::LowercasePrimitiveTypeName(config.output));
    auto* b = benchmark::RegisterBenchmark(
        name.c_str(), [i](benchmark::State& state) {
          BM_GroupedConv2D(state, HloBenchmarkOptions(), i);
        });
    b->MeasureProcessCPUTime();
    b->ArgNames({"b", "h", "w", "i", "kh", "kw", "s", "o", "g"});
    AddGroupedConv2DArgs(b);
  }

  for (int i = 0; i < configs.size(); ++i) {
    const auto& config = configs[i];
    std::string name = absl::StrFormat(
        "BM_Conv1D_%s_%s_%s",
        primitive_util::LowercasePrimitiveTypeName(config.input),
        primitive_util::LowercasePrimitiveTypeName(config.kernel),
        primitive_util::LowercasePrimitiveTypeName(config.output));
    auto* b = benchmark::RegisterBenchmark(
        name.c_str(), [i](benchmark::State& state) {
          BM_Conv1D(state, HloBenchmarkOptions(), i);
        });
    b->MeasureProcessCPUTime();
    b->ArgNames({"b", "w", "i", "kw", "s", "o", "pad"});
    AddConv1DArgs(b);
  }

  for (int i = 0; i < configs.size(); ++i) {
    const auto& config = configs[i];
    std::string name = absl::StrFormat(
        "BM_GroupedConv1D_%s_%s_%s",
        primitive_util::LowercasePrimitiveTypeName(config.input),
        primitive_util::LowercasePrimitiveTypeName(config.kernel),
        primitive_util::LowercasePrimitiveTypeName(config.output));
    auto* b = benchmark::RegisterBenchmark(
        name.c_str(), [i](benchmark::State& state) {
          BM_GroupedConv1D(state, HloBenchmarkOptions(), i);
        });
    b->MeasureProcessCPUTime();
    b->ArgNames({"b", "w", "i", "kw", "s", "o", "g"});
    AddGroupedConv1DArgs(b);
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
