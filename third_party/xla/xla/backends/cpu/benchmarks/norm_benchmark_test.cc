/* Copyright 2026 The OpenXLA Authors.

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
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "benchmark/benchmark.h"
#include "xla/backends/cpu/benchmarks/aot_benchmark_helper.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/stacktrace_handler.h"

ABSL_FLAG(std::string, shapes,
          "{(f32[1024,1024], s32[1]), (f32[1024,4096], s32[1]), "
          "(f32[8,1024,1024], s32[2])}",
          "List of shapes encoding RMS norm (ab)using the shape parser. "
          "The first shape is the input shape, the second shape's dimensions "
          "are the dimensions to reduce over.");

ABSL_FLAG(int32_t, num_executions, 1,
          "Number of times to execute the HLO within a single benchmark "
          "iteration.");

ABSL_FLAG(bool, aot_compiled_execution, false,
          "If true, when running the benchmark, the HLO will be compiled AOT.");

ABSL_FLAG(std::string, xla_flags, "", "Flags to append to XLA_FLAGS");

namespace xla::cpu {

namespace {

void Set_XLA_FLAGS() {
  const char* env_xla_flags = std::getenv("XLA_FLAGS");
  std::string xla_flags = absl::StrCat(env_xla_flags ? env_xla_flags : "",
                                       absl::GetFlag(FLAGS_xla_flags));
  tsl::setenv("XLA_FLAGS", xla_flags.data(), /*overwrite=*/1);
}

struct NormShape {
  Shape input_shape;
  std::vector<int64_t> reduction_dims;

  PrimitiveType GetDType() const { return input_shape.element_type(); }

  Shape GetReductionShape() const {
    Shape reduction_shape = input_shape;
    reduction_shape.DeleteDimensions(reduction_dims);
    return reduction_shape;
  }

  std::vector<int64_t> GetKeptDims() const {
    std::vector<int64_t> kept_dims;
    for (int64_t i = 0; i < input_shape.dimensions().size(); ++i) {
      if (!absl::c_linear_search(reduction_dims, i)) {
        kept_dims.push_back(i);
      }
    }
    return kept_dims;
  }
};

NormShape ParseShape(const Shape& s) {
  NormShape result;
  CHECK(s.IsTuple());
  CHECK_EQ(s.tuple_shapes().size(), 2);

  result.input_shape = s.tuple_shapes(0);

  const Shape& dims_shape = s.tuple_shapes(1);
  absl::Span<const int64_t> dims = dims_shape.dimensions();
  result.reduction_dims.assign(dims.begin(), dims.end());

  return result;
}

Literal GetRandomLiteral(const Shape& shape) {
  double mean = 1.0f;
  double stddev = 0.1f;
  std::minstd_rand0 engine;
  PrimitiveType dtype = shape.element_type();
  switch (dtype) {
    case F32:
      return *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, mean,
                                                    stddev);
    case BF16:
      return *LiteralUtil::CreateRandomLiteral<BF16>(shape, &engine, mean,
                                                     stddev);
    default:
      LOG(FATAL) << "Add dtype to the if-else block before use: " << dtype;
  }
}

void BM_RmsNorm(benchmark::State& state, const NormShape& shape) {
  const std::string input_shape_str = shape.input_shape.ToString();
  const std::string reduction_dims_str =
      absl::StrJoin(shape.reduction_dims, ",");
  const std::string dtype_str =
      primitive_util::LowercasePrimitiveTypeName(shape.GetDType());
  const std::string reduction_shape_str = shape.GetReductionShape().ToString();

  Shape input_shape_f32 = ShapeUtil::ChangeElementType(shape.input_shape, F32);
  const std::string input_shape_f32_str = input_shape_f32.ToString();

  Shape reduction_shape_f32 =
      ShapeUtil::ChangeElementType(shape.GetReductionShape(), F32);
  const std::string reduction_shape_f32_str = reduction_shape_f32.ToString();

  int64_t reduction_size = 1;
  for (int64_t d : shape.reduction_dims) {
    reduction_size *= shape.input_shape.dimensions(d);
  }

  const std::string kept_dims_str = absl::StrJoin(shape.GetKeptDims(), ",");

  absl::string_view hlo = R"(
  reducer_add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT sum = f32[] add(lhs, rhs)
  }

  ENTRY main {
    input = $input_shape parameter(0)

    input_f32 = $input_shape_f32 convert(input)
    input_squared = $input_shape_f32 multiply(input_f32, input_f32)
    zero_f32 = f32[] constant(0)
    sum_input_squared = $reduction_shape_f32 reduce(input_squared, zero_f32),
        dimensions={$reduction_dims}, to_apply=reducer_add

    dim_size_f32 = f32[] constant($reduction_size)
    dim_size_br = $reduction_shape_f32 broadcast(dim_size_f32), dimensions={}
    mean_input_squared =
        $reduction_shape_f32 divide(sum_input_squared, dim_size_br)

    epsilon_f32 = f32[] constant(1e-6)
    epsilon_br = $reduction_shape_f32 broadcast(epsilon_f32), dimensions={}
    mean_input_squared_eps =
        $reduction_shape_f32 add(mean_input_squared, epsilon_br)
    rms_f32 = $reduction_shape_f32 sqrt(mean_input_squared_eps)

    rms_br_f32 = $input_shape_f32 broadcast(rms_f32), dimensions={$kept_dims}

    output_f32 = $input_shape_f32 divide(input_f32, rms_br_f32)
    ROOT output = $input_shape convert(output_f32)
  }
  )";

  HloBenchmarkOptions benchmark_options;
  benchmark_options.num_executions = absl::GetFlag(FLAGS_num_executions);
  benchmark_options.aot_options = absl::GetFlag(FLAGS_aot_compiled_execution)
                                      ? GetAotCompilationOptions()
                                      : nullptr;

  Literal input = GetRandomLiteral(shape.input_shape);

  CHECK_OK(RunHloBenchmark(state, hlo, {&input},
                           {{"$input_shape", input_shape_str},
                            {"$input_shape_f32", input_shape_f32_str},
                            {"$reduction_shape_f32", reduction_shape_f32_str},
                            {"$reduction_dims", reduction_dims_str},
                            {"$reduction_size", absl::StrCat(reduction_size)},
                            {"$kept_dims", kept_dims_str},
                            {"$dtype", dtype_str}},
                           benchmark_options));
}

void BM_Softmax(benchmark::State& state, const NormShape& shape) {
  const std::string input_shape_str = shape.input_shape.ToString();
  const std::string reduction_dims_str =
      absl::StrJoin(shape.reduction_dims, ",");
  const std::string dtype_str =
      primitive_util::LowercasePrimitiveTypeName(shape.GetDType());

  Shape input_shape_f32 = ShapeUtil::ChangeElementType(shape.input_shape, F32);
  const std::string input_shape_f32_str = input_shape_f32.ToString();

  Shape reduction_shape_f32 =
      ShapeUtil::ChangeElementType(shape.GetReductionShape(), F32);
  const std::string reduction_shape_f32_str = reduction_shape_f32.ToString();

  const std::string kept_dims_str = absl::StrJoin(shape.GetKeptDims(), ",");

  absl::string_view hlo = R"(
  HloModule softmax

  reducer_max {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT max = f32[] maximum(lhs, rhs)
  }

  reducer_add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT sum = f32[] add(lhs, rhs)
  }

  ENTRY main {
    input = $input_shape parameter(0)
    input_f32 = $input_shape_f32 convert(input)

    neg_inf = f32[] constant(-inf)
    max_val = $reduction_shape_f32 reduce(input_f32, neg_inf),
        dimensions={$reduction_dims}, to_apply=reducer_max
    max_br = $input_shape_f32 broadcast(max_val), dimensions={$kept_dims}

    input_centered = $input_shape_f32 subtract(input_f32, max_br)
    input_exp = $input_shape_f32 exponential(input_centered)

    zero = f32[] constant(0)
    sum_exp = $reduction_shape_f32 reduce(input_exp, zero),
        dimensions={$reduction_dims}, to_apply=reducer_add
    sum_exp_br = $input_shape_f32 broadcast(sum_exp), dimensions={$kept_dims}

    output_f32 = $input_shape_f32 divide(input_exp, sum_exp_br)
    ROOT output = $input_shape convert(output_f32)
  }
  )";

  HloBenchmarkOptions benchmark_options;
  benchmark_options.num_executions = absl::GetFlag(FLAGS_num_executions);
  benchmark_options.aot_options = absl::GetFlag(FLAGS_aot_compiled_execution)
                                      ? GetAotCompilationOptions()
                                      : nullptr;

  Literal input = GetRandomLiteral(shape.input_shape);

  CHECK_OK(RunHloBenchmark(state, hlo, {&input},
                           {{"$input_shape", input_shape_str},
                            {"$input_shape_f32", input_shape_f32_str},
                            {"$reduction_shape_f32", reduction_shape_f32_str},
                            {"$reduction_dims", reduction_dims_str},
                            {"$kept_dims", kept_dims_str},
                            {"$dtype", dtype_str}},
                           benchmark_options));
}

void BM_ZScore(benchmark::State& state, const NormShape& shape) {
  const std::string input_shape_str = shape.input_shape.ToString();
  const std::string reduction_dims_str =
      absl::StrJoin(shape.reduction_dims, ",");
  const std::string dtype_str =
      primitive_util::LowercasePrimitiveTypeName(shape.GetDType());
  const std::string reduction_shape_str = shape.GetReductionShape().ToString();

  Shape input_shape_f32 = ShapeUtil::ChangeElementType(shape.input_shape, F32);
  const std::string input_shape_f32_str = input_shape_f32.ToString();

  Shape reduction_shape_f32 =
      ShapeUtil::ChangeElementType(shape.GetReductionShape(), F32);
  const std::string reduction_shape_f32_str = reduction_shape_f32.ToString();

  int64_t reduction_size = 1;
  for (int64_t d : shape.reduction_dims) {
    reduction_size *= shape.input_shape.dimensions(d);
  }

  const std::string kept_dims_str = absl::StrJoin(shape.GetKeptDims(), ",");

  absl::string_view hlo = R"(
  reducer_add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT sum = f32[] add(lhs, rhs)
  }

  ENTRY main {
    input = $input_shape parameter(0)
    input_f32 = $input_shape_f32 convert(input)

    zero_f32 = f32[] constant(0)
    sum_input = $reduction_shape_f32 reduce(input_f32, zero_f32),
        dimensions={$reduction_dims}, to_apply=reducer_add

    dim_size_f32 = f32[] constant($reduction_size)
    dim_size_br = $reduction_shape_f32 broadcast(dim_size_f32), dimensions={}
    mean = $reduction_shape_f32 divide(sum_input, dim_size_br)

    mean_br = $input_shape_f32 broadcast(mean), dimensions={$kept_dims}
    input_centered = $input_shape_f32 subtract(input_f32, mean_br)

    input_centered_squared =
        $input_shape_f32 multiply(input_centered, input_centered)
    sum_input_centered_squared =
        $reduction_shape_f32 reduce(input_centered_squared, zero_f32),
        dimensions={$reduction_dims}, to_apply=reducer_add

    variance =
        $reduction_shape_f32 divide(sum_input_centered_squared, dim_size_br)

    epsilon_f32 = f32[] constant(1e-6)
    epsilon_br = $reduction_shape_f32 broadcast(epsilon_f32), dimensions={}
    variance_eps = $reduction_shape_f32 add(variance, epsilon_br)
    std_dev = $reduction_shape_f32 sqrt(variance_eps)

    std_dev_br = $input_shape_f32 broadcast(std_dev), dimensions={$kept_dims}

    output_f32 = $input_shape_f32 divide(input_centered, std_dev_br)
    ROOT output = $input_shape convert(output_f32)
  }
  )";

  HloBenchmarkOptions benchmark_options;
  benchmark_options.num_executions = absl::GetFlag(FLAGS_num_executions);
  benchmark_options.aot_options = absl::GetFlag(FLAGS_aot_compiled_execution)
                                      ? GetAotCompilationOptions()
                                      : nullptr;

  Literal input = GetRandomLiteral(shape.input_shape);

  CHECK_OK(RunHloBenchmark(state, hlo, {&input},
                           {{"$input_shape", input_shape_str},
                            {"$input_shape_f32", input_shape_f32_str},
                            {"$reduction_shape_f32", reduction_shape_f32_str},
                            {"$reduction_dims", reduction_dims_str},
                            {"$reduction_size", absl::StrCat(reduction_size)},
                            {"$kept_dims", kept_dims_str},
                            {"$dtype", dtype_str}},
                           benchmark_options));
}

void RegisterBenchmarks() {
  std::vector<Shape> list = ParseShapeList(absl::GetFlag(FLAGS_shapes)).value();
  for (const auto& s : list) {
    NormShape shape = ParseShape(s);

    std::string shape_str =
        absl::StrCat(shape.input_shape.ToString(), "_{",
                     absl::StrJoin(shape.reduction_dims, ","), "}");

    benchmark::RegisterBenchmark("BM_RmsNorm/" + shape_str, BM_RmsNorm, shape)
        ->MeasureProcessCPUTime();

    benchmark::RegisterBenchmark("BM_Softmax/" + shape_str, BM_Softmax, shape)
        ->MeasureProcessCPUTime();

    benchmark::RegisterBenchmark("BM_ZScore/" + shape_str, BM_ZScore, shape)
        ->MeasureProcessCPUTime();
  }
}

}  // namespace

}  // namespace xla::cpu

GTEST_API_ int main(int argc, char** argv) {
  // Only run benchmarks if `--benchmark_filter` is set.
  for (int i = 1; i < argc; ++i) {
    if (absl::StartsWith(argv[i], "--benchmark_filter=")) {
      tsl::testing::InstallStacktraceHandler();
      ::benchmark::Initialize(&argc, argv);
      testing::InitGoogleTest(&argc, argv);
      xla::cpu::Set_XLA_FLAGS();
      xla::cpu::RegisterBenchmarks();
      ::benchmark::RunSpecifiedBenchmarks();
      return 0;
    }
  }
}
