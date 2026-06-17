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
    case F64:
      return *LiteralUtil::CreateRandomLiteral<F64>(shape, &engine, mean,
                                                    stddev);
    case F32:
      return *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, mean,
                                                    stddev);
    case BF16:
      return *LiteralUtil::CreateRandomLiteral<BF16>(shape, &engine, mean,
                                                     stddev);
    case F16:
      return *LiteralUtil::CreateRandomLiteral<F16>(shape, &engine, mean,
                                                    stddev);
    default:
      LOG(FATAL) << "Add dtype to the if-else block before use: " << dtype;
  }
}

PrimitiveType GetComputeDType(PrimitiveType dtype) {
  return dtype == BF16 || dtype == F16 ? F32 : dtype;
}

void BM_RmsNorm(benchmark::State& state, const NormShape& shape) {
  const std::string input_shape_str = shape.input_shape.ToString();
  const std::string reduction_dims_str =
      absl::StrJoin(shape.reduction_dims, ",");
  const std::string dtype_str =
      primitive_util::LowercasePrimitiveTypeName(shape.GetDType());

  PrimitiveType compute_dtype = GetComputeDType(shape.GetDType());
  const std::string compute_dtype_str =
      primitive_util::LowercasePrimitiveTypeName(compute_dtype);

  Shape compute_input_shape =
      ShapeUtil::ChangeElementType(shape.input_shape, compute_dtype);
  const std::string compute_input_shape_str = compute_input_shape.ToString();

  Shape compute_reduction_shape =
      ShapeUtil::ChangeElementType(shape.GetReductionShape(), compute_dtype);
  const std::string compute_reduction_shape_str =
      compute_reduction_shape.ToString();

  int64_t reduction_size = 1;
  for (int64_t d : shape.reduction_dims) {
    reduction_size *= shape.input_shape.dimensions(d);
  }

  const std::string kept_dims_str = absl::StrJoin(shape.GetKeptDims(), ",");

  absl::string_view hlo = R"(
  reducer_add {
    lhs = $compute_dtype[] parameter(0)
    rhs = $compute_dtype[] parameter(1)
    ROOT sum = $compute_dtype[] add(lhs, rhs)
  }

  ENTRY main {
    input = $input_shape parameter(0)

    input_compute = $compute_input_shape convert(input)
    input_squared = $compute_input_shape multiply(input_compute, input_compute)
    zero = $compute_dtype[] constant(0)
    sum_input_squared = $compute_reduction_shape reduce(input_squared, zero),
        dimensions={$reduction_dims}, to_apply=reducer_add

    dim_size = $compute_dtype[] constant($reduction_size)
    dim_size_br = $compute_reduction_shape broadcast(dim_size), dimensions={}
    mean_input_squared =
        $compute_reduction_shape divide(sum_input_squared, dim_size_br)

    epsilon = $compute_dtype[] constant(1e-6)
    epsilon_br = $compute_reduction_shape broadcast(epsilon), dimensions={}
    mean_input_squared_eps =
        $compute_reduction_shape add(mean_input_squared, epsilon_br)
    rms = $compute_reduction_shape sqrt(mean_input_squared_eps)

    rms_br = $compute_input_shape broadcast(rms), dimensions={$kept_dims}

    output_compute = $compute_input_shape divide(input_compute, rms_br)
    ROOT output = $input_shape convert(output_compute)
  }
  )";

  HloBenchmarkOptions benchmark_options;
  benchmark_options.num_executions = absl::GetFlag(FLAGS_num_executions);
  benchmark_options.aot_options = absl::GetFlag(FLAGS_aot_compiled_execution)
                                      ? GetAotCompilationOptions()
                                      : nullptr;

  Literal input = GetRandomLiteral(shape.input_shape);

  CHECK_OK(RunHloBenchmark(
      state, hlo, {&input},
      {{"$input_shape", input_shape_str},
       {"$compute_input_shape", compute_input_shape_str},
       {"$compute_reduction_shape", compute_reduction_shape_str},
       {"$reduction_dims", reduction_dims_str},
       {"$reduction_size", absl::StrCat(reduction_size)},
       {"$kept_dims", kept_dims_str},
       {"$compute_dtype", compute_dtype_str},
       {"$dtype", dtype_str}},
      benchmark_options));
}

void BM_Softmax(benchmark::State& state, const NormShape& shape) {
  const std::string input_shape_str = shape.input_shape.ToString();
  const std::string reduction_dims_str =
      absl::StrJoin(shape.reduction_dims, ",");
  const std::string dtype_str =
      primitive_util::LowercasePrimitiveTypeName(shape.GetDType());

  PrimitiveType compute_dtype = GetComputeDType(shape.GetDType());
  const std::string compute_dtype_str =
      primitive_util::LowercasePrimitiveTypeName(compute_dtype);

  Shape compute_input_shape =
      ShapeUtil::ChangeElementType(shape.input_shape, compute_dtype);
  const std::string compute_input_shape_str = compute_input_shape.ToString();

  Shape compute_reduction_shape =
      ShapeUtil::ChangeElementType(shape.GetReductionShape(), compute_dtype);
  const std::string compute_reduction_shape_str =
      compute_reduction_shape.ToString();

  const std::string kept_dims_str = absl::StrJoin(shape.GetKeptDims(), ",");

  absl::string_view hlo = R"(
  HloModule softmax

  reducer_max {
    lhs = $compute_dtype[] parameter(0)
    rhs = $compute_dtype[] parameter(1)
    ROOT max = $compute_dtype[] maximum(lhs, rhs)
  }

  reducer_add {
    lhs = $compute_dtype[] parameter(0)
    rhs = $compute_dtype[] parameter(1)
    ROOT sum = $compute_dtype[] add(lhs, rhs)
  }

  ENTRY main {
    input = $input_shape parameter(0)
    input_compute = $compute_input_shape convert(input)

    neg_inf = $compute_dtype[] constant(-inf)
    max_val = $compute_reduction_shape reduce(input_compute, neg_inf),
        dimensions={$reduction_dims}, to_apply=reducer_max
    max_br = $compute_input_shape broadcast(max_val), dimensions={$kept_dims}

    input_centered = $compute_input_shape subtract(input_compute, max_br)
    input_exp = $compute_input_shape exponential(input_centered)

    zero = $compute_dtype[] constant(0)
    sum_exp = $compute_reduction_shape reduce(input_exp, zero),
        dimensions={$reduction_dims}, to_apply=reducer_add
    sum_exp_br = $compute_input_shape broadcast(sum_exp),
        dimensions={$kept_dims}

    output_compute = $compute_input_shape divide(input_exp, sum_exp_br)
    ROOT output = $input_shape convert(output_compute)
  }
  )";

  HloBenchmarkOptions benchmark_options;
  benchmark_options.num_executions = absl::GetFlag(FLAGS_num_executions);
  benchmark_options.aot_options = absl::GetFlag(FLAGS_aot_compiled_execution)
                                      ? GetAotCompilationOptions()
                                      : nullptr;

  Literal input = GetRandomLiteral(shape.input_shape);

  CHECK_OK(RunHloBenchmark(
      state, hlo, {&input},
      {{"$input_shape", input_shape_str},
       {"$compute_input_shape", compute_input_shape_str},
       {"$compute_reduction_shape", compute_reduction_shape_str},
       {"$reduction_dims", reduction_dims_str},
       {"$kept_dims", kept_dims_str},
       {"$compute_dtype", compute_dtype_str},
       {"$dtype", dtype_str}},
      benchmark_options));
}

void BM_ZScore(benchmark::State& state, const NormShape& shape) {
  const std::string input_shape_str = shape.input_shape.ToString();
  const std::string reduction_dims_str =
      absl::StrJoin(shape.reduction_dims, ",");
  const std::string dtype_str =
      primitive_util::LowercasePrimitiveTypeName(shape.GetDType());

  PrimitiveType compute_dtype = GetComputeDType(shape.GetDType());
  const std::string compute_dtype_str =
      primitive_util::LowercasePrimitiveTypeName(compute_dtype);

  Shape compute_input_shape =
      ShapeUtil::ChangeElementType(shape.input_shape, compute_dtype);
  const std::string compute_input_shape_str = compute_input_shape.ToString();

  Shape compute_reduction_shape =
      ShapeUtil::ChangeElementType(shape.GetReductionShape(), compute_dtype);
  const std::string compute_reduction_shape_str =
      compute_reduction_shape.ToString();

  int64_t reduction_size = 1;
  for (int64_t d : shape.reduction_dims) {
    reduction_size *= shape.input_shape.dimensions(d);
  }

  const std::string kept_dims_str = absl::StrJoin(shape.GetKeptDims(), ",");

  absl::string_view hlo = R"(
  reducer_add {
    lhs = $compute_dtype[] parameter(0)
    rhs = $compute_dtype[] parameter(1)
    ROOT sum = $compute_dtype[] add(lhs, rhs)
  }

  ENTRY main {
    input = $input_shape parameter(0)
    input_compute = $compute_input_shape convert(input)

    zero = $compute_dtype[] constant(0)
    sum_input = $compute_reduction_shape reduce(input_compute, zero),
        dimensions={$reduction_dims}, to_apply=reducer_add

    dim_size = $compute_dtype[] constant($reduction_size)
    dim_size_br = $compute_reduction_shape broadcast(dim_size), dimensions={}
    mean = $compute_reduction_shape divide(sum_input, dim_size_br)

    mean_br = $compute_input_shape broadcast(mean), dimensions={$kept_dims}
    input_centered = $compute_input_shape subtract(input_compute, mean_br)

    input_centered_squared =
        $compute_input_shape multiply(input_centered, input_centered)
    sum_input_centered_squared =
        $compute_reduction_shape reduce(input_centered_squared, zero),
        dimensions={$reduction_dims}, to_apply=reducer_add

    variance =
        $compute_reduction_shape divide(sum_input_centered_squared, dim_size_br)

    epsilon = $compute_dtype[] constant(1e-6)
    epsilon_br = $compute_reduction_shape broadcast(epsilon), dimensions={}
    variance_eps = $compute_reduction_shape add(variance, epsilon_br)
    std_dev = $compute_reduction_shape sqrt(variance_eps)

    std_dev_br = $compute_input_shape broadcast(std_dev),
        dimensions={$kept_dims}

    output_compute = $compute_input_shape divide(input_centered, std_dev_br)
    ROOT output = $input_shape convert(output_compute)
  }
  )";

  HloBenchmarkOptions benchmark_options;
  benchmark_options.num_executions = absl::GetFlag(FLAGS_num_executions);
  benchmark_options.aot_options = absl::GetFlag(FLAGS_aot_compiled_execution)
                                      ? GetAotCompilationOptions()
                                      : nullptr;

  Literal input = GetRandomLiteral(shape.input_shape);

  CHECK_OK(RunHloBenchmark(
      state, hlo, {&input},
      {{"$input_shape", input_shape_str},
       {"$compute_input_shape", compute_input_shape_str},
       {"$compute_reduction_shape", compute_reduction_shape_str},
       {"$reduction_dims", reduction_dims_str},
       {"$reduction_size", absl::StrCat(reduction_size)},
       {"$kept_dims", kept_dims_str},
       {"$compute_dtype", compute_dtype_str},
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
