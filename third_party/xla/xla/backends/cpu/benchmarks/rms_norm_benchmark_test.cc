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
#include <string>
#include <utility>
#include <vector>

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

struct RmsNorm {
  Shape input_shape;
  std::vector<int64_t> reduction_dims;

  PrimitiveType GetDType() const { return input_shape.element_type(); }

  Shape GetReductionShape() const {
    Shape reduction_shape = input_shape;
    reduction_shape.DeleteDimensions(reduction_dims);
    return reduction_shape;
  }

  std::string GetBenchmarkName() const {
    return absl::StrCat("BM_RmsNorm/", input_shape.ToString(), "_{",
                        absl::StrJoin(reduction_dims, ","), "}");
  }
};

RmsNorm ParseRmsNorm(const Shape& s) {
  RmsNorm rms_norm;
  CHECK(s.IsTuple());
  CHECK_EQ(s.tuple_shapes().size(), 2);

  rms_norm.input_shape = s.tuple_shapes(0);

  const Shape& dims_shape = s.tuple_shapes(1);
  absl::Span<const int64_t> dims = dims_shape.dimensions();
  rms_norm.reduction_dims.assign(dims.begin(), dims.end());

  return rms_norm;
}

void BM_RmsNorm(benchmark::State& state, const RmsNorm& rms_norm) {
  const std::string input_shape_str = rms_norm.input_shape.ToString();
  const std::string reduction_dims_str =
      absl::StrJoin(rms_norm.reduction_dims, ",");
  const std::string dtype_str =
      primitive_util::LowercasePrimitiveTypeName(rms_norm.GetDType());
  const std::string reduction_shape_str =
      rms_norm.GetReductionShape().ToString();

  Shape input_shape_f32 =
      ShapeUtil::ChangeElementType(rms_norm.input_shape, F32);
  const std::string input_shape_f32_str = input_shape_f32.ToString();

  Shape reduction_shape_f32 =
      ShapeUtil::ChangeElementType(rms_norm.GetReductionShape(), F32);
  const std::string reduction_shape_f32_str = reduction_shape_f32.ToString();

  int64_t reduction_size = 1;
  for (int64_t d : rms_norm.reduction_dims) {
    reduction_size *= rms_norm.input_shape.dimensions(d);
  }

  std::vector<int64_t> kept_dims;
  for (int64_t i = 0; i < rms_norm.input_shape.dimensions_size(); ++i) {
    bool is_reduced = false;
    for (int64_t d : rms_norm.reduction_dims) {
      if (i == d) {
        is_reduced = true;
        break;
      }
    }
    if (!is_reduced) {
      kept_dims.push_back(i);
    }
  }
  const std::string kept_dims_str = absl::StrJoin(kept_dims, ",");

  absl::string_view hlo_template = R"(
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

  std::string hlo_data = absl::StrReplaceAll(
      hlo_template, {{"$input_shape", input_shape_str},
                     {"$input_shape_f32", input_shape_f32_str},
                     {"$reduction_shape_f32", reduction_shape_f32_str},
                     {"$reduction_dims", reduction_dims_str},
                     {"$reduction_size", absl::StrCat(reduction_size)},
                     {"$kept_dims", kept_dims_str},
                     {"$dtype", dtype_str}});

  HloBenchmarkOptions benchmark_options;
  benchmark_options.num_executions = absl::GetFlag(FLAGS_num_executions);
  benchmark_options.aot_options = absl::GetFlag(FLAGS_aot_compiled_execution)
                                      ? GetAotCompilationOptions()
                                      : nullptr;

  TF_ASSERT_OK_AND_ASSIGN(
      auto module_and_iteration_literals,
      LoadHloModuleAndMaybeIterationLiteralsFromString(hlo_data));

  std::unique_ptr<HloModule> hlo_module =
      std::move(module_and_iteration_literals.first);

  std::vector<Literal> args;
  args.reserve(module_and_iteration_literals.second->arguments_size());
  for (const auto& arg : module_and_iteration_literals.second->arguments()) {
    TF_ASSERT_OK_AND_ASSIGN(args.emplace_back(), Literal::CreateFromProto(arg));
  }

  std::vector<Literal*> arg_ptrs;
  arg_ptrs.reserve(args.size());
  for (auto& arg : args) {
    arg_ptrs.push_back(&arg);
  }

  CHECK_OK(RunHloBenchmark(state, std::move(hlo_module), arg_ptrs,
                           benchmark_options));
}

void RegisterBenchmarks() {
  std::vector<Shape> list = ParseShapeList(absl::GetFlag(FLAGS_shapes)).value();
  for (const auto& s : list) {
    RmsNorm rms_norm = ParseRmsNorm(s);

    benchmark::RegisterBenchmark(rms_norm.GetBenchmarkName(), BM_RmsNorm,
                                 rms_norm)
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
