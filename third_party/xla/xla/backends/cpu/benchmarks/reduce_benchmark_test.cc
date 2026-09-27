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
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/stacktrace_handler.h"

ABSL_FLAG(std::string, shapes, "{(f32[1024,1024], s32[1,0])}",
          "List of shapes encoding reductions (ab)using the shape parser.");

ABSL_FLAG(int32_t, num_executions, 1,
          "Number of times to execute the HLO within a single benchmark "
          "iteration. By overlapping multiple independent execution we can "
          "measure how well XLA runtime handles concurrent requests, which is "
          "similar to production inference workloads.");

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

struct Reduction {
  Shape input_shape;
  std::vector<int64_t> dims;

  PrimitiveType GetDType() const { return input_shape.element_type(); }

  Shape GetOutputShape() const {
    Shape output_shape = input_shape;
    output_shape.DeleteDimensions(dims);
    return output_shape;
  }

  std::string GetBenchmarkName() const {
    return absl::StrCat("BM_Reduce/", input_shape.ToString(), "_{",
                        absl::StrJoin(dims, ","), "}");
  }
};

Reduction ParseReduction(const Shape& s) {
  Reduction reduction;
  CHECK(s.IsTuple());
  CHECK_EQ(s.tuple_shapes().size(), 2);

  reduction.input_shape = s.tuple_shapes(0);

  const Shape& dims_shape = s.tuple_shapes(1);
  absl::Span<const int64_t> dims = dims_shape.dimensions();
  reduction.dims.assign(dims.begin(), dims.end());

  return reduction;
}

void BM_Reduce(benchmark::State& state, const Reduction& reduction) {
  const std::string input_shape_str = reduction.input_shape.ToString();
  const std::string dims_list_str = absl::StrJoin(reduction.dims, ",");
  const std::string dtype_str =
      primitive_util::LowercasePrimitiveTypeName(reduction.GetDType());
  const std::string output_shape_str = reduction.GetOutputShape().ToString();

  absl::string_view hlo_template = R"(
  reducer_add {
    lhs = $dtype[] parameter(0)
    rhs = $dtype[] parameter(1)
    ROOT sum = $dtype[] add(lhs, rhs)
  }

  ENTRY main {
    input = $input_shape parameter(0)
    c = $dtype[] constant(0)
    ROOT output = $output_shape reduce(input, c), dimensions={$dims_list}, to_apply=reducer_add
  }
  )";

  std::string hlo_data =
      absl::StrReplaceAll(hlo_template, {{"$input_shape", input_shape_str},
                                         {"$output_shape", output_shape_str},
                                         {"$dims_list", dims_list_str},
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
    Reduction reduction = ParseReduction(s);

    benchmark::RegisterBenchmark(reduction.GetBenchmarkName(), BM_Reduce,
                                 reduction)
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
