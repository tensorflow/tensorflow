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
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "benchmark/benchmark.h"
#include "xla/backends/cpu/benchmarks/aot_benchmark_helper.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/backends/cpu/benchmarks/multi_benchmark_config.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/stacktrace_handler.h"

using xla::primitive_util::LowercasePrimitiveTypeName;

ABSL_FLAG(std::string, shapes, "",
          "Comma-separated list of dot shapes to benchmark. Shapes are "
          "interpreted as M,K,N.");

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

HloBenchmarkOptions GetBenchmarkOptions() {
  HloBenchmarkOptions options;
  options.num_executions = absl::GetFlag(FLAGS_num_executions);
  options.aot_options = absl::GetFlag(FLAGS_aot_compiled_execution)
                            ? GetAotCompilationOptions()
                            : nullptr;
  return options;
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
    case S8:
      return *LiteralUtil::CreateRandomLiteral<S8>(shape, &engine, mean,
                                                   stddev);
    case S32:
      return *LiteralUtil::CreateRandomLiteral<S32>(shape, &engine, mean,
                                                    stddev);
    default:
      LOG(FATAL) << "Add dtype to the if-else block before use: " << dtype;
  }
}

struct BatchedDot {
  PrimitiveType in_dtype;
  PrimitiveType out_dtype;
  int64_t d0;
  int64_t d1;
};

static void BM_BatchedDot(benchmark::State& state, BatchedDot info) {
  absl::string_view hlo = R"(
    HloModule dot_$dtype_b$d0_d$d1

    ENTRY e {
      p0 = $dtype[$d0,$d1,$d1] parameter(0)
      p1 = $dtype[$d0,$d1,$d1] parameter(1)
      ROOT dot = $out_dtype[$d0,$d1,$d1] dot(p0, p1),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={1}
    }
  )";

  auto shape = ShapeUtil::MakeShape(info.in_dtype, {info.d0, info.d1, info.d1});
  Literal p0 = GetRandomLiteral(shape);
  Literal p1 = GetRandomLiteral(shape);

  std::vector<const Literal*> args = {&p0, &p1};
  CHECK_OK(RunHloBenchmark(
      state, hlo, args,
      {{"$dtype", primitive_util::LowercasePrimitiveTypeName(info.in_dtype)},
       {"$out_dtype",
        primitive_util::LowercasePrimitiveTypeName(info.out_dtype)},
       {"$d0", absl::StrCat(info.d0)},
       {"$d1", absl::StrCat(info.d1)}},
      GetBenchmarkOptions()));
}

// LINT.IfChange
struct GenericDot {
  std::string name;
  PrimitiveType lhs_type;
  std::vector<int64_t> lhs_shape;
  PrimitiveType rhs_type;
  std::vector<int64_t> rhs_shape;
  PrimitiveType out_type;
  std::vector<int64_t> out_shape;
  std::vector<int64_t> lhs_batch_dims;
  std::vector<int64_t> rhs_batch_dims;
  std::vector<int64_t> lhs_contracting_dims;
  std::vector<int64_t> rhs_contracting_dims;
};
// LINT.ThenChange(//tensorflow/compiler/xla/tools/extract_dots_for_benchmark.cc)

void BM_GenericDot(benchmark::State& state, GenericDot info) {
  HloComputation::Builder builder("BM_GenericDot");
  auto lhs_shape = ShapeUtil::MakeShape(info.lhs_type, info.lhs_shape);
  auto rhs_shape = ShapeUtil::MakeShape(info.rhs_type, info.rhs_shape);
  auto out_shape = ShapeUtil::MakeShape(info.out_type, info.out_shape);
  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "lhs"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, rhs_shape, "rhs"));

  DotDimensionNumbers dot_dnums;
  for (int64_t dim : info.lhs_batch_dims) {
    dot_dnums.add_lhs_batch_dimensions(dim);
  }
  for (int64_t dim : info.rhs_batch_dims) {
    dot_dnums.add_rhs_batch_dimensions(dim);
  }
  for (int64_t dim : info.lhs_contracting_dims) {
    dot_dnums.add_lhs_contracting_dimensions(dim);
  }
  for (int64_t dim : info.rhs_contracting_dims) {
    dot_dnums.add_rhs_contracting_dimensions(dim);
  }
  builder.AddInstruction(HloInstruction::CreateDot(
      out_shape, lhs, rhs, dot_dnums, PrecisionConfig()));
  std::unique_ptr<HloComputation> computation = builder.Build();

  Literal lhs_lit = GetRandomLiteral(lhs_shape);
  Literal rhs_lit = GetRandomLiteral(rhs_shape);
  std::vector<const Literal*> args = {&lhs_lit, &rhs_lit};
  CHECK_OK(RunHloBenchmark(state, std::move(computation), args,
                           GetBenchmarkOptions()));
}

std::vector<GenericDot> GetGenericDotList() {
  std::vector<GenericDot> list;
  // clang-format off
  // NOLINTBEGIN
  // Generate dot entries from an HLO module file by running
  // //xla/tools:extract_dots_for_benchmark
  std::string name = "Gemma3_1B_Call";
  list.insert(list.end(), {
    GenericDot{name, BF16, {1,11,1152}, BF16, {2,6912,1152}, BF16, {1,11,2,6912}, {}, {}, {2}, {2}},
    GenericDot{name, BF16, {1,11,1152}, BF16, {4,1152,256}, BF16, {1,11,4,256}, {}, {}, {2}, {1}},
    GenericDot{name, BF16, {1,11,4,11}, BF16, {1,11,256}, BF16, {1,11,4,256}, {0}, {0}, {3}, {1}},
    GenericDot{name, BF16, {1,11,4,256}, BF16, {1,11,256}, BF16, {1,11,4,11}, {0}, {0}, {3}, {2}},
    GenericDot{name, BF16, {1,11,4,256}, BF16, {4,256,1152}, BF16, {1,11,1152}, {}, {}, {3,2}, {1,0}},
    GenericDot{name, BF16, {1,11,6912}, BF16, {6912,1152}, BF16, {1,11,1152}, {}, {}, {2}, {0}},
    GenericDot{name, BF16, {1,1152}, BF16, {1152,262144}, BF16, {1,262144}, {}, {}, {1}, {0}},
    GenericDot{name, BF16, {2,1,1152,256}, BF16, {1,11,1152}, BF16, {2,1,256,1,11}, {}, {}, {2}, {2}}
  });
  name = "Gemma3_1B_SampleLoop";
  list.insert(list.end(), {
    GenericDot{name, BF16, {1,1,1152}, BF16, {1152,262144}, BF16, {1,1,262144}, {}, {}, {2}, {0}},
    GenericDot{name, BF16, {1,1,1152}, BF16, {2,6912,1152}, BF16, {1,1,2,6912}, {}, {}, {2}, {2}},
    GenericDot{name, BF16, {1,1,1152}, BF16, {4,1152,256}, BF16, {1,1,4,256}, {}, {}, {2}, {1}},
    GenericDot{name, BF16, {1,1,4,256}, BF16, {1,4096,256}, BF16, {1,1,4,4096}, {0}, {0}, {3}, {2}},
    GenericDot{name, BF16, {1,1,4,256}, BF16, {4,256,1152}, BF16, {1,1,1152}, {}, {}, {3,2}, {1,0}},
    GenericDot{name, BF16, {1,1,4,4096}, BF16, {1,4096,256}, BF16, {1,1,4,256}, {0}, {0}, {3}, {1}},
    GenericDot{name, BF16, {1,1,6912}, BF16, {6912,1152}, BF16, {1,1,1152}, {}, {}, {2}, {0}},
    GenericDot{name, BF16, {2,1,1152,256}, BF16, {1,1,1152}, BF16, {2,1,256,1,1}, {}, {}, {2}, {2}}
  });
  // NOLINTEND
  // clang-format on
  return list;
}

std::string GenericDotBenchmarkName(const GenericDot& dot) {
  auto dtype_str = absl::AsciiStrToUpper(absl::StrCat(
      PrimitiveType_Name(dot.lhs_type), "_", PrimitiveType_Name(dot.rhs_type),
      "_", PrimitiveType_Name(dot.out_type)));
  return absl::StrCat("BM_", dot.name, "/", dtype_str, "_",
                      absl::StrJoin(dot.lhs_shape, "x"), "_",
                      absl::StrJoin(dot.rhs_shape, "x"), "_",
                      absl::StrJoin(dot.out_shape, "x"));
}

PrimitiveType GetAccumulatorType(PrimitiveType type) {
  switch (type) {
    case F64:
      return F64;
    case F32:
    case BF16:
    case F16:
      return F32;
    case S8:
      return S32;
    default:
      LOG(FATAL) << "Unsupported type: " << type;
  }
}

void BM_Dot(benchmark::State& state, const Shape& shape) {
  absl::string_view hlo_template = R"(
    HloModule benchmark
    ENTRY main {
      p0 = $a_type[$m,$k] parameter(0)
      p1 = $b_type[$k,$n] parameter(1)
      ROOT %result = $c_type[$m,$n] dot(p0, p1), lhs_contracting_dims={1},
          rhs_contracting_dims={0}
    }
  )";

  PrimitiveType input_type = shape.element_type();
  PrimitiveType output_type = GetAccumulatorType(input_type);

  std::string hlo_data = absl::StrReplaceAll(
      hlo_template, {{"$m", absl::StrCat(shape.dimensions(0))},
                     {"$k", absl::StrCat(shape.dimensions(1))},
                     {"$n", absl::StrCat(shape.dimensions(2))},
                     {"$a_type", LowercasePrimitiveTypeName(input_type)},
                     {"$b_type", LowercasePrimitiveTypeName(input_type)},
                     {"$c_type", LowercasePrimitiveTypeName(output_type)}});

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
                           GetBenchmarkOptions()));
}

void RegisterBenchmarks() {
  std::string shapes_arg = absl::GetFlag(FLAGS_shapes);

  if (shapes_arg.empty()) {
    //===------------------------------------------------------------------===//
    // BM_BatchedDot
    //===------------------------------------------------------------------===//
    // Pairs of input-output data types.
    std::vector<std::pair<PrimitiveType, PrimitiveType>> dtype_pairs = {
        {F32, F32}, {BF16, F32}, {S8, S32}, {S32, S32}};
    for (auto [in_dtype, out_dtype] : dtype_pairs) {
      std::string in_dtype_str = PrimitiveType_Name(in_dtype);
      std::string out_dtype_str = PrimitiveType_Name(out_dtype);
      for (int64_t d0 : {1, 2, 4, 8}) {
        for (int64_t d1 : {2, 32, 64, 128, 256, 512}) {
          benchmark::RegisterBenchmark(
              absl::StrCat("BM_BatchedDot_", in_dtype_str, "_", out_dtype_str,
                           "_", d0, "x", d1, "x", d1),
              BM_BatchedDot, BatchedDot{in_dtype, out_dtype, d0, d1})
              ->MeasureProcessCPUTime();
        }
      }
    }

    //===------------------------------------------------------------------===//
    // BM_GenericDot
    //===------------------------------------------------------------------===//
    for (const GenericDot& dot : GetGenericDotList()) {
      benchmark::RegisterBenchmark(GenericDotBenchmarkName(dot), BM_GenericDot,
                                   dot)
          ->MeasureProcessCPUTime();
    }
  } else {
    std::vector<Shape> shapes = ParseShapeList(shapes_arg).value();

    for (const Shape& shape : shapes) {
      if (shape.dimensions().size() != 3) {
        LOG(ERROR) << "Shape must have 3 dimensions M,K,N: "
                   << shape.ToString();
        continue;
      }
      benchmark::RegisterBenchmark(absl::StrCat("BM_Dot/", shape.ToString()),
                                   BM_Dot, shape)
          ->MeasureProcessCPUTime();
    }
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
