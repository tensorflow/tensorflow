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
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/backends/cpu/benchmarks/multi_benchmark_config.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/stacktrace_handler.h"

namespace xla::cpu {

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

static void BM_BatchedDot(benchmark::State& state,
                          HloBenchmarkOptions options) {
  PrimitiveType dtype = static_cast<PrimitiveType>(state.range(0));
  PrimitiveType out_dtype = F32;
  int64_t d0 = state.range(1);
  int64_t d1 = state.range(2);

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

  auto shape = ShapeUtil::MakeShape(dtype, {d0, d1, d1});
  Literal p0 = GetRandomLiteral(shape);
  Literal p1 = GetRandomLiteral(shape);

  std::vector<const Literal*> args = {&p0, &p1};
  CHECK_OK(RunHloBenchmark(
      state, hlo, args,
      {{"$dtype", primitive_util::LowercasePrimitiveTypeName(dtype)},
       {"$out_dtype", primitive_util::LowercasePrimitiveTypeName(out_dtype)},
       {"$d0", absl::StrCat(d0)},
       {"$d1", absl::StrCat(d1)}},
      options));
}

#define BENCHMARK_BATCHED_DOT(dtype) \
  XLA_CPU_BENCHMARK(BM_BatchedDot)   \
      ->MeasureProcessCPUTime()      \
      ->Args({dtype, 1, 2})          \
      ->Args({dtype, 1, 32})         \
      ->Args({dtype, 1, 64})         \
      ->Args({dtype, 1, 128})        \
      ->Args({dtype, 1, 256})        \
      ->Args({dtype, 1, 512})        \
      ->Args({dtype, 2, 2})          \
      ->Args({dtype, 2, 32})         \
      ->Args({dtype, 2, 64})         \
      ->Args({dtype, 2, 128})        \
      ->Args({dtype, 2, 256})        \
      ->Args({dtype, 2, 512})        \
      ->Args({dtype, 4, 2})          \
      ->Args({dtype, 4, 32})         \
      ->Args({dtype, 4, 64})         \
      ->Args({dtype, 4, 128})        \
      ->Args({dtype, 4, 256})        \
      ->Args({dtype, 4, 512})        \
      ->Args({dtype, 8, 2})          \
      ->Args({dtype, 8, 32})         \
      ->Args({dtype, 8, 64})         \
      ->Args({dtype, 8, 128})        \
      ->Args({dtype, 8, 256})        \
      ->Args({dtype, 8, 512})

BENCHMARK_BATCHED_DOT(F32);   // Shown as "11" in the benchmark name.
BENCHMARK_BATCHED_DOT(BF16);  // Shown as "16" in the benchmark name.

namespace {

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
  CHECK_OK(RunHloBenchmark(state, std::move(computation), args));
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

std::string BenchmarkName(const GenericDot& dot) {
  auto dtype_str = absl::AsciiStrToUpper(absl::StrCat(
      primitive_util::LowercasePrimitiveTypeName(dot.lhs_type), "_",
      primitive_util::LowercasePrimitiveTypeName(dot.rhs_type), "_",
      primitive_util::LowercasePrimitiveTypeName(dot.out_type)));
  return absl::StrCat("BM_", dot.name, "/", dtype_str, "_",
                      absl::StrJoin(dot.lhs_shape, "x"), "_",
                      absl::StrJoin(dot.rhs_shape, "x"), "_",
                      absl::StrJoin(dot.out_shape, "x"));
}

void RegisterBenchmarks() {
  for (const GenericDot& dot : GetGenericDotList()) {
    benchmark::RegisterBenchmark(BenchmarkName(dot), BM_GenericDot, dot)
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
      xla::cpu::RegisterBenchmarks();
      ::benchmark::RunSpecifiedBenchmarks();
      return 0;
    }
  }
}
