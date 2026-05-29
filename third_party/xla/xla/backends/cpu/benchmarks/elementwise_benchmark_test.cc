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
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/backends/cpu/benchmarks/multi_benchmark_config.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

static void BM_AddF32(benchmark::State& state, HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule add_f32_$d0

    ENTRY e {
      p0 = f32[1,2,1,$d0,256] parameter(0)
      p1 = f32[1,2,1,$d0,256] parameter(1)
      ROOT add = f32[1,2,1,$d0,256] add(p0, p1)
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(F32, {1, 2, 1, d0, 256});
  ASSERT_OK_AND_ASSIGN(Literal p0, LiteralUtil::CreateRandomLiteral<F32>(
                                       shape, &engine, 1.0f, 0.1f));
  ASSERT_OK_AND_ASSIGN(Literal p1, LiteralUtil::CreateRandomLiteral<F32>(
                                       shape, &engine, 1.0f, 0.1f));

  std::vector<const Literal*> args = {&p0, &p1};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

static void BM_AddBF16(benchmark::State& state, HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule add_bf16_$d0

    ENTRY e {
      p0 = bf16[1,2,1,$d0,256] parameter(0)
      p1 = bf16[1,2,1,$d0,256] parameter(1)
      ROOT add = bf16[1,2,1,$d0,256] add(p0, p1)
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(BF16, {1, 2, 1, d0, 256});
  ASSERT_OK_AND_ASSIGN(Literal p0, LiteralUtil::CreateRandomLiteral<BF16>(
                                       shape, &engine, 1.0f, 0.1f));
  ASSERT_OK_AND_ASSIGN(Literal p1, LiteralUtil::CreateRandomLiteral<BF16>(
                                       shape, &engine, 1.0f, 0.1f));

  std::vector<const Literal*> args = {&p0, &p1};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

static void BM_UnaryOp(benchmark::State& state,
                       const HloBenchmarkOptions& options, PrimitiveType type,
                       HloOpcode op) {
  int64_t d0 = state.range(0);
  std::string type_name = absl::AsciiStrToLower(PrimitiveType_Name(type));
  absl::string_view op_name = xla::HloOpcodeString(op);

  absl::string_view hlo = R"(
    HloModule $op_$type_$d0

    ENTRY e {
      p0 = $type[1,2,1,$d0,256] parameter(0)
      ROOT root = $type[1,2,1,$d0,256] $op(p0)
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(type, {1, 2, 1, d0, 256});

  auto p0_status = [&]() -> absl::StatusOr<Literal> {
    switch (type) {
      case F32:
        return LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f,
                                                     0.1f);
      case BF16:
        return LiteralUtil::CreateRandomLiteral<BF16>(shape, &engine, 1.0f,
                                                      0.1f);
      case F64:
        return LiteralUtil::CreateRandomLiteral<F64>(shape, &engine, 1.0, 0.1);
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported type: ", PrimitiveType_Name(type)));
    }
  }();

  ASSERT_OK(p0_status.status());
  Literal p0 = std::move(p0_status).value();

  std::vector<const Literal*> args = {&p0};
  CHECK_OK(RunHloBenchmark(
      state, hlo, args,
      {{"$d0", absl::StrCat(d0)}, {"$type", type_name}, {"$op", op_name}},
      options));
}

static void BM_ConvertF32ToBF16(benchmark::State& state,
                                HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule convert_f32_to_bf16_$d0

    ENTRY e {
      p0 = f32[1,2,1,$d0,256] parameter(0)
      ROOT convert = bf16[1,2,1,$d0,256] convert(p0)
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(F32, {1, 2, 1, d0, 256});
  ASSERT_OK_AND_ASSIGN(Literal p0, LiteralUtil::CreateRandomLiteral<F32>(
                                       shape, &engine, 1.0f, 0.1f));

  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

#define BENCHMARK_SIZES(NAME)   \
  XLA_CPU_BENCHMARK(NAME)       \
      ->MeasureProcessCPUTime() \
      ->Arg(128)                \
      ->Arg(256)                \
      ->Arg(512)                \
      ->Arg(1024)               \
      ->Arg(8192)               \
      ->Arg(16384)              \
      ->Arg(32768)

BENCHMARK_SIZES(BM_AddF32);
BENCHMARK_SIZES(BM_AddBF16);
BENCHMARK_SIZES(BM_ConvertF32ToBF16);

#define BM_UNARY_OP(OP, TYPE)                              \
  static void BM_##OP##TYPE(benchmark::State& state,       \
                            HloBenchmarkOptions options) { \
    BM_UnaryOp(state, options, TYPE, HloOpcode::k##OP);    \
  }                                                        \
  BENCHMARK_SIZES(BM_##OP##TYPE);

#define BM_UNARY_OP_ALL_TYPES(OP) \
  BM_UNARY_OP(OP, F32)            \
  BM_UNARY_OP(OP, F64)

BM_UNARY_OP_ALL_TYPES(Cbrt);
BM_UNARY_OP_ALL_TYPES(Cos);
BM_UNARY_OP_ALL_TYPES(Erf);
BM_UNARY_OP_ALL_TYPES(Exp);
BM_UNARY_OP_ALL_TYPES(Expm1);
BM_UNARY_OP_ALL_TYPES(Log);
BM_UNARY_OP_ALL_TYPES(Log1p);
BM_UNARY_OP_ALL_TYPES(Logistic);
BM_UNARY_OP_ALL_TYPES(Rsqrt);
BM_UNARY_OP_ALL_TYPES(Sin);
BM_UNARY_OP_ALL_TYPES(Sqrt);
BM_UNARY_OP_ALL_TYPES(Tan);
BM_UNARY_OP_ALL_TYPES(Tanh);

}  // namespace xla::cpu
