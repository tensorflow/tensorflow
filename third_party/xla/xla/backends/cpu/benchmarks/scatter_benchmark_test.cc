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

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/array2d.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/backends/cpu/benchmarks/multi_benchmark_config.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/scatter_simplifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

namespace {

// Generate random indices to be used in the scatter.
// To make them unique, we randomly select from the range [0, slice_size)
// by first shuffling the full range and then taking the first slice_size
// elements
Literal createR1ScatterIndices(int64_t domain_size, int64_t scatter_size,
                               std::minstd_rand0& engine) {
  Literal scatter_indices;

  std::vector<int32_t> scatter_indices_vector(domain_size);
  std::iota(scatter_indices_vector.begin(), scatter_indices_vector.end(), 0);
  std::shuffle(scatter_indices_vector.begin(), scatter_indices_vector.end(),
               engine);
  scatter_indices_vector.resize(scatter_size);
  scatter_indices = LiteralUtil::CreateR1<int32_t>(scatter_indices_vector);

  return scatter_indices;
}

// Creates a 2D literal to reduce from num_elems * step to num_elems elements.
Literal CreateReduceIndices(int32_t num_elems, int32_t step) {
  CHECK_GE(num_elems, step);
  CHECK_EQ(num_elems % step, 0);
  Array2D<int32_t> array(num_elems * step, 1);
  for (int i = 0; i < num_elems; ++i) {
    for (int j = 0; j < step; ++j) {
      array(i * step + j, 0) = i;
    }
  }
  return LiteralUtil::CreateR2FromArray2D(array);
}

void BM_ScatterS32_R1(benchmark::State& state, HloBenchmarkOptions options) {
  const int64_t d0 = state.range(0);
  const int64_t slice_size = state.range(1);

  const std::string hlo = R"(
    HloModule BM_ScatterS32_R1

    assign (lhs: s32[], rhs: s32[]) -> s32[] {
      lhs = s32[] parameter(0)
      ROOT rhs = s32[] parameter(1) // just assign the value
    }

    ENTRY main {
      operand = s32[$d0] parameter(0)
      indices = s32[$slice_size] parameter(1)
      updates = s32[$slice_size] parameter(2)
      ROOT scatter = s32[$d0] scatter(operand, indices, updates),
          to_apply=assign,
          update_window_dims={},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    }
    )";

  std::minstd_rand0 engine;

  const Shape operand_shape = ShapeUtil::MakeShape(S32, {d0});
  const Literal operand = *LiteralUtil::CreateRandomLiteral<S32>(
      operand_shape, &engine, /*mean=*/50, /*stddev=*/10);

  const Literal scatter_indices =
      createR1ScatterIndices(d0, slice_size, engine);

  const Shape update_shape = ShapeUtil::MakeShape(S32, {slice_size});
  const Literal update = *LiteralUtil::CreateRandomLiteral<S32>(
      update_shape, &engine, /*mean=*/50, /*stddev=*/10);

  std::vector<const Literal*> args = {&operand, &scatter_indices, &update};
  CHECK_OK(RunHloBenchmark(
      state, hlo, args,
      {{"$d0", absl::StrCat(d0)}, {"$slice_size", absl::StrCat(slice_size)}},
      options));

  state.SetComplexityN(state.range(1));
}

void BM_ScatterS32_R2(benchmark::State& state, HloBenchmarkOptions options) {
  const int64_t d0 = state.range(0);
  const int64_t d1 = d0;
  const int64_t slice_size = state.range(1);

  const std::string hlo = R"(
    HloModule BM_ScatterS32_R2

    assign (lhs: s32[], rhs: s32[]) -> s32[] {
      lhs = s32[] parameter(0)
      ROOT rhs = s32[] parameter(1) // just assign the value
    }

    ENTRY main {
      operand = s32[$d0,$d1] parameter(0)
      indices = s32[$slice_size] parameter(1)
      updates = s32[$slice_size,$d1] parameter(2)
      ROOT scatter = s32[$d0,$d1] scatter(operand, indices, updates),
          to_apply=assign,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    }
    )";

  std::minstd_rand0 engine;

  const Shape operand_shape = ShapeUtil::MakeShape(S32, {d0, d1});
  const Literal operand = *LiteralUtil::CreateRandomLiteral<S32>(
      operand_shape, &engine, /*mean=*/50, /*stddev=*/10);

  const Literal scatter_indices =
      createR1ScatterIndices(d0, slice_size, engine);
  const Shape update_shape = ShapeUtil::MakeShape(S32, {slice_size, d1});
  const Literal update = *LiteralUtil::CreateRandomLiteral<S32>(
      update_shape, &engine, /*mean=*/50, /*stddev=*/10);

  std::vector<const Literal*> args = {&operand, &scatter_indices, &update};
  CHECK_OK(RunHloBenchmark(state, hlo, args,
                           {{"$d0", absl::StrCat(d0)},
                            {"$d1", absl::StrCat(d1)},
                            {"$slice_size", absl::StrCat(slice_size)}},
                           options));
}

void BM_ScatterS32_R3(benchmark::State& state, HloBenchmarkOptions options) {
  const int64_t d0 = state.range(0);
  const int64_t d1 = d0;
  const int64_t d2 = d0;
  const int64_t slice_size = state.range(1);

  const std::string hlo = R"(
    HloModule BM_ScatterS32_R3

    assign (lhs: s32[], rhs: s32[]) -> s32[] {
      lhs = s32[] parameter(0)
      ROOT rhs = s32[] parameter(1) // just assign the value
    }

    ENTRY main {
      operand = s32[$d0,$d1,$d2] parameter(0)
      indices = s32[$slice_size] parameter(1)
      updates = s32[$slice_size,$d1,$d2] parameter(2)
      ROOT scatter = s32[$d0,$d1,$d2] scatter(operand, indices, updates),
          to_apply=assign,
          update_window_dims={1, 2},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    }
    )";

  std::minstd_rand0 engine;

  const Shape operand_shape = ShapeUtil::MakeShape(S32, {d0, d1, d2});
  const Literal operand = *LiteralUtil::CreateRandomLiteral<S32>(
      operand_shape, &engine, /*mean=*/50, /*stddev=*/10);

  const Literal scatter_indices =
      createR1ScatterIndices(d0, slice_size, engine);

  const Shape update_shape = ShapeUtil::MakeShape(S32, {slice_size, d1, d2});
  const Literal update = *LiteralUtil::CreateRandomLiteral<S32>(
      update_shape, &engine, /*mean=*/50, /*stddev=*/10);

  std::vector<const Literal*> args = {&operand, &scatter_indices, &update};
  CHECK_OK(RunHloBenchmark(state, hlo, args,
                           {{"$d0", absl::StrCat(d0)},
                            {"$d1", absl::StrCat(d1)},
                            {"$d2", absl::StrCat(d2)},
                            {"$slice_size", absl::StrCat(slice_size)}},
                           options));
}

void BM_SimpleScatterReduceF32_R3(benchmark::State& state,
                                  HloBenchmarkOptions options) {
  const int64_t d0 = state.range(0);
  const int64_t d1 = state.range(1);
  const int64_t d2 = state.range(2);
  const int64_t num_reduce_elems = state.range(3);
  const int64_t num_slices = d0 * num_reduce_elems;

  constexpr absl::string_view hlo_string = R"(
    HloModule m
    add {
      %lhs = f32[] parameter(0)
      %rhs = f32[] parameter(1)
      ROOT %add.2 = f32[] add(%lhs, %rhs)
    }
    ENTRY main {
      %operand = f32[$d0,$d1,$d2] parameter(0)
      %indices = s32[$num_slices, 1]{1,0} parameter(1)
      %updates = f32[$num_slices,1,$d1,$d2] parameter(2)
      ROOT %scatter = f32[$d0,$d1,$d2] scatter(%operand, %indices, %updates),
        update_window_dims={1,2,3},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=add
    })";
  std::string hlo = absl::StrReplaceAll(
      hlo_string, {{"$d0", absl::StrCat(d0)},
                   {"$d1", absl::StrCat(d1)},
                   {"$d2", absl::StrCat(d2)},
                   {"$num_slices", absl::StrCat(num_slices)}});

  {
    // Make sure this is a simple scatter.
    auto module = ParseAndReturnUnverifiedModule(hlo).value();
    auto scatter = module->entry_computation()->root_instruction();
    CHECK(ScatterSimplifier::IsSimplifiedScatter(
        Cast<HloScatterInstruction>(scatter)));
  }

  std::minstd_rand0 engine;

  const Shape operand_shape = ShapeUtil::MakeShape(F32, {d0, d1, d2});
  const Literal operand = *LiteralUtil::CreateRandomLiteral<F32>(
      operand_shape, &engine, /*mean=*/50, /*stddev=*/10);

  const Literal indices = CreateReduceIndices(d0, num_reduce_elems);

  const Shape update_shape = ShapeUtil::MakeShape(F32, {num_slices, d1, d2});
  const Literal update = *LiteralUtil::CreateRandomLiteral<F32>(
      update_shape, &engine, /*mean=*/50, /*stddev=*/10);

  std::vector<const Literal*> args = {&operand, &indices, &update};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {}, options));
}

// these all have the same number of elements in the operand
// (2^18) == (2^9)^2 == (2^6)^3
XLA_CPU_BENCHMARK(BM_ScatterS32_R1)
    ->MeasureProcessCPUTime()
    ->Args({1 << 18, 1 << 18});
XLA_CPU_BENCHMARK(BM_ScatterS32_R2)
    ->MeasureProcessCPUTime()
    ->Args({1 << 9, 1 << 9});
XLA_CPU_BENCHMARK(BM_ScatterS32_R3)
    ->MeasureProcessCPUTime()
    ->Args({1 << 6, 1 << 6});

XLA_CPU_BENCHMARK(BM_SimpleScatterReduceF32_R3)
    ->MeasureProcessCPUTime()
    ->ArgNames({"d0", "d1", "d2", "num_slices"})
    ->Args({1, 64, 8, 1})
    ->Args({50, 64, 8, 10})
    ->Args({500, 64, 8, 100});

}  // namespace
}  // namespace xla::cpu
