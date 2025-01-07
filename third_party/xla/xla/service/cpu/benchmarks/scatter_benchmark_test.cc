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
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::cpu {

namespace {

// Generate random indices to be used in the scatter.
// They must be unique so we randomly select from the range [0, slice_size)
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

// For simplicity all these benchamrks use square operands and only scatter on
// the first dimension.
// This may not be representative of all use cases but should work for now.

void BM_ScatterS32_R1(benchmark::State& state) {
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
          index_vector_dim=1,
          unique_indices=true
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
      {{"$d0", absl::StrCat(d0)}, {"$slice_size", absl::StrCat(slice_size)}}));

  state.SetComplexityN(state.range(1));
}

void BM_ScatterS32_R2(benchmark::State& state) {
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
          index_vector_dim=1,
          unique_indices=true
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
                            {"$slice_size", absl::StrCat(slice_size)}}));
}

void BM_ScatterS32_R3(benchmark::State& state) {
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
          index_vector_dim=1,
          unique_indices=true
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
                            {"$slice_size", absl::StrCat(slice_size)}}));
}

// these all have the same number of elements in the operand
// (2^18) == (2^9)^2 == (2^6)^3
BENCHMARK(BM_ScatterS32_R1)->MeasureProcessCPUTime()->Args({1 << 18, 1 << 18});
BENCHMARK(BM_ScatterS32_R2)->MeasureProcessCPUTime()->Args({1 << 9, 1 << 9});
BENCHMARK(BM_ScatterS32_R3)->MeasureProcessCPUTime()->Args({1 << 6, 1 << 6});

}  // namespace
}  // namespace xla::cpu
