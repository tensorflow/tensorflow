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

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/array2d.h"
#include "xla/backends/cpu/benchmarks/aot_benchmark_helper.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

static void BM_GatherS32(benchmark::State& state) {
  int64_t d0 = state.range(0);
  int64_t d1 = state.range(1);
  int64_t slice_size = state.range(2);
  bool is_aot = static_cast<bool>(state.range(3));

  absl::string_view hlo = R"(
    HloModule gather_s32_d$d0_d$d1_s$slice_size

    ENTRY e {
      operand = s32[$d0,$d1] parameter(0)
      indices = s32[$slice_size, 1] parameter(1)
      ROOT gather = s32[$slice_size, $d1] gather(operand, indices),
          offset_dims={1},
          collapsed_slice_dims={0},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={1, $d1}
    }
  )";

  std::minstd_rand0 engine;

  auto operand_shape = ShapeUtil::MakeShape(S32, {d0, d1});
  auto indices_shape = ShapeUtil::MakeShape(S32, {slice_size, 1});
  auto operand = *LiteralUtil::CreateRandomLiteral<S32>(
      operand_shape, &engine, /*mean=*/50, /*stddev=*/10);

  // Generate random indices to be used in the gather
  std::vector<int32_t> random_indices(slice_size);
  std::uniform_int_distribution<int32_t> dist(0, d0 - 1);
  absl::c_generate(random_indices, [&]() { return dist(engine); });

  // Transform the indices into a 2D array - as expected by the gather op
  Array2D<int32_t> indices_2d(slice_size, 1);
  for (int i = 0; i < slice_size; ++i) {
    indices_2d(i, 0) = random_indices[i];
  }
  auto indices = LiteralUtil::CreateR2FromArray2D(indices_2d);

  std::vector<const Literal*> args = {&operand, &indices};

  HloBenchmarkOptions benchmark_options;
  benchmark_options.aot_options = is_aot ? GetAotCompilationOptions() : nullptr;

  CHECK_OK(RunHloBenchmark(state, hlo, args,
                           {{"$d0", absl::StrCat(d0)},
                            {"$d1", absl::StrCat(d1)},
                            {"$slice_size", absl::StrCat(slice_size)}},
                           benchmark_options));
}

void GenerateGatherArgs(benchmark::internal::Benchmark* benchmark) {
  benchmark->MeasureProcessCPUTime();
  benchmark->ArgNames({"d0", "d1", "slice_size", "is_aot"});
  const std::vector<std::array<int64_t, 3>> args_values = {
      {3, 3, 1},     {3, 3, 2},      {3, 3, 4},      {3, 32, 1},
      {3, 32, 2},    {3, 32, 8},     {3, 64, 1},     {3, 64, 2},
      {3, 64, 16},   {3, 128, 1},    {3, 128, 2},    {3, 128, 32},
      {3, 256, 1},   {3, 256, 2},    {3, 256, 64},   {3, 512, 1},
      {3, 512, 2},   {3, 512, 128},  {10, 3, 1},     {10, 3, 2},
      {10, 3, 4},    {10, 32, 1},    {10, 32, 2},    {10, 32, 8},
      {10, 64, 1},   {10, 64, 2},    {10, 64, 16},   {10, 128, 1},
      {10, 128, 2},  {10, 128, 32},  {10, 256, 1},   {10, 256, 2},
      {10, 256, 64}, {10, 512, 1},   {10, 512, 2},   {10, 512, 128},
      {100, 3, 1},   {100, 3, 2},    {100, 3, 4},    {100, 32, 1},
      {100, 32, 2},  {100, 32, 8},   {100, 64, 1},   {100, 64, 2},
      {100, 64, 16}, {100, 128, 1},  {100, 128, 2},  {100, 128, 32},
      {100, 256, 1}, {100, 256, 2},  {100, 256, 64}, {100, 512, 1},
      {100, 512, 2}, {100, 512, 128}};

  const std::vector<bool> is_aot_values = {false, true};

  for (const auto& arg_value : args_values) {
    for (const auto& is_aot : is_aot_values) {
      std::vector<int64_t> all_arg_values(arg_value.begin(), arg_value.end());
      all_arg_values.push_back(is_aot);
      benchmark->Args(all_arg_values);
    }
  }
}

BENCHMARK(BM_GatherS32)->Apply(GenerateGatherArgs);

}  // namespace xla::cpu
