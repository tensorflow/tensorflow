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

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/array2d.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::cpu {

static void BM_GatherS32(benchmark::State& state) {
  int64_t d0 = state.range(0);
  int64_t d1 = state.range(1);
  int64_t slice_size = state.range(2);

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
  CHECK_OK(RunHloBenchmark(state, hlo, args,
                           {{"$d0", absl::StrCat(d0)},
                            {"$d1", absl::StrCat(d1)},
                            {"$slice_size", absl::StrCat(slice_size)}}));
}

BENCHMARK(BM_GatherS32)
    ->MeasureProcessCPUTime()
    ->Args({3, 3, 1})
    ->Args({3, 3, 2})
    ->Args({3, 3, 4})
    ->Args({3, 32, 1})
    ->Args({3, 32, 2})
    ->Args({3, 32, 8})
    ->Args({3, 64, 1})
    ->Args({3, 64, 2})
    ->Args({3, 64, 16})
    ->Args({3, 128, 1})
    ->Args({3, 128, 2})
    ->Args({3, 128, 32})
    ->Args({3, 256, 1})
    ->Args({3, 256, 2})
    ->Args({3, 256, 64})
    ->Args({3, 512, 1})
    ->Args({3, 512, 2})
    ->Args({3, 512, 128})
    ->Args({10, 3, 1})
    ->Args({10, 3, 2})
    ->Args({10, 3, 4})
    ->Args({10, 32, 1})
    ->Args({10, 32, 2})
    ->Args({10, 32, 8})
    ->Args({10, 64, 1})
    ->Args({10, 64, 2})
    ->Args({10, 64, 16})
    ->Args({10, 128, 1})
    ->Args({10, 128, 2})
    ->Args({10, 128, 32})
    ->Args({10, 256, 1})
    ->Args({10, 256, 2})
    ->Args({10, 256, 64})
    ->Args({10, 512, 1})
    ->Args({10, 512, 2})
    ->Args({10, 512, 128})
    ->Args({100, 3, 1})
    ->Args({100, 3, 2})
    ->Args({100, 3, 4})
    ->Args({100, 32, 1})
    ->Args({100, 32, 2})
    ->Args({100, 32, 8})
    ->Args({100, 64, 1})
    ->Args({100, 64, 2})
    ->Args({100, 64, 16})
    ->Args({100, 128, 1})
    ->Args({100, 128, 2})
    ->Args({100, 128, 32})
    ->Args({100, 256, 1})
    ->Args({100, 256, 2})
    ->Args({100, 256, 64})
    ->Args({100, 512, 1})
    ->Args({100, 512, 2})
    ->Args({100, 512, 128});

}  // namespace xla::cpu
