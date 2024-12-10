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

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::cpu {

static void BM_TopKCustomCall_F32(benchmark::State& state) {
  int64_t k = state.range(0);
  int64_t batch = state.range(1);
  int64_t length = state.range(2);
  CHECK_LE(k, length);

  absl::string_view hlo = R"(
    HloModule topk_custom_call

    ENTRY test {
      x = f32[$batch,$length] parameter(0)
      ROOT topk = (f32[$batch,$k], s32[$batch,$k]) custom-call(x),
            custom_call_target="TopK"
    }
  )";

  // Fixed seed to avoid too inconsistent runs
  std::minstd_rand0 engine(/*seed=*/0xCAFEFEED);
  auto x = LiteralUtil::CreateRandomLiteral<F32>(
               ShapeUtil::MakeShape(F32, {batch, length}), &engine, 1.0f, 0.1f)
               .value();

  CHECK_OK(RunHloBenchmark(state, hlo, {&x},
                           {{"$batch", absl::StrCat(batch)},
                            {"$length", absl::StrCat(length)},
                            {"$k", absl::StrCat(k)}}));
}

static void BM_TopK_BF16(benchmark::State& state) {
  int64_t k = state.range(0);
  int64_t batch = state.range(1);
  int64_t length = state.range(2);
  CHECK_LE(k, length);

  absl::string_view hlo = R"(
    HloModule topk

    ENTRY test {
      x = bf16[$batch,$length] parameter(0)
      ROOT topk = (bf16[$batch,$k], s32[$batch,$k]) topk(x), k=$k, largest=true
    }
  )";

  // Fixed seed to avoid too inconsistent runs
  std::minstd_rand0 engine(/*seed=*/0xCAFEFEED);
  auto x = LiteralUtil::CreateRandomLiteral<BF16>(
               ShapeUtil::MakeShape(BF16, {batch, length}), &engine, 1.0f, 0.1f)
               .value();

  CHECK_OK(RunHloBenchmark(state, hlo, {&x},
                           {{"$batch", absl::StrCat(batch)},
                            {"$length", absl::StrCat(length)},
                            {"$k", absl::StrCat(k)}}));
}

#define BENCHMARK_TOPK(name)               \
  BENCHMARK(name)                          \
      ->MeasureProcessCPUTime()            \
      ->ArgNames({"k", "batch", "length"}) \
      ->Args({4, 4, 64})                   \
      ->Args({4, 16, 16})                  \
      ->Args({4, 64, 4})                   \
      ->Args({16, 4, 64})                  \
      ->Args({16, 16, 16})                 \
      ->Args({16, 64, 16})                 \
      ->Args({64, 4, 64})                  \
      ->Args({64, 16, 64})                 \
      ->Args({64, 64, 64})

BENCHMARK_TOPK(BM_TopKCustomCall_F32);
BENCHMARK_TOPK(BM_TopK_BF16);

}  // namespace xla::cpu
