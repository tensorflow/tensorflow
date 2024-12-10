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
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::cpu {

static void BM_FusionF32(benchmark::State& state) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule fusion_f32_$d0

    ENTRY e {
      p0 = f32[1,2,1,$d0,256] parameter(0)
      p1 = f32[1,2,1,$d0,256] parameter(1)
      p2 = f32[] parameter(2)
      c1 = f32[] constant(1)
      bcast = f32[1,2,1,$d0,256] broadcast(p2), dimensions={}
      multiply = f32[1,2,1,$d0,256] multiply(bcast, p1)
      subtract = f32[] subtract(c1, p2)
      bcast1 = f32[1,2,1,$d0,256] broadcast(subtract), dimensions={}
      multiply1 = f32[1,2,1,$d0,256] multiply(p0, p0)
      multiply2 = f32[1,2,1,$d0,256] multiply(bcast1, multiply1)
      ROOT add = f32[1,2,1,$d0,256] add(multiply, multiply2)
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(F32, {1, 2, 1, d0, 256});
  auto scalar = ShapeUtil::MakeShape(F32, {});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);
  auto p1 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);
  auto p2 = *LiteralUtil::CreateRandomLiteral<F32>(scalar, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0, &p1, &p2};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}));
}

static void BM_FusionF32_2(benchmark::State& state) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule fusion_f32_2_$d0

    ENTRY e {
      p0 = f32[$d0,1] parameter(0)
      p1 = f32[$d0,1] parameter(1)
      p2 = f32[$d0,1] parameter(2)
      p3 = s32[$d0,1] parameter(3)
      p4 = s32[$d0,1] parameter(4)
      p5 = f32[7,6] parameter(5)
      p6 = f32[7,3] parameter(6)
      c0 = s32[] constant(0)
      c1 = s32[] constant(7)
      c2 = s32[] constant(6)
      c3 = f32[] constant(nan)
      bcast0 = s32[$d0,1] broadcast(c0), dimensions={}
      cmp0 = pred[$d0,1] compare(p3, bcast0), direction=LT
      bcast1 = s32[$d0,1] broadcast(c1), dimensions={}
      add0 = s32[$d0,1] add(p3, bcast1)
      select0 = s32[$d0,1] select(cmp0, add0, p3)
      reshape0 = s32[$d0,1,1] reshape(select0)
      bcast2 = s32[$d0,1,1] broadcast(c0), dimensions={}
      cmp1 = pred[$d0,1,1] compare(reshape0, bcast2), direction=GE
      bcast3 = s32[$d0,1,1] broadcast(c2), dimensions={}
      cmp2 = pred[$d0,1,1] compare(reshape0, bcast3), direction=LE
      and0 = pred[$d0,1,1] and(cmp1, cmp2)
      reshape1 = pred[$d0] reshape(and0)
      bcast4 = pred[$d0,1,6] broadcast(reshape1), dimensions={0}
      gather0 = f32[$d0,1,6] gather(p5, reshape0), offset_dims={2},
        collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=2,
        slice_sizes={1,6}
      bcast5 = f32[$d0,1,6] broadcast(c3), dimensions={}
      select1 = f32[$d0,1,6] select(bcast4, gather0, bcast5)
      reshape2 = f32[$d0,6] reshape(select1)
      cmp3 = pred[$d0,1] compare(p4, bcast0), direction=LT
      add1 = s32[$d0,1] add(p4, bcast1)
      select2 = s32[$d0,1] select(cmp3, add1, p4)
      reshape3 = s32[$d0,1,1] reshape(select2)
      cmp4 = pred[$d0,1,1] compare(reshape3, bcast2), direction=GE
      cmp5 = pred[$d0,1,1] compare(reshape3, bcast3), direction=LE
      and1 = pred[$d0,1,1] and(cmp4, cmp5)
      reshape4 = pred[$d0] reshape(and1)
      bcast6 = pred[$d0,1,3] broadcast(reshape4), dimensions={0}
      gather1 = f32[$d0,1,3] gather(p6, reshape3), offset_dims={2},
        collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=2,
        slice_sizes={1,3}
      bcast7 = f32[$d0,1,3] broadcast(c3), dimensions={}
      select3 = f32[$d0,1,3] select(bcast6, gather1, bcast7)
      reshape5 = f32[$d0,3] reshape(select3)
      ROOT concat = f32[$d0,12] concatenate(p0, p1, p2, reshape2, reshape5),
        dimensions={1}
    }
  )";

  std::minstd_rand0 engine;

  auto shape0 = ShapeUtil::MakeShape(F32, {d0, 1});
  auto shape1 = ShapeUtil::MakeShape(S32, {d0, 1});
  auto shape2 = ShapeUtil::MakeShape(F32, {7, 6});
  auto shape3 = ShapeUtil::MakeShape(F32, {7, 3});

  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape0, &engine, 1.0f, 0.1f);
  auto p1 = *LiteralUtil::CreateRandomLiteral<F32>(shape0, &engine, 1.0f, 0.1f);
  auto p2 = *LiteralUtil::CreateRandomLiteral<F32>(shape0, &engine, 1.0f, 0.1f);
  auto p3 = *LiteralUtil::CreateRandomLiteral<S32>(shape1, &engine, 1.0f, 0.1f);
  auto p4 = *LiteralUtil::CreateRandomLiteral<S32>(shape1, &engine, 1.0f, 0.1f);
  auto p5 = *LiteralUtil::CreateRandomLiteral<F32>(shape2, &engine, 1.0f, 0.1f);
  auto p6 = *LiteralUtil::CreateRandomLiteral<F32>(shape3, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0, &p1, &p2, &p3, &p4, &p5, &p6};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}));
}

static void BM_BcastFusionF32(benchmark::State& state) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule fusion_f32_$d0

    ENTRY e {
      p0 = f32[1,2,1,$d0,256] parameter(0)
      p1 = f32[] parameter(1)
      bcast = f32[1,2,1,$d0,256] broadcast(p1), dimensions={}
      ROOT multiply = f32[1,2,1,$d0,256] multiply(bcast, p0)
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(F32, {1, 2, 1, d0, 256});
  auto scalar = ShapeUtil::MakeShape(F32, {});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);
  auto p1 = *LiteralUtil::CreateRandomLiteral<F32>(scalar, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0, &p1};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}));
}

static void BM_DynamicUpdateSliceFusionF32(benchmark::State& state) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule dynamic_update_slice_fusion_f32_$d0

    ENTRY e {
      p0 = f32[$d0,256] parameter(0)
      p1 = s32[] parameter(1)
      p2 = s32[] parameter(2)
      slice = f32[1,1] dynamic-slice(p0, p1, p2), dynamic_slice_sizes={1,1}
      add = f32[1,1] add(slice, slice)
      ROOT update = f32[$d0,256] dynamic-update-slice(p0, add, p1, p2)
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(F32, {d0, 256});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);
  auto p1 = LiteralUtil::CreateR0<int32_t>(0);
  auto p2 = LiteralUtil::CreateR0<int32_t>(0);

  std::vector<const Literal*> args = {&p0, &p1, &p2};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}));
}

static void BM_ChainOfAddF32(benchmark::State& state) {
  int64_t size = state.range(0);

  // In this benchmark we create a chain of additions starting from `p2` and
  // ending with `p$size`. The chain is fused into a single fusion node.
  absl::string_view hlo = R"(
    HloModule chain_of_add_f32_$size

    ENTRY e {
      p0 = f32[3844] parameter(0)
      p1 = f32[3844] parameter(1)
      $parameters
      $additions
      bcast_p0 = f32[12,3844] broadcast(p0), dimensions={1}
      bcast_p1 = f32[12,3844] broadcast(p1), dimensions={1}
      bcast_add = f32[12,3844] broadcast(add$size), dimensions={0}
      mul = f32[12,3844] multiply(bcast_p0, bcast_add)
      ROOT sub = f32[12,3844] subtract(mul, bcast_p1)
    }
  )";

  // Initialize [`p2`, `p$size`] parameters.
  std::string parameters;
  for (int i = 2; i <= size; ++i) {
    parameters += absl::StrFormat("\n p%d = f32[12] parameter(%d)", i, i);
  }

  // Create a chain of additions starting from `p2`.
  std::string additions = "add2 = f32[12] add(p2, p2)";
  for (int i = 3; i <= size; ++i) {
    additions +=
        absl::StrFormat("\n add%d = f32[12] add(add%d, p%d)", i, i - 1, i);
  }

  std::minstd_rand0 engine;

  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(
      ShapeUtil::MakeShape(F32, {3844}), &engine, 1.0f, 0.1f);
  auto p1 = *LiteralUtil::CreateRandomLiteral<F32>(
      ShapeUtil::MakeShape(F32, {3844}), &engine, 1.0f, 0.1f);
  auto pN = *LiteralUtil::CreateRandomLiteral<F32>(
      ShapeUtil::MakeShape(F32, {12}), &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0, &p1};
  for (int i = 2; i <= size; ++i) args.push_back(&pN);

  CHECK_OK(RunHloBenchmark(state, hlo, args,
                           {{"$size", absl::StrCat(size)},
                            {"$parameters", parameters},
                            {"$additions", additions}}));
}

BENCHMARK(BM_FusionF32)
    ->MeasureProcessCPUTime()
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(8192)
    ->Arg(16384);

BENCHMARK(BM_FusionF32_2)
    ->MeasureProcessCPUTime()
    ->Arg(40)
    ->Arg(80)
    ->Arg(160)
    ->Arg(240);

BENCHMARK(BM_BcastFusionF32)
    ->MeasureProcessCPUTime()
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(8192)
    ->Arg(16384);

BENCHMARK(BM_DynamicUpdateSliceFusionF32)
    ->MeasureProcessCPUTime()
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(8192)
    ->Arg(16384);

BENCHMARK(BM_ChainOfAddF32)
    ->MeasureProcessCPUTime()
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024);

}  // namespace xla::cpu
