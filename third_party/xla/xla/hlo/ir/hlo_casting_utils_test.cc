/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_casting_utils.h"

#include <memory>

#include <gtest/gtest.h>
#include "benchmark/benchmark.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

std::unique_ptr<HloInstruction> CreateCP() {
  Shape shape = ShapeUtil::MakeShape(F32, {4, 4});
  std::unique_ptr<HloInstruction> p0 =
      HloInstruction::CreateParameter(0, shape, "param");
  std::unique_ptr<HloInstruction> cp =
      HloInstruction::CreateCollectivePermute(shape, p0.get(), {{0, 1}}, 1);
  cp->SetAndSanitizeName("test_cp");
  return cp;
}

const char* kWrongCastError =
    ".*ClassOf.*'test_cp'.*HloCollectivePermuteInstruction.*"
    "HloAllReduceInstruction.*";
const char* kNullptrError = ".*nullptr.*";

TEST(HloCastingUtilsTest, Cast) {
  std::unique_ptr<HloInstruction> cp = CreateCP();
  EXPECT_NE(Cast<HloCollectivePermuteInstruction>(cp.get()), nullptr);
  EXPECT_DEATH(Cast<HloAllReduceInstruction>(cp.get()), kWrongCastError);
  cp.reset();
  EXPECT_DEATH(Cast<HloCollectivePermuteInstruction>(cp.get()), kNullptrError);
}

TEST(HloCastingUtilsTest, CastConst) {
  std::unique_ptr<const HloInstruction> cp = CreateCP();
  EXPECT_NE(Cast<const HloCollectivePermuteInstruction>(cp.get()), nullptr);
  EXPECT_DEATH(Cast<const HloAllReduceInstruction>(cp.get()), kWrongCastError);
  cp.reset();
  EXPECT_DEATH(Cast<const HloCollectivePermuteInstruction>(cp.get()),
               kNullptrError);
}

TEST(HloCastingUtilsTest, DynCast) {
  std::unique_ptr<HloInstruction> cp = CreateCP();
  EXPECT_NE(DynCast<HloCollectivePermuteInstruction>(cp.get()), nullptr);
  EXPECT_EQ(DynCast<HloAllReduceInstruction>(cp.get()), nullptr);
  cp.reset();
  EXPECT_DEATH(DynCast<HloCollectivePermuteInstruction>(cp.get()),
               kNullptrError);
}

TEST(HloCastingUtilsTest, DynCastConst) {
  std::unique_ptr<const HloInstruction> cp = CreateCP();
  EXPECT_NE(DynCast<const HloCollectivePermuteInstruction>(cp.get()), nullptr);
  EXPECT_EQ(DynCast<const HloAllReduceInstruction>(cp.get()), nullptr);
  cp.reset();
  EXPECT_DEATH(DynCast<const HloCollectivePermuteInstruction>(cp.get()),
               kNullptrError);
}

void BM_Cast(benchmark::State& state) {
  std::unique_ptr<HloInstruction> cp = CreateCP();
  for (auto s : state) {
    HloInstruction* source = cp.get();
    HloCollectivePermuteInstruction* casted =
        Cast<HloCollectivePermuteInstruction>(source);
    benchmark::DoNotOptimize(casted);
  }
}

void BM_Cast_Const(benchmark::State& state) {
  std::unique_ptr<const HloInstruction> cp = CreateCP();
  for (auto s : state) {
    const HloInstruction* source = cp.get();
    const HloCollectivePermuteInstruction* casted =
        Cast<HloCollectivePermuteInstruction>(source);
    benchmark::DoNotOptimize(casted);
  }
}

void BM_DynCast(benchmark::State& state) {
  std::unique_ptr<HloInstruction> cp = CreateCP();
  for (auto s : state) {
    HloCollectivePermuteInstruction* casted =
        DynCast<HloCollectivePermuteInstruction>(cp.get());
    benchmark::DoNotOptimize(casted);
  }
}

void BM_DynCast_Const(benchmark::State& state) {
  std::unique_ptr<const HloInstruction> cp = CreateCP();
  for (auto s : state) {
    const HloCollectivePermuteInstruction* casted =
        DynCast<HloCollectivePermuteInstruction>(cp.get());
    benchmark::DoNotOptimize(casted);
  }
}

void BM_dynamic_cast(benchmark::State& state) {
  std::unique_ptr<HloInstruction> cp = CreateCP();
  for (auto s : state) {
    HloCollectivePermuteInstruction* casted =
        dynamic_cast<HloCollectivePermuteInstruction*>(cp.get());
    benchmark::DoNotOptimize(casted);
  }
}

void BM_static_cast(benchmark::State& state) {
  std::unique_ptr<HloInstruction> cp = CreateCP();
  for (auto s : state) {
    HloCollectivePermuteInstruction* casted =
        static_cast<HloCollectivePermuteInstruction*>(cp.get());
    benchmark::DoNotOptimize(casted);
  }
}

void BM_down_cast(benchmark::State& state) {
  std::unique_ptr<HloInstruction> cp = CreateCP();
  for (auto s : state) {
    HloCollectivePermuteInstruction* casted =
        tsl::down_cast<HloCollectivePermuteInstruction*>(cp.get());
    benchmark::DoNotOptimize(casted);
  }
}

// reference benchmarks
BENCHMARK(BM_dynamic_cast);
BENCHMARK(BM_static_cast);
BENCHMARK(BM_down_cast);

BENCHMARK(BM_Cast);
BENCHMARK(BM_Cast_Const);
BENCHMARK(BM_DynCast);
BENCHMARK(BM_DynCast_Const);

}  // namespace
}  // namespace xla
