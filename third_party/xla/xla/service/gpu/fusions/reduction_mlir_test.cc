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
#include "xla/service/gpu/fusions/reduction_mlir.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/error_spec.h"
#include "xla/service/gpu/fusions/mlir_emitter_test_base.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::SizeIs;

template <typename EmitterType>
class ReductionTest : public MlirEmitterTestBase<EmitterType> {
 protected:
  absl::Status TestBijection(const IndexingMap& map,
                             absl::Span<int64_t const> shape) {
    std::vector<Interval> intervals;
    for (int64_t size : shape) {
      intervals.push_back({0, size - 1});
    }
    auto status = VerifyBijection(map, intervals);
    if (status.ok()) return status;
    return absl::FailedPreconditionError(
        absl::StrCat(status.message(), " in map ", map.ToString()));
  }
};

using MlirMultiRowReductionTest = ReductionTest<MlirMultiRowReductionFusion>;

constexpr auto kMultiRowReductionX8 = R"(
    Add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }
    fused_computation {
      param_0 = f32[1024,4] parameter(0)
      param_1 = f32[] parameter(1)
      ROOT reduce = f32[1024] reduce(param_0, param_1), dimensions={1}, to_apply=Add
    }
    ENTRY main {
      a = f32[1024,4] parameter(0)
      c = f32[] constant(0)
      ROOT fusion = f32[1024] fusion(a, c), kind=kInput, calls=fused_computation
    })";

constexpr auto kMultiRowReductionX2VectorX4 = R"(
    or {
      tmp_0 = pred[] parameter(0)
      tmp_1 = pred[] parameter(1)
      ROOT tmp_2 = pred[] or(tmp_0, tmp_1)
    }

    fusion {
      tmp_0 = f32[76800,16]{1,0} parameter(0)
      tmp_1 = f32[] constant(-1.70141173e+38)
      tmp_2 = f32[76800,16]{1,0} broadcast(tmp_1), dimensions={}
      tmp_3 = pred[76800,16]{1,0} compare(tmp_0, tmp_2), direction=GT
      tmp_4 = pred[] constant(false)
      tmp_5 = pred[76800]{0} reduce(tmp_3, tmp_4), dimensions={1}, to_apply=or
      tmp_6 = f32[76800,16]{1,0} parameter(1)
      tmp_7 = pred[76800,16]{1,0} compare(tmp_6, tmp_2), direction=GT
      tmp_8 = pred[76800]{0} reduce(tmp_7, tmp_4), dimensions={1}, to_apply=or
      ROOT tmp_9 = (pred[76800]{0}, pred[76800]{0}) tuple(tmp_5, tmp_8)
    }

    ENTRY main {
      p0 = f32[76800,16]{1,0} parameter(0)
      p1 = f32[76800,16]{1,0} parameter(1)

      ROOT fusion = (pred[76800]{0}, pred[76800]{0}) fusion(p0, p1), kind=kInput, calls=fusion
    })";

constexpr auto kMultiRowReductionX16VectorX2 = R"(
    or {
      tmp_0 = pred[] parameter(0)
      tmp_1 = pred[] parameter(1)
      ROOT tmp_2 = pred[] or(tmp_0, tmp_1)
    }

    fusion {
      p0 = pred[76800,2] parameter(0)
      c0 = pred[] constant(false)
      ROOT reduce = pred[76800] reduce(p0, c0), dimensions={1}, to_apply=or
    }

    ENTRY main {
      p0 = pred[76800,2] parameter(0)
      ROOT fusion = pred[76800] fusion(p0), kind=kInput, calls=fusion
    })";

TEST_F(MlirMultiRowReductionTest, MultiRowReductionIndexing) {
  auto fusion = GetEmitter(kMultiRowReductionX8);

  TF_EXPECT_OK(TestBijection(
      *fusion->ComputeThreadIdToInputIndexing(0, 0, &mlir_context_),
      {1024, 4}));
  TF_EXPECT_OK(TestBijection(
      *fusion->ComputeThreadIdToOutputIndexing(0, &mlir_context_), {1024}));
  EXPECT_EQ(Product(GetLoopTripCounts(
                *fusion->ComputeThreadIdToInputIndexing(0, 0, &mlir_context_))),
            1);
}

TEST_F(MlirMultiRowReductionTest, MultiRowReductionIr) {
  // Multi-row reductions don't use shared memory.
  TF_ASSERT_OK(EmitAndCheckIR(kMultiRowReductionX8, R"(
    // CHECK: shuffle_reduce {{.*}} to 2
    // CHECK-NOT: allocate_shared
  )"));
}

TEST_F(MlirMultiRowReductionTest, MultiRowReductionCorrectness) {
  EXPECT_TRUE(RunAndCompareNoHloPasses(kMultiRowReductionX8, ErrorSpec{1e-3}));
}

TEST_F(MlirMultiRowReductionTest, TwoGroups) {
  auto module = ParseAndReturnVerifiedModule(R"(
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    fusion {
      %p0 = f32[2] parameter(0)
      %p1 = f32[2] parameter(1)
      %c0 = f32[] constant(-inf)
      %r0 = f32[] reduce(%p0, %c0), dimensions={0}, to_apply=add
      %c1 = f32[] constant(inf)
      %r1 = f32[] reduce(%p1, %c1), dimensions={0}, to_apply=add
      ROOT %tuple = (f32[], f32[]) tuple(%r0, %r1)
    }
    ENTRY entry {
      %p0 = f32[2] parameter(0)
      %p1 = f32[2] parameter(1)
      ROOT %fusion = (f32[], f32[]) fusion(%p0, %p1), kind=kInput, calls=fusion
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = HloFusionAnalysis::Create(*root, device_info_);
  MlirMultiRowReductionFusion fusion(analysis);

  EXPECT_THAT(fusion.GetGroups().grouped_roots,
              ElementsAre(ElementsAre(&analysis.fusion_root(0).instruction()),
                          ElementsAre(&analysis.fusion_root(1).instruction())));
}

TEST_F(MlirMultiRowReductionTest, OneGroup) {
  auto module = ParseAndReturnVerifiedModule(R"(
    %add {
      %p0 = c128[] parameter(0)
      %p1 = c128[] parameter(1)
      ROOT %add.35 = c128[] add(c128[] %p0, c128[] %p1)
    }
    %fusion {
      %p0 = c128[1,2] parameter(0)
      %c0 = c128[] constant((0, 0))
      %reduce = c128[] reduce(%p0, %c0), dimensions={0,1}, to_apply=%add
      %real = f64[] real(c128[] %reduce)
      %imag = f64[] imag(c128[] %reduce)
      %negate = f64[] negate(f64[] %imag)
      ROOT %tuple.29 = (f64[], f64[]) tuple(f64[] %real, f64[] %negate)
    }
    ENTRY entry {
      %p0 = c128[1,2] parameter(0)
      ROOT %fusion = (f64[], f64[]) fusion(%p0), kind=kInput, calls=fusion
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = HloFusionAnalysis::Create(*root, device_info_);

  MlirMultiRowReductionFusion mlir_fusion(analysis);
  EXPECT_THAT(mlir_fusion.GetGroups().grouped_roots, SizeIs(1));
}

TEST_F(MlirMultiRowReductionTest, VectorizedX4Indexing) {
  auto fusion = GetEmitter(kMultiRowReductionX2VectorX4);

  TF_EXPECT_OK(TestBijection(
      *fusion->ComputeThreadIdToInputIndexing(0, 0, &mlir_context_),
      {76800, 16}));
  TF_EXPECT_OK(TestBijection(
      *fusion->ComputeThreadIdToOutputIndexing(0, &mlir_context_), {76800}));
  EXPECT_THAT(GetLoopTripCounts(*fusion->ComputeThreadIdToInputIndexing(
                  0, 0, &mlir_context_)),
              ElementsAre(1 /* major reduced */, 4 /* vector size */));
}

TEST_F(MlirMultiRowReductionTest, LimitedVectorizationCorrectness) {
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kMultiRowReductionX16VectorX2, ErrorSpec{1e-3}));
}

TEST_F(MlirMultiRowReductionTest, VectorizedX4Correctness) {
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kMultiRowReductionX2VectorX4, ErrorSpec{1e-3}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
