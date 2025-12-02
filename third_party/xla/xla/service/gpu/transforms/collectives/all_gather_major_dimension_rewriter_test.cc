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

#include "xla/service/gpu/transforms/collectives/all_gather_major_dimension_rewriter.h"

#include <optional>

#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla {
namespace gpu {
namespace {

class AllGatherMajorDimensionRewriterTest
    : public HloHardwareIndependentTestBase {
 protected:
  void CheckRewrite(absl::string_view hlo,
                    std::optional<absl::string_view> expected) {
    RunAndFilecheckHloRewrite(hlo, AllGatherMajorDimensionRewriter(), expected);
  }
};

TEST_F(AllGatherMajorDimensionRewriterTest,
       AllGatherToNonDefaultLayoutIsTransformedCorrectly) {
  CheckRewrite(R"(
e {
  a = s4[3,7,4]{2,1,0} parameter(0)
  c1 = s4[3,7,4]{1,0,2} copy(a)
  ag = s4[3,7,8]{1,0,2} all-gather(c1), dimensions={2}
  c2 = s4[3,7,8]{0,2,1} copy(ag)
})",
               R"(
// CHECK: %[[AG:.*]] = s4[6,7,4]{2,1,0} all-gather(%a), replica_groups={}, dimensions={0}
// CHECK: %[[BITCAST1:.*]] = s4[2,3,7,4]{3,2,1,0} bitcast(%[[AG]])
// CHECK: %[[COPY:.*]] = s4[2,3,7,4]{1,3,0,2} copy(%[[BITCAST1]])
// CHECK: ROOT %{{.*}} = s4[3,7,8]{0,2,1} bitcast(%[[COPY]])
)");
}

TEST_F(AllGatherMajorDimensionRewriterTest,
       AllGatherOnMiddleDimensionIsTransformedCorrectly) {
  CheckRewrite(R"(
e {
  a = s4[8,16,32]{2,1,0} parameter(0)
  c1 = s4[8,16,32]{2,0,1} copy(a)
  ag = s4[8,64,32]{2,0,1} all-gather(c1), dimensions={1}
  c2 = s4[8,64,32]{2,1,0} copy(ag)
})",
               R"(
// CHECK: %[[AG:.*]] = s4[32,16,32]{2,1,0} all-gather(%a), replica_groups={}, dimensions={0}
// CHECK: %[[BITCAST1:.*]] = s4[4,8,16,32]{3,2,1,0} bitcast(%[[AG]])
// CHECK: %[[COPY:.*]] = s4[4,8,16,32]{3,2,0,1} copy(%[[BITCAST1]])
// CHECK: ROOT %{{.*}} = s4[8,64,32]{2,1,0} bitcast(%[[COPY]])
)");
}

TEST_F(AllGatherMajorDimensionRewriterTest,
       AllGatherFromNonDefaultLayoutIsTransformedCorrectly) {
  CheckRewrite(R"(
e {
  a = s4[3,7,4]{2,0,1} parameter(0)
  c1 = s4[3,7,4]{1,0,2} copy(a)
  ag = s4[3,7,8]{1,0,2} all-gather(c1), dimensions={2}
  c2 = s4[3,7,8]{2,1,0} copy(ag)
})",
               R"(
// CHECK: %[[AG:.*]] = s4[3,14,4]{2,0,1} all-gather(%a), replica_groups={}, dimensions={1}
// CHECK: %[[BITCAST1:.*]] = s4[3,2,7,4]{3,0,2,1} bitcast(%[[AG]])
// CHECK: %[[COPY:.*]] = s4[3,2,7,4]{3,1,2,0} copy(%[[BITCAST1]])
// CHECK: ROOT %{{.*}} = s4[3,7,8]{2,1,0} bitcast(%[[COPY]])
)");
}

TEST_F(AllGatherMajorDimensionRewriterTest,
       AllGatherFromNonDefaultLayoutIsTransformedCorrectly2) {
  CheckRewrite(R"(
e {
a = s4[3,7,4]{2,0,1} parameter(0)
c1 = s4[3,7,4]{0,1,2} copy(a)
ag = s4[3,7,8]{0,1,2} all-gather(c1), dimensions={2}
c2 = s4[3,7,8]{2,1,0} copy(ag)
})",
               R"(
// CHECK: %[[AG:.*]] = s4[3,14,4]{2,0,1} all-gather(%a), replica_groups={}, dimensions={1}
// CHECK: %[[BITCAST1:.*]] = s4[3,2,7,4]{3,0,2,1} bitcast(%[[AG]])
// CHECK: %[[COPY:.*]] = s4[3,2,7,4]{3,1,2,0} copy(%[[BITCAST1]])
// CHECK: ROOT %{{.*}} = s4[3,7,8]{2,1,0} bitcast(%[[COPY]])
)");
}

TEST_F(AllGatherMajorDimensionRewriterTest, VariadicAllGatherIsNotSupported) {
  CheckRewrite(R"(
e {
  a = f32[8,16]{1,0} parameter(0)
  ac = f32[8,16]{0,1} copy(a)
  b = s32[8,16]{0,1} parameter(1)
  bc = s32[8,16]{0,1} copy(b)
  ag = (f32[8,64]{0,1}, s32[8,64]{0,1}) all-gather(ac, bc), dimensions={1}
})",
               std::nullopt);
}

TEST_F(AllGatherMajorDimensionRewriterTest, AllGatherOperandHasToBeCopy) {
  CheckRewrite(R"(
e {
  a = s4[3,7,4]{2,1,0} parameter(0)
  ag = s4[3,7,8]{2,1,0} all-gather(a), dimensions={2}
  c2 = s4[3,7,8]{2,1,0} copy(ag)
})",
               std::nullopt);
}

TEST_F(AllGatherMajorDimensionRewriterTest, AllGatherHasToBeTheOnlyUser) {
  CheckRewrite(R"(
e {
  a = s4[3,7,4]{2,1,0} parameter(0)
  c1 = s4[3,7,4]{1,0,2} copy(a)
  ag = s4[3,7,8]{1,0,2} all-gather(c1), dimensions={2}
  c2 = s4[3,7,8]{2,1,0} copy(ag)
  s = s4[3,7,4]{1,0,2} copy(c1)
  r = (s4[3,7,8]{2,1,0}, s4[3,7,4]{1,0,2}) tuple(c2, s)
})",
               std::nullopt);
}

TEST_F(AllGatherMajorDimensionRewriterTest, AllGatherNeedsSingleCopyUser) {
  CheckRewrite(R"(
e {
  a = f32[3,7,4]{2,1,0} parameter(0)
  c1 = f32[3,7,4]{1,0,2} copy(a)
  ag = f32[3,7,8]{1,0,2} all-gather(c1), dimensions={2}
  c2 = f32[3,7,8]{2,1,0} copy(ag)
  s = f32[3,7,8]{1,0,2} exponential(ag)
  r = (f32[3,7,8]{2,1,0}, f32[3,7,8]{1,0,2}) tuple(c2, s)
})",
               std::nullopt);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
