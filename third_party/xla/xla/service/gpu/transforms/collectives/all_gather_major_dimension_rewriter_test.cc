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
       AllGatherOnMajorDimensionIsSkipped) {
  CheckRewrite(R"(
e {
a = s4[8,16,32] parameter(0)
ag = s4[32,16,32] all-gather(a), dimensions={0}
})",
               std::nullopt);
}

TEST_F(AllGatherMajorDimensionRewriterTest,
       AllGatherOnNonDefaultLayoutIsSkipped) {
  CheckRewrite(R"(
e {
  a = s4[3,3]{0,1} parameter(0)
  b = s8[3,3]{0,1} parameter(1)
  ag = (s4[3,3], s8[3,3]) all-gather(a, b), dimensions={1}
})",
               std::nullopt);
}

TEST_F(AllGatherMajorDimensionRewriterTest,
       AllGatherOnLastDimensionIsTransformedCorrectly) {
  CheckRewrite(R"(
e {
  a = s4[3,7,5] parameter(0)
  ag = s4[3,7,20] all-gather(a), dimensions={2}
})",
               R"(
// CHECK:      [[a_0:%[^ ]+]] = s4[3,7,5]{2,1,0} parameter(0)
// CHECK-NEXT: [[ag_1:%[^ ]+]] = s4[12,7,5]{2,1,0} all-gather([[a_0]]), replica_groups={{.*}}, dimensions={0}
// CHECK-NEXT: [[reshape_2:%[^ ]+]] = s4[4,3,7,5]{3,2,1,0} reshape([[ag_1]])
// CHECK-NEXT: [[transpose_3:%[^ ]+]] = s4[3,7,4,5]{3,2,1,0} transpose([[reshape_2]]), dimensions={1,2,0,3}
// CHECK-NEXT: ROOT [[reshape_4:%[^ ]+]] = s4[3,7,20]{2,1,0} reshape([[transpose_3]])
)");
}

TEST_F(AllGatherMajorDimensionRewriterTest,
       AllGatherOnMiddleDimensionIsTransformedCorrectly) {
  CheckRewrite(R"(
e {
  a = s4[8,16,32] parameter(0)
  ag = s4[8,64,32] all-gather(a), dimensions={1}
})",
               R"(
// CHECK:      [[a_0:%[^ ]+]] = s4[8,16,32]{2,1,0} parameter(0)
// CHECK-NEXT: [[ag_1:%[^ ]+]] = s4[32,16,32]{2,1,0} all-gather([[a_0]]), replica_groups={{.*}}, dimensions={0}
// CHECK-NEXT: [[reshape_2:%[^ ]+]] = s4[4,8,16,32]{3,2,1,0} reshape([[ag_1]])
// CHECK-NEXT: [[transpose_3:%[^ ]+]] = s4[8,4,16,32]{3,2,1,0} transpose([[reshape_2]]), dimensions={1,0,2,3}
// CHECK-NEXT: ROOT [[reshape_4:%[^ ]+]] = s4[8,64,32]{2,1,0} reshape([[transpose_3]])
)");
}

TEST_F(AllGatherMajorDimensionRewriterTest,
       VariadicAllGatherIsTransformedCorrectly) {
  CheckRewrite(R"(
e {
  a = f32[8,16] parameter(0)
  b = s32[8,16] parameter(1)
  ag = (f32[8,64], s32[8,64]) all-gather(a, b), dimensions={1}
})",
               R"(
// CHECK:      [[a_0:%[^ ]+]] = f32[8,16]{1,0} parameter(0)
// CHECK-NEXT: [[b_1:%[^ ]+]] = s32[8,16]{1,0} parameter(1)
// CHECK-NEXT: [[ag_2:%[^ ]+]] = (f32[32,16]{1,0}, s32[32,16]{1,0}) all-gather([[a_0]], [[b_1]]), replica_groups={{.*}}, dimensions={0}
// CHECK-NEXT: [[gte_3:%[^ ]+]] = f32[32,16]{1,0} get-tuple-element([[ag_2]]), index=0
// CHECK-NEXT: [[reshape_4:%[^ ]+]] = f32[4,8,16]{2,1,0} reshape([[gte_3]])
// CHECK-NEXT: [[transpose_5:%[^ ]+]] = f32[8,4,16]{2,1,0} transpose([[reshape_4]]), dimensions={1,0,2}
// CHECK-NEXT: [[reshape_6:%[^ ]+]] = f32[8,64]{1,0} reshape([[transpose_5]])
// CHECK-NEXT: [[gte_7:%[^ ]+]] = s32[32,16]{1,0} get-tuple-element([[ag_2]]), index=1
// CHECK-NEXT: [[reshape_8:%[^ ]+]] = s32[4,8,16]{2,1,0} reshape([[gte_7]])
// CHECK-NEXT: [[transpose_9:%[^ ]+]] = s32[8,4,16]{2,1,0} transpose([[reshape_8]]), dimensions={1,0,2}
// CHECK-NEXT: [[reshape_10:%[^ ]+]] = s32[8,64]{1,0} reshape([[transpose_9]])
)");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
