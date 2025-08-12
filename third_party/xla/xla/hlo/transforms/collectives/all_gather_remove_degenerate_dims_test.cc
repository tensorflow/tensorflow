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

#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include "xla/hlo/transforms/collectives/all_gather_remove_degenerate_dims.h"

#include <optional>

#include <gtest/gtest.h>
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla {
namespace {

using AllGatherRemoveDegenerateDimsTest = HloHardwareIndependentTestBase;

TEST_F(AllGatherRemoveDegenerateDimsTest, RemovesDegenerateDims) {
  RunAndFilecheckHloRewrite(R"(
      ENTRY entry {
        // CHECK:      %[[P0:.*]] = f32[1,4,1,128]{3,2,1,0} parameter(0)
        // CHECK-NEXT: %[[RESHAPE:.*]] = f32[4,128]{1,0} reshape(%[[P0]])
        // CHECK-NEXT: %[[ALLGATHER:.*]] = f32[16,128]{1,0}
        // CHECK-SAME:     all-gather(%[[RESHAPE]]){{.*}}, dimensions={0}
        // CHECK-NEXT: ROOT {{.*}} = f32[1,16,1,128]{3,2,1,0}
        // CHECK-SAME:     reshape(%[[ALLGATHER]])
        %p0 = f32[1,4,1,128] parameter(0)
        ROOT %all_gather = f32[1,16,1,128] all-gather(p0), dimensions={1}
      })",
                            AllGatherRemoveDegenerateDims());
}

TEST_F(AllGatherRemoveDegenerateDimsTest,
       DoesNotRemoveGatherDimInDegenerateAllGather) {
  RunAndFilecheckHloRewrite(R"(
      ENTRY entry {
        // CHECK:      %[[P0:.*]] = f32[1,1,128]{2,1,0} parameter(0)
        // CHECK-NEXT: %[[RESHAPE:.*]] = f32[1,128]{1,0} reshape(%[[P0]])
        // CHECK-NEXT: %[[ALLGATHER:.*]] = f32[1,128]{1,0}
        // CHECK-SAME:     all-gather(%[[RESHAPE]]){{.*}}, dimensions={0}
        // CHECK-NEXT: ROOT {{.*}} = f32[1,1,128]{2,1,0}
        // CHECK-SAME:     reshape(%[[ALLGATHER]])
        %p0 = f32[1,1,128] parameter(0)
        ROOT %all_gather = f32[1,1,128] all-gather(p0), dimensions={1}
      })",
                            AllGatherRemoveDegenerateDims());
}

TEST_F(AllGatherRemoveDegenerateDimsTest,
       DoesNotChangeFullyDegenerateAllGather) {
  RunAndFilecheckHloRewrite(R"(
      ENTRY entry {
        %p0 = f32[1,128] parameter(0)
        ROOT %all_gather = f32[1,128] all-gather(p0), dimensions={0}
      })",
                            AllGatherRemoveDegenerateDims(), std::nullopt);
}

TEST_F(AllGatherRemoveDegenerateDimsTest, KeepsAllGatherWithConstrainedLayout) {
  RunAndFilecheckHloRewrite(R"(
      ENTRY entry {
        %p0 = f32[1,4,1,128] parameter(0)
        ROOT %all_gather = f32[1,16,1,128] all-gather(p0), dimensions={1},
            constrain_layout=true
      })",
                            AllGatherRemoveDegenerateDims(), std::nullopt);
}

TEST_F(AllGatherRemoveDegenerateDimsTest, DropsDimensionsFromTupleAllGather) {
  RunAndFilecheckHloRewrite(R"(
      ENTRY entry {
        // CHECK-DAG:   %[[P0:.*]] = {{.*}} parameter(0)
        // CHECK-DAG:   %[[P1:.*]] = {{.*}} parameter(1)
        // CHECK-DAG:   %[[RP0:.*]] = f32[1,128]{1,0} reshape(%[[P0]])
        // CHECK-DAG:   %[[RP1:.*]] = f32[1,64]{1,0} reshape(%[[P1]])
        // CHECK:       %[[AG:.*]] = (f32[4,128]{1,0}, f32[4,64]{1,0})
        // CHECK-SAME:      all-gather(%[[RP0]], %[[RP1]])
        // CHECK-DAG:   %[[T0:.*]] = {{.*}} get-tuple-element(%[[AG]]), index=0
        // CHECK-DAG:   %[[T1:.*]] = {{.*}} get-tuple-element(%[[AG]]), index=1
        // CHECK-DAG:   %[[RT0:.*]] = f32[1,4,128]{2,1,0} reshape(%[[T0]])
        // CHECK-DAG:   %[[RT1:.*]] = f32[1,4,64,1]{3,2,1,0} reshape(%[[T1]])
        // CHECK:       ROOT {{.*}} tuple(%[[RT0]], %[[RT1]])

        %p0 = f32[1,1,128] parameter(0)
        %p1 = f32[1,1,64,1] parameter(1)
        ROOT %all_gather = (f32[1,4,128], f32[1,4,64,1]) all-gather(p0, p1),
          dimensions={1}
      })",
                            AllGatherRemoveDegenerateDims());
}

TEST_F(AllGatherRemoveDegenerateDimsTest, OnlyDropsMixedMinorDims) {
  RunAndFilecheckHloRewrite(R"(
      ENTRY entry {
        // CHECK-DAG:   %[[P0:.*]] = {{.*}} parameter(0)
        // CHECK-DAG:   %[[P1:.*]] = {{.*}} parameter(1)
        // CHECK-DAG:   %[[RP0:.*]] = f32[1,1,64]{2,1,0} reshape(%[[P0]])
        // CHECK-DAG:   %[[RP1:.*]] = f32[2,1,64]{2,1,0} reshape(%[[P1]])
        // CHECK:       %[[AG:.*]] = (f32[1,4,64]{2,1,0}, f32[2,4,64]{2,1,0})
        // CHECK-SAME:      all-gather(%[[RP0]], %[[RP1]])
        // CHECK-DAG:   %[[T0:.*]] = {{.*}} get-tuple-element(%[[AG]]), index=0
        // CHECK-DAG:   %[[T1:.*]] = {{.*}} get-tuple-element(%[[AG]]), index=1
        // CHECK-DAG:   %[[RT0:.*]] = f32[1,4,1,64]{3,2,1,0} reshape(%[[T0]])
        // CHECK-DAG:   %[[RT1:.*]] = f32[2,4,64,1]{3,2,1,0} reshape(%[[T1]])
        // CHECK:       ROOT {{.*}} tuple(%[[RT0]], %[[RT1]])

        %p0 = f32[1,1,1,64] parameter(0)
        %p1 = f32[2,1,64,1] parameter(1)
        ROOT %all_gather = (f32[1,4,1,64], f32[2,4,64,1]) all-gather(p0, p1),
          dimensions={1}
      })",
                            AllGatherRemoveDegenerateDims());
}

}  // namespace
}  // namespace xla
