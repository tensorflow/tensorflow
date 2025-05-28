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

#include <memory>
#include <optional>

#include <gmock/gmock.h>
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

}  // namespace
}  // namespace xla
