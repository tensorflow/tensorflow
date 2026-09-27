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

#include "xla/hlo/transforms/collectives/collective_permute_cse.h"

#include <optional>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

class CollectivePermuteCSETest : public HloHardwareIndependentTestBase {
 public:
  void CheckCollectivePermuteCSE(absl::string_view hlo,
                                 std::optional<absl::string_view> expected) {
    auto config = GetModuleConfigForTest();
    RunAndFilecheckHloRewrite(
        hlo, CollectivePermuteCSE{}, expected, [](HloModule*) {}, &config);
  }
};

TEST_F(CollectivePermuteCSETest, RemovesIdenticalPermutes) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[100] parameter(0)
  c1 = f32[100] collective-permute(p0), channel_id=1, source_target_pairs={{0,1}}
  c2 = f32[100] collective-permute(p0), channel_id=1, source_target_pairs={{0,1}}
  ROOT root = (f32[100], f32[100]) tuple(c1, c2)
}
)";
  const absl::string_view expected = R"(
// CHECK: %[[P0:.*]] = {{.*}} parameter(0)
// CHECK: %[[C1:.*]] = {{.*}} collective-permute(%[[P0]])
// CHECK: ROOT {{.*}} tuple(%[[C1]], %[[C1]])
)";
  CheckCollectivePermuteCSE(hlo_string, expected);
}

TEST_F(CollectivePermuteCSETest, SortedPairsMatch) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[100] parameter(0)
  c1 = f32[100] collective-permute(p0), channel_id=1, source_target_pairs={{0,1}, {1,2}}
  c2 = f32[100] collective-permute(p0), channel_id=1, source_target_pairs={{1,2}, {0,1}}
  ROOT root = (f32[100], f32[100]) tuple(c1, c2)
}
)";
  const absl::string_view expected = R"(
// CHECK: %[[P0:.*]] = {{.*}} parameter(0)
// CHECK: %[[C1:.*]] = {{.*}} collective-permute(%[[P0]])
// CHECK: ROOT {{.*}} tuple(%[[C1]], %[[C1]])
)";
  CheckCollectivePermuteCSE(hlo_string, expected);
}

TEST_F(CollectivePermuteCSETest, SubsetPermuteSlices) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[100] parameter(0)
  slice0 = f32[50] slice(p0), slice={[0:50]}
  c1 = f32[100] collective-permute(p0), channel_id=1, source_target_pairs={{0,1}}
  c2 = f32[50] collective-permute(slice0), channel_id=1, source_target_pairs={{0,1}}
  ROOT root = (f32[100], f32[50]) tuple(c1, c2)
}
)";
  const absl::string_view expected = R"(
// CHECK: %[[P0:.*]] = {{.*}} parameter(0)
// CHECK: %[[C1:.*]] = {{.*}} collective-permute(%[[P0]])
// CHECK: %[[SLICE:.*]] = {{.*}} slice(%[[C1]])
// CHECK: ROOT {{.*}} tuple(%[[C1]], %[[SLICE]])
)";
  CheckCollectivePermuteCSE(hlo_string, expected);
}

TEST_F(CollectivePermuteCSETest, SubsetPermuteSlicesSmallBeforeLarge) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[100] parameter(0)
  slice0 = f32[50] slice(p0), slice={[0:50]}
  c2 = f32[50] collective-permute(slice0), channel_id=1, source_target_pairs={{0,1}}
  c1 = f32[100] collective-permute(p0), channel_id=1, source_target_pairs={{0,1}}
  ROOT root = (f32[100], f32[50]) tuple(c1, c2)
}
)";
  const absl::string_view expected = R"(
// CHECK: %[[P0:.*]] = {{.*}} parameter(0)
// CHECK: %[[C1:.*]] = {{.*}} collective-permute(%[[P0]])
// CHECK: %[[SLICE:.*]] = {{.*}} slice(%[[C1]])
// CHECK: ROOT {{.*}} tuple(%[[C1]], %[[SLICE]])
)";
  CheckCollectivePermuteCSE(hlo_string, expected);
}

TEST_F(CollectivePermuteCSETest, SubsetPermuteSlicesSmallBeforeLargeMultiple) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[100] parameter(0)
  slice0 = f32[50] slice(p0), slice={[0:50]}
  c2 = f32[50] collective-permute(slice0), channel_id=1, source_target_pairs={{0,1}}
  c1 = f32[100] collective-permute(p0), channel_id=1, source_target_pairs={{0,1}}
  c3 = f32[100] collective-permute(p0), channel_id=1, source_target_pairs={{0,1}}
  ROOT root = (f32[100], f32[50], f32[100]) tuple(c1, c2, c3)
}
)";
  const absl::string_view expected = R"(
// CHECK: %[[P0:.*]] = {{.*}} parameter(0)
// CHECK: %[[C1:.*]] = {{.*}} collective-permute(%[[P0]])
// CHECK: %[[SLICE:.*]] = {{.*}} slice(%[[C1]])
// CHECK: ROOT {{.*}} tuple(%[[C1]], %[[SLICE]], %[[C1]])
)";
  CheckCollectivePermuteCSE(hlo_string, expected);
}

TEST_F(CollectivePermuteCSETest,
       SubsetPermuteSlicesSmallBeforeLargeReachability) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[100] parameter(0)
  slice0 = f32[50] slice(p0), slice={[0:50]}
  c2 = f32[50] collective-permute(slice0), channel_id=1, source_target_pairs={{0,1}}
  c1 = f32[100] collective-permute(p0), channel_id=1, source_target_pairs={{0,1}}, control-predecessors={c2}
  ROOT root = (f32[100], f32[50]) tuple(c1, c2)
}
)";
  const absl::string_view expected = R"(
// CHECK: %[[P0:.*]] = {{.*}} parameter(0)
// CHECK: %[[C1:.*]] = {{.*}} collective-permute(%[[P0]])
// CHECK: %[[SLICE:.*]] = {{.*}} slice(%[[C1]])
// CHECK: ROOT {{.*}} tuple(%[[C1]], %[[SLICE]])
)";
  CheckCollectivePermuteCSE(hlo_string, expected);
}

TEST_F(CollectivePermuteCSETest, SubsetPermuteTwoSlices) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[100] parameter(0)
  slice_large = f32[80] slice(p0), slice={[10:90]}
  slice_small = f32[50] slice(p0), slice={[20:70]}
  c_large = f32[80] collective-permute(slice_large), channel_id=1, source_target_pairs={{0,1}}
  c_small = f32[50] collective-permute(slice_small), channel_id=1, source_target_pairs={{0,1}}
  ROOT root = (f32[80], f32[50]) tuple(c_large, c_small)
}
)";
  const absl::string_view expected = R"(
// CHECK: %[[P0:.*]] = {{.*}} parameter(0)
// CHECK: %[[S1:.*]] = {{.*}} slice(%[[P0]]), slice={[10:90]}
// CHECK: %[[C1:.*]] = {{.*}} collective-permute(%[[S1]])
// CHECK: %[[S2:.*]] = {{.*}} slice(%[[C1]])
// CHECK: ROOT {{.*}} tuple(%[[C1]], %[[S2]])
)";
  CheckCollectivePermuteCSE(hlo_string, expected);
}

}  // namespace
}  // namespace xla
