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
                                 std::optional<absl::string_view> expected,
                                 bool enable_enzyme_comms_opt = false) {
    auto config = GetModuleConfigForTest();
    config.mutable_debug_options().set_xla_enable_enzyme_comms_opt(
        enable_enzyme_comms_opt);
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

TEST_F(CollectivePermuteCSETest, NestedSliceContained) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[4,100] parameter(0)
  large = f32[4,50] slice(p0), slice={[0:4], [0:50]}
  c1 = f32[4,50] collective-permute(large), channel_id=1, source_target_pairs={{0,1}}
  levels = f32[2,100] slice(p0), slice={[1:3], [0:100]}
  small = f32[1,50] slice(levels), slice={[0:1], [0:50]}
  c2 = f32[1,50] collective-permute(small), channel_id=2, source_target_pairs={{0,1}}
  ROOT root = (f32[4,50], f32[1,50]) tuple(c1, c2)
}
)";
  const absl::string_view expected = R"(
// CHECK: %[[LARGE:.*]] = {{.*}} slice(%{{.*}}), slice={[0:4], [0:50]}
// CHECK: %[[C1:.*]] = {{.*}} collective-permute(%[[LARGE]])
// CHECK: %[[SLICE:.*]] = f32[1,50]{{.*}} slice(%[[C1]]), slice={[1:2], [0:50]}
// CHECK: ROOT {{.*}} tuple(%[[C1]], %[[SLICE]])
)";
  CheckCollectivePermuteCSE(hlo_string, expected);
}

TEST_F(CollectivePermuteCSETest, NestedSlicesSameWindow) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[4,100] parameter(0)
  w1 = f32[1,100] slice(p0), slice={[1:2], [0:100]}
  c1 = f32[1,100] collective-permute(w1), channel_id=1, source_target_pairs={{0,1}}
  levels = f32[2,100] slice(p0), slice={[1:3], [0:100]}
  w2 = f32[1,100] slice(levels), slice={[0:1], [0:100]}
  c2 = f32[1,100] collective-permute(w2), channel_id=2, source_target_pairs={{0,1}}
  ROOT root = (f32[1,100], f32[1,100]) tuple(c1, c2)
}
)";
  const absl::string_view expected = R"(
// CHECK: %[[C1:.*]] = {{.*}} collective-permute(
// CHECK-NOT: collective-permute(
// CHECK: ROOT {{.*}} tuple(%[[C1]], %[[C1]])
)";
  CheckCollectivePermuteCSE(hlo_string, expected);
}

TEST_F(CollectivePermuteCSETest, AdjacentWindowsUnionIsWholeArray) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[4,100] parameter(0)
  w1 = f32[3,100] slice(p0), slice={[0:3], [0:100]}
  w2 = f32[3,100] slice(p0), slice={[1:4], [0:100]}
  c1 = f32[3,100] collective-permute(w1), channel_id=1, source_target_pairs={{0,1}}
  c2 = f32[3,100] collective-permute(w2), channel_id=2, source_target_pairs={{0,1}}
  ROOT root = (f32[3,100], f32[3,100]) tuple(c1, c2)
}
)";
  const absl::string_view expected = R"(
// CHECK: %[[P0:.*]] = f32[4,100]{{.*}} parameter(0)
// CHECK: %[[CP:.*]] = f32[4,100]{{.*}} collective-permute(%[[P0]])
// CHECK-NOT: collective-permute(
// CHECK-DAG: %[[A:.*]] = f32[3,100]{{.*}} slice(%[[CP]]), slice={[0:3], [0:100]}
// CHECK-DAG: %[[B:.*]] = f32[3,100]{{.*}} slice(%[[CP]]), slice={[1:4], [0:100]}
// CHECK: ROOT {{.*}} tuple(%[[A]], %[[B]])
)";
  CheckCollectivePermuteCSE(hlo_string, expected);
}

TEST_F(CollectivePermuteCSETest, AdjacentWindowsUnionIsSlice) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[4,100] parameter(0)
  w1 = f32[1,50] slice(p0), slice={[0:1], [0:50]}
  w2 = f32[1,50] slice(p0), slice={[1:2], [0:50]}
  c1 = f32[1,50] collective-permute(w1), channel_id=1, source_target_pairs={{0,1}}
  c2 = f32[1,50] collective-permute(w2), channel_id=2, source_target_pairs={{0,1}}
  ROOT root = (f32[1,50], f32[1,50]) tuple(c1, c2)
}
)";
  const absl::string_view expected = R"(
// CHECK: %[[P0:.*]] = f32[4,100]{{.*}} parameter(0)
// CHECK: %[[U:.*]] = f32[2,50]{{.*}} slice(%[[P0]]), slice={[0:2], [0:50]}
// CHECK: %[[CP:.*]] = f32[2,50]{{.*}} collective-permute(%[[U]])
// CHECK-NOT: collective-permute(
// CHECK-DAG: %[[A:.*]] = f32[1,50]{{.*}} slice(%[[CP]]), slice={[0:1], [0:50]}
// CHECK-DAG: %[[B:.*]] = f32[1,50]{{.*}} slice(%[[CP]]), slice={[1:2], [0:50]}
// CHECK: ROOT {{.*}} tuple(%[[A]], %[[B]])
)";
  CheckCollectivePermuteCSE(hlo_string, expected);
}

TEST_F(CollectivePermuteCSETest, ThreeWindowsUnion) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[4,100] parameter(0)
  w1 = f32[1,100] slice(p0), slice={[0:1], [0:100]}
  w2 = f32[1,100] slice(p0), slice={[1:2], [0:100]}
  w3 = f32[3,100] slice(p0), slice={[1:4], [0:100]}
  c1 = f32[1,100] collective-permute(w1), channel_id=1, source_target_pairs={{0,1}}
  c2 = f32[1,100] collective-permute(w2), channel_id=2, source_target_pairs={{0,1}}
  c3 = f32[3,100] collective-permute(w3), channel_id=3, source_target_pairs={{0,1}}
  ROOT root = (f32[1,100], f32[1,100], f32[3,100]) tuple(c1, c2, c3)
}
)";
  const absl::string_view expected = R"(
// CHECK: %[[P0:.*]] = f32[4,100]{{.*}} parameter(0)
// CHECK: %[[CP:.*]] = f32[4,100]{{.*}} collective-permute(%[[P0]])
// CHECK-NOT: collective-permute(
// CHECK-DAG: %[[A:.*]] = f32[1,100]{{.*}} slice(%[[CP]]), slice={[0:1], [0:100]}
// CHECK-DAG: %[[B:.*]] = f32[1,100]{{.*}} slice(%[[CP]]), slice={[1:2], [0:100]}
// CHECK-DAG: %[[C:.*]] = f32[3,100]{{.*}} slice(%[[CP]]), slice={[1:4], [0:100]}
// CHECK: ROOT {{.*}} tuple(%[[A]], %[[B]], %[[C]])
)";
  CheckCollectivePermuteCSE(hlo_string, expected);
}

TEST_F(CollectivePermuteCSETest, DisjointWindowsNotMerged) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[4,100] parameter(0)
  w1 = f32[1,100] slice(p0), slice={[0:1], [0:100]}
  w2 = f32[1,100] slice(p0), slice={[2:3], [0:100]}
  c1 = f32[1,100] collective-permute(w1), channel_id=1, source_target_pairs={{0,1}}
  c2 = f32[1,100] collective-permute(w2), channel_id=2, source_target_pairs={{0,1}}
  ROOT root = (f32[1,100], f32[1,100]) tuple(c1, c2)
}
)";
  CheckCollectivePermuteCSE(hlo_string, std::nullopt);
}

TEST_F(CollectivePermuteCSETest, WindowsDifferingInTwoDimsNotMerged) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[4,100] parameter(0)
  w1 = f32[1,50] slice(p0), slice={[0:1], [0:50]}
  w2 = f32[1,50] slice(p0), slice={[1:2], [50:100]}
  c1 = f32[1,50] collective-permute(w1), channel_id=1, source_target_pairs={{0,1}}
  c2 = f32[1,50] collective-permute(w2), channel_id=2, source_target_pairs={{0,1}}
  ROOT root = (f32[1,50], f32[1,50]) tuple(c1, c2)
}
)";
  CheckCollectivePermuteCSE(hlo_string, std::nullopt);
}

TEST_F(CollectivePermuteCSETest, AdjacentWindowsDifferentPairsNotMerged) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[4,100] parameter(0)
  w1 = f32[1,100] slice(p0), slice={[0:1], [0:100]}
  w2 = f32[1,100] slice(p0), slice={[1:2], [0:100]}
  c1 = f32[1,100] collective-permute(w1), channel_id=1, source_target_pairs={{0,1}}
  c2 = f32[1,100] collective-permute(w2), channel_id=2, source_target_pairs={{1,0}}
  ROOT root = (f32[1,100], f32[1,100]) tuple(c1, c2)
}
)";
  CheckCollectivePermuteCSE(hlo_string, std::nullopt);
}

TEST_F(CollectivePermuteCSETest, StridedSliceChainNotComposed) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[8,100] parameter(0)
  even = f32[4,100] slice(p0), slice={[0:8:2], [0:100]}
  w1 = f32[1,100] slice(even), slice={[0:1], [0:100]}
  w2 = f32[1,100] slice(p0), slice={[1:2], [0:100]}
  c1 = f32[1,100] collective-permute(w1), channel_id=1, source_target_pairs={{0,1}}
  c2 = f32[1,100] collective-permute(w2), channel_id=2, source_target_pairs={{0,1}}
  ROOT root = (f32[1,100], f32[1,100]) tuple(c1, c2)
}
)";
  CheckCollectivePermuteCSE(hlo_string, std::nullopt);
}

TEST_F(CollectivePermuteCSETest, PairedHiLoSlicesFromX64SplitFused) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = u64[4,100] parameter(0)
  lo = f32[4,100] custom-call(p0), custom_call_target="X64SplitLow"
  hi = f32[4,100] custom-call(p0), custom_call_target="X64SplitHigh"
  s_lo = f32[4,2] slice(lo), slice={[0:4], [0:2]}
  s_hi = f32[4,2] slice(hi), slice={[0:4], [0:2]}
  c_lo = f32[4,2] collective-permute(s_lo), channel_id=1, source_target_pairs={{0,1}}
  c_hi = f32[4,2] collective-permute(s_hi), channel_id=2, source_target_pairs={{0,1}}
  ROOT root = (f32[4,2], f32[4,2]) tuple(c_lo, c_hi)
}
)";
  const absl::string_view expected = R"(
// CHECK: %[[CONCAT:.*]] = f32[8,2]{{.*}} concatenate(%{{.*}}, %{{.*}}), dimensions={0}
// CHECK: %[[CP:.*]] = f32[8,2]{{.*}} collective-permute(%[[CONCAT]])
// CHECK-NOT: collective-permute(
// CHECK-DAG: %[[A:.*]] = f32[4,2]{{.*}} slice(%[[CP]]), slice={[0:4], [0:2]}
// CHECK-DAG: %[[B:.*]] = f32[4,2]{{.*}} slice(%[[CP]]), slice={[4:8], [0:2]}
// CHECK: ROOT {{.*}} tuple(%[[A]], %[[B]])
)";
  CheckCollectivePermuteCSE(hlo_string, expected,
                            /*enable_enzyme_comms_opt=*/true);
}

TEST_F(CollectivePermuteCSETest, PairedHiLoSlicesFromFastTwoSumFused) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  a = f32[4,100] parameter(0)
  b = f32[4,100] parameter(1)
  cond = pred[4,100] parameter(2)
  zero = f32[4,100] parameter(3)
  hi_raw = f32[4,100] add(a, b)
  t = f32[4,100] subtract(hi_raw, a)
  lo_raw = f32[4,100] subtract(b, t)
  hi = f32[4,100] select(cond, hi_raw, zero)
  lo = f32[4,100] select(cond, lo_raw, zero)
  s_hi = f32[4,2] slice(hi), slice={[0:4], [0:2]}
  s_lo = f32[4,2] slice(lo), slice={[0:4], [0:2]}
  c_hi = f32[4,2] collective-permute(s_hi), channel_id=1, source_target_pairs={{0,1}}
  c_lo = f32[4,2] collective-permute(s_lo), channel_id=2, source_target_pairs={{0,1}}
  ROOT root = (f32[4,2], f32[4,2]) tuple(c_hi, c_lo)
}
)";
  const absl::string_view expected = R"(
// CHECK: %[[CONCAT:.*]] = f32[8,2]{{.*}} concatenate(%{{.*}}, %{{.*}}), dimensions={0}
// CHECK: %[[CP:.*]] = f32[8,2]{{.*}} collective-permute(%[[CONCAT]])
// CHECK-NOT: collective-permute(
// CHECK-DAG: %[[A:.*]] = f32[4,2]{{.*}} slice(%[[CP]]), slice={[0:4], [0:2]}
// CHECK-DAG: %[[B:.*]] = f32[4,2]{{.*}} slice(%[[CP]]), slice={[4:8], [0:2]}
// CHECK: ROOT {{.*}} tuple(%[[A]], %[[B]])
)";
  CheckCollectivePermuteCSE(hlo_string, expected,
                            /*enable_enzyme_comms_opt=*/true);
}

TEST_F(CollectivePermuteCSETest, PairedHiLoSlicesFromKnuthTwoSumFused) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  lhs = f32[4,100] parameter(0)
  rhs = f32[4,100] parameter(1)
  hi = f32[4,100] add(lhs, rhs)
  rhs_virtual = f32[4,100] subtract(hi, lhs)
  lhs_virtual = f32[4,100] subtract(hi, rhs_virtual)
  lhs_error = f32[4,100] subtract(lhs, lhs_virtual)
  rhs_error = f32[4,100] subtract(rhs, rhs_virtual)
  lo = f32[4,100] add(lhs_error, rhs_error)
  s_hi = f32[4,2] slice(hi), slice={[0:4], [0:2]}
  s_lo = f32[4,2] slice(lo), slice={[0:4], [0:2]}
  c_hi = f32[4,2] collective-permute(s_hi), channel_id=1, source_target_pairs={{0,1}}
  c_lo = f32[4,2] collective-permute(s_lo), channel_id=2, source_target_pairs={{0,1}}
  ROOT root = (f32[4,2], f32[4,2]) tuple(c_hi, c_lo)
}
)";
  const absl::string_view expected = R"(
// CHECK: %[[CONCAT:.*]] = f32[8,2]{{.*}} concatenate(%{{.*}}, %{{.*}}), dimensions={0}
// CHECK: %[[CP:.*]] = f32[8,2]{{.*}} collective-permute(%[[CONCAT]])
// CHECK-NOT: collective-permute(
// CHECK-DAG: %[[A:.*]] = f32[4,2]{{.*}} slice(%[[CP]]), slice={[0:4], [0:2]}
// CHECK-DAG: %[[B:.*]] = f32[4,2]{{.*}} slice(%[[CP]]), slice={[4:8], [0:2]}
// CHECK: ROOT {{.*}} tuple(%[[A]], %[[B]])
)";
  CheckCollectivePermuteCSE(hlo_string, expected,
                            /*enable_enzyme_comms_opt=*/true);
}

TEST_F(CollectivePermuteCSETest,
       PairedHiLoSlicesFromAdjacentGetTupleElementFused) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  t = (f32[4,100], f32[4,100]) parameter(0)
  hi = f32[4,100] get-tuple-element(t), index=0
  lo = f32[4,100] get-tuple-element(t), index=1
  s_hi = f32[4,2] slice(hi), slice={[0:4], [98:100]}
  s_lo = f32[4,2] slice(lo), slice={[0:4], [98:100]}
  c_hi = f32[4,2] collective-permute(s_hi), channel_id=1, source_target_pairs={{0,1}}
  c_lo = f32[4,2] collective-permute(s_lo), channel_id=2, source_target_pairs={{0,1}}
  ROOT root = (f32[4,2], f32[4,2]) tuple(c_hi, c_lo)
}
)";
  const absl::string_view expected = R"(
// CHECK: %[[CONCAT:.*]] = f32[8,2]{{.*}} concatenate(%{{.*}}, %{{.*}}), dimensions={0}
// CHECK: %[[CP:.*]] = f32[8,2]{{.*}} collective-permute(%[[CONCAT]])
// CHECK-NOT: collective-permute(
// CHECK-DAG: %[[A:.*]] = f32[4,2]{{.*}} slice(%[[CP]]), slice={[0:4], [0:2]}
// CHECK-DAG: %[[B:.*]] = f32[4,2]{{.*}} slice(%[[CP]]), slice={[4:8], [0:2]}
// CHECK: ROOT {{.*}} tuple(%[[A]], %[[B]])
)";
  CheckCollectivePermuteCSE(hlo_string, expected,
                            /*enable_enzyme_comms_opt=*/true);
}

TEST_F(CollectivePermuteCSETest, PairedHiLoSlicesNotFusedWhenFlagDisabled) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = u64[4,100] parameter(0)
  lo = f32[4,100] custom-call(p0), custom_call_target="X64SplitLow"
  hi = f32[4,100] custom-call(p0), custom_call_target="X64SplitHigh"
  s_lo = f32[4,2] slice(lo), slice={[0:4], [0:2]}
  s_hi = f32[4,2] slice(hi), slice={[0:4], [0:2]}
  c_lo = f32[4,2] collective-permute(s_lo), channel_id=1, source_target_pairs={{0,1}}
  c_hi = f32[4,2] collective-permute(s_hi), channel_id=2, source_target_pairs={{0,1}}
  ROOT root = (f32[4,2], f32[4,2]) tuple(c_lo, c_hi)
}
)";
  CheckCollectivePermuteCSE(hlo_string, std::nullopt,
                            /*enable_enzyme_comms_opt=*/false);
}

TEST_F(CollectivePermuteCSETest, UnpairedParametersNotFused) {
  const absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[4,100] parameter(0)
  p1 = f32[4,100] parameter(1)
  slice0 = f32[4,2] slice(p0), slice={[0:4], [0:2]}
  slice1 = f32[4,2] slice(p1), slice={[0:4], [0:2]}
  c0 = f32[4,2] collective-permute(slice0), channel_id=1, source_target_pairs={{0,1}}
  c1 = f32[4,2] collective-permute(slice1), channel_id=2, source_target_pairs={{0,1}}
  ROOT root = (f32[4,2], f32[4,2]) tuple(c0, c1)
}
)";
  CheckCollectivePermuteCSE(hlo_string, std::nullopt,
                            /*enable_enzyme_comms_opt=*/true);
}

}  // namespace
}  // namespace xla
