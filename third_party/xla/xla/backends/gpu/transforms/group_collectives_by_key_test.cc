/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/backends/gpu/transforms/group_collectives_by_key.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/transforms/explicit_collectives_group_async_wrapper.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla::gpu {
namespace {

using GroupCollectivesByKeyTest = HloHardwareIndependentTestBase;

void ExpectFileCheck(absl::string_view input, absl::string_view pattern) {
  ASSERT_OK_AND_ASSIGN(bool matches, RunFileCheck(std::string(input), pattern));
  EXPECT_TRUE(matches);
}

// Input mimics combiner output: multi-operand AG + tuple output + GTE.
TEST_F(GroupCollectivesByKeyTest, GroupsCombinedAgAndRs) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    w0 = f32[8,8] parameter(0)
    w1 = f32[8,8] parameter(1)
    g0 = f32[32,8] parameter(2)
    g1 = f32[32,8] parameter(3)
    ag = (f32[32,8], f32[32,8]) all-gather(w0, w1), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    ag0 = f32[32,8] get-tuple-element(ag), index=0
    ag1 = f32[32,8] get-tuple-element(ag), index=1
    rs = (f32[8,8], f32[8,8]) reduce-scatter(g0, g1), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g0"}
    rs0 = f32[8,8] get-tuple-element(rs), index=0
    rs1 = f32[8,8] get-tuple-element(rs), index=1
    ROOT result = (f32[32,8], f32[32,8], f32[8,8], f32[8,8])
        tuple(ag0, ag1, rs0, rs1)
  }
  )";

  const absl::string_view expected_hlo = R"(
  // CHECK-LABEL: %collectives_group (
  //       CHECK: = {{.*}} all-gather(
  //  CHECK-SAME:   frontend_attributes={collective_group_key="g0"}
  //       CHECK: = {{.*}} reduce-scatter(
  //  CHECK-SAME:   frontend_attributes={collective_group_key="g0"}

  // CHECK-LABEL: ENTRY %main
  //       CHECK: %[[START:[^ ]+]] = {{.*}} async-start
  //  CHECK-SAME:   calls=%collectives_group
  //  CHECK-SAME:   frontend_attributes={_collectives_group="",
  //  CHECK-SAME:   collective_group_key="g0"}
  //       CHECK: %[[DONE:[^ ]+]] = {{.*}} async-done(%[[START]])
  )";

  RunAndFilecheckHloRewrite(hlo_string, GroupCollectivesByKey(), expected_hlo);
}

TEST_F(GroupCollectivesByKeyTest, SkipsUnpairedKeys) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    w1 = f32[8,8] parameter(0)
    g1 = f32[32,8] parameter(1)
    w2 = f32[8,8] parameter(2)
    g2 = f32[32,8] parameter(3)
    ag1 = f32[32,8] all-gather(w1), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    rs1 = f32[8,8] reduce-scatter(g1), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g1"}
    ag2 = f32[32,8] all-gather(w2), dimensions={0},
        replica_groups={{0,1,2,3}}
    rs2 = f32[8,8] reduce-scatter(g2), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add
    ROOT result = (f32[32,8], f32[8,8], f32[32,8], f32[8,8])
        tuple(ag1, rs1, ag2, rs2)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);

  const absl::string_view expected_hlo = R"(
  // CHECK-LABEL: ENTRY %main
  //       CHECK: %ag1 = {{.*}} all-gather(
  //       CHECK: %rs1 = {{.*}} reduce-scatter(
  //       CHECK: %ag2 = {{.*}} all-gather(
  //       CHECK: %rs2 = {{.*}} reduce-scatter(
  )";
  ExpectFileCheck(module->ToString(), expected_hlo);
}

// Unannotated collectives are never grouped.
TEST_F(GroupCollectivesByKeyTest, UnannotatedCollectivesUnchanged) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    w = f32[8,8] parameter(0)
    g = f32[32,8] parameter(1)
    ag = f32[32,8] all-gather(w), dimensions={0},
        replica_groups={{0,1,2,3}}
    rs = f32[8,8] reduce-scatter(g), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add
    ROOT result = (f32[32,8], f32[8,8]) tuple(ag, rs)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);

  const absl::string_view expected_hlo = R"(
  // CHECK-LABEL: ENTRY %main
  //       CHECK: %ag = {{.*}} all-gather(
  //       CHECK: %rs = {{.*}} reduce-scatter(
  )";
  ExpectFileCheck(module->ToString(), expected_hlo);
}

// AG and RS may belong to different communicators (different replica_groups).
// NCCL's group call fuses collectives across comms, so the pass should still
// group them when the key matches.
TEST_F(GroupCollectivesByKeyTest, GroupsAcrossDifferentReplicaGroups) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    weights = f32[8,8] parameter(0)
    grads = f32[32,8] parameter(1)
    ag = f32[16,8] all-gather(weights), dimensions={0},
        replica_groups={{0,1},{2,3}},
        frontend_attributes={collective_group_key="g0"}
    rs = f32[8,8] reduce-scatter(grads), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[16,8], f32[8,8]) tuple(ag, rs)
  }
  )";

  const absl::string_view expected_hlo = R"(
  // CHECK-LABEL: %collectives_group (
  //       CHECK: = {{.*}} all-gather(
  //       CHECK: = {{.*}} reduce-scatter(

  // CHECK-LABEL: ENTRY %main
  //       CHECK: %[[START:[^ ]+]] = {{.*}} async-start
  //  CHECK-SAME:   calls=%collectives_group
  //  CHECK-SAME:   frontend_attributes={_collectives_group="",
  //  CHECK-SAME:   collective_group_key="g0"}
  //       CHECK: %[[DONE:[^ ]+]] = {{.*}} async-done(%[[START]])
  )";

  RunAndFilecheckHloRewrite(hlo_string, GroupCollectivesByKey(), expected_hlo);
}

// Two AllGathers with the same key must be paired even though neither is a
// ReduceScatter.
TEST_F(GroupCollectivesByKeyTest, GroupsTwoAllGathers) {
  const absl::string_view hlo_string = R"(
  HloModule test

  ENTRY main {
    w1 = f32[8,8] parameter(0)
    w2 = f32[8,8] parameter(1)
    ag1 = f32[16,8] all-gather(w1), dimensions={0},
        replica_groups={{0,1},{2,3}},
        frontend_attributes={collective_group_key="g0"}
    ag2 = f32[32,8] all-gather(w2), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[16,8], f32[32,8]) tuple(ag1, ag2)
  }
  )";

  const absl::string_view expected_hlo = R"(
  // CHECK-LABEL: %collectives_group (
  //       CHECK: = {{.*}} all-gather(
  //       CHECK: = {{.*}} all-gather(

  // CHECK-LABEL: ENTRY %main
  //       CHECK: %[[START:[^ ]+]] = {{.*}} async-start
  //  CHECK-SAME:   calls=%collectives_group
  //  CHECK-SAME:   frontend_attributes={_collectives_group="",
  //  CHECK-SAME:   collective_group_key="g0"}
  //       CHECK: %[[DONE:[^ ]+]] = {{.*}} async-done(%[[START]])
  )";

  RunAndFilecheckHloRewrite(hlo_string, GroupCollectivesByKey(), expected_hlo);
}

// Two AllReduces with the same key group together (all-reduce support).
TEST_F(GroupCollectivesByKeyTest, GroupsTwoAllReduces) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    x = f32[8,8] parameter(0)
    y = f32[8,8] parameter(1)
    ar1 = f32[8,8] all-reduce(x), replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g0"}
    ar2 = f32[8,8] all-reduce(y), replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[8,8], f32[8,8]) tuple(ar1, ar2)
  }
  )";

  const absl::string_view expected_hlo = R"(
  // CHECK-LABEL: %collectives_group (
  //       CHECK: = {{.*}} all-reduce(
  //       CHECK: = {{.*}} all-reduce(

  // CHECK-LABEL: ENTRY %main
  //       CHECK: %[[START:[^ ]+]] = {{.*}} async-start
  //  CHECK-SAME:   calls=%collectives_group
  //  CHECK-SAME:   frontend_attributes={_collectives_group="",
  //  CHECK-SAME:   collective_group_key="g0"}
  //       CHECK: %[[DONE:[^ ]+]] = {{.*}} async-done(%[[START]])
  )";

  RunAndFilecheckHloRewrite(hlo_string, GroupCollectivesByKey(), expected_hlo);
}

// A custom predicate narrows the eligible opcode set: here only all-gathers
// are groupable, so a same-key AG+RS pair is left ungrouped (RS is not a
// candidate, leaving the AG as an unpaired singleton).
TEST_F(GroupCollectivesByKeyTest, CustomPredicateRestrictsToAllGather) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    w = f32[8,8] parameter(0)
    g = f32[32,8] parameter(1)
    ag = f32[32,8] all-gather(w), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    rs = f32[8,8] reduce-scatter(g), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[32,8], f32[8,8]) tuple(ag, rs)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass(HloPredicateIsOp<HloOpcode::kAllGather>);
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);

  const absl::string_view expected_hlo = R"(
  // CHECK-LABEL: ENTRY %main
  //       CHECK: %ag = {{.*}} all-gather(
  //       CHECK: %rs = {{.*}} reduce-scatter(
  )";
  ExpectFileCheck(module->ToString(), expected_hlo);
}

// With the same all-gather-only predicate, two same-key all-gathers still
// group; the restriction only excludes non-matching opcodes.
TEST_F(GroupCollectivesByKeyTest, CustomPredicateStillGroupsMatchingOpcodes) {
  const absl::string_view hlo_string = R"(
  HloModule test

  ENTRY main {
    w1 = f32[8,8] parameter(0)
    w2 = f32[8,8] parameter(1)
    ag1 = f32[16,8] all-gather(w1), dimensions={0},
        replica_groups={{0,1},{2,3}},
        frontend_attributes={collective_group_key="g0"}
    ag2 = f32[32,8] all-gather(w2), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[16,8], f32[32,8]) tuple(ag1, ag2)
  }
  )";

  const absl::string_view expected_hlo = R"(
  // CHECK-LABEL: %collectives_group (
  //       CHECK: = {{.*}} all-gather(
  //       CHECK: = {{.*}} all-gather(

  // CHECK-LABEL: ENTRY %main
  //       CHECK: %[[START:[^ ]+]] = {{.*}} async-start
  //  CHECK-SAME:   calls=%collectives_group
  //  CHECK-SAME:   frontend_attributes={_collectives_group="",
  //  CHECK-SAME:   collective_group_key="g0"}
  //       CHECK: %[[DONE:[^ ]+]] = {{.*}} async-done(%[[START]])
  )";

  RunAndFilecheckHloRewrite(
      hlo_string,
      GroupCollectivesByKey(HloPredicateIsOp<HloOpcode::kAllGather>),
      expected_hlo);
}

// Three independent collectives sharing the same key fuse into a single group,
// not a pair plus a singleton.
TEST_F(GroupCollectivesByKeyTest, GroupsThreeCollectives) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    w1 = f32[8,8] parameter(0)
    w2 = f32[8,8] parameter(1)
    g1 = f32[32,8] parameter(2)
    ag1 = f32[16,8] all-gather(w1), dimensions={0},
        replica_groups={{0,1},{2,3}},
        frontend_attributes={collective_group_key="g0"}
    ag2 = f32[32,8] all-gather(w2), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    rs1 = f32[8,8] reduce-scatter(g1), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[16,8], f32[32,8], f32[8,8]) tuple(ag1, ag2, rs1)
  }
  )";

  const absl::string_view expected_hlo = R"(
  // CHECK-LABEL: %collectives_group (
  //       CHECK: = {{.*}} all-gather(
  //       CHECK: = {{.*}} all-gather(
  //       CHECK: = {{.*}} reduce-scatter(

  // CHECK-LABEL: ENTRY %main
  //       CHECK: %[[START:[^ ]+]] = {{.*}} async-start
  //  CHECK-SAME:   calls=%collectives_group
  //  CHECK-SAME:   frontend_attributes={_collectives_group="",
  //  CHECK-SAME:   collective_group_key="g0"}
  //       CHECK: %[[DONE:[^ ]+]] = {{.*}} async-done(%[[START]])
  )";

  RunAndFilecheckHloRewrite(hlo_string, GroupCollectivesByKey(), expected_hlo);
}

TEST_F(GroupCollectivesByKeyTest, MultipleKeyPairsGroupedIndependently) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    w1 = f32[8,8] parameter(0)
    g1 = f32[32,8] parameter(1)
    w2 = f32[8,8] parameter(2)
    g2 = f32[32,8] parameter(3)
    ag1 = f32[32,8] all-gather(w1), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    rs1 = f32[8,8] reduce-scatter(g1), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g0"}
    ag2 = f32[32,8] all-gather(w2), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g1"}
    rs2 = f32[8,8] reduce-scatter(g2), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g1"}
    ROOT result = (f32[32,8], f32[8,8], f32[32,8], f32[8,8])
        tuple(ag1, rs1, ag2, rs2)
  }
  )";

  const absl::string_view expected_hlo = R"(
  // CHECK-LABEL: %collectives_group (
  //       CHECK: = {{.*}} all-gather(
  //  CHECK-SAME:   frontend_attributes={collective_group_key="g0"}
  //       CHECK: = {{.*}} reduce-scatter(
  //  CHECK-SAME:   frontend_attributes={collective_group_key="g0"}

  // CHECK-LABEL: %collectives_group.1 (
  //       CHECK: = {{.*}} all-gather(
  //  CHECK-SAME:   frontend_attributes={collective_group_key="g1"}
  //       CHECK: = {{.*}} reduce-scatter(
  //  CHECK-SAME:   frontend_attributes={collective_group_key="g1"}

  // CHECK-LABEL: ENTRY %main
  //       CHECK: %[[START0:[^ ]+]] = {{.*}} async-start
  //  CHECK-SAME:   calls=%collectives_group
  //  CHECK-SAME:   frontend_attributes={_collectives_group="",
  //  CHECK-SAME:   collective_group_key="g0"}
  //       CHECK: %[[DONE0:[^ ]+]] = {{.*}} async-done(%[[START0]])
  //       CHECK: %[[START1:[^ ]+]] = {{.*}} async-start
  //  CHECK-SAME:   calls=%collectives_group.1
  //  CHECK-SAME:   frontend_attributes={_collectives_group="",
  //  CHECK-SAME:   collective_group_key="g1"}
  //       CHECK: %[[DONE1:[^ ]+]] = {{.*}} async-done(%[[START1]])
  )";

  RunAndFilecheckHloRewrite(hlo_string, GroupCollectivesByKey(), expected_hlo);
}

TEST_F(GroupCollectivesByKeyTest, PreservesWrapperProperties) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    x = f32[8,8] parameter(0), sharding={replicated}
    y = f32[8,8] parameter(1), sharding={replicated}
    ar0 = f32[8,8] all-reduce(x),
        replica_groups={{0,1}},
        to_apply=add,
        frontend_attributes={collective_group_key="g0"},
        metadata={op_name="first_collective"},
        sharding={replicated},
        backend_config={"collective_backend_config":{"is_pipelined":true}}
    ar1 = f32[8,8] all-reduce(y),
        replica_groups={{0,1}},
        to_apply=add,
        frontend_attributes={collective_group_key="g0"},
        sharding={replicated}
    ROOT result = (f32[8,8], f32[8,8]) tuple(ar0, ar1)
  }
  )";

  const absl::string_view expected_hlo = R"(
  // CHECK-LABEL: %collectives_group (
  //       CHECK: ROOT %{{[^ ]+}} = {{.*}} tuple
  //  CHECK-SAME:   sharding={
  //  CHECK-SAME:   {replicated}, {replicated}}

  // CHECK-LABEL: ENTRY %main
  //       CHECK: %[[START:[^ ]+]] = {{.*}} async-start
  //  CHECK-SAME:   calls=%collectives_group
  //  CHECK-SAME:   sharding={
  //  CHECK-SAME:   {replicated}, {replicated},
  //  CHECK-SAME:   {replicated}, {replicated}}
  //  CHECK-SAME:   frontend_attributes={_collectives_group="",
  //  CHECK-SAME:   collective_group_key="g0"}
  //  CHECK-SAME:   metadata={op_name="first_collective"}
  //  CHECK-SAME:   backend_config={"collective_backend_config":
  //  CHECK-SAME:   {"is_pipelined":true}}
  //       CHECK: %[[DONE:[^ ]+]] = {{.*}} async-done(%[[START]])
  //  CHECK-SAME:   sharding={
  //  CHECK-SAME:   {replicated}, {replicated}}
  //  CHECK-SAME:   frontend_attributes={_collectives_group="",
  //  CHECK-SAME:   collective_group_key="g0"}
  //  CHECK-SAME:   metadata={op_name="first_collective"}
  //  CHECK-SAME:   backend_config={"collective_backend_config":
  //  CHECK-SAME:   {"is_pipelined":true}}
  )";

  RunAndFilecheckHloRewrite(hlo_string, GroupCollectivesByKey(), expected_hlo);
}

TEST_F(GroupCollectivesByKeyTest, PreservesExecutionThread) {
  const absl::string_view hlo_string = R"(
  HloModule test

  ENTRY main {
    x = f32[8,8] parameter(0)
    y = f32[8,8] parameter(1)
    ag0 = f32[16,8] all-gather(x), dimensions={0},
        replica_groups={{0,1}},
        frontend_attributes={collective_group_key="g0"}
    ag1 = f32[16,8] all-gather(y), dimensions={0},
        replica_groups={{0,1}},
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[16,8], f32[16,8]) tuple(ag0, ag1)
  }, execution_thread="worker"
  )";

  const absl::string_view expected_hlo = R"(
  // CHECK-LABEL: %collectives_group (
  //       CHECK: }, execution_thread="worker"

  // CHECK-LABEL: ENTRY %main
  //       CHECK: %[[START:[^ ]+]] = {{.*}} async-start
  //  CHECK-SAME:   async_execution_thread="worker"
  //  CHECK-SAME:   calls=%collectives_group
  //  CHECK-SAME:   frontend_attributes={_collectives_group="",
  //  CHECK-SAME:   collective_group_key="g0"}
  //       CHECK: %[[DONE:[^ ]+]] = {{.*}} async-done(%[[START]])
  //       CHECK: }, execution_thread="worker"
  )";

  RunAndFilecheckHloRewrite(hlo_string, GroupCollectivesByKey(), expected_hlo);
}

// External control dependencies are relayed onto the async pair.
TEST_F(GroupCollectivesByKeyTest, PreservesExternalControlDeps) {
  const absl::string_view hlo_string = R"(
  HloModule test

  ENTRY main {
    w1 = f32[8,8] parameter(0)
    w2 = f32[8,8] parameter(1)
    barrier = f32[8,8] add(w1, w1)
    ag1 = f32[16,8] all-gather(w1), dimensions={0},
        replica_groups={{0,1},{2,3}}, control-predecessors={barrier},
        frontend_attributes={collective_group_key="g0"}
    ag2 = f32[32,8] all-gather(w2), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[16,8], f32[32,8]) tuple(ag1, ag2)
  }
  )";

  const absl::string_view expected_hlo = R"(
  // CHECK-LABEL: ENTRY %main
  //       CHECK: %barrier = {{.*}} add(
  //       CHECK: %[[START:[^ ]+]] = {{.*}} async-start
  //  CHECK-SAME:   calls=%collectives_group
  //  CHECK-SAME:   frontend_attributes={_collectives_group="",
  //  CHECK-SAME:   collective_group_key="g0"}
  //  CHECK-SAME:   control-predecessors={%barrier}
  //       CHECK: %[[DONE:[^ ]+]] = {{.*}} async-done(%[[START]])
  )";

  RunAndFilecheckHloRewrite(hlo_string, GroupCollectivesByKey(), expected_hlo);
}

// Running the pass twice is idempotent: the second run makes no change.
TEST_F(GroupCollectivesByKeyTest, Idempotent) {
  const absl::string_view hlo_string = R"(
  HloModule test

  ENTRY main {
    w1 = f32[8,8] parameter(0)
    w2 = f32[8,8] parameter(1)
    ag1 = f32[16,8] all-gather(w1), dimensions={0},
        replica_groups={{0,1},{2,3}},
        frontend_attributes={collective_group_key="g0"}
    ag2 = f32[32,8] all-gather(w2), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[16,8], f32[32,8]) tuple(ag1, ag2)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(bool changed_again, pass.Run(module.get()));
  EXPECT_FALSE(changed_again);

  const absl::string_view expected_hlo = R"(
  // CHECK-LABEL: %collectives_group (
  //       CHECK: = {{.*}} all-gather(
  //       CHECK: = {{.*}} all-gather(

  // CHECK-LABEL: ENTRY %main
  //       CHECK: %[[START:[^ ]+]] = {{.*}} async-start
  //  CHECK-SAME:   calls=%collectives_group
  //  CHECK-SAME:   frontend_attributes={_collectives_group="",
  //  CHECK-SAME:   collective_group_key="g0"}
  //       CHECK: %[[DONE:[^ ]+]] = {{.*}} async-done(%[[START]])
  )";
  ExpectFileCheck(module->ToString(), expected_hlo);
}

// The pass output is consumed cleanly: the wrapper does not re-wrap the groups
// this pass already formed, leaving exactly the async pairs it created.
TEST_F(GroupCollectivesByKeyTest, EndToEndWithWrapper) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    w = f32[8,8] parameter(0)
    g = f32[32,8] parameter(1)
    ag = f32[32,8] all-gather(w), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    rs = f32[8,8] reduce-scatter(g), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[32,8], f32[8,8]) tuple(ag, rs)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey group_pass;
  ASSERT_OK_AND_ASSIGN(bool grouped, group_pass.Run(module.get()));
  EXPECT_TRUE(grouped);

  ExplicitCollectivesGroupAsyncWrapper wrapper;
  ASSERT_OK_AND_ASSIGN(bool wrapped, wrapper.Run(module.get()));
  EXPECT_FALSE(wrapped);

  const absl::string_view expected_hlo = R"(
  // CHECK-LABEL: %collectives_group (
  //       CHECK: = {{.*}} all-gather(
  //       CHECK: = {{.*}} reduce-scatter(

  // CHECK-LABEL: ENTRY %main
  //       CHECK: %[[START:[^ ]+]] = {{.*}} async-start
  //  CHECK-SAME:   calls=%collectives_group
  //  CHECK-SAME:   frontend_attributes={_collectives_group="",
  //  CHECK-SAME:   collective_group_key="g0"}
  //       CHECK: %[[DONE:[^ ]+]] = {{.*}} async-done(%[[START]])
  )";
  ExpectFileCheck(module->ToString(), expected_hlo);
}

// Invalid collective groups fail without mutating the module.
TEST_F(GroupCollectivesByKeyTest, ErrorsWhenDependencyExists) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    weights = f32[8,8] parameter(0)
    ag = f32[32,8] all-gather(weights), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    rs = f32[8,8] reduce-scatter(ag), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g0"}
    ROOT result = f32[8,8] copy(rs)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass;
  absl::Status status = pass.Run(module.get()).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);

  const absl::string_view expected_error = R"(
  //       CHECK: Collectives ag and rs
  //  CHECK-SAME:   collective_group_key=g0
  //       CHECK: Computation: main
  //       CHECK: Reachability chain ag -> rs:
  )";
  ExpectFileCheck(status.message(), expected_error);

  const absl::string_view expected_hlo = R"(
  // CHECK-LABEL: ENTRY %main
  //       CHECK: %ag = {{.*}} all-gather(
  //       CHECK: %rs = {{.*}} reduce-scatter(
  )";
  ExpectFileCheck(module->ToString(), expected_hlo);
}

// A later key's invalid dependency path runs through an earlier key's members.
// Keys are processed in sorted order, so "g0" (valid, independent) would be
// grouped before "g1" (invalid) is validated. Validation must therefore happen
// for every key against a single mutation-free reachability map: otherwise the
// "g1" diagnostic would walk users() of instructions whose "g0" neighbors have
// already been replaced by async/GTE nodes absent from the stale map (crash),
// and the "g0" group would be left half-transformed behind the error.
TEST_F(GroupCollectivesByKeyTest, ErrorsOnLaterKeyWithoutMutatingEarlierKey) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    w1 = f32[8,8] parameter(0)
    g = f32[32,8] parameter(1)
    // g1 member; feeds an all-gather that carries key g0.
    rs1 = f32[8,8] reduce-scatter(g), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g1"}
    // g0 members: independent of each other, so g0 is valid on its own. But
    // ag0a depends on rs1 and feeds ag1, so it sits on the g1 dependency path.
    ag0a = f32[32,8] all-gather(rs1), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    ag0b = f32[32,8] all-gather(w1), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    // g1 member reachable from rs1 through ag0a: g1 is not independent.
    ag1 = f32[128,8] all-gather(ag0a), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g1"}
    ROOT result = (f32[32,8], f32[128,8]) tuple(ag0b, ag1)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass;
  absl::Status status = pass.Run(module.get()).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);

  const absl::string_view expected_error = R"(
  //       CHECK: Collectives rs1 and ag1
  //  CHECK-SAME:   collective_group_key=g1
  //       CHECK: Reachability chain rs1 -> ag1:
  )";
  ExpectFileCheck(status.message(), expected_error);

  const absl::string_view expected_hlo = R"(
  // CHECK-LABEL: ENTRY %main
  //       CHECK: %ag0b = {{.*}} all-gather(
  //       CHECK: %rs1 = {{.*}} reduce-scatter(
  //       CHECK: %ag0a = {{.*}} all-gather(
  //       CHECK: %ag1 = {{.*}} all-gather(
  )";
  ExpectFileCheck(module->ToString(), expected_hlo);
}

TEST_F(GroupCollectivesByKeyTest, ErrorsWhenGroupedNodeGraphHasCycle) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    x = f32[8,8] parameter(0)
    y = f32[8,8] parameter(1)
    g0_first = f32[8,8] all-reduce(x), replica_groups={{0,1,2,3}},
        to_apply=add, frontend_attributes={collective_group_key="g0"}
    g1_first = f32[8,8] all-reduce(g0_first),
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g1"}
    g1_second = f32[8,8] all-reduce(y), replica_groups={{0,1,2,3}},
        to_apply=add, frontend_attributes={collective_group_key="g1"}
    g0_second = f32[8,8] all-reduce(g1_second),
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[8,8], f32[8,8]) tuple(g1_first, g0_second)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass;
  absl::Status status = pass.Run(module.get()).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);

  const absl::string_view expected_error = R"(
  //       CHECK: dependency cycle among collective_group_key values {g0, g1}
  )";
  ExpectFileCheck(status.message(), expected_error);

  const absl::string_view expected_hlo = R"(
  // CHECK-LABEL: ENTRY %main
  //       CHECK: %g0_first = {{.*}} all-reduce(
  //       CHECK: %g1_first = {{.*}} all-reduce(
  //       CHECK: %g1_second = {{.*}} all-reduce(
  //       CHECK: %g0_second = {{.*}} all-reduce(
  )";
  ExpectFileCheck(module->ToString(), expected_hlo);
}

}  // namespace
}  // namespace xla::gpu
