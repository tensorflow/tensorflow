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

#include "xla/hlo/transforms/collectives/all_reduce_reduce_scatter_reorder.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::xla::RunFileCheck;

class AllReduceReduceScatterReorderTest
    : public HloHardwareIndependentTestBase {
 public:
  AllReduceReduceScatterReorder pass_;
};

TEST_F(AllReduceReduceScatterReorderTest, KeepingReplicaGroups) {
  absl::string_view hlo_text = R"(
  sum {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    p0 = f32[8] parameter(0)
    ar = f32[8] all-reduce(p0), replica_groups={{0,1}, {2,3}}, to_apply=sum
    ROOT rs = f32[4] reduce-scatter(ar), dimensions={0}, replica_groups={{0,2}, {1,3}}, to_apply=sum
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass_, module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* param = module->entry_computation()->parameter_instruction(0);
  HloInstruction* new_rs = param->users().front();
  HloInstruction* new_ar = new_rs->users().front();

  EXPECT_EQ(new_rs->replica_groups().size(), 2);
  EXPECT_THAT(new_rs->replica_groups()[0].replica_ids(),
              ::testing::ElementsAre(0, 2));
  EXPECT_THAT(new_rs->replica_groups()[1].replica_ids(),
              ::testing::ElementsAre(1, 3));

  EXPECT_EQ(new_ar->replica_groups().size(), 2);
  EXPECT_THAT(new_ar->replica_groups()[0].replica_ids(),
              ::testing::ElementsAre(0, 1));
  EXPECT_THAT(new_ar->replica_groups()[1].replica_ids(),
              ::testing::ElementsAre(2, 3));

  EXPECT_EQ(new_rs->opcode(), HloOpcode::kReduceScatter);
  EXPECT_EQ(new_ar->opcode(), HloOpcode::kAllReduce);

  EXPECT_EQ(new_rs->shape().dimensions(0), 4);
  EXPECT_EQ(new_ar->shape().dimensions(0), 4);

  EXPECT_EQ(new_ar, module->entry_computation()->root_instruction());
}

TEST_F(AllReduceReduceScatterReorderTest, AllReduceMultipleUsers) {
  absl::string_view hlo_text = R"(
  sum {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    p0 = f32[8] parameter(0)
    ar = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
    rs = f32[4] reduce-scatter(ar), dimensions={0}, replica_groups={}, to_apply=sum
    ROOT tuple = (f32[8], f32[4]) tuple(ar, rs)
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass_, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(AllReduceReduceScatterReorderTest, DifferentReductionFunctions) {
  absl::string_view hlo_text = R"(
  sum {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  product {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT product = f32[] multiply(a, b)
  }

  ENTRY main {
    p0 = f32[8] parameter(0)
    ar = f32[8] all-reduce(p0), replica_groups={{0,1}, {2,3}}, to_apply=sum
    ROOT rs = f32[4] reduce-scatter(ar), dimensions={0}, replica_groups={{0,2}, {1,3}}, to_apply=product
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass_, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(AllReduceReduceScatterReorderTest, NonOrthogonalReplicaGroups) {
  absl::string_view hlo_text = R"(
  sum {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    p0 = f32[8] parameter(0)
    ar = f32[8] all-reduce(p0), replica_groups={{0,1}, {2,3}}, to_apply=sum
    ROOT rs = f32[4] reduce-scatter(ar), dimensions={0}, replica_groups={{0,1}, {2,3}}, to_apply=sum
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass_, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(AllReduceReduceScatterReorderTest, NonOrthogonalSwappedScatterOrder) {
  absl::string_view hlo_text = R"(
  sum {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    p0 = f32[8] parameter(0)
    ar = f32[8] all-reduce(p0), replica_groups={{0,1}, {2,3}}, to_apply=sum
    ROOT rs = f32[4] reduce-scatter(ar), dimensions={0}, replica_groups={{0,2}, {3,1}}, to_apply=sum
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass_, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(AllReduceReduceScatterReorderTest, UnrecognizedReductionComputation) {
  absl::string_view hlo_text = R"(
  sub {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT diff = f32[] subtract(a, b)
  }

  ENTRY main {
    p0 = f32[8] parameter(0)
    ar = f32[8] all-reduce(p0), replica_groups={{0,1}, {2,3}}, to_apply=sub
    ROOT rs = f32[4] reduce-scatter(ar), dimensions={0}, replica_groups={{0,2}, {1,3}}, to_apply=sub
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass_, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(AllReduceReduceScatterReorderTest, VariadicReduceScatter) {
  absl::string_view hlo_text = R"(
  sum {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    p0 = f32[8] parameter(0)
    p1 = f32[8] parameter(1)
    ar = f32[8] all-reduce(p0), replica_groups={{0,1}, {2,3}}, to_apply=sum
    ROOT rs = (f32[4], f32[4]) reduce-scatter(ar, p1), dimensions={0}, replica_groups={{0,2}, {1,3}}, to_apply=sum
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass_, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(AllReduceReduceScatterReorderTest, MismatchedReduceScatterGroupSize) {
  absl::string_view hlo_text = R"(
  sum {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    p0 = f32[8] parameter(0)
    ar = f32[8] all-reduce(p0), replica_groups={{0}, {1}, {2}, {3}}, to_apply=sum
    ROOT rs = f32[4] reduce-scatter(ar), dimensions={0}, replica_groups={{0,1}}, to_apply=sum
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass_, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(AllReduceReduceScatterReorderTest, DoNotCrossCallBoundaries) {
  absl::string_view hlo_string = R"(
  HloModule module

  sum {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  foo {
    p = f32[8] parameter(0)
    ROOT rs = f32[4] reduce-scatter(p), dimensions={0}, replica_groups={{0,2}, {1,3}}, to_apply=sum
  }

  ENTRY main {
    p0 = f32[8] parameter(0)
    ar = f32[8] all-reduce(p0), replica_groups={{0,1}, {2,3}}, to_apply=sum
    ROOT call = f32[4] call(ar), to_apply=foo
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_THAT(pass_.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(AllReduceReduceScatterReorderTest,
       TransformationInsideCalledComputation) {
  absl::string_view hlo_string = R"(
  HloModule module

  sum {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  // CHECK: %foo (
  // CHECK:   %[[P:.*]] = f32[8]{0} parameter(0)
  // CHECK:   %[[RS:.*]] = f32[4]{0} reduce-scatter(%[[P]]){{.*}}, dimensions={0}, to_apply=%sum
  // CHECK:   ROOT %[[AR:.*]] = f32[4]{0} all-reduce(%[[RS]]){{.*}}, to_apply=%sum
  // CHECK: }
  foo {
    p = f32[8] parameter(0)
    ar = f32[8] all-reduce(p), replica_groups={{0,1}, {2,3}}, to_apply=sum
    ROOT rs = f32[4] reduce-scatter(ar), dimensions={0}, replica_groups={{0,2}, {1,3}}, to_apply=sum
  }

  // CHECK: ENTRY %main (
  // CHECK:   %[[P0:.*]] = f32[8]{0} parameter(0)
  // CHECK:   ROOT %[[CALL:.*]] = f32[4]{0} call(%[[P0]]), to_apply=%foo
  // CHECK: }
  ENTRY main {
    p0 = f32[8] parameter(0)
    ROOT call = f32[4] call(p0), to_apply=foo
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_THAT(pass_.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(module->ToString(), hlo_string), IsOkAndHolds(true));
}

TEST_F(AllReduceReduceScatterReorderTest,
       EntryCallsFooAndBarFooCallsBarTransformationInsideBar) {
  absl::string_view hlo_string = R"(
  HloModule module

  sum {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  // CHECK: %bar (
  // CHECK:   %[[P_BAR:.*]] = f32[8]{0} parameter(0)
  // CHECK:   %[[RS:.*]] = f32[4]{0} reduce-scatter(%[[P_BAR]]){{.*}}, dimensions={0}, to_apply=%sum
  // CHECK:   ROOT %[[AR:.*]] = f32[4]{0} all-reduce(%[[RS]]){{.*}}, to_apply=%sum
  // CHECK: }
  bar {
    p_bar = f32[8] parameter(0)
    ar = f32[8] all-reduce(p_bar), replica_groups={{0,1}, {2,3}}, to_apply=sum
    ROOT rs = f32[4] reduce-scatter(ar), dimensions={0}, replica_groups={{0,2}, {1,3}}, to_apply=sum
  }

  // CHECK: %foo (
  // CHECK:   %[[P_FOO:.*]] = f32[8]{0} parameter(0)
  // CHECK:   ROOT %[[CALL_BAR:.*]] = f32[4]{0} call(%[[P_FOO]]), to_apply=%bar
  // CHECK: }
  foo {
    p_foo = f32[8] parameter(0)
    ROOT call_bar = f32[4] call(p_foo), to_apply=bar
  }

  // CHECK: ENTRY %main (
  // CHECK:     %[[P0:.*]] = f32[8]{0} parameter(0)
  // CHECK-DAG: %[[CALL_FOO:.*]] = f32[4]{0} call(%[[P0]]), to_apply=%foo
  // CHECK-DAG: %[[CALL_BAR_MAIN:.*]] = f32[4]{0} call(%[[P0]]), to_apply=%bar
  // CHECK:     ROOT %[[ADD:.*]] = f32[4]{0} add(%[[CALL_FOO]], %[[CALL_BAR_MAIN]])
  // CHECK: }
  ENTRY main {
    p0 = f32[8] parameter(0)
    call_foo = f32[4] call(p0), to_apply=foo
    call_bar = f32[4] call(p0), to_apply=bar
    ROOT result = f32[4] add(call_foo, call_bar)
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_THAT(pass_.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(module->ToString(), hlo_string), IsOkAndHolds(true));
}

}  // namespace
}  // namespace xla
