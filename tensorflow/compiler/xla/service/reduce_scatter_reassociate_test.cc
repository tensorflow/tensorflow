/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/reduce_scatter_reassociate.h"

#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace m = xla::testing::opcode_matchers;

class ReduceScatterReassociateTest : public HloTestBase {
 public:
  StatusOr<std::unique_ptr<HloModule>> RunPass(absl::string_view hlo_module,
                                               bool expect_change) {
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo_module));
    auto changed = ReduceScatterReassociate().Run(module.get());
    if (!changed.ok()) {
      return changed.status();
    }
    EXPECT_EQ(changed.value(), expect_change);
    return StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }

  size_t ReduceScatterCount(std::unique_ptr<HloModule>& module) {
    return absl::c_count_if(module->entry_computation()->instructions(),
                            [](const HloInstruction* inst) {
                              return inst->opcode() ==
                                     HloOpcode::kReduceScatter;
                            });
  }
};

TEST_F(ReduceScatterReassociateTest, Simple) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  rs0 = f32[4] reduce-scatter(p0), dimensions={0}, to_apply=sum
  rs1 = f32[4] reduce-scatter(p1), dimensions={0}, to_apply=sum
  ROOT add = f32[4] add(rs0, rs1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::ReduceScatter(m::Add(m::Parameter(0), m::Parameter(1))));
  EXPECT_EQ(ReduceScatterCount(module), 1);
}

TEST_F(ReduceScatterReassociateTest, SimpleWithConstrainLayout) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  rs0 = f32[4] reduce-scatter(p0), dimensions={0}, constrain_layout=true, to_apply=sum
  rs1 = f32[4] reduce-scatter(p1), dimensions={0}, constrain_layout=true, to_apply=sum
  ROOT add = f32[4] add(rs0, rs1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

// Checks whether a linear chain of adds of RSs is reassociated in a single
// pass.
TEST_F(ReduceScatterReassociateTest, SimpleChain) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  p2 = f32[8] parameter(2)
  p3 = f32[8] parameter(3)
  rs0 = f32[4] reduce-scatter(p0), dimensions={0}, to_apply=sum
  rs1 = f32[4] reduce-scatter(p1), dimensions={0}, to_apply=sum
  rs2 = f32[4] reduce-scatter(p2), dimensions={0}, to_apply=sum
  rs3 = f32[4] reduce-scatter(p3), dimensions={0}, to_apply=sum
  add0 = f32[4] add(rs0, rs1)
  add1 = f32[4] add(add0, rs2)
  ROOT add2 = f32[4] add(add1, rs3)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      m::ReduceScatter(m::Add(
          m::Add(m::Add(m::Parameter(0), m::Parameter(1)), m::Parameter(2)),
          m::Parameter(3))));
  EXPECT_EQ(ReduceScatterCount(module), 1);
}

// Checks whether a tree of add of RSs is reassociated in a single pass.
TEST_F(ReduceScatterReassociateTest, SimpleTree) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  p2 = f32[8] parameter(2)
  p3 = f32[8] parameter(3)
  rs0 = f32[4] reduce-scatter(p0), dimensions={0}, to_apply=sum
  rs1 = f32[4] reduce-scatter(p1), dimensions={0}, to_apply=sum
  rs2 = f32[4] reduce-scatter(p2), dimensions={0}, to_apply=sum
  rs3 = f32[4] reduce-scatter(p3), dimensions={0}, to_apply=sum
  add0 = f32[4] add(rs0, rs1)
  add1 = f32[4] add(rs2, rs3)
  ROOT add2 = f32[4] add(add0, add1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      m::ReduceScatter(m::Add(m::Add(m::Parameter(0), m::Parameter(1)),
                              m::Add(m::Parameter(2), m::Parameter(3)))));
  EXPECT_EQ(ReduceScatterCount(module), 1);
}

TEST_F(ReduceScatterReassociateTest, MismatchOp0) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

max {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT r = f32[] maximum(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  rs0 = f32[4] reduce-scatter(p0), dimensions={0}, to_apply=sum
  rs1 = f32[4] reduce-scatter(p1), dimensions={0}, to_apply=max
  ROOT add = f32[4] add(rs0, rs1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(ReduceScatterReassociateTest, MismatchOp1) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

max {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT r = f32[] maximum(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  rs0 = f32[4] reduce-scatter(p0), dimensions={0}, to_apply=max
  rs1 = f32[4] reduce-scatter(p1), dimensions={0}, to_apply=max
  ROOT add = f32[4] add(rs0, rs1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(ReduceScatterReassociateTest, MismatchDimension) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8,8] parameter(0)
  p1 = f32[8,8] parameter(1)
  rs0 = f32[8,8] reduce-scatter(p0), dimensions={0}, to_apply=sum
  rs1 = f32[8,8] reduce-scatter(p1), dimensions={1}, to_apply=sum
  ROOT add = f32[8,8] add(rs0, rs1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(ReduceScatterReassociateTest, MismatchReplicaGroups) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  rs0 = f32[4] reduce-scatter(p0), dimensions={0}, replica_groups={{0}}, to_apply=sum
  rs1 = f32[4] reduce-scatter(p1), dimensions={0}, replica_groups={}, to_apply=sum
  ROOT add = f32[4] add(rs0, rs1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(ReduceScatterReassociateTest, MismatchHasChannelId) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  rs0 = f32[4] reduce-scatter(p0), dimensions={0}, channel_id=3, to_apply=sum
  rs1 = f32[4] reduce-scatter(p1), dimensions={0}, to_apply=sum
  ROOT add = f32[4] add(rs0, rs1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(ReduceScatterReassociateTest, MismatchUseGlobalDeviceId) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  rs0 = f32[4] reduce-scatter(p0), dimensions={0}, replica_groups={{0,1}}, channel_id=3, use_global_device_ids=true, to_apply=sum
  rs1 = f32[4] reduce-scatter(p1), dimensions={0}, replica_groups={{0,1}}, channel_id=4, to_apply=sum
  ROOT add = f32[4] add(rs0, rs1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(ReduceScatterReassociateTest, NotSingleUser) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  rs0 = f32[4] reduce-scatter(p0), dimensions={0}, to_apply=sum
  rs1 = f32[4] reduce-scatter(p1), dimensions={0}, to_apply=sum
  add = f32[4] add(rs0, rs1)
  ROOT t = (f32[4], f32[4]) tuple(rs0, add)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(ReduceScatterReassociateTest, DoubleUse) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  rs0 = f32[4] reduce-scatter(p0), dimensions={0}, to_apply=sum
  add = f32[4] add(rs0, rs0)
  ROOT c = f32[4] copy(add)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
}

}  // namespace
}  // namespace xla
