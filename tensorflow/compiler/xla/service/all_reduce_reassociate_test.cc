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

#include "tensorflow/compiler/xla/service/all_reduce_reassociate.h"

#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace m = xla::testing::opcode_matchers;

class AllReduceSimplifierTest : public HloTestBase {
 public:
  StatusOr<std::unique_ptr<HloModule>> RunPass(absl::string_view hlo_module,
                                               bool expect_change) {
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo_module));
    auto changed = AllReduceReassociate().Run(module.get());
    if (!changed.ok()) {
      return changed.status();
    }
    EXPECT_EQ(changed.value(), expect_change);
    return StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }

  size_t AllReduceCount(std::unique_ptr<HloModule>& module) {
    return absl::c_count_if(module->entry_computation()->instructions(),
                            [](const HloInstruction* inst) {
                              return inst->opcode() == HloOpcode::kAllReduce;
                            });
  }
};

TEST_F(AllReduceSimplifierTest, Simple) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::AllReduce(m::Add(m::Parameter(0), m::Parameter(1))));
  EXPECT_EQ(AllReduceCount(module), 1);
}

TEST_F(AllReduceSimplifierTest, SimpleWithChannelId) {
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
  ar0 = f32[8] all-reduce(p0), channel_id=1, replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), channel_id=1, replica_groups={}, to_apply=sum
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::AllReduce(m::Add(m::Parameter(0), m::Parameter(1))));
  EXPECT_EQ(AllReduceCount(module), 1);
}

// Checks whether a linear chain of adds of ARs is reassociated iin a single
// pass.
TEST_F(AllReduceSimplifierTest, SimpleChain) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  ar2 = f32[8] all-reduce(p2), replica_groups={}, to_apply=sum
  ar3 = f32[8] all-reduce(p3), replica_groups={}, to_apply=sum
  add0 = f32[8] add(ar0, ar1)
  add1 = f32[8] add(add0, ar2)
  ROOT add2 = f32[8] add(add1, ar3)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      m::AllReduce(m::Add(
          m::Add(m::Add(m::Parameter(0), m::Parameter(1)), m::Parameter(2)),
          m::Parameter(3))));
  EXPECT_EQ(AllReduceCount(module), 1);
}

// Checks whether a tree of add of ARs is reassociated in a single pass.
TEST_F(AllReduceSimplifierTest, SimpleTree) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  ar2 = f32[8] all-reduce(p2), replica_groups={}, to_apply=sum
  ar3 = f32[8] all-reduce(p3), replica_groups={}, to_apply=sum
  add0 = f32[8] add(ar0, ar1)
  add1 = f32[8] add(ar2, ar3)
  ROOT add2 = f32[8] add(add0, add1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::AllReduce(m::Add(m::Add(m::Parameter(0), m::Parameter(1)),
                                  m::Add(m::Parameter(2), m::Parameter(3)))));
  EXPECT_EQ(AllReduceCount(module), 1);
}

TEST_F(AllReduceSimplifierTest, MismatchOp0) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=max
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceSimplifierTest, MismatchOp1) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=max
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=max
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceSimplifierTest, MismatchReplicaGroups) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={{0}}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceSimplifierTest, MismatchHasChannelId) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={}, channel_id=3, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceSimplifierTest, MismatchUseGlobalDeviceId) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={{0, 1}}, channel_id=3, use_global_device_ids=true, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={{0, 1}}, channel_id=4, to_apply=sum
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceSimplifierTest, NotSingleUser) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  add = f32[8] add(ar0, ar1)
  ROOT t = (f32[8], f32[8]) tuple(ar0, add)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceSimplifierTest, DoubleUse) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  add = f32[8] add(ar0, ar0)
  ROOT c = f32[8] copy(add)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
}

}  // namespace
}  // namespace xla
