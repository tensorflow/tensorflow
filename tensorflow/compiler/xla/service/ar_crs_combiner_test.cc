/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/ar_crs_combiner.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class ArCrsCombinerTest : public HloTestBase {};

TEST_F(ArCrsCombinerTest, SameValueTestBasecase) {
  const char* module_str = R"(
HloModule foobar

ENTRY %entrycomp (p: f32[2,2]) -> (f32[2,2], f32[2,2]) {
  %p = f32[2,2] parameter(0)
  %constant.f32.1 = f32[2,2] constant(f32[2,2] {{1, 2}, {3, 4}})
  %constant.f32.2 = f32[2,2] constant(f32[2,2] {{1, 2}, {3, 4}})
  ROOT %tuple = (f32[2,2], f32[2,2]) tuple(%constant.f32.1, %constant.f32.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_tuple = module->entry_computation()->root_instruction();
  auto i1 = root_tuple->operands()[0];
  auto i2 = root_tuple->operands()[1];
  EXPECT_FALSE(ArCrsCombiner::TestInstructionsComputeSameValue(
      i1, module->entry_computation()->parameter_instruction(0)));
  EXPECT_TRUE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, SameValueTestNumOperands) {
  const char* module_str = R"(
HloModule foobar

ENTRY %entrycomp (p: f32[2,2]) -> ((f32[2,2]), (f32[2,2], f32[2,2])) {
  %p = f32[2,2] parameter(0)
  %constant.f32 = f32[2,2] constant(f32[2,2] {{1, 2}, {3, 4}})
  %tuple1 = (f32[2,2]) tuple(%constant.f32)
  %tuple2 = (f32[2,2], f32[2,2]) tuple(%constant.f32, %constant.f32)
  ROOT %tuple = ((f32[2,2]), (f32[2,2], f32[2,2])) tuple(%tuple1, %tuple2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_tuple = module->entry_computation()->root_instruction();
  auto i1 = root_tuple->operands()[0];
  auto i2 = root_tuple->operands()[1];
  EXPECT_FALSE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, SameValueTestTupleElementSameIndex) {
  const char* module_str = R"(
HloModule foobar

ENTRY %entrycomp (p: f32[2,2]) -> (f32[2,2], f32[2,2]) {
  %p = f32[2,2] parameter(0)
  %constant.f32 = f32[2,2] constant(f32[2,2] {{1, 2}, {3, 4}})
  %tuple.1 = (f32[2,2], f32[2,2]) tuple(%constant.f32, %constant.f32)
  %get-tuple-element.1 = f32[2,2] get-tuple-element(%tuple.1), index=0
  %get-tuple-element.2 = f32[2,2] get-tuple-element(%tuple.1), index=0
  ROOT %tuple = (f32[2,2], f32[2,2]) tuple(%get-tuple-element.1, %get-tuple-element.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_tuple = module->entry_computation()->root_instruction();
  auto i1 = root_tuple->operands()[0];
  auto i2 = root_tuple->operands()[1];
  EXPECT_TRUE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, SameValueTestTupleElementDifferentIndex1) {
  const char* module_str = R"(
HloModule foobar

ENTRY %entrycomp (p: f32[2,2]) -> (f32[2,2], f32[2,2]) {
  %p = f32[2,2] parameter(0)
  %constant.f32 = f32[2,2] constant(f32[2,2] {{1, 2}, {3, 4}})
  %tuple.1 = (f32[2,2], f32[2,2]) tuple(%constant.f32, %constant.f32)
  %get-tuple-element.1 = f32[2,2] get-tuple-element(%tuple.1), index=0
  %get-tuple-element.2 = f32[2,2] get-tuple-element(%tuple.1), index=1
  ROOT %tuple = (f32[2,2], f32[2,2]) tuple(%get-tuple-element.1, %get-tuple-element.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_tuple = module->entry_computation()->root_instruction();
  auto i1 = root_tuple->operands()[0];
  auto i2 = root_tuple->operands()[1];
  EXPECT_TRUE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, SameValueTestTupleElementDifferentIndex2) {
  const char* module_str = R"(
HloModule foobar

ENTRY %entrycomp (p: f32[2,2]) -> (f32[2,2], f32[2,2]) {
  %p = f32[2,2] parameter(0)
  %constant.f32.1 = f32[2,2] constant(f32[2,2] {{1, 2}, {3, 4}})
  %constant.f32.2 = f32[2,2] constant(f32[2,2] {{2, 3}, {4, 5}})
  %tuple.1 = (f32[2,2], f32[2,2]) tuple(%constant.f32.1, %constant.f32.2)
  %get-tuple-element.1 = f32[2,2] get-tuple-element(%tuple.1), index=0
  %get-tuple-element.2 = f32[2,2] get-tuple-element(%tuple.1), index=1
  ROOT %tuple = (f32[2,2], f32[2,2]) tuple(%get-tuple-element.1, %get-tuple-element.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_tuple = module->entry_computation()->root_instruction();
  auto i1 = root_tuple->operands()[0];
  auto i2 = root_tuple->operands()[1];
  EXPECT_FALSE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, SameValueTestWhile1) {
  const char* module_str = R"(
HloModule foobar

%condition (x: (f32[2,2], f32[2,2])) -> pred[] {
  %x = (f32[2,2], f32[2,2]) parameter(0)
  %constant.0 = s32[] constant(0)
  %constant.1 = s32[] constant(1)
  ROOT %greater-than = pred[] greater-than(s32[] %constant.1, s32[] %constant.0)
}

%body (x: (f32[2,2], f32[2,2])) -> (f32[2,2], f32[2,2]) {
  %x = (f32[2,2], f32[2,2]) parameter(0)
  %constant.f32 = f32[2,2] constant(f32[2,2] {{1, 2}, {3, 4}})
  %get-tuple-element.1 = f32[2,2] get-tuple-element(%x), index=0
  %get-tuple-element.2 = f32[2,2] get-tuple-element(%x), index=1
  %add.1 = f32[2,2] add(%get-tuple-element.1, %constant.f32)
  %add.2 = f32[2,2] add(%get-tuple-element.2, %constant.f32)
  ROOT %tuple = (f32[2,2], f32[2,2]) tuple(%add.1, %add.2)
}

ENTRY %WhileLoop () -> (f32[2,2], f32[2,2]) {
  %constant.f32 = f32[2,2] constant(f32[2,2] {{3, 4}, {5, 6}})
  %init.tuple = (f32[2,2], f32[2,2]) tuple(%constant.f32, %constant.f32)
  ROOT %while = (f32[2,2], f32[2,2]) while(%init.tuple), condition=%condition, body=%body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_while = module->entry_computation()->root_instruction();
  auto body_tuple = root_while->while_body()->root_instruction();
  auto i1 = body_tuple->operands()[0];
  auto i2 = body_tuple->operands()[1];
  EXPECT_TRUE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, SameValueTestWhile2) {
  const char* module_str = R"(
HloModule foobar

%condition (x: (f32[2,2], f32[2,2])) -> pred[] {
  %x = (f32[2,2], f32[2,2]) parameter(0)
  %constant.0 = s32[] constant(0)
  %constant.1 = s32[] constant(1)
  ROOT %greater-than = pred[] greater-than(s32[] %constant.1, s32[] %constant.0)
}

%body (x: (f32[2,2], f32[2,2])) -> (f32[2,2], f32[2,2]) {
  %x = (f32[2,2], f32[2,2]) parameter(0)
  %constant.f32 = f32[2,2] constant(f32[2,2] {{1, 2}, {3, 4}})
  %get-tuple-element.1 = f32[2,2] get-tuple-element(%x), index=0
  %get-tuple-element.2 = f32[2,2] get-tuple-element(%x), index=1
  %add.1 = f32[2,2] add(%get-tuple-element.1, %constant.f32)
  %add.2 = f32[2,2] add(%get-tuple-element.2, %constant.f32)
  ROOT %tuple = (f32[2,2], f32[2,2]) tuple(%add.1, %add.2)
}

ENTRY %WhileLoop () -> (f32[2,2], f32[2,2]) {
  %constant.f32.1 = f32[2,2] constant(f32[2,2] {{3, 4}, {5, 6}})
  %constant.f32.2 = f32[2,2] constant(f32[2,2] {{3, 4}, {7, 8}})
  %init.tuple = (f32[2,2], f32[2,2]) tuple(%constant.f32.1, %constant.f32.2)
  ROOT %while = (f32[2,2], f32[2,2]) while(%init.tuple), condition=%condition, body=%body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_while = module->entry_computation()->root_instruction();
  auto body_tuple = root_while->while_body()->root_instruction();
  auto i1 = body_tuple->operands()[0];
  auto i2 = body_tuple->operands()[1];
  EXPECT_FALSE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, SameValueTestWhile3) {
  const char* module_str = R"(
HloModule foobar

%condition (x: (f32[2,2], f32[2,2])) -> pred[] {
  %x = (f32[2,2], f32[2,2]) parameter(0)
  %constant.0 = s32[] constant(0)
  %constant.1 = s32[] constant(1)
  ROOT %greater-than = pred[] greater-than(s32[] %constant.1, s32[] %constant.0)
}

%body (x: (f32[2,2], f32[2,2])) -> (f32[2,2], f32[2,2]) {
  %x = (f32[2,2], f32[2,2]) parameter(0)
  %constant.f32.1 = f32[2,2] constant(f32[2,2] {{1, 2}, {3, 4}})
  %constant.f32.2 = f32[2,2] constant(f32[2,2] {{3, 4}, {1, 2}})
  %get-tuple-element.1 = f32[2,2] get-tuple-element(%x), index=0
  %get-tuple-element.2 = f32[2,2] get-tuple-element(%x), index=1
  %add.1 = f32[2,2] add(%get-tuple-element.1, %constant.f32.1)
  %add.2 = f32[2,2] add(%get-tuple-element.2, %constant.f32.2)
  ROOT %tuple = (f32[2,2], f32[2,2]) tuple(%add.1, %add.2)
}

ENTRY %WhileLoop () -> (f32[2,2], f32[2,2]) {
  %constant.f32 = f32[2,2] constant(f32[2,2] {{3, 4}, {5, 6}})
  %init.tuple = (f32[2,2], f32[2,2]) tuple(%constant.f32, %constant.f32)
  ROOT %while = (f32[2,2], f32[2,2]) while(%init.tuple), condition=%condition, body=%body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto root_while = module->entry_computation()->root_instruction();
  auto body_tuple = root_while->while_body()->root_instruction();
  auto i1 = body_tuple->operands()[0]->operands()[0];  // %get-tuple-element.1
  auto i2 = body_tuple->operands()[1]->operands()[0];  // %get-tuple-element.2
  EXPECT_FALSE(ArCrsCombiner::TestInstructionsComputeSameValue(i1, i2));
}

TEST_F(ArCrsCombinerTest, RewritePatternArConvertAddCrs) {
  const char* module_str = R"(
HloModule foobar

%binary_add (a: bf16[], b: bf16[]) -> bf16[] {
  %a = bf16[] parameter(0)
  %b = bf16[] parameter(1)
  ROOT %add = bf16[] add(%a, %b)
}

%sum.f32 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: f32[2,2]) -> (f32[2,2], f32[2,2]) {
  %p = f32[2,2] parameter(0)
  %constant.bf16 = bf16[2,2] constant(bf16[2,2] {{1, 2}, {3, 4}})
  %constant.f32 = f32[2,2] constant(f32[2,2] {{1, 2}, {3, 4}})

  %cross-replica-sum.ar.1 = bf16[2,2]
      cross-replica-sum(%constant.bf16),
      replica_groups={{0},{1}},
      all_reduce_id=1,
      to_apply=%binary_add,
      sharding={maximal device=0}
  %convert.1 = f32[2,2]
      convert(%cross-replica-sum.ar.1),
      sharding={maximal device=0}
  %add.1 = f32[2,2]
      add(%constant.f32, %convert.1),
      sharding={maximal device=0}
  %cross-replica-sum.1 = f32[2,2]
      cross-replica-sum(%add.1),
      replica_groups={{0,1}},
      to_apply=%sum.f32,
      sharding={maximal device=0}

  %cross-replica-sum.ar.2 = bf16[2,2]
      cross-replica-sum(%constant.bf16),
      replica_groups={{0},{1}},
      all_reduce_id=1,
      to_apply=%binary_add,
      sharding={maximal device=1}
  %convert.2 = f32[2,2]
      convert(%cross-replica-sum.ar.2),
      sharding={maximal device=1}
  %add.2 = f32[2,2]
      add(%constant.f32, %convert.2),
      sharding={maximal device=1}
  %cross-replica-sum.2 = f32[2,2]
      cross-replica-sum(%add.2),
      replica_groups={{0,1}},
      to_apply=%sum.f32,
      sharding={maximal device=1}

  ROOT %tuple = (f32[2,2], f32[2,2])
      tuple(%cross-replica-sum.1, %cross-replica-sum.2),
      sharding={{maximal device=0}, {maximal device=1}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  auto crs_before =
      module->entry_computation()->root_instruction()->operands()[0];
  auto replica_groups_before = crs_before->replica_groups();
  ArCrsCombiner combiner(2, 1);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::Subtract(op::CrossReplicaSum(),
                             op::Multiply(op::Constant(), op::Constant())),
                op::Subtract(op::CrossReplicaSum(),
                             op::Multiply(op::Constant(), op::Constant()))));
  auto sub = module->entry_computation()->root_instruction()->operands()[0];
  auto crs_after = sub->operands()[0];
  auto replica_groups_after = crs_after->replica_groups();
  ASSERT_EQ(replica_groups_before.size(), replica_groups_after.size());
  for (int i = 0; i < replica_groups_before.size(); ++i) {
    // Somewhat verbose way to compare the replica_ids, because EqualsProto
    // is not available in the open-source build.
    auto group_before = replica_groups_before[i];
    std::vector<int64> ids_before(group_before.replica_ids().begin(),
                                  group_before.replica_ids().end());
    auto group_after = replica_groups_after[i];
    std::vector<int64> ids_after(group_after.replica_ids().begin(),
                                 group_after.replica_ids().end());
    EXPECT_EQ(ids_before, ids_after);
  }
}

TEST_F(ArCrsCombinerTest, OtherSummandNotTheSameDontRewrite) {
  const char* module_str = R"(
HloModule foobar

%binary_add (a: bf16[], b: bf16[]) -> bf16[] {
  %a = bf16[] parameter(0)
  %b = bf16[] parameter(1)
  ROOT %add = bf16[] add(%a, %b)
}

%sum.f32 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY %entrycomp (p: f32[2,2]) -> (f32[2,2], f32[2,2]) {
  %p = f32[2,2] parameter(0)
  %constant.bf16 = bf16[2,2] constant(bf16[2,2] {{1, 2}, {3, 4}})
  %constant.f32.1 = f32[2,2] constant(f32[2,2] {{1, 2}, {3, 4}})
  %constant.f32.2 = f32[2,2] constant(f32[2,2] {{3, 4}, {5, 6}})

  %cross-replica-sum.ar.1 = bf16[2,2]
      cross-replica-sum(%constant.bf16),
      replica_groups={{0},{1}},
      all_reduce_id=1,
      to_apply=%binary_add,
      sharding={maximal device=0}
  %convert.1 = f32[2,2]
      convert(%cross-replica-sum.ar.1),
      sharding={maximal device=0}
  %add.1 = f32[2,2]
      add(%constant.f32.1, %convert.1),
      sharding={maximal device=0}
  %cross-replica-sum.1 = f32[2,2]
      cross-replica-sum(%add.1),
      replica_groups={{0,1}},
      to_apply=%sum.f32,
      sharding={maximal device=0}

  %cross-replica-sum.ar.2 = bf16[2,2]
      cross-replica-sum(%constant.bf16),
      replica_groups={{0},{1}},
      all_reduce_id=1,
      to_apply=%binary_add,
      sharding={maximal device=1}
  %convert.2 = f32[2,2]
      convert(%cross-replica-sum.ar.2),
      sharding={maximal device=1}
  %add.2 = f32[2,2]
      add(%constant.f32.2, %convert.2),
      sharding={maximal device=1}
  %cross-replica-sum.2 = f32[2,2]
      cross-replica-sum(%add.2),
      replica_groups={{0,1}},
      to_apply=%sum.f32,
      sharding={maximal device=1}

  ROOT %tuple = (f32[2,2], f32[2,2])
      tuple(%cross-replica-sum.1, %cross-replica-sum.2),
      sharding={{maximal device=0}, {maximal device=1}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  ArCrsCombiner combiner(2, 1);
  auto changed = combiner.Run(module.get()).ValueOrDie();
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla
