/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/all_reduce_combiner.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

using std::nullopt;
using ::testing::AllOf;
namespace op = xla::testing::opcode_matchers;
int64_t kMaxCombineCount = 256;

int64_t AllReduceCount(const HloModule& module) {
  int64_t count = 0;
  for (HloComputation* computation : module.computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (HloInstruction* hlo : computation->instructions()) {
      if (hlo->opcode() == HloOpcode::kAllReduce) {
        ++count;
      }
    }
  }
  return count;
}

// inputs[i] will be some op producing a shape of size sizes_in_kib[i] which
// feeds into all reduce op in all_reduces[i]. Returns a tuple
// of the all_reduces.
HloInstruction* MakeCrossReplicaReductions(
    std::vector<int64_t> sizes_in_kib, std::vector<HloComputation*> reductions,
    std::vector<HloInstruction*>* inputs, HloComputation::Builder* b) {
  CHECK_EQ(reductions.size(), sizes_in_kib.size());
  std::vector<HloInstruction*> all_reduces;
  for (int i = 0; i < sizes_in_kib.size(); i++) {
    int64_t size_in_kib = sizes_in_kib[i];
    HloComputation* reduction = reductions[i];
    auto constant = b->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0(42.3)));
    Shape shape = ShapeUtil::MakeShape(
        F32, {static_cast<int32_t>(size_in_kib * 1024 / sizeof(float))});
    auto input =
        b->AddInstruction(HloInstruction::CreateBroadcast(shape, constant, {}));
    inputs->push_back(input);
    all_reduces.push_back(b->AddInstruction(HloInstruction::CreateAllReduce(
        shape, {input}, reduction, /*replica_groups=*/{},
        /*constrain_layout=*/false, /*channel_id=*/nullopt,
        /*use_global_device_ids=*/false)));
  }
  return b->AddInstruction(HloInstruction::CreateTuple(all_reduces));
}

// Create and add a reduction computation in the given type to the module.
HloComputation* MakeReduction(const HloOpcode type, HloModule* module) {
  HloComputation::Builder sum_builder(HloOpcodeString(type));
  auto x = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {}), "x"));
  auto y = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {}), "y"));
  sum_builder.AddInstruction(
      HloInstruction::CreateBinary(ShapeUtil::MakeShape(F32, {}), type, x, y));
  HloComputation* reduction =
      module->AddEmbeddedComputation(sum_builder.Build());
  return reduction;
}

// Creates replica groups for AllReduce. groups[i] represents replica ids
// for group 'i'.
std::vector<ReplicaGroup> CreateReplicaGroups(
    absl::Span<const std::vector<int64_t>> groups) {
  std::vector<ReplicaGroup> replica_groups(groups.size());
  for (int64_t i = 0; i < groups.size(); ++i) {
    *replica_groups[i].mutable_replica_ids() = {groups[i].begin(),
                                                groups[i].end()};
  }
  return replica_groups;
}

using AllReduceCombinerTest = HloTestBase;

// Tests combination of several AllReduce instructions.
TEST_F(AllReduceCombinerTest, CombineAllReduces) {
  auto module = CreateNewVerifiedModule();
  HloComputation* sum = MakeReduction(HloOpcode::kAdd, module.get());

  HloComputation::Builder b(TestName());
  std::vector<HloInstruction*> inputs;
  auto root = MakeCrossReplicaReductions(
      {1, 2, 10, 7, 6}, {sum, sum, sum, sum, sum}, &inputs, &b);
  auto computation = module->AddEntryComputation(b.Build());

  // Run the AllReduce combiner optimization pass.
  AllReduceCombiner combine(10 * 1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(AllReduceCount(*module), inputs.size());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  ASSERT_EQ(AllReduceCount(*module), 1);
  EXPECT_TRUE(changed);

  ASSERT_EQ(root, computation->root_instruction());
  ASSERT_EQ(inputs.size(), root->operands().size());

  HloInstruction* combined = nullptr;
  for (int64_t i = 0; i < root->operands().size(); ++i) {
    HloInstruction* hlo = root->mutable_operand(i);
    ASSERT_TRUE(hlo->opcode() == HloOpcode::kGetTupleElement);
    EXPECT_EQ(hlo->tuple_index(), i);
    EXPECT_TRUE(ShapeUtil::Equal(inputs[i]->shape(), hlo->shape()));

    if (combined == nullptr) {
      // Verify the combined all reduce instruction.
      combined = hlo->mutable_operand(0);
      ASSERT_TRUE(combined->opcode() == HloOpcode::kAllReduce);
      EXPECT_TRUE(ShapeUtil::Equal(root->shape(), combined->shape()));
      ASSERT_EQ(combined->operands().size(), inputs.size());
    }
    EXPECT_EQ(combined, hlo->operand(0));
    EXPECT_TRUE(ShapeUtil::Equal(inputs[i]->shape(), hlo->shape()));
    EXPECT_EQ(combined->operand(i), inputs[i]);
    EXPECT_EQ(1, inputs[i]->users().size());
  }
  ASSERT_NE(combined, nullptr);
}

// Tests combination of several cross replica reduction instructions in
// different types.k
TEST_F(AllReduceCombinerTest, CombineCrossReplicaReductionsInGroups) {
  auto module = CreateNewVerifiedModule();
  HloComputation* sum = MakeReduction(HloOpcode::kAdd, module.get());
  HloComputation* min = MakeReduction(HloOpcode::kMinimum, module.get());
  HloComputation* max = MakeReduction(HloOpcode::kMaximum, module.get());
  HloComputation* sum_2 = MakeReduction(HloOpcode::kAdd, module.get());

  HloComputation::Builder b(TestName());
  std::vector<HloInstruction*> inputs;
  MakeCrossReplicaReductions(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {sum, sum_2, min, min, min, max, max, max, sum, sum_2}, &inputs, &b);
  module->AddEntryComputation(b.Build());

  // Run the AllReduce combiner optimization pass.
  AllReduceCombiner combine(10 * 1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(AllReduceCount(*module), inputs.size());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  ASSERT_EQ(AllReduceCount(*module), 3)
      << "expects 3 groups for 3 reduction types.";
  EXPECT_TRUE(changed);
}

// Tests that the combination threshold is respected.
TEST_F(AllReduceCombinerTest, RespectThreshold) {
  auto module = CreateNewVerifiedModule();
  HloComputation* sum = MakeReduction(HloOpcode::kAdd, module.get());

  HloComputation::Builder b(TestName());
  std::vector<HloInstruction*> inputs;
  MakeCrossReplicaReductions({8, 4}, {sum, sum}, &inputs, &b);
  module->AddEntryComputation(b.Build());

  // Run the AllReduce combiner optimization pass with threshold less than
  // the combined size of the all reduce ops so that the combination
  // cannot occur.
  {
    AllReduceCombiner combine((8 + 4) * 1024 - 1, kMaxCombineCount);
    ASSERT_EQ(AllReduceCount(*module), inputs.size());
    TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
    EXPECT_EQ(AllReduceCount(*module), inputs.size());
    EXPECT_FALSE(changed);
  }

  // Run the AllReduce combiner optimization pass again with a slightly
  // higher threshold so that the combination can occur.
  {
    AllReduceCombiner combine((8 + 4) * 1024, kMaxCombineCount);
    ASSERT_EQ(AllReduceCount(*module), inputs.size());
    TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
    EXPECT_EQ(AllReduceCount(*module), 1);
    EXPECT_TRUE(changed);
  }
}

// Tests that dependent all reduces are not combined.
TEST_F(AllReduceCombinerTest, NoDependentCombination) {
  auto module = CreateNewVerifiedModule();
  HloComputation* reduction = MakeReduction(HloOpcode::kAdd, module.get());

  HloComputation::Builder b(TestName());
  auto constant = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(42.3)));
  auto all_reduce = b.AddInstruction(HloInstruction::CreateAllReduce(
      constant->shape(), {constant}, reduction, /*replica_groups=*/{},
      /*constrain_layout=*/false, /*channel_id=*/nullopt,
      /*use_global_device_ids=*/false));
  b.AddInstruction(HloInstruction::CreateAllReduce(
      constant->shape(), {all_reduce}, reduction,
      /*replica_groups=*/{}, /*constrain_layout=*/false,
      /*channel_id=*/nullopt, /*use_global_device_ids=*/false));

  module->AddEntryComputation(b.Build());

  AllReduceCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(AllReduceCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllReduceCount(*module), 2);
  EXPECT_FALSE(changed);
}

// Tests that AllReduce ops with different groups are not combined.
TEST_F(AllReduceCombinerTest, GroupAllReduce) {
  auto module = CreateNewVerifiedModule(TestName(), /*replica_count=*/4);
  HloComputation::Builder b(TestName());
  HloComputation* reduction = MakeReduction(HloOpcode::kAdd, module.get());

  auto constant = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(42.3)));
  auto crs0 = b.AddInstruction(HloInstruction::CreateAllReduce(
      constant->shape(), {constant}, reduction,
      CreateReplicaGroups({{0, 1}, {2, 3}}),
      /*constrain_layout=*/false,
      /*channel_id=*/nullopt, /*use_global_device_ids=*/false));
  auto crs1 = b.AddInstruction(HloInstruction::CreateAllReduce(
      constant->shape(), {constant}, reduction,
      CreateReplicaGroups({{0, 2}, {1, 3}}),
      /*constrain_layout=*/false,
      /*channel_id=*/nullopt, /*use_global_device_ids=*/false));
  b.AddInstruction(HloInstruction::CreateTuple({crs0, crs1}));

  module->AddEntryComputation(b.Build());

  AllReduceCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(AllReduceCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllReduceCount(*module), 2);
  EXPECT_FALSE(changed);
}

TEST_F(AllReduceCombinerTest, DomainPreventsCombining) {
  const char* const hlo_string = R"(
HloModule Module

summit {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY entry {
  param0 = f32[128] parameter(0), sharding={maximal device=0}
  param1 = f32[128] parameter(1), sharding={maximal device=1}
  crs0 = f32[128] all-reduce(param0),
    replica_groups={}, to_apply=summit, sharding={maximal device=0}
  crs1 = f32[128] all-reduce(param1),
    replica_groups={}, to_apply=summit, sharding={maximal device=1}
  domain0 = f32[128] domain(crs0),
    domain={kind="sharding", entry={{maximal device=0}, {maximal device=1}}, exit={maximal device=0}}
  domain1 = f32[128] domain(crs1),
    domain={kind="sharding", entry={{maximal device=0}, {maximal device=1}}, exit={maximal device=1}}
  ROOT tuple = (f32[128], f32[128]) tuple(domain0, domain1),
    sharding={{maximal device=0}, {maximal device=1}}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LOG(INFO) << "Original module:\n" << module->ToString();

  AllReduceCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(AllReduceCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllReduceCount(*module), 2);
  EXPECT_FALSE(changed);
}

// This test checks that two CRS instructions that are in separate domains
// but with the same domain metadata can be combined.
TEST_F(AllReduceCombinerTest, CombineFromTwoDomainsWithSameMetadata) {
  const char* const hlo_string = R"(
HloModule Module

summit {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY entry {
  param0 = f32[128] parameter(0), sharding={maximal device=0}
  param1 = f32[128] parameter(1), sharding={maximal device=1}
  param2 = f32[128] parameter(2), sharding={maximal device=1}
  crs0 = f32[128] all-reduce(param0),
    replica_groups={}, to_apply=summit, sharding={maximal device=0}
  crs1 = f32[128] all-reduce(param1),
    replica_groups={}, to_apply=summit, sharding={maximal device=1}
  crs2 = f32[128] all-reduce(param2),
    replica_groups={}, to_apply=summit, sharding={maximal device=0}
  domain0 = f32[128] domain(crs0),
    domain={kind="sharding", entry={{maximal device=0}, {maximal device=1},
    {maximal device=0}}, exit={maximal device=0}}
  domain1 = f32[128] domain(crs1),
    domain={kind="sharding", entry={{maximal device=0}, {maximal device=1},
    {maximal device=0}}, exit={maximal device=1}}
  domain2 = f32[128] domain(crs2),
    domain={kind="sharding", entry={{maximal device=0}, {maximal device=1},
    {maximal device=0}}, exit={maximal device=0}}
  ROOT tuple = (f32[128], f32[128], f32[128]) tuple(domain0, domain1, domain2),
    sharding={{maximal device=0}, {maximal device=1}, {maximal device=0}}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllReduceCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(AllReduceCount(*module), 3);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllReduceCount(*module), 2);
  EXPECT_TRUE(changed);

  // Verify that the sharding is combined correctly.
  const HloInstruction* param0 =
      module->entry_computation()->parameter_instruction(0);
  ASSERT_EQ(param0->user_count(), 1);
  const HloInstruction* combined_ar = param0->users().front();
  ASSERT_EQ(combined_ar->opcode(), HloOpcode::kAllReduce);
  EXPECT_THAT(combined_ar, testing::opcode_matchers::Sharding(
                               "{{maximal device=0}, {maximal device=0}}"));
}

TEST_F(AllReduceCombinerTest, DoNotCombineCrossShardAndCrossReplicaInSPMD) {
  const char* const hlo_string = R"(
HloModule Module

summit {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY entry {
  param0 = f32[128] parameter(0), sharding={maximal device=0}
  param1 = f32[128] parameter(1), sharding={maximal device=1}
  cross_shard_ar = f32[128] all-reduce(param0),
    replica_groups={{0}}, to_apply=summit, channel_id=1
  cross_replica_ar = f32[128] all-reduce(param1),
    replica_groups={{0}}, to_apply=summit, sharding={maximal device=1}
  ROOT tuple = (f32[128], f32[128]) tuple(cross_shard_ar, cross_replica_ar)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllReduceCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(AllReduceCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllReduceCount(*module), 2);
  EXPECT_FALSE(changed);
}

TEST_F(AllReduceCombinerTest, CrossCoreAllReduce) {
  const char* const hlo_string = R"(
HloModule Module

summit {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY entry {
  param0 = f32[128] parameter(0), sharding={maximal device=0}
  param1 = f32[128] parameter(1), sharding={maximal device=1}
  crs00 = f32[128] all-reduce(param0),
    replica_groups={{0}}, channel_id=1, to_apply=summit,
    sharding={maximal device=0}
  crs01 = f32[128] all-reduce(param1),
    replica_groups={{0}}, channel_id=1, to_apply=summit,
    sharding={maximal device=1}
  crs10 = f32[128] all-reduce(param0),
    replica_groups={{0}}, channel_id=2, to_apply=summit,
    sharding={maximal device=0}
  crs11 = f32[128] all-reduce(param1),
    replica_groups={{0}}, channel_id=2, to_apply=summit,
    sharding={maximal device=1}
  domain0 = f32[128] domain(crs00),
    domain={kind="sharding", entry={maximal device=0}, exit={maximal device=1}}
  ROOT add = f32[128] add(domain0, crs11),
    sharding={maximal device=1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllReduceCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(AllReduceCount(*module), 4);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllReduceCount(*module), 2);
  EXPECT_TRUE(changed);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Add(op::Domain(op::GetTupleElement(AllOf(
                          op::AllReduce(op::Parameter(0), op::Parameter(0)),
                          op::Shape("(f32[128], f32[128])")))),
                      op::GetTupleElement(AllOf(
                          op::AllReduce(op::Parameter(1), op::Parameter(1)),
                          op::Shape("(f32[128], f32[128])")))));
}

TEST_F(AllReduceCombinerTest, CrossCombineGroupCycle) {
  const char* const hlo_string = R"(
HloModule module

%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

%max {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] maximum(lhs, rhs)
}
ENTRY %comp {
  p0 = f32[128] parameter(0)
  p1 = f32[128] parameter(1)

  crs00 = f32[128] all-reduce(p0), to_apply=add
  crs10 = f32[128] all-reduce(p1), to_apply=max

  crs01 = f32[128] all-reduce(crs00), to_apply=max
  crs11 = f32[128] all-reduce(crs10), to_apply=add
  add0 = f32[128] add(crs01, crs11)

  crs02 = f32[128] all-reduce(add0), to_apply=add
  crs12 = f32[128] all-reduce(crs11), to_apply=add
  ROOT tuple = (f32[128], f32[128]) tuple(crs02, crs12)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllReduceCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(AllReduceCount(*module), 6);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllReduceCount(*module), 4);
  EXPECT_TRUE(changed);

  auto crs0 = op::AllReduce(op::Parameter(0), op::AllReduce(op::Parameter(1)));
  auto add = op::Add(op::AllReduce(op::GetTupleElement(crs0, 0)),
                     op::GetTupleElement(crs0, 1));
  auto crs1 = op::AllReduce(add, op::GetTupleElement(crs0));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::GetTupleElement(crs1, 0), op::GetTupleElement(crs1, 1)));
}

}  // namespace
}  // namespace xla
