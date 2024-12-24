/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/analysis/hlo_replication_analysis.h"

#include <memory>
#include <string>
#include <vector>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/types.h"

namespace xla {
namespace {

class HloReplicationAnalysisTest : public HloHardwareIndependentTestBase {
 public:
  std::vector<ReplicaGroup> CreateReplicaGroups(
      std::vector<std::vector<int>> replica_ids) {
    std::vector<ReplicaGroup> replica_groups(replica_ids.size());
    for (int i = 0; i < replica_ids.size(); ++i) {
      for (int id : replica_ids[i]) {
        replica_groups[i].add_replica_ids(id);
      }
    }
    return replica_groups;
  }
};

TEST_F(HloReplicationAnalysisTest, NoControlFlow) {
  const std::string module_str = R"(
HloModule NoControlFlow

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

sum.u32 {
  a = u32[] parameter(0)
  b = u32[] parameter(1)
  ROOT add.2 = u32[] add(a, b)
}

ENTRY entry {
  param = (f32[4096,4096]{1,0}, f32[4096,4096]{1,0}) parameter(0)
  get-tuple-element.2 = f32[4096,4096]{1,0} get-tuple-element(param), index=0
  get-tuple-element.3 = f32[4096,4096]{1,0} get-tuple-element(param), index=1
  after-all.1 = token[] after-all()
  replica-id = u32[] replica-id()
  infeed = (f32[4096,4096]{1,0}, token[]) infeed(after-all.1)
  get-tuple-element.5 = f32[4096,4096]{1,0} get-tuple-element(infeed), index=0
  dot = f32[4096,4096]{1,0} dot(get-tuple-element.5, get-tuple-element.3),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  all-reduce = f32[4096,4096]{1,0} all-reduce(dot), replica_groups={},
    to_apply=sum
  subtract = f32[4096,4096]{1,0} subtract(get-tuple-element.3, all-reduce)
  all-reduce-partitions = u32[] all-reduce(replica-id), channel_id=1,
    to_apply=sum.u32, replica_groups={{0},{1},{2},{3}}
  all-reduce-subgroup = u32[] all-reduce(replica-id),
    replica_groups={{0,1},{2,3}}, to_apply=sum.u32
  ROOT add = f32[4096,4096]{1,0} add(get-tuple-element.2, subtract)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           module_str, /*replica_count=*/4));
  auto param = module->entry_computation()->parameter_instruction(0);
  param->set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool>{false, true});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::Run(
                              module.get(), /*cross_partition_spmd=*/false));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "get-tuple-element.2"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "get-tuple-element.3"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "get-tuple-element.5"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "dot"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "all-reduce"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "subtract"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "add"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "replica-id"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "all-reduce-partitions"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "all-reduce-subgroup"), {}));
}

TEST_F(HloReplicationAnalysisTest, NoControlFlowSPMD) {
  const std::string module_str = R"(
HloModule NoControlFlow

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

sum.u32 {
  a = u32[] parameter(0)
  b = u32[] parameter(1)
  ROOT add.2 = u32[] add(a, b)
}

ENTRY entry {
  param = (f32[4096,4096]{1,0}, f32[4096,4096]{1,0}, f32[4096,4096]{1,0})
    parameter(0), sharding={{maximal device=0}, {replicated}, {replicated}}
  get-tuple-element.2 = f32[4096,4096]{1,0} get-tuple-element(param), index=0
  get-tuple-element.3 = f32[4096,4096]{1,0} get-tuple-element(param), index=1
  get-tuple-element.4 = f32[4096,4096]{1,0} get-tuple-element(param), index=2
  after-all.1 = token[] after-all()
  replica-id = u32[] replica-id()
  partition-id = u32[] partition-id()
  infeed = ((f32[4096,4096]{1,0}, f32[8,8]{1,0}), token[]) infeed(after-all.1),
    sharding={{maximal device=0}, {replicated}, {maximal device=0}}
  infeed-data = (f32[4096,4096]{1,0}, f32[8,8]{1,0}) get-tuple-element(infeed),
    index=0
  get-tuple-element.5 = f32[4096,4096]{1,0} get-tuple-element(infeed-data),
    index=0
  get-tuple-element.6 = f32[8,8]{1,0} get-tuple-element(infeed-data), index=1
  dot = f32[4096,4096]{1,0} dot(get-tuple-element.5, get-tuple-element.3),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot.2 = f32[4096,4096]{1,0} dot(get-tuple-element.4, get-tuple-element.3),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  all-reduce = f32[4096,4096]{1,0} all-reduce(dot), replica_groups={},
    to_apply=sum
  all-reduce.2 = f32[4096,4096]{1,0} all-reduce(dot.2), replica_groups={},
    to_apply=sum
  all-reduce-subgroup = f32[4096,4096]{1,0} all-reduce(dot),
    replica_groups={{0,1},{2,3}}, to_apply=sum
  all-reduce-partitions = f32[4096,4096]{1,0} all-reduce(get-tuple-element.2),
    channel_id=1, to_apply=sum
  all-reduce-partitions.2 = f32[4096,4096]{1,0} all-reduce(get-tuple-element.4),
    channel_id=1, to_apply=sum
  subtract = f32[4096,4096]{1,0} subtract(get-tuple-element.3,
    all-reduce-partitions)
  subtract.2 = f32[4096,4096]{1,0} subtract(get-tuple-element.3,
    all-reduce-partitions.2)
  all-reduce-same-operand = u32[] all-reduce(replica-id), to_apply=sum.u32
  all-reduce-same-operand-subgroup = u32[] all-reduce(replica-id),
    replica_groups={{0,1},{2,3}}, to_apply=sum.u32
  all-reduce-different-operand = u32[] all-reduce(partition-id),
    to_apply=sum.u32
  add = f32[4096,4096]{1,0} add(get-tuple-element.2, subtract)
  ROOT add.2 = f32[4096,4096]{1,0} add(get-tuple-element.4, subtract.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           module_str, /*replica_count=*/4));
  auto param = module->entry_computation()->parameter_instruction(0);
  param->set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool>{false, true, false});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloReplicationAnalysis> analysis,
      HloReplicationAnalysis::Run(module.get(), /*cross_partition_spmd=*/true));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "get-tuple-element.2"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "get-tuple-element.3"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "get-tuple-element.4"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "get-tuple-element.5"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "get-tuple-element.6"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "dot"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "dot.2"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "all-reduce"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "all-reduce.2"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "all-reduce-partitions"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "all-reduce-partitions.2"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "subtract"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "subtract.2"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "add"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "replica-id"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "partition-id"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "all-reduce-same-operand"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "all-reduce-same-operand-subgroup"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "all-reduce-different-operand"), {}));
}

TEST_F(HloReplicationAnalysisTest, NestedCall) {
  const std::string module_str = R"(
HloModule NestedCall

fusion_computation {
  fusion_p0 = f32[] parameter(0)
  fusion_p1 = f32[] parameter(1)
  add = f32[] add(fusion_p0, fusion_p0)
  multiply = f32[] multiply(add, fusion_p1)
  ROOT tuple = (f32[], f32[]) tuple(add, multiply)
}

call_body {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT fusion = (f32[], f32[]) fusion(a, b), kind=kLoop, calls=fusion_computation
}

ENTRY entry {
  param = (f32[], f32[]) parameter(0)
  get-tuple-element = f32[] get-tuple-element(param), index=0
  get-tuple-element.1 = f32[] get-tuple-element(param), index=1
  ROOT call = (f32[], f32[]) call(get-tuple-element, get-tuple-element.1), to_apply=call_body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_str));
  auto param = module->entry_computation()->parameter_instruction(0);
  param->set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool>{true, false});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::Run(
                              module.get(), /*cross_partition_spmd=*/false));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "get-tuple-element"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "get-tuple-element.1"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "add"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "multiply"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "fusion"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "fusion"), {1}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "call"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "call"), {1}));
}

TEST_F(HloReplicationAnalysisTest, SimpleWhileLoop) {
  const std::string module_str = R"(
HloModule SimpleWhileLoop

cond {
  cond_param = (f32[4096,4096]{1,0}, u32[]) parameter(0)
  get-tuple-element = u32[] get-tuple-element(cond_param), index=1
  constant.3 = u32[] constant(5)
  ROOT greater-than = pred[] compare(get-tuple-element, constant.3), direction=LT
}

body {
  body_param = (f32[4096,4096]{1,0}, u32[]) parameter(0)
  get-tuple-element.1 = f32[4096,4096]{1,0} get-tuple-element(body_param), index=0
  multiply = f32[4096,4096]{1,0} multiply(get-tuple-element.1, get-tuple-element.1)
  get-tuple-element.6 = u32[] get-tuple-element(body_param), index=1
  constant.1 = u32[] constant(1)
  add = u32[] add(get-tuple-element.6, constant.1)
  ROOT tuple = (f32[4096,4096]{1,0}, u32[]) tuple(multiply, add)
}

ENTRY SimpleWhileLoop {
  param = (f32[4096,4096]{1,0}, u32[]) parameter(0)
  ROOT while = (f32[4096,4096]{1,0}, u32[]) while(param), condition=cond, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_str));
  auto param = module->entry_computation()->parameter_instruction(0);
  param->set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool>{true, true});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::Run(
                              module.get(), /*cross_partition_spmd=*/false));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple"), {0}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple"), {1}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "while"), {0}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "while"), {1}));
}

TEST_F(HloReplicationAnalysisTest,
       WhileLoopParameterAliasingNonReplicatedOutput) {
  const std::string module_str = R"(
HloModule WhileLoopParameterAliasingNonReplicatedOutput

cond {
  cond_param = (f32[4096,4096]{1,0}, u32[]) parameter(0)
  get-tuple-element = u32[] get-tuple-element(cond_param), index=1
  constant.3 = u32[] constant(5)
  ROOT greater-than = pred[] compare(get-tuple-element, constant.3), direction=LT
}

body {
  body_param = (f32[4096,4096]{1,0}, u32[]) parameter(0)
  get-tuple-element.1 = f32[4096,4096]{1,0} get-tuple-element(body_param), index=0
  multiply = f32[4096,4096]{1,0} multiply(get-tuple-element.1, get-tuple-element.1)
  after-all.1 = token[] after-all()
  infeed = (f32[4096,4096]{1,0}, token[]) infeed(after-all.1)
  get-tuple-element.5 = f32[4096,4096]{1,0} get-tuple-element(infeed), index=0
  subtract = f32[4096,4096]{1,0} subtract(get-tuple-element.5, multiply)
  get-tuple-element.6 = u32[] get-tuple-element(body_param), index=1
  constant.1 = u32[] constant(1)
  add = u32[] add(get-tuple-element.6, constant.1)
  ROOT tuple = (f32[4096,4096]{1,0}, u32[]) tuple(subtract, add)
}

ENTRY WhileLoopParameterAliasingNonReplicatedOutput {
  param = (f32[4096,4096]{1,0}, u32[]) parameter(0)
  ROOT while = (f32[4096,4096]{1,0}, u32[]) while(param), condition=cond, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_str));
  auto param = module->entry_computation()->parameter_instruction(0);
  param->set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool>{true, true});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::Run(
                              module.get(), /*cross_partition_spmd=*/false));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "multiply"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple"), {0}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple"), {1}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "while"), {0}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "while"), {1}));
}

TEST_F(HloReplicationAnalysisTest, WhileLoopDifferentCondition) {
  const std::string module_str = R"(
HloModule WhileLoopDifferentCondition

cond {
  cond_param = (f32[4096,4096]{1,0}, u32[]) parameter(0)
  get-tuple-element = u32[] get-tuple-element(cond_param), index=1
  constant.3 = u32[] constant(5)
  ROOT greater-than = pred[] compare(get-tuple-element, constant.3), direction=LT
}

body {
  body_param = (f32[4096,4096]{1,0}, u32[]) parameter(0)
  get-tuple-element.1 = f32[4096,4096]{1,0} get-tuple-element(body_param), index=0
  multiply = f32[4096,4096]{1,0} multiply(get-tuple-element.1, get-tuple-element.1)
  get-tuple-element.6 = u32[] get-tuple-element(body_param), index=1
  replica-id = u32[] replica-id()
  add = u32[] add(get-tuple-element.6, replica-id)
  ROOT tuple = (f32[4096,4096]{1,0}, u32[]) tuple(multiply, add)
}

ENTRY WhileLoopDifferentCondition {
  param = (f32[4096,4096]{1,0}, u32[]) parameter(0)
  ROOT while = (f32[4096,4096]{1,0}, u32[]) while(param), condition=cond, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_str));
  auto param = module->entry_computation()->parameter_instruction(0);
  param->set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool>{true, true});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::Run(
                              module.get(), /*cross_partition_spmd=*/false));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "while"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "while"), {1}));
}

TEST_F(HloReplicationAnalysisTest, SimpleConditional) {
  const std::string module_str = R"(
HloModule SimpleConditional

Negate {
  x = (f32[], f32[]) parameter(0)
  get-tuple-element = f32[] get-tuple-element(x), index=0
  negate = f32[] negate(get-tuple-element)
  get-tuple-element.1 = f32[] get-tuple-element(x), index=1
  negate.1 = f32[] negate(get-tuple-element.1)
  ROOT tuple = (f32[], f32[]) tuple(negate, negate.1)
}

Identity {
  ROOT y = (f32[], f32[]) parameter(0)
}

Floor {
  z = (f32[], f32[]) parameter(0)
  get-tuple-element.2 = f32[] get-tuple-element(z), index=0
  floor = f32[] floor(get-tuple-element.2)
  get-tuple-element.3 = f32[] get-tuple-element(z), index=1
  floor.1 = f32[] floor(get-tuple-element.3)
  ROOT tuple.1 = (f32[], f32[]) tuple(floor, floor.1)
}

ENTRY entry {
  param = ((f32[], f32[]), (f32[], f32[]), (f32[], f32[]), s32[]) parameter(0)
  get-tuple-element.4 = (f32[], f32[]) get-tuple-element(param), index=0
  get-tuple-element.5 = (f32[], f32[]) get-tuple-element(param), index=1
  get-tuple-element.6 = (f32[], f32[]) get-tuple-element(param), index=2
  get-tuple-element.7 = s32[] get-tuple-element(param), index=3
  ROOT conditional = (f32[], f32[]) conditional(get-tuple-element.7, get-tuple-element.4, get-tuple-element.5, get-tuple-element.6), branch_computations={Negate, Identity, Floor}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_str));
  auto param = module->entry_computation()->parameter_instruction(0);
  param->set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool>{true, true, true, true, false, true, true});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::Run(
                              module.get(), /*cross_partition_spmd=*/false));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple"), {0}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple"), {1}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "y"), {0}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "y"), {1}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple.1"), {0}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple.1"), {1}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "conditional"), {0}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "conditional"), {1}));
}

TEST_F(HloReplicationAnalysisTest, ConditionalWithDifferentPredicates) {
  const std::string module_str = R"(
HloModule ConditionalWithDifferentPredicates

Negate {
  x = (f32[], f32[]) parameter(0)
  get-tuple-element = f32[] get-tuple-element(x), index=0
  negate = f32[] negate(get-tuple-element)
  get-tuple-element.1 = f32[] get-tuple-element(x), index=1
  negate.1 = f32[] negate(get-tuple-element.1)
  ROOT tuple = (f32[], f32[]) tuple(negate, negate.1)
}

Identity {
  ROOT y = (f32[], f32[]) parameter(0)
}

Floor {
  z = (f32[], f32[]) parameter(0)
  get-tuple-element.2 = f32[] get-tuple-element(z), index=0
  floor = f32[] floor(get-tuple-element.2)
  get-tuple-element.3 = f32[] get-tuple-element(z), index=1
  floor.1 = f32[] floor(get-tuple-element.3)
  ROOT tuple.1 = (f32[], f32[]) tuple(floor, floor.1)
}

ENTRY entry {
  param = ((f32[], f32[]), (f32[], f32[]), (f32[], f32[])) parameter(0)
  get-tuple-element.4 = (f32[], f32[]) get-tuple-element(param), index=0
  get-tuple-element.5 = (f32[], f32[]) get-tuple-element(param), index=1
  get-tuple-element.6 = (f32[], f32[]) get-tuple-element(param), index=2
  replica-id = u32[] replica-id()
  id = s32[] bitcast-convert(replica-id)
  ROOT conditional = (f32[], f32[]) conditional(id, get-tuple-element.4,
    get-tuple-element.5, get-tuple-element.6),
    branch_computations={Negate, Identity, Floor}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_str));
  auto param = module->entry_computation()->parameter_instruction(0);
  param->set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool>{true, true, true, true, true, true});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::Run(
                              module.get(), /*cross_partition_spmd=*/false));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple"), {1}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "y"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "y"), {1}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple.1"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple.1"), {1}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "conditional"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "conditional"), {1}));
}

TEST_F(HloReplicationAnalysisTest, X64SplitCombine) {
  const std::string module_str = R"(
HloModule SimpleX64SplitCombine

ENTRY entry {
  param = (f64[]) parameter(0)
  gte = f64[] get-tuple-element(param), index=0
  param-low = f32[] custom-call(gte), custom_call_target="X64SplitLow"
  param-high = f32[] custom-call(gte), custom_call_target="X64SplitHigh"
  ROOT result-combine = f64[] custom-call(param-low, param-high), custom_call_target="X64Combine"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_str));
  auto param = module->entry_computation()->parameter_instruction(0);
  param->set_parameter_replicated_at_leaf_buffers(absl::Span<const bool>{true});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::Run(
                              module.get(), /*cross_partition_spmd=*/false));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "gte"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "param-low"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "param-high"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "result-combine"), {}));
}

TEST_F(HloReplicationAnalysisTest, CrossModuleAndReplicaAllReduce) {
  const std::string module_str = R"(
HloModule CrossModuleAndReplicaAllReduce

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  param = (f32[], f32[]) parameter(0)
  get-tuple-element.0 = f32[] get-tuple-element(param), index=0
  get-tuple-element.1 = f32[] get-tuple-element(param), index=1
  ar0 = f32[] all-reduce(get-tuple-element.0), to_apply=sum, replica_groups={{0,1}}
  ar1 = f32[] all-reduce(get-tuple-element.1), to_apply=sum, replica_groups={{0},{1}}
  ROOT tuple = (f32[], f32[]) tuple(ar0, ar1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           module_str, /*replica_count=*/2));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::Run(
                              module.get(), /*cross_partition_spmd=*/false));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "ar0"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "ar1"), {}));
}

TEST_F(HloReplicationAnalysisTest, GlobalIdAllGather) {
  const std::string module_str = R"(
HloModule GlobalIdAllGather

ENTRY entry {
  param = f32[1] parameter(0)
  ag1 = f32[2] all-gather(param), replica_groups={{0,1},{2,3}}, dimensions={0},
    use_global_device_ids=true, channel_id=1
  ag2 = f32[2] all-gather(param), replica_groups={{0,2},{1,3}}, dimensions={0},
    use_global_device_ids=true, channel_id=2
  ag3 = f32[4] all-gather(param), replica_groups={{0,1,2,3}}, dimensions={0},
    use_global_device_ids=true, channel_id=3
  ag4 = f32[2] all-gather(param), replica_groups={{0,3},{1,2}}, dimensions={0},
    use_global_device_ids=true, channel_id=4
  ROOT tuple = (f32[2], f32[2], f32[4], f32[2]) tuple(ag1, ag2, ag3, ag4)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2,
                                                /*num_partitions=*/2));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloReplicationAnalysis> replica_analysis,
      HloReplicationAnalysis::Run(module.get(),
                                  /*cross_partition_spmd=*/false));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloReplicationAnalysis> partition_analysis,
      HloReplicationAnalysis::Run(module.get(),
                                  /*cross_partition_spmd=*/true));
  EXPECT_FALSE(replica_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "ag1"), {}));
  EXPECT_TRUE(replica_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "ag2"), {}));
  EXPECT_TRUE(replica_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "ag3"), {}));
  EXPECT_FALSE(replica_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "ag4"), {}));

  EXPECT_TRUE(partition_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "ag1"), {}));
  EXPECT_FALSE(partition_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "ag2"), {}));
  EXPECT_TRUE(partition_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "ag3"), {}));
  EXPECT_FALSE(partition_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "ag4"), {}));
}

TEST_F(HloReplicationAnalysisTest, PartiallyReplicatedDynamicSlice) {
  const std::string module_str = R"(
HloModule PartiallyReplicatedDynamicSlice

ENTRY entry {
  constant = s32[8] constant({1, 3, 9, 10, 1, 3, 9, 10})
  replica-id = u32[] replica-id()
  ROOT dynamic-slice = s32[1] dynamic-slice(constant, replica-id), dynamic_slice_sizes={1}
}
)";
  const int replica_count = 8;
  const int num_partitions = 1;
  const bool cross_partition_spmd = false;
  const std::vector<ReplicaGroup> replica_groups0 =
      CreateReplicaGroups({{0, 4}, {1, 5}, {2, 6}, {3, 7}});
  const std::vector<ReplicaGroup> replica_groups1 =
      CreateReplicaGroups({{0, 1, 2, 3}, {4, 5, 6, 7}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(module_str, replica_count, num_partitions));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloReplicationAnalysis> replica_analysis,
      HloReplicationAnalysis::RunWithPartialReplication(module.get(),
                                                        cross_partition_spmd));

  EXPECT_FALSE(replica_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "dynamic-slice"), {}));

  EXPECT_TRUE(replica_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "dynamic-slice"), {}, replica_groups0));

  EXPECT_FALSE(replica_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "dynamic-slice"), {}, replica_groups1));
}

TEST_F(HloReplicationAnalysisTest,
       PartiallyReplicatedAllGatherFlattenedIDReplicaAnalysis) {
  const std::string module_str = R"(
HloModule PartiallyReplicatedAllGatherFlattenedIDReplicaAnalysis

ENTRY entry {
  param = s32[2] parameter(0)
  all-gather0 = s32[4] all-gather(param), dimensions={0}, replica_groups={{0,2},{4,6},{1,3},{5,7}}, channel_id=1, use_global_device_ids=true
  all-gather1 = s32[4] all-gather(param), dimensions={0}, replica_groups={{0,4},{2,6},{1,5},{3,7}}, channel_id=2, use_global_device_ids=true
  ROOT tuple = (s32[4], s32[4]) tuple(all-gather0, all-gather1)
}
)";
  const int replica_count = 4;
  const int num_partitions = 2;
  const bool cross_partition_spmd = false;
  const std::vector<ReplicaGroup> replica_groups0 =
      CreateReplicaGroups({{0, 1}, {2, 3}});
  const std::vector<ReplicaGroup> replica_groups1 =
      CreateReplicaGroups({{0, 2}, {1, 3}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto module_replica_analysis,
      ParseAndReturnVerifiedModule(module_str, replica_count, num_partitions));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloReplicationAnalysis> replica_analysis,
      HloReplicationAnalysis::RunWithPartialReplication(
          module_replica_analysis.get(), cross_partition_spmd));

  EXPECT_FALSE(replica_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module_replica_analysis.get(), "all-gather0"), {}));

  EXPECT_TRUE(replica_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module_replica_analysis.get(), "all-gather0"), {},
      replica_groups0));

  EXPECT_FALSE(replica_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module_replica_analysis.get(), "all-gather0"), {},
      replica_groups1));

  EXPECT_FALSE(replica_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module_replica_analysis.get(), "all-gather1"), {}));

  EXPECT_TRUE(replica_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module_replica_analysis.get(), "all-gather1"), {},
      replica_groups1));

  EXPECT_FALSE(replica_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module_replica_analysis.get(), "all-gather1"), {},
      replica_groups0));
}

TEST_F(HloReplicationAnalysisTest,
       PartiallyReplicatedAllGatherFlattenedIDPartitionAnalysis) {
  const std::string module_str = R"(
HloModule PartiallyReplicatedAllGatherFlattenedIDPartitionAnalysis

ENTRY entry {
  param = s32[2] parameter(0)
  all-gather0 = s32[4] all-gather(param), dimensions={0}, replica_groups={{0,1},{2,3},{4,5},{6,7}}, channel_id=1, use_global_device_ids=true
  all-gather1 = s32[4] all-gather(param), dimensions={0}, replica_groups={{0,2},{1,3},{4,6},{5,7}}, channel_id=2, use_global_device_ids=true
  ROOT tuple = (s32[4], s32[4]) tuple(all-gather0, all-gather1)
}
)";
  const int replica_count = 2;
  const int num_partitions = 4;
  const bool cross_partition_spmd = true;
  const std::vector<ReplicaGroup> replica_groups0 =
      CreateReplicaGroups({{0, 1}, {2, 3}});
  const std::vector<ReplicaGroup> replica_groups1 =
      CreateReplicaGroups({{0, 2}, {1, 3}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto module_partition_analysis,
      ParseAndReturnVerifiedModule(module_str, replica_count, num_partitions));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloReplicationAnalysis> partition_analysis,
      HloReplicationAnalysis::RunWithPartialReplication(
          module_partition_analysis.get(), cross_partition_spmd));

  EXPECT_FALSE(partition_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module_partition_analysis.get(), "all-gather0"), {}));

  EXPECT_TRUE(partition_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module_partition_analysis.get(), "all-gather0"), {},
      replica_groups0));

  EXPECT_FALSE(partition_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module_partition_analysis.get(), "all-gather0"), {},
      replica_groups1));

  EXPECT_FALSE(partition_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module_partition_analysis.get(), "all-gather1"), {}));

  EXPECT_TRUE(partition_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module_partition_analysis.get(), "all-gather1"), {},
      replica_groups1));

  EXPECT_FALSE(partition_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module_partition_analysis.get(), "all-gather1"), {},
      replica_groups0));
}

TEST_F(
    HloReplicationAnalysisTest,
    PartiallyReplicatedAllGatherFlattenedIDPartitionAnalysisAsymmetricGroups) {
  const std::string module_str = R"(
HloModule GlobalIdAllGather

ENTRY entry {
  param = f32[1] parameter(0)
  ROOT all_gather = f32[6] all-gather(param), replica_groups={{0,1,2,3,6,7},{4,5,8,9,10,11}}, dimensions={0}, use_global_device_ids=true, channel_id=1
}
)";
  const int replica_count = 2;
  const int num_partitions = 6;
  const bool cross_partition_spmd = true;
  const std::vector<ReplicaGroup> replica_groups0 =
      CreateReplicaGroups({{0, 1}, {2, 3}, {4, 5}});
  const std::vector<ReplicaGroup> replica_groups1 =
      CreateReplicaGroups({{0, 1, 2}, {3, 4, 5}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(module_str, replica_count, num_partitions));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloReplicationAnalysis> partition_analysis,
      HloReplicationAnalysis::RunWithPartialReplication(module.get(),
                                                        cross_partition_spmd));

  EXPECT_TRUE(partition_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "all_gather"), {}, replica_groups0));
  EXPECT_FALSE(partition_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "all_gather"), {}, replica_groups1));
}

TEST_F(HloReplicationAnalysisTest,
       PartiallyReplicatedAllGatherFlattenedIDReplicaAnalysisAsymmetricGroups) {
  const std::string module_str = R"(
HloModule GlobalIdAllGather

ENTRY entry {
  param = f32[1] parameter(0)
  ROOT all_gather = f32[6] all-gather(param), replica_groups={{0,1,2,3,4,6},{5,7,8,9,10,11}}, dimensions={0}, use_global_device_ids=true, channel_id=1
}
)";
  const int replica_count = 6;
  const int num_partitions = 2;
  const bool cross_partition_spmd = false;
  const std::vector<ReplicaGroup> replica_groups0 =
      CreateReplicaGroups({{0, 1}, {2, 3}, {4, 5}});
  const std::vector<ReplicaGroup> replica_groups1 =
      CreateReplicaGroups({{0, 1, 2}, {3, 4, 5}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(module_str, replica_count, num_partitions));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloReplicationAnalysis> partition_analysis,
      HloReplicationAnalysis::RunWithPartialReplication(module.get(),
                                                        cross_partition_spmd));

  EXPECT_TRUE(partition_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "all_gather"), {}, replica_groups0));
  EXPECT_FALSE(partition_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "all_gather"), {}, replica_groups1));
}

TEST_F(
    HloReplicationAnalysisTest,
    PartiallyReplicatedAllGatherFlattenedIDPartitionAnalysisAsymmetricPartial) {
  const std::string module_str = R"(
HloModule GlobalIdAllGather

ENTRY entry {
  param = f32[1] parameter(0)
  ROOT all_gather = f32[6] all-gather(param), replica_groups={{0,1,2,3,6,7},{4,5,8,9,10,11},{12,13,14,15,16,17}}, dimensions={0}, use_global_device_ids=true, channel_id=1
}
)";
  const int replica_count = 3;
  const int num_partitions = 6;
  const bool cross_partition_spmd = true;
  const std::vector<ReplicaGroup> replica_groups0 =
      CreateReplicaGroups({{0, 1}, {2, 3}, {4, 5}});
  const std::vector<ReplicaGroup> replica_groups1 =
      CreateReplicaGroups({{0, 1, 2}, {3, 4, 5}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(module_str, replica_count, num_partitions));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloReplicationAnalysis> partition_analysis,
      HloReplicationAnalysis::RunWithPartialReplication(module.get(),
                                                        cross_partition_spmd));

  EXPECT_TRUE(partition_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "all_gather"), {}, replica_groups0));
  EXPECT_FALSE(partition_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "all_gather"), {}, replica_groups1));
}

TEST_F(HloReplicationAnalysisTest,
       PartiallyReplicatedAllGatherFlattenedIDPartitionAnalysisAsymmetricAll) {
  const std::string module_str = R"(
HloModule GlobalIdAllGather

ENTRY entry {
  param = f32[1] parameter(0)
  ROOT all_gather = f32[4] all-gather(param), replica_groups={{0,2,5,7},{1,3,4,6}}, dimensions={0}, use_global_device_ids=true, channel_id=1
}
)";
  const int replica_count = 2;
  const int num_partitions = 4;
  const bool cross_partition_spmd = true;
  const std::vector<ReplicaGroup> replica_groups =
      CreateReplicaGroups({{0, 1}, {2, 3}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(module_str, replica_count, num_partitions));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloReplicationAnalysis> partition_analysis,
      HloReplicationAnalysis::RunWithPartialReplication(module.get(),
                                                        cross_partition_spmd));

  EXPECT_FALSE(partition_analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "all_gather"), {}, replica_groups));
}

TEST_F(HloReplicationAnalysisTest,
       PartiallyReplicatedAllGatherFlattenedIDPartitionAnalysisMerge) {
  const std::string module_str = R"(
  HloModule module

  ENTRY entry {
    param0 = f32[2] parameter(0)
    param1 = f32[4] parameter(1)
    all_gather0 = f32[8] all-gather(param0), dimensions={0}, replica_groups={{0,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,15}}, use_global_device_ids=true, channel_id=1
    all_gather1 = f32[8] all-gather(param1), dimensions={0}, replica_groups={{0,1},{2,3},{4,5},{6,7},{8,9},{10,11},{12,13},{14,15}}, use_global_device_ids=true, channel_id=2
    all_gather2 = f32[8] all-gather(param0), dimensions={0}, replica_groups={{0,3,4,5},{1,2,6,7},{8,11,12,13},{9,10,14,15}}, use_global_device_ids=true, channel_id=3
    add0 = f32[8] add(all_gather0, all_gather1)
    add1 = f32[8] add(all_gather0, all_gather2)
    ROOT tuple = (f32[8], f32[8]) tuple(add0, add1)
    }
  )";
  const int replica_count = 2;
  const int num_partitions = 8;
  const bool cross_partition_spmd = true;
  const std::vector<ReplicaGroup> replica_groups0 =
      CreateReplicaGroups({{0, 1, 2, 3}, {4, 5, 6, 7}});
  const std::vector<ReplicaGroup> replica_groups1 =
      CreateReplicaGroups({{0, 1}, {2, 3}, {4, 5}, {6, 7}});
  const std::vector<ReplicaGroup> replica_groups2 =
      CreateReplicaGroups({{1, 2}, {0, 3}, {4, 5}, {6, 7}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(module_str, replica_count, num_partitions));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::RunWithPartialReplication(
                              module.get(), cross_partition_spmd));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "add0"), {}, replica_groups0));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "add0"), {}, replica_groups1));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "add1"), {}, replica_groups0));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "add1"), {}, replica_groups2));
}

TEST_F(HloReplicationAnalysisTest, OptimizationBarrier) {
  const std::string module_str = R"(
HloModule OptimizationBarrier

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  param = (f32[], f32[]) parameter(0)
  get-tuple-element.0 = f32[] get-tuple-element(param), index=0
  get-tuple-element.1 = f32[] get-tuple-element(param), index=1
  ar0 = f32[] all-reduce(get-tuple-element.0), to_apply=sum, replica_groups={{0,1}}
  ar1 = f32[] all-reduce(get-tuple-element.1), to_apply=sum, replica_groups={{0},{1}}
  tuple = (f32[], f32[]) tuple(ar0, ar1)
  opt-barrier = (f32[], f32[]) opt-barrier(tuple)
  gte.0 = f32[] get-tuple-element(opt-barrier), index=0
  gte.1 = f32[] get-tuple-element(opt-barrier), index=1
  ROOT tuple.1 = (f32[], f32[]) tuple(gte.0, gte.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           module_str, /*replica_count=*/2));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::Run(
                              module.get(), /*cross_partition_spmd=*/false));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "gte.0"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "gte.1"), {}));
}

}  // namespace
}  // namespace xla
