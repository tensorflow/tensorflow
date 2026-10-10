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

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

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

TEST_F(HloReplicationAnalysisTest,
       CrossPartitionSpmdWithModuleParameterShardings) {
  const std::string module_str = R"(
HloModule CrossPartitionSpmdWithModuleParameterShardings

ENTRY entry {
  param = (f32[4096,4096]{1,0}, f32[4096,4096]{1,0})
    parameter(0), sharding={{maximal device=0}, {replicated}}
  gte0 = f32[4096,4096]{1,0} get-tuple-element(param), index=0
  gte1 = f32[4096,4096]{1,0} get-tuple-element(param), index=1
  ROOT add = f32[4096,4096]{1,0} add(gte0, gte1)
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        module_str, /*replica_count=*/4));
  HloInstruction* param = module->entry_computation()->parameter_instruction(0);
  module->set_spmd_parameters_shardings({param->sharding()});
  param->clear_sharding();

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloReplicationAnalysis> analysis,
      HloReplicationAnalysis::Run(module.get(), /*cross_partition_spmd=*/true));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "gte0"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "gte1"), {}));
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

TEST_F(HloReplicationAnalysisTest, MultipleCallSites) {
  const std::string module_str = R"(
HloModule MultipleCallSites

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
  call0 = (f32[], f32[]) call(get-tuple-element, get-tuple-element.1), to_apply=call_body
  call1 = (f32[], f32[]) call(get-tuple-element.1, get-tuple-element), to_apply=call_body
  ROOT ret = tuple(call0, call1)
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
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "add"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "multiply"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "fusion"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "fusion"), {1}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "call0"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "call0"), {1}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "call1"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "call1"), {1}));
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

// Loop carried values that are partially replicated with two different
// groupings, unique, and replicated, at their own tuple indices. The loop
// needs a second iteration to settle: the partially replicated all-reduce
// results only reach the parameter after the first pass over the body.
TEST_F(HloReplicationAnalysisTest, WhileLoopPartialReplicationPerIndex) {
  const std::string module_str = R"hlo(
HloModule WhileLoopPartialReplicationPerIndex

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

cond {
  cond_param = (f32[8], f32[8], f32[8], u32[]) parameter(0)
  i = u32[] get-tuple-element(cond_param), index=3
  limit = u32[] constant(4)
  ROOT lt = pred[] compare(i, limit), direction=LT
}

body {
  body_param = (f32[8], f32[8], f32[8], u32[]) parameter(0)
  x = f32[8] get-tuple-element(body_param), index=0
  y = f32[8] get-tuple-element(body_param), index=1
  u = f32[8] get-tuple-element(body_param), index=2
  i = u32[] get-tuple-element(body_param), index=3
  xu = f32[8] add(x, u)
  yu = f32[8] add(y, u)
  ar0 = f32[8] all-reduce(xu), to_apply=sum, replica_groups={{0,1},{2,3}}
  ar1 = f32[8] all-reduce(yu), to_apply=sum, replica_groups={{0,2},{1,3}}
  mixed = f32[8] add(ar0, ar1)
  one = u32[] constant(1)
  next_i = u32[] add(i, one)
  ROOT tuple = (f32[8], f32[8], f32[8], u32[]) tuple(ar0, ar1, mixed, next_i)
}

ENTRY entry {
  p0 = f32[8] parameter(0), parameter_replication={true}
  p1 = f32[8] parameter(1), parameter_replication={true}
  p2 = f32[8] parameter(2), parameter_replication={false}
  zero = u32[] constant(0)
  init = (f32[8], f32[8], f32[8], u32[]) tuple(p0, p1, p2, zero)
  ROOT while = (f32[8], f32[8], f32[8], u32[]) while(init), condition=cond, body=body
}
)hlo";
  const std::vector<ReplicaGroup> pairs01_23 =
      CreateReplicaGroups({{0, 1}, {2, 3}});
  const std::vector<ReplicaGroup> pairs02_13 =
      CreateReplicaGroups({{0, 2}, {1, 3}});

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        module_str, /*replica_count=*/4));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                       HloReplicationAnalysis::RunWithPartialReplication(
                           module.get(), /*cross_partition_spmd=*/false));

  for (const char* name : {"while", "body_param", "tuple"}) {
    const HloInstruction* inst = FindInstruction(module.get(), name);
    EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(inst, {0})) << name;
    EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(inst, {0}, pairs01_23))
        << name;
    EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(inst, {0}, pairs02_13))
        << name;
    EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(inst, {1})) << name;
    EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(inst, {1}, pairs01_23))
        << name;
    EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(inst, {1}, pairs02_13))
        << name;
    EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(inst, {2})) << name;
    EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(inst, {2}, pairs01_23))
        << name;
    EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(inst, {2}, pairs02_13))
        << name;
    EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(inst, {3})) << name;
    EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(inst, {3}, pairs01_23))
        << name;
  }
  // u is unique, so the all-reduce operands are unique in every pass and each
  // all-reduce result is replicated within its own groups only.
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "xu"), {}, pairs01_23));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "yu"), {}, pairs02_13));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "ar0"), {}, pairs01_23));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "ar1"), {}, pairs02_13));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "mixed"), {}, pairs01_23));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "mixed"), {}, pairs02_13));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "lt"), {}));
}

// The loop state enters through a tuple instruction, whose tree is replicated
// at the tuple root as well as at the leaves. The condition turns
// non replicated after the first pass, and the body parameter must then be
// unique at every index, the tuple root included.
TEST_F(HloReplicationAnalysisTest,
       NonReplicatedConditionMarksBodyAtEveryIndex) {
  const std::string module_str = R"hlo(
HloModule NonReplicatedConditionMarksBodyAtEveryIndex

cond {
  cond_param = (f32[8], u32[]) parameter(0)
  i = u32[] get-tuple-element(cond_param), index=1
  limit = u32[] constant(5)
  ROOT lt = pred[] compare(i, limit), direction=LT
}

body {
  body_param = (f32[8], u32[]) parameter(0)
  x = f32[8] get-tuple-element(body_param), index=0
  i = u32[] get-tuple-element(body_param), index=1
  replica-id = u32[] replica-id()
  next_i = u32[] add(i, replica-id)
  ROOT tuple = (f32[8], u32[]) tuple(x, next_i)
}

ENTRY entry {
  p0 = f32[8] parameter(0), parameter_replication={true}
  zero = u32[] constant(0)
  init = (f32[8], u32[]) tuple(p0, zero)
  ROOT while = (f32[8], u32[]) while(init), condition=cond, body=body
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        module_str, /*replica_count=*/2));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                       HloReplicationAnalysis::Run(
                           module.get(), /*cross_partition_spmd=*/false));
  const HloInstruction* body_param =
      FindInstruction(module.get(), "body_param");
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(body_param, {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(body_param, {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(body_param, {1}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "x"), {}));
  // The body root tuple is unique at its own index too, and so are the loop
  // result and the condition parameter that merge it.
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "while"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "cond_param"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "cond_param"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "while"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "while"), {1}));
  // The loop input is outside the body and keeps its replicated tuple root.
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "init"), {}));
}

// The trip count depends on the partition, so the partitions run the infeed in
// the body a different number of times. Its data is unique although its
// sharding is replicated, and so is the loop result that it feeds.
TEST_F(HloReplicationAnalysisTest, InfeedInDivergentLoopIsNotReplicatedSPMD) {
  const std::string module_str = R"hlo(
HloModule InfeedInDivergentLoopIsNotReplicatedSPMD

cond {
  cond_param = (f32[8], u32[]) parameter(0)
  i = u32[] get-tuple-element(cond_param), index=1
  pid = u32[] partition-id()
  cmp_val = u32[] add(i, pid)
  limit = u32[] constant(4)
  ROOT lt = pred[] compare(cmp_val, limit), direction=LT
}

body {
  body_param = (f32[8], u32[]) parameter(0)
  tok = token[] after-all()
  infeed = (f32[8], token[]) infeed(tok),
    sharding={{replicated}, {maximal device=0}}
  infeed_data = f32[8] get-tuple-element(infeed), index=0
  i = u32[] get-tuple-element(body_param), index=1
  one = u32[] constant(1)
  next_i = u32[] add(i, one)
  ROOT tuple = (f32[8], u32[]) tuple(infeed_data, next_i)
}

ENTRY entry {
  p0 = f32[8] parameter(0), sharding={replicated}
  zero = u32[] constant(0)
  init = (f32[8], u32[]) tuple(p0, zero)
  ROOT while = (f32[8], u32[]) while(init), condition=cond, body=body
}
)hlo";
  ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(module_str, /*replica_count=*/1,
                                                /*num_partitions=*/2));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloReplicationAnalysis> analysis,
      HloReplicationAnalysis::Run(module.get(), /*cross_partition_spmd=*/true));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "lt"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "infeed"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "infeed_data"), {}));
  for (const char* name : {"tuple", "while"}) {
    EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
        FindInstruction(module.get(), name), {0}))
        << name;
  }
  // The replicated parameter outside the loop stays replicated.
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "init"), {0}));
}

// Two loops in a called computation share their body. The second loop makes
// the shared body unique at index 0 after the first loop took its result, and
// nothing that feeds the call changes. The next pass of the outer loop still
// has to visit the call and the first loop again for their results to become
// unique.
TEST_F(HloReplicationAnalysisTest, SharedLoopBodyChangesAfterItsFirstCaller) {
  const std::string module_str = R"hlo(
HloModule SharedLoopBodyChangesAfterItsFirstCaller

cond {
  cond_param = (f32[], u32[]) parameter(0)
  i = u32[] get-tuple-element(cond_param), index=1
  limit = u32[] constant(4)
  ROOT lt = pred[] compare(i, limit), direction=LT
}

body {
  body_param = (f32[], u32[]) parameter(0)
  x = f32[] get-tuple-element(body_param), index=0
  i = u32[] get-tuple-element(body_param), index=1
  one = u32[] constant(1)
  next_i = u32[] add(i, one)
  ROOT tuple = (f32[], u32[]) tuple(x, next_i)
}

callee {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  zero = u32[] constant(0)
  init1 = (f32[], u32[]) tuple(a, zero)
  while1 = (f32[], u32[]) while(init1), condition=cond, body=body
  init2 = (f32[], u32[]) tuple(b, zero)
  while2 = (f32[], u32[]) while(init2), condition=cond, body=body
  r1 = f32[] get-tuple-element(while1), index=0
  r2 = f32[] get-tuple-element(while2), index=0
  ROOT out = (f32[], f32[]) tuple(r1, r2)
}

outer_cond {
  outer_cond_param = (f32[], f32[], u32[]) parameter(0)
  j = u32[] get-tuple-element(outer_cond_param), index=2
  limit = u32[] constant(3)
  ROOT lt = pred[] compare(j, limit), direction=LT
}

outer_body {
  outer_param = (f32[], f32[], u32[]) parameter(0)
  outer_a = f32[] get-tuple-element(outer_param), index=0
  outer_b = f32[] get-tuple-element(outer_param), index=1
  outer_j = u32[] get-tuple-element(outer_param), index=2
  call = (f32[], f32[]) call(outer_a, outer_b), to_apply=callee
  call_r1 = f32[] get-tuple-element(call), index=0
  one = u32[] constant(1)
  next_j = u32[] add(outer_j, one)
  ROOT outer_tuple = (f32[], f32[], u32[]) tuple(call_r1, outer_b, next_j)
}

ENTRY entry {
  p0 = f32[] parameter(0), parameter_replication={true}
  p1 = f32[] parameter(1), parameter_replication={false}
  zero = u32[] constant(0)
  init = (f32[], f32[], u32[]) tuple(p0, p1, zero)
  ROOT outer = (f32[], f32[], u32[]) while(init), condition=outer_cond,
      body=outer_body
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        module_str, /*replica_count=*/2));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                       HloReplicationAnalysis::Run(
                           module.get(), /*cross_partition_spmd=*/false));
  for (const char* name : {"while1", "call", "outer"}) {
    EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
        FindInstruction(module.get(), name), {0}))
        << name;
  }
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "outer_a"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "while1"), {1}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "outer_j"), {}));
}

// Two loops in different called computations share a body that returns its
// parameter, and nothing else. The second loop makes the body unique at index
// 0 after the first loop took its result. Only the body root changes, so the
// first loop and its call have to be visited again when the outer loop
// repeats.
TEST_F(HloReplicationAnalysisTest, IdentityLoopBodySharedAcrossCalls) {
  const std::string module_str = R"hlo(
HloModule IdentityLoopBodySharedAcrossCalls

cond1 {
  cond1_param = (f32[], u32[]) parameter(0)
  i = u32[] get-tuple-element(cond1_param), index=1
  limit = u32[] constant(4)
  ROOT lt = pred[] compare(i, limit), direction=LT
}

cond2 {
  cond2_param = (f32[], u32[]) parameter(0)
  i = u32[] get-tuple-element(cond2_param), index=1
  limit = u32[] constant(4)
  ROOT lt = pred[] compare(i, limit), direction=LT
}

body {
  ROOT body_param = (f32[], u32[]) parameter(0)
}

callee1 {
  a = f32[] parameter(0)
  zero = u32[] constant(0)
  init1 = (f32[], u32[]) tuple(a, zero)
  while1 = (f32[], u32[]) while(init1), condition=cond1, body=body
  ROOT r1 = f32[] get-tuple-element(while1), index=0
}

callee2 {
  b = f32[] parameter(0)
  zero = u32[] constant(0)
  init2 = (f32[], u32[]) tuple(b, zero)
  while2 = (f32[], u32[]) while(init2), condition=cond2, body=body
  ROOT r2 = f32[] get-tuple-element(while2), index=0
}

outer_cond {
  outer_cond_param = (f32[], f32[], u32[], f32[]) parameter(0)
  j = u32[] get-tuple-element(outer_cond_param), index=2
  limit = u32[] constant(3)
  ROOT lt = pred[] compare(j, limit), direction=LT
}

outer_body {
  outer_param = (f32[], f32[], u32[], f32[]) parameter(0)
  outer_a = f32[] get-tuple-element(outer_param), index=0
  outer_b = f32[] get-tuple-element(outer_param), index=1
  outer_j = u32[] get-tuple-element(outer_param), index=2
  call1 = f32[] call(outer_a), to_apply=callee1
  one = u32[] constant(1)
  next_j = u32[] add(outer_j, one)
  call2 = f32[] call(outer_b), to_apply=callee2
  ROOT outer_tuple = (f32[], f32[], u32[], f32[]) tuple(call1, outer_b, next_j,
      call2)
}

ENTRY entry {
  p0 = f32[] parameter(0), parameter_replication={true}
  p1 = f32[] parameter(1), parameter_replication={false}
  zero = u32[] constant(0)
  init = (f32[], f32[], u32[], f32[]) tuple(p0, p1, zero, p1)
  ROOT outer = (f32[], f32[], u32[], f32[]) while(init), condition=outer_cond,
      body=outer_body
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        module_str, /*replica_count=*/2));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                       HloReplicationAnalysis::Run(
                           module.get(), /*cross_partition_spmd=*/false));
  for (const char* name : {"while1", "outer"}) {
    EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
        FindInstruction(module.get(), name), {0}))
        << name;
  }
  for (const char* name : {"call1", "outer_a"}) {
    EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
        FindInstruction(module.get(), name), {}))
        << name;
  }
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "while1"), {1}));
}

// An all-reduce over groups of one device yields a partially replicated value
// whose device sets are all singletons. Merging it with itself gives unique, so
// in a loop body it turns unique when the body is visited again: with partial
// replication every visit evaluates every instruction.
TEST_F(HloReplicationAnalysisTest, SingletonGroupsInLoopBodyBecomeUnique) {
  const std::string module_str = R"hlo(
HloModule SingletonGroupsInLoopBodyBecomeUnique

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

cond {
  cond_param = (f32[], u32[]) parameter(0)
  i = u32[] get-tuple-element(cond_param), index=1
  limit = u32[] constant(4)
  ROOT lt = pred[] compare(i, limit), direction=LT
}

body {
  body_param = (f32[], u32[]) parameter(0)
  x = f32[] get-tuple-element(body_param), index=0
  i = u32[] get-tuple-element(body_param), index=1
  all-reduce = f32[] all-reduce(x), replica_groups={{0},{1}}, to_apply=sum
  one = u32[] constant(1)
  next_i = u32[] add(i, one)
  ROOT tuple = (f32[], u32[]) tuple(all-reduce, next_i)
}

ENTRY entry {
  p0 = f32[] parameter(0), parameter_replication={false}
  zero = u32[] constant(0)
  init = (f32[], u32[]) tuple(p0, zero)
  ROOT while = (f32[], u32[]) while(init), condition=cond, body=body
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        module_str, /*replica_count=*/2));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                       HloReplicationAnalysis::RunWithPartialReplication(
                           module.get(), /*cross_partition_spmd=*/false));
  const std::vector<ReplicaGroup> singletons = CreateReplicaGroups({{0}, {1}});
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "all-reduce"), {}, singletons));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple"), {0}, singletons));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple"), {1}));
}

}  // namespace
}  // namespace xla
