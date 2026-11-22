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

#include "xla/hlo/transforms/collectives/all_reduce_simplifier.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/pattern_matcher.h"

namespace xla {
namespace {

namespace m = match;

using AllReduceSimplifierTest = HloHardwareIndependentTestBase;

TEST_F(AllReduceSimplifierTest, ReplicatedParameters) {
  const char* kModuleStr = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

max {
  a.1 = f32[] parameter(0)
  b.1 = f32[] parameter(1)
  ROOT max = f32[] maximum(a.1, b.1)
}

min {
  a.2 = f32[] parameter(0)
  b.2 = f32[] parameter(1)
  ROOT min = f32[] minimum(a.2, b.2)
}

sum.1 {
  a.3 = f32[] parameter(0)
  b.3 = f32[] parameter(1)
  ROOT add.1 = f32[] add(a.3, b.3)
}

test {
  p0 = f32[8,16] parameter(0), parameter_replication={true}
  p1 = f32[8,16] parameter(1), parameter_replication={false}
  p2 = f32[] parameter(2), parameter_replication={true}
  all-reduce = f32[8,16] all-reduce(p0), replica_groups={}, to_apply=sum
  all-reduce.1 = f32[8,16] all-reduce(p0), replica_groups={}, to_apply=max
  all-reduce.2 = f32[8,16] all-reduce(p1), replica_groups={}, to_apply=min
  all-reduce.3 = f32[] all-reduce(p2), replica_groups={}, to_apply=sum.1
  ROOT tuple = (f32[8,16], f32[8,16], f32[8,16], f32[]) tuple(all-reduce, all-reduce.1, all-reduce.2, all-reduce.3)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleStr, /*replica_count=*/8));
  AllReduceSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::MultiplyAnyOrder(m::Parameter(0),
                              m::Broadcast(m::Convert(m::ConstantScalar(8)))),
          m::Parameter(0), m::AllReduce(m::Parameter(1)),
          m::MultiplyAnyOrder(m::Parameter(2),
                              m::Convert(m::ConstantScalar(8))))));
}

TEST_F(AllReduceSimplifierTest, AllReduceAfterAllReduce) {
  const char* kModuleStr = R"(
HloModule m

max {
  a.1 = f32[] parameter(0)
  b.1 = f32[] parameter(1)
  ROOT max = f32[] maximum(a.1, b.1)
}

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

test {
  p0 = f32[8,16] parameter(0), parameter_replication={false}
  all-reduce = f32[8,16] all-reduce(p0), replica_groups={}, to_apply=max
  ROOT all-reduce.1 = f32[8,16] all-reduce(all-reduce), replica_groups={}, to_apply=sum
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleStr, /*replica_count=*/8));
  AllReduceSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::MultiplyAnyOrder(
                  m::AllReduce(m::Parameter(0)),
                  m::Broadcast(m::Convert(m::ConstantScalar(8))))));
}

TEST_F(AllReduceSimplifierTest, SubgroupAllReduce) {
  const char* kModuleStr = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

max {
  a.1 = f32[] parameter(0)
  b.1 = f32[] parameter(1)
  ROOT max = f32[] maximum(a.1, b.1)
}

min {
  a.2 = f32[] parameter(0)
  b.2 = f32[] parameter(1)
  ROOT min = f32[] minimum(a.2, b.2)
}

test {
  p0 = f32[8,16] parameter(0), parameter_replication={true}
  p1 = f32[8,16] parameter(1), parameter_replication={false}
  all-reduce = f32[8,16] all-reduce(p0), replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=sum
  all-reduce.1 = f32[8,16] all-reduce(p0), replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=max
  all-reduce.2 = f32[8,16] all-reduce(p1), replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=min
  ROOT tuple = (f32[8,16], f32[8,16], f32[8,16]) tuple(all-reduce, all-reduce.1, all-reduce.2)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleStr, /*replica_count=*/8));
  AllReduceSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::MultiplyAnyOrder(m::Parameter(0),
                              m::Broadcast(m::Convert(m::ConstantScalar(4)))),
          m::Parameter(0), m::AllReduce(m::Parameter(1)))));
}

TEST_F(AllReduceSimplifierTest, TrivialSubgroupAllReduce) {
  const char* kModuleStr = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}


test {
  p0 = f32[8,16] parameter(0), parameter_replication={false}
  ROOT all-reduce = f32[8,16] all-reduce(p0),
    replica_groups={{0},{1},{2},{3},{4},{5},{6},{7}},
    to_apply=sum
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleStr, /*replica_count=*/8));
  AllReduceSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

TEST_F(AllReduceSimplifierTest, TrivialSubgroupNonCrossReplicaAllReduce) {
  const char* kModuleStr = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}


test {
  p0 = f32[8,16] parameter(0), parameter_replication={false}
  ROOT all-reduce = f32[8,16] all-reduce(p0),
    channel_id=1,
    use_global_device_ids=true,
    replica_groups={{0},{1},{2},{3},{4},{5},{6},{7}},
    to_apply=sum
}
)";
  ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, /*replica_count=*/1,
                                                /*num_partitions=*/8));
  module->mutable_config().set_use_spmd_partitioning(true);
  AllReduceSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

TEST_F(AllReduceSimplifierTest, NonCrossReplicaAllReduceAfterAllReduce) {
  const char* kModuleStr = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}


test {
  p0 = f32[8,16] parameter(0), parameter_replication={false}
  all-reduce = f32[8,16] all-reduce(p0),
    channel_id=1,
    use_global_device_ids=true,
    replica_groups={{0,2},{1,3},{4,6},{5,7}},
    to_apply=sum
  ROOT all-reduce.1 = f32[8,16] all-reduce(all-reduce),
    channel_id=2,
    use_global_device_ids=true,
    replica_groups={{0,4},{1,5},{2,6},{3,7}},
    to_apply=sum
}
)";
  ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, /*replica_count=*/1,
                                                /*num_partitions=*/8));
  module->mutable_config().set_use_spmd_partitioning(true);
  AllReduceSimplifier simplifier;
  EXPECT_FALSE(simplifier.Run(module.get()).value());
}

TEST_F(AllReduceSimplifierTest, MPMDNonCrossReplicaAllReduce) {
  const char* kModuleStr = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

test {
  p0 = f32[8,16] parameter(0), parameter_replication={false}
  ROOT all-reduce = f32[8,16] all-reduce(p0),
    channel_id=1,
    replica_groups={{0},{1}},
    to_apply=sum
}
)";
  ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, /*replica_count=*/2,
                                                /*num_partitions=*/1));
  // Mark as MPMD.
  module->mutable_config().set_use_spmd_partitioning(false);
  AllReduceSimplifier simplifier;
  EXPECT_FALSE(simplifier.Run(module.get()).value());
}

TEST_F(AllReduceSimplifierTest, AllGatherSameInputOutputShape) {
  const char* kModuleStr = R"(
HloModule m

test {
  p0 = f32[8,16] parameter(0)
  ROOT all-gather = f32[8,16] all-gather(p0), dimensions={0},
    replica_groups={{0},{1},{2},{3},{4},{5},{6},{7}}
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleStr, /*replica_count=*/8));
  AllReduceSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

TEST_F(AllReduceSimplifierTest, ReduceScatterSameInputOutputShape) {
  const char* kModuleStr = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

test {
  p0 = f32[8,16] parameter(0)
  ROOT reduce-scatter = f32[8,16] reduce-scatter(p0), dimensions={0},
    replica_groups={{0},{1},{2},{3},{4},{5},{6},{7}}, to_apply=sum
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleStr, /*replica_count=*/8));
  AllReduceSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

TEST_F(AllReduceSimplifierTest, ReplicatedParameterWithOr) {
  const char* kModuleStr = R"(
HloModule m

or {
  a = pred[] parameter(0)
  b = pred[] parameter(1)
  ROOT or = pred[] or(a, b)
}

test {
  p0 = pred[8,16] parameter(0), parameter_replication={true}
  ROOT all-reduce = pred[8,16] all-reduce(p0), replica_groups={}, to_apply=or
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleStr, /*replica_count=*/8));
  AllReduceSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

TEST_F(AllReduceSimplifierTest, ReplicatedParameterWithAnd) {
  const char* kModuleStr = R"(
HloModule m

and {
  a = pred[] parameter(0)
  b = pred[] parameter(1)
  ROOT and = pred[] and(a, b)
}

test {
  p0 = pred[8,16] parameter(0), parameter_replication={true}
  ROOT all-reduce = pred[8,16] all-reduce(p0), replica_groups={}, to_apply=and
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleStr, /*replica_count=*/8));
  AllReduceSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

TEST_F(AllReduceSimplifierTest, AsyncAllGatherSameInputOutputShape) {
  const char* kModuleStr = R"(
HloModule m

async_all_gather {
  p0 = f32[8,16] parameter(0)
  ROOT all-gather = f32[8,16] all-gather(p0), dimensions={0},
    replica_groups={{0},{1},{2},{3},{4},{5},{6},{7}}
}

ENTRY test {
  p = f32[8,16] parameter(0)
  async-start = ((f32[8,16]), f32[8,16], u32[]) async-start(p), calls=async_all_gather
  ROOT async-done = f32[8,16] async-done(async-start), calls=async_all_gather
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleStr, /*replica_count=*/8));
  AllReduceSimplifier simplifier;
  ASSERT_OK_AND_ASSIGN(bool changed, simplifier.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

TEST_F(AllReduceSimplifierTest, ReplicatedProductPreservesControlDependencies) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"hlo(
HloModule m, replica_count=2

product {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT result = f32[] multiply(lhs, rhs)
}

ENTRY main {
  p0 = f32[8] parameter(0), parameter_replication={true}
  before = f32[8] negate(p0)
  ar = f32[8] all-reduce(p0), replica_groups={{0,1}}, to_apply=product,
    control-predecessors={before}, metadata={op_name="replicated_product"},
    frontend_attributes={test_attr="value"}
  after = f32[8] negate(p0), control-predecessors={ar}
  ROOT result = (f32[8], f32[8]) tuple(ar, after)
}
)hlo"));
  ASSERT_OK_AND_ASSIGN(bool changed, AllReduceSimplifier().Run(module.get()));
  ASSERT_TRUE(changed);
  const HloInstruction* replacement =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_THAT(replacement,
              GmockMatch(m::Multiply(m::Parameter(0), m::Parameter(0))));
  EXPECT_THAT(replacement->control_predecessors(),
              ::testing::ElementsAre(FindInstruction(module.get(), "before")));
  EXPECT_THAT(FindInstruction(module.get(), "after")->control_predecessors(),
              ::testing::ElementsAre(replacement));
  EXPECT_EQ(replacement->metadata().op_name(), "replicated_product");
  EXPECT_EQ(replacement->frontend_attributes().map().at("test_attr"), "value");
}

struct ProductAllReduceTestCase {
  const char* name;
  const char* element_type;
  const char* dimensions;
  const char* replica_count;
  const char* replica_groups;
  const char* replicated;
  bool expect_change;
};

class ProductAllReduceSimplifierTest
    : public HloHardwareIndependentTestBase,
      public ::testing::WithParamInterface<ProductAllReduceTestCase> {};

TEST_P(ProductAllReduceSimplifierTest, SimplifyOnlyReplicatedPairs) {
  const ProductAllReduceTestCase& test = GetParam();
  ASSERT_OK_AND_ASSIGN(auto module,
                       RunAndCheckHloRewrite(
                           R"hlo(
HloModule m, replica_count=$replicas

product {
  lhs = $type[] parameter(0)
  rhs = $type[] parameter(1)
  ROOT result = $type[] multiply(lhs, rhs)
}

ENTRY main {
  p0 = $type$dims parameter(0), parameter_replication={$replicated}
  ROOT ar = $type$dims all-reduce(p0), replica_groups=$groups, to_apply=product
}
)hlo",
                           AllReduceSimplifier(), test.expect_change,
                           {{"$type", test.element_type},
                            {"$dims", test.dimensions},
                            {"$replicas", test.replica_count},
                            {"$groups", test.replica_groups},
                            {"$replicated", test.replicated}}));
  if (test.expect_change) {
    EXPECT_THAT(module->entry_computation()->root_instruction(),
                GmockMatch(m::Multiply(m::Parameter(0), m::Parameter(0))));
  } else {
    EXPECT_THAT(module->entry_computation()->root_instruction(),
                GmockMatch(m::AllReduce(m::Parameter(0))));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ProductAllReduce, ProductAllReduceSimplifierTest,
    ::testing::Values(
        ProductAllReduceTestCase{"FloatArray", "f32", "[8,16]", "2", "{{0,1}}",
                                 "true", true},
        ProductAllReduceTestCase{"FloatScalar", "f32", "[]", "2", "{}", "true",
                                 true},
        ProductAllReduceTestCase{"IntegerScalar", "s32", "[]", "2", "{{0,1}}",
                                 "true", true},
        ProductAllReduceTestCase{"TwoReplicaSubgroups", "f32", "[8]", "4",
                                 "{{0,2},{1,3}}", "true", true},
        ProductAllReduceTestCase{"NonReplicatedInput", "f32", "[8]", "2",
                                 "{{0,1}}", "false", false},
        ProductAllReduceTestCase{"ThreeReplicas", "f32", "[8]", "3", "{}",
                                 "true", false},
        ProductAllReduceTestCase{"FourReplicas", "f32", "[8]", "4", "{}",
                                 "true", false},
        ProductAllReduceTestCase{"UnequalGroups", "f32", "[8]", "3",
                                 "{{0},{1,2}}", "true", false}),
    [](const ::testing::TestParamInfo<ProductAllReduceTestCase>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace xla
