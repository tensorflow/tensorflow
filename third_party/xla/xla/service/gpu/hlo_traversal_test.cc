/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/service/gpu/hlo_traversal.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;

MATCHER_P(InstructionAdaptorName, name, "") { return arg.name() == name; }

class HloTraversalTest : public HloTestBase {};

const char kTestModule[] = R"(
    HloModule test

    scalar_add_computation {
      scalar_lhs.0 = f32[] parameter(0)
      scalar_rhs.0 = f32[] parameter(1)
      ROOT add.0 = f32[] add(scalar_lhs.0, scalar_rhs.0)
    }

    fused_computation {
      p0.1 = f32[] parameter(0)
      p1.1 = f32[128] parameter(1)
      mul = f32[128] multiply(p1.1, p1.1)
      ROOT reduce.1 = f32[] reduce(mul, p0.1), dimensions={0}, to_apply=scalar_add_computation
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1 = f32[128] parameter(1)
      sum = f32[128] add(p1, p1)
      log = f32[128] log(sum)
      negate = f32[128] negate(log)
      fusion = f32[] fusion(p0, negate), kind=kLoop, calls=fused_computation
      ROOT difference = f32[] subtract(fusion, p0)
    })";

TEST_F(HloTraversalTest, AdaptorOperands) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();

  HloInstructionAdaptor instr{
      *module->entry_computation()->GetInstructionWithName("difference")};

  auto operands = instr.GetOperands();
  ASSERT_EQ(operands.size(), 2);
  EXPECT_EQ(operands[0].name(), "reduce.1");
  EXPECT_EQ(operands[1].name(), "p0");
}

TEST_F(HloTraversalTest, AdaptorUsers) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    fused_computation {
      p0 = f32[] parameter(0)
      neg = f32[] negate(p0)
      add = f32[] add(p0, neg)
      ROOT t = (f32[], f32[]) tuple(neg, add)
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      fusion = (f32[], f32[]) fusion(p0), kind=kLoop, calls=fused_computation
      ROOT gte = f32[] get-tuple-element(fusion), index=0
    }
  )")
                    .value();

  auto get_single_user = [](auto instr) {
    auto users = instr.GetUsers();
    EXPECT_EQ(users.size(), 1);
    return users[0];
  };

  HloInstructionAdaptor add{*module->GetComputationWithName("fused_computation")
                                 ->GetInstructionWithName("add")};
  EXPECT_EQ(get_single_user(add).name(), "t");
  EXPECT_EQ(get_single_user(get_single_user(add)).name(), "gte");
}

TEST_F(HloTraversalTest, TraverseFusion) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> visited_nodes;
  std::vector<std::string> visited_args;
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion"));
  HloBfsConsumersFirstTraversal(
      fusion->GetRoots(), *fusion,
      [&](HloInstructionAdaptor node) {
        visited_nodes.emplace_back(node.name());
        return TraversalResult::kVisitOperands;
      },
      [&](HloInstructionAdaptor arg) {
        visited_args.emplace_back(arg.name());
      });

  EXPECT_THAT(visited_nodes, ElementsAre("reduce.1", "mul"));
  EXPECT_THAT(visited_args, ElementsAre("p0", "negate"));
}

TEST_F(HloTraversalTest, AbortTraversal) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion"));
  std::vector<std::string> visited_nodes;
  HloBfsConsumersFirstTraversal(fusion->GetRoots(), *fusion,
                                [&](HloInstructionAdaptor node) {
                                  visited_nodes.emplace_back(node.name());
                                  return node.opcode() == HloOpcode::kReduce
                                             ? TraversalResult::kVisitOperands
                                             : TraversalResult::kAbortTraversal;
                                });

  EXPECT_THAT(visited_nodes, ElementsAre("reduce.1", "mul"));
}

TEST_F(HloTraversalTest, FindArguments) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion"));
  std::vector<std::string> producers;
  FindFusionArguments(*fusion, [&](HloInstructionAdaptor producer) {
    producers.emplace_back(producer.name());
  });
  EXPECT_THAT(producers, ElementsAre("p0", "negate"));
}

TEST_F(HloTraversalTest, FindArgumentsAfterFusion) {
  // Verifies that we correctly find the arguments after fusing the negation.
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  auto producer = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("negate"));
  auto consumer = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion"));
  std::vector<std::string> producers;
  FindFusionArguments(
      ProducerConsumerFusion(std::move(producer), std::move(consumer)),
      [&](HloInstructionAdaptor producer) {
        producers.emplace_back(producer.name());
      });
  EXPECT_THAT(producers, ElementsAre("p0", "log"));
}

TEST_F(HloTraversalTest, FindIf) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion"));
  std::vector<std::string> visited_nodes;
  auto result =
      HloFindIf(fusion->GetRoots(), *fusion, [&](HloInstructionAdaptor node) {
        return node.opcode() == HloOpcode::kMultiply;
      });
  ASSERT_NE(result, std::nullopt);
  ASSERT_EQ(result->name(), "mul");
}

TEST_F(HloTraversalTest, NotFound) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion"));
  std::vector<std::string> visited_nodes;
  auto result = HloFindIf(fusion->GetRoots(), *fusion,
                          [&](HloInstructionAdaptor node) { return false; });
  ASSERT_EQ(result, std::nullopt);
}

const char kTwoFusions[] = R"(
    HloModule test

    scalar_add_computation {
      scalar_lhs.0 = f32[] parameter(0)
      scalar_rhs.0 = f32[] parameter(1)
      ROOT add.0 = f32[] add(scalar_lhs.0, scalar_rhs.0)
    }

    fused_computation_1 {
      p0.1 = f32[] parameter(0)
      p1.1 = f32[128] parameter(1)
      mul = f32[128] multiply(p1.1, p1.1)
      ROOT reduce.1 = f32[] reduce(mul, p0.1), dimensions={0}, to_apply=scalar_add_computation
    }

    fused_computation_2 {
      p0.2 = f32[] parameter(0)
      p1.2 = f32[128] parameter(1)
      ROOT reduce.2 = f32[] reduce(p1.2, p0.2), dimensions={0}, to_apply=scalar_add_computation
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1 = f32[128] parameter(1)
      sum = f32[128] add(p1, p1)
      negate = f32[128] negate(sum)
      fusion.1 = f32[] fusion(p0, negate), kind=kLoop, calls=fused_computation_1
      fusion.2 = f32[] fusion(fusion.1, negate), kind=kLoop, calls=fused_computation_2
      ROOT difference = f32[] subtract(fusion.2, p0)
    })";

TEST_F(HloTraversalTest, FuseFusionConsumer) {
  auto module = ParseAndReturnVerifiedModule(kTwoFusions).value();

  auto producer = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("negate"));
  auto consumer = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion.1"));
  ProducerConsumerFusion fusion(std::move(producer), std::move(consumer));

  std::vector<std::string> nodes;
  std::vector<std::string> params;
  HloBfsConsumersFirstTraversal(
      fusion.GetRoots(), fusion,
      [&](HloInstructionAdaptor node) {
        nodes.emplace_back(node.name());
        return TraversalResult::kVisitOperands;
      },
      [&](HloInstructionAdaptor param) { params.emplace_back(param.name()); });

  EXPECT_THAT(nodes, ElementsAre("reduce.1", "mul", "negate"));
  EXPECT_THAT(params, ElementsAre("p0", "sum"));
}

TEST_F(HloTraversalTest, FuseFusionProducer) {
  auto module = ParseAndReturnVerifiedModule(kTwoFusions).value();

  auto producer = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion.2"));
  auto consumer = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("difference"));
  ProducerConsumerFusion fusion(std::move(producer), std::move(consumer));

  std::vector<std::string> nodes;
  std::vector<std::string> params;
  HloBfsConsumersFirstTraversal(
      fusion.GetRoots(), fusion,
      [&](HloInstructionAdaptor node) {
        nodes.emplace_back(node.name());
        return TraversalResult::kVisitOperands;
      },
      [&](HloInstructionAdaptor arg) { params.emplace_back(arg.name()); });

  EXPECT_THAT(nodes, ElementsAre("difference", "reduce.2"));
  EXPECT_THAT(params, ElementsAre("p0", "negate", "reduce.1"));
}

TEST_F(HloTraversalTest, FuseFusionConsumerAndProducer) {
  auto module = ParseAndReturnVerifiedModule(kTwoFusions).value();
  auto producer = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion.1"));
  auto consumer = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion.2"));
  ProducerConsumerFusion fusion(std::move(producer), std::move(consumer));

  std::vector<std::string> nodes;
  HloBfsConsumersFirstTraversal(fusion.GetRoots(), fusion,
                                [&](HloInstructionAdaptor node) {
                                  nodes.emplace_back(node.name());
                                  return TraversalResult::kVisitOperands;
                                });
  std::vector<std::string> params;
  FindFusionArguments(fusion, [&](const HloInstructionAdaptor& param) {
    params.emplace_back(param.name());
  });

  EXPECT_THAT(nodes, ElementsAre("reduce.2", "reduce.1", "mul"));
  EXPECT_THAT(params, ElementsAre("negate", "p0"));
}

TEST_F(HloTraversalTest, FuseNonFusionConsumerAndProducer) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();

  auto producer = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("log"));
  auto consumer = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("negate"));
  ProducerConsumerFusion fusion(std::move(producer), std::move(consumer));

  std::vector<std::string> nodes;
  HloBfsConsumersFirstTraversal(fusion.GetRoots(), fusion,
                                [&](HloInstructionAdaptor node) {
                                  nodes.emplace_back(node.name());
                                  return TraversalResult::kVisitOperands;
                                });

  EXPECT_THAT(nodes, ElementsAre("negate", "log"));
}

TEST_F(HloTraversalTest, SingleInstructionFusionOfFusion) {
  auto module = ParseAndReturnVerifiedModule(kTwoFusions).value();
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion.1"));

  std::vector<std::string> nodes;
  HloBfsConsumersFirstTraversal(fusion->GetRoots(), *fusion,
                                [&](HloInstructionAdaptor node) {
                                  nodes.emplace_back(node.name());
                                  return TraversalResult::kVisitOperands;
                                });

  EXPECT_THAT(nodes, ElementsAre("reduce.1", "mul"));
}

TEST_F(HloTraversalTest, SingleInstructionFusionOfInstruction) {
  auto module = ParseAndReturnVerifiedModule(kTwoFusions).value();
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("negate"));

  std::vector<std::string> nodes;
  HloBfsConsumersFirstTraversal(fusion->GetRoots(), *fusion,
                                [&](HloInstructionAdaptor node) {
                                  nodes.emplace_back(node.name());
                                  return TraversalResult::kVisitOperands;
                                });

  EXPECT_THAT(nodes, ElementsAre("negate"));
}

TEST_F(HloTraversalTest, MakeInstructionsPostOrder_SingleInstruction) {
  auto module = ParseAndReturnVerifiedModule(kTwoFusions).value();
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("negate"));

  auto nodes = fusion->MakeInstructionPostOrder();
  EXPECT_THAT(nodes, ElementsAre(InstructionAdaptorName("negate")));
}

TEST_F(HloTraversalTest, MakeInstructionsPostOrder_TwoFusions) {
  auto module = ParseAndReturnVerifiedModule(kTwoFusions).value();
  auto fusion = ProducerConsumerFusion(
      module->entry_computation()->GetInstructionWithName("fusion.1"),
      module->entry_computation()->GetInstructionWithName("fusion.2"));

  auto nodes = fusion.MakeInstructionPostOrder();
  EXPECT_THAT(nodes, ElementsAre(InstructionAdaptorName("mul"),
                                 InstructionAdaptorName("reduce.1"),
                                 InstructionAdaptorName("reduce.2")));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
