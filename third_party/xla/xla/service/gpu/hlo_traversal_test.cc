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

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;

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
      negate = f32[128] negate(sum)
      fusion = f32[] fusion(p0, negate), kind=kLoop, calls=fused_computation
      ROOT difference = f32[] subtract(fusion, p0)
    })";

TEST_F(HloTraversalTest, TraverseFusion) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> visited_nodes;
  HloBfsConsumersFirstTraversal(
      {module->GetComputationWithName("fused_computation")->root_instruction()},
      DefaultFusionBoundaryFn, [&](const HloInstruction& node) {
        visited_nodes.emplace_back(node.name());
        return TraversalResult::kVisitOperands;
      });

  EXPECT_THAT(visited_nodes, ElementsAre("reduce.1", "mul", "p0.1", "p1.1"));
}

TEST_F(HloTraversalTest, TraverseFusionPartially) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> visited_nodes;
  HloBfsConsumersFirstTraversal(
      {module->GetComputationWithName("fused_computation")->root_instruction()},
      DefaultFusionBoundaryFn, [&](const HloInstruction& node) {
        visited_nodes.emplace_back(node.name());
        return node.opcode() == HloOpcode::kReduce
                   ? TraversalResult::kVisitOperands
                   : TraversalResult::kDoNotVisitOperands;
      });

  EXPECT_THAT(visited_nodes, ElementsAre("reduce.1", "mul", "p0.1"));
}

TEST_F(HloTraversalTest, AbortTraversal) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> visited_nodes;
  HloBfsConsumersFirstTraversal(
      {module->GetComputationWithName("fused_computation")->root_instruction()},
      DefaultFusionBoundaryFn, [&](const HloInstruction& node) {
        visited_nodes.emplace_back(node.name());
        return node.opcode() == HloOpcode::kReduce
                   ? TraversalResult::kVisitOperands
                   : TraversalResult::kAbortTraversal;
      });

  EXPECT_THAT(visited_nodes, ElementsAre("reduce.1", "mul"));
}

TEST_F(HloTraversalTest, TraversePartialFusion) {
  // Verifies that we correctly traverse the fusion that would result if we
  // fused the negation into fused_computation.
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> visited_nodes;

  auto* fused_computation = module->GetComputationWithName("fused_computation");
  HloBfsConsumersFirstTraversal(
      {fused_computation->root_instruction()},
      [&](const HloInstruction& producer, const HloInstruction& consumer) {
        return &consumer == fused_computation->parameter_instruction(0) ||
               consumer.opcode() == HloOpcode::kNegate;
      },
      [&](const HloInstruction& node) {
        visited_nodes.emplace_back(node.name());
        return TraversalResult::kVisitOperands;
      });

  EXPECT_THAT(visited_nodes,
              ElementsAre("reduce.1", "mul", "p0.1", "p1.1", "negate"));
}

TEST_F(HloTraversalTest, FindArguments) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> producers;
  FindFusionArguments(
      {module->GetComputationWithName("fused_computation")->root_instruction()},
      DefaultFusionBoundaryFn, [&](const HloInstruction& producer) {
        producers.emplace_back(producer.name());
      });
  EXPECT_THAT(producers, ElementsAre("p0", "negate"));
}

TEST_F(HloTraversalTest, FindArgumentsAfterFusion) {
  // Verifies that we correctly find the arguments after fusing the negation.
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> producers;
  auto* fused_computation = module->GetComputationWithName("fused_computation");
  FindFusionArguments(
      {fused_computation->root_instruction()},
      [&](const HloInstruction& producer, const HloInstruction& consumer) {
        return &consumer == fused_computation->parameter_instruction(0) ||
               consumer.opcode() == HloOpcode::kNegate;
      },
      [&](const HloInstruction& producer) {
        producers.emplace_back(producer.name());
      });
  EXPECT_THAT(producers, ElementsAre("p0", "sum"));
}

TEST_F(HloTraversalTest, FuseEverything) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> producers;
  auto* fused_computation = module->GetComputationWithName("fused_computation");
  FindFusionArguments(
      {fused_computation->root_instruction()},
      [&](const HloInstruction& producer, const HloInstruction& consumer) {
        return producer.opcode() == HloOpcode::kParameter &&
               producer.parent()->IsEntryComputation();
      },
      [&](const HloInstruction& producer) {
        producers.emplace_back(producer.name());
      });
  EXPECT_THAT(producers, ElementsAre("p0", "p1"));
}

TEST_F(HloTraversalTest, FuseConsumer) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> visited_nodes;
  HloBfsConsumersFirstTraversal(
      {module->entry_computation()->root_instruction()},
      [](const HloInstruction& producer, const HloInstruction& consumer) {
        return consumer.opcode() == HloOpcode::kParameter ||
               (producer.opcode() == HloOpcode::kParameter &&
                consumer.opcode() == HloOpcode::kSubtract);
      },
      [&](const HloInstruction& node) {
        visited_nodes.emplace_back(node.name());
        return TraversalResult::kVisitOperands;
      });
  EXPECT_THAT(visited_nodes, ElementsAre("difference", "fusion", "reduce.1",
                                         "mul", "p0.1", "p1.1"));
}

TEST_F(HloTraversalTest, FindIf) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> visited_nodes;
  auto* result = HloFindIf(
      {module->GetComputationWithName("fused_computation")->root_instruction()},
      DefaultFusionBoundaryFn, [&](const HloInstruction& node) {
        return node.opcode() == HloOpcode::kMultiply;
      });
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->name(), "mul");
}

TEST_F(HloTraversalTest, NotFound) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> visited_nodes;
  auto* result = HloFindIf(
      {module->GetComputationWithName("fused_computation")->root_instruction()},
      DefaultFusionBoundaryFn,
      [&](const HloInstruction& node) { return false; });
  ASSERT_EQ(result, nullptr);
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
  auto* producer =
      module->entry_computation()->GetInstructionWithName("negate");
  auto* consumer =
      module->entry_computation()->GetInstructionWithName("fusion.1");

  auto roots = GetFusionRoots(*consumer->fused_instructions_computation());
  auto boundary = MakeProducerConsumerFusion(*producer, *consumer);
  std::vector<std::string> nodes;
  HloBfsConsumersFirstTraversal(roots, boundary,
                                [&](const HloInstruction& node) {
                                  nodes.emplace_back(node.name());
                                  return TraversalResult::kVisitOperands;
                                });
  std::vector<std::string> params;
  FindFusionArguments(roots, boundary, [&](const HloInstruction& param) {
    params.emplace_back(param.name());
  });

  EXPECT_THAT(nodes, ElementsAre("reduce.1", "mul", "p0.1", "p1.1", "negate"));
  EXPECT_THAT(params, ElementsAre("p0", "sum"));
}

TEST_F(HloTraversalTest, FuseFusionProducer) {
  auto module = ParseAndReturnVerifiedModule(kTwoFusions).value();
  auto* producer =
      module->entry_computation()->GetInstructionWithName("fusion.2");
  auto* consumer =
      module->entry_computation()->GetInstructionWithName("difference");

  auto boundary = MakeProducerConsumerFusion(*producer, *consumer);
  std::vector<std::string> nodes;
  HloBfsConsumersFirstTraversal({consumer}, boundary,
                                [&](const HloInstruction& node) {
                                  nodes.emplace_back(node.name());
                                  return TraversalResult::kVisitOperands;
                                });
  std::vector<std::string> params;
  FindFusionArguments({consumer}, boundary, [&](const HloInstruction& param) {
    params.emplace_back(param.name());
  });

  EXPECT_THAT(
      nodes, ElementsAre("difference", "fusion.2", "reduce.2", "p1.2", "p0.2"));
  EXPECT_THAT(params, ElementsAre("p0", "negate", "fusion.1"));
}

TEST_F(HloTraversalTest, FuseFusionConsumerAndProducer) {
  auto module = ParseAndReturnVerifiedModule(kTwoFusions).value();
  auto* producer =
      module->entry_computation()->GetInstructionWithName("fusion.1");
  auto* consumer =
      module->entry_computation()->GetInstructionWithName("fusion.2");

  auto roots = GetFusionRoots(*consumer->fused_instructions_computation());
  auto boundary = MakeProducerConsumerFusion(*producer, *consumer);
  std::vector<std::string> nodes;
  HloBfsConsumersFirstTraversal(roots, boundary,
                                [&](const HloInstruction& node) {
                                  nodes.emplace_back(node.name());
                                  return TraversalResult::kVisitOperands;
                                });
  std::vector<std::string> params;
  FindFusionArguments(roots, boundary, [&](const HloInstruction& param) {
    params.emplace_back(param.name());
  });

  EXPECT_THAT(nodes, ElementsAre("reduce.2", "p1.2", "p0.2", "fusion.1",
                                 "reduce.1", "mul", "p0.1", "p1.1"));
  EXPECT_THAT(params, ElementsAre("negate", "p0"));
}

TEST_F(HloTraversalTest, SingleInstructionFusionOfFusion) {
  auto module = ParseAndReturnVerifiedModule(kTwoFusions).value();
  auto* fusion =
      module->entry_computation()->GetInstructionWithName("fusion.1");

  auto boundary = MakeSingleInstructionFusion(*fusion);
  std::vector<std::string> nodes;
  HloBfsConsumersFirstTraversal({fusion}, boundary,
                                [&](const HloInstruction& node) {
                                  nodes.emplace_back(node.name());
                                  return TraversalResult::kVisitOperands;
                                });

  EXPECT_THAT(nodes,
              ElementsAre("fusion.1", "reduce.1", "mul", "p0.1", "p1.1"));
}

TEST_F(HloTraversalTest, SingleInstructionFusionOfInstruction) {
  auto module = ParseAndReturnVerifiedModule(kTwoFusions).value();
  auto* negate = module->entry_computation()->GetInstructionWithName("negate");

  auto boundary = MakeSingleInstructionFusion(*negate);
  std::vector<std::string> nodes;
  HloBfsConsumersFirstTraversal({negate}, boundary,
                                [&](const HloInstruction& node) {
                                  nodes.emplace_back(node.name());
                                  return TraversalResult::kVisitOperands;
                                });

  EXPECT_THAT(nodes, ElementsAre("negate"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
