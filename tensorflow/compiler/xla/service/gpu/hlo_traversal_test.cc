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
#include "tensorflow/compiler/xla/service/gpu/hlo_traversal.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

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

bool FusionInstructionBoundary(const HloInstruction& producer,
                               const HloInstruction& consumer) {
  return consumer.opcode() == HloOpcode::kParameter;
}

TEST_F(HloTraversalTest, TraverseFusion) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> visited_nodes;
  HloBfsConsumersFirstTraversal(
      *module->GetComputationWithName("fused_computation")->root_instruction(),
      FusionInstructionBoundary, [&](const HloInstruction& node) {
        visited_nodes.push_back(std::string(node.name()));
        return TraversalResult::kVisitOperands;
      });

  EXPECT_THAT(visited_nodes, ElementsAre("reduce.1", "mul", "p0.1", "p1.1"));
}

TEST_F(HloTraversalTest, TraverseFusionPartially) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> visited_nodes;
  HloBfsConsumersFirstTraversal(
      *module->GetComputationWithName("fused_computation")->root_instruction(),
      FusionInstructionBoundary, [&](const HloInstruction& node) {
        visited_nodes.push_back(std::string(node.name()));
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
      *module->GetComputationWithName("fused_computation")->root_instruction(),
      FusionInstructionBoundary, [&](const HloInstruction& node) {
        visited_nodes.push_back(std::string(node.name()));
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
      *fused_computation->root_instruction(),
      [&](const HloInstruction& producer, const HloInstruction& consumer) {
        return &consumer == fused_computation->parameter_instruction(0) ||
               consumer.opcode() == HloOpcode::kNegate;
      },
      [&](const HloInstruction& node) {
        visited_nodes.push_back(std::string(node.name()));
        return TraversalResult::kVisitOperands;
      });

  EXPECT_THAT(visited_nodes,
              ElementsAre("reduce.1", "mul", "p0.1", "p1.1", "negate"));
}

TEST_F(HloTraversalTest, FindParameters) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> producers;
  FindFusionParameters(
      *module->GetComputationWithName("fused_computation")->root_instruction(),
      FusionInstructionBoundary, [&](const HloInstruction& producer) {
        producers.push_back(std::string(producer.name()));
      });
  EXPECT_THAT(producers, ElementsAre("p0", "negate"));
}

TEST_F(HloTraversalTest, FindParametersAfterFusion) {
  // Verifies that we correctly find the parameters after fusing the negation.
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> producers;
  auto* fused_computation = module->GetComputationWithName("fused_computation");
  FindFusionParameters(
      *fused_computation->root_instruction(),
      [&](const HloInstruction& producer, const HloInstruction& consumer) {
        return &consumer == fused_computation->parameter_instruction(0) ||
               consumer.opcode() == HloOpcode::kNegate;
      },
      [&](const HloInstruction& producer) {
        producers.push_back(std::string(producer.name()));
      });
  EXPECT_THAT(producers, ElementsAre("p0", "sum"));
}

TEST_F(HloTraversalTest, FuseEverything) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> producers;
  auto* fused_computation = module->GetComputationWithName("fused_computation");
  FindFusionParameters(
      *fused_computation->root_instruction(),
      [&](const HloInstruction& producer, const HloInstruction& consumer) {
        return producer.opcode() == HloOpcode::kParameter &&
               producer.parent()->IsEntryComputation();
      },
      [&](const HloInstruction& producer) {
        producers.push_back(std::string(producer.name()));
      });
  EXPECT_THAT(producers, ElementsAre("p0", "p1"));
}

TEST_F(HloTraversalTest, FuseConsumer) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> visited_nodes;
  HloBfsConsumersFirstTraversal(
      *module->entry_computation()->root_instruction(),
      [](const HloInstruction& producer, const HloInstruction& consumer) {
        return consumer.opcode() == HloOpcode::kParameter ||
               (producer.opcode() == HloOpcode::kParameter &&
                consumer.opcode() == HloOpcode::kSubtract);
      },
      [&](const HloInstruction& node) {
        visited_nodes.push_back(std::string(node.name()));
        return TraversalResult::kVisitOperands;
      });
  EXPECT_THAT(visited_nodes, ElementsAre("difference", "fusion", "reduce.1",
                                         "mul", "p0.1", "p1.1"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
