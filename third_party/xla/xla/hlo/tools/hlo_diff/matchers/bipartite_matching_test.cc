/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/tools/hlo_diff/matchers/bipartite_matching.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/utils/test_util.h"
#include "xla/service/call_graph.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::hlo_diff {
namespace {

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

class BipartiteMatcherUtilsTest : public HloHardwareIndependentTestBase {};

TEST_F(BipartiteMatcherUtilsTest, MatchSameTypeInstructionsIgnorePosition) {
  const char* hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  c20 = bf16[2]{0} constant({1.1, 2.2})
  c21 = bf16[2]{0} constant({1.1, 2.2})
  c22 = bf16[2]{0} constant({1.1, 2.2})
  c23 = bf16[2]{0} constant({5.5, 6.6})
  c24 = u32[2]{0} constant({1, 2}), metadata={op_name="first-phase"}
  c25 = bf16[1] constant(0.0), metadata={source_file="test.cc", source_line=42}
  c26 = s32[4]{0} constant({1, 2, 3, 4})

  add21 = bf16[2]{0} add(c22, c23)
  add22 = bf16[2]{0} add(c22, c23)
  add23 = bf16[2]{0} add(add21, add22)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> left_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> right_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> left_gumgraph,
                          HloGumgraph::Create(left_module.get()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> right_gumgraph,
                          HloGumgraph::Create(right_module.get()));
  auto mappings = std::make_unique<HloGumgraphMappings>();
  const CallGraphNode& left_entry_computation =
      left_gumgraph->GetCallGraph().GetNode(left_module->entry_computation());
  const CallGraphNode& right_entry_computation =
      right_gumgraph->GetCallGraph().GetNode(right_module->entry_computation());

  mappings->MapComputationsIfAbsent(left_entry_computation,
                                    right_entry_computation,
                                    ComputationMatchType::kSignature);
  std::vector<const HloInstructionNode*> left_constants, right_constants;
  for (const HloInstruction* instruction :
       left_entry_computation.computation()->instructions()) {
    if (instruction->IsConstant()) {
      left_constants.push_back(left_gumgraph->GetNode(instruction));
    }
  }
  for (const HloInstruction* instruction :
       right_entry_computation.computation()->instructions()) {
    if (instruction->IsConstant()) {
      right_constants.push_back(right_gumgraph->GetNode(instruction));
    }
  }

  MatchSameTypeInstructions(*left_gumgraph, *right_gumgraph, left_constants,
                            right_constants, *mappings,
                            MatcherType::kComputationGraphExactSignatureMatcher,
                            /*map_by_position=*/false);

  auto matched_params = ExtractMappedInstructionNames(*mappings);
  EXPECT_THAT(matched_params,
              UnorderedElementsAre(Pair("c22", "c22"), Pair("c23", "c23"),
                                   Pair("c24", "c24"), Pair("c25", "c25"),
                                   Pair("c26", "c26")));
}

TEST_F(BipartiteMatcherUtilsTest, MatchSameTypeInstructionsByPosition) {
  const char* hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  c20 = bf16[2]{0} constant({1.1, 2.2})
  c21 = bf16[2]{0} constant({1.1, 2.2})
  c22 = bf16[2]{0} constant({1.1, 2.2})
  c23 = bf16[2]{0} constant({5.5, 6.6})
  c24 = u32[2]{0} constant({1, 2}), metadata={op_name="first-phase"}
  c25 = bf16[1] constant(0.0), metadata={source_file="test.cc", source_line=42}
  c26 = s32[4]{0} constant({1, 2, 3, 4})

  add21 = bf16[2]{0} add(c22, c23)
  add22 = bf16[2]{0} add(c22, c23)
  add23 = bf16[2]{0} add(add21, add22)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> left_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> right_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> left_gumgraph,
                          HloGumgraph::Create(left_module.get()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> right_gumgraph,
                          HloGumgraph::Create(right_module.get()));
  auto mappings = std::make_unique<HloGumgraphMappings>();
  const CallGraphNode& left_entry_computation =
      left_gumgraph->GetCallGraph().GetNode(left_module->entry_computation());
  const CallGraphNode& right_entry_computation =
      right_gumgraph->GetCallGraph().GetNode(right_module->entry_computation());

  mappings->MapComputationsIfAbsent(left_entry_computation,
                                    right_entry_computation,
                                    ComputationMatchType::kSignature);
  std::vector<const HloInstructionNode*> left_constants, right_constants;
  for (const HloInstruction* instruction :
       left_entry_computation.computation()->instructions()) {
    if (instruction->IsConstant()) {
      left_constants.push_back(left_gumgraph->GetNode(instruction));
    }
  }
  for (const HloInstruction* instruction :
       right_entry_computation.computation()->instructions()) {
    if (instruction->IsConstant()) {
      right_constants.push_back(right_gumgraph->GetNode(instruction));
    }
  }

  MatchSameTypeInstructions(*left_gumgraph, *right_gumgraph, left_constants,
                            right_constants, *mappings,
                            MatcherType::kComputationGraphExactSignatureMatcher,
                            /*map_by_position=*/true);

  auto matched_params = ExtractMappedInstructionNames(*mappings);
  EXPECT_THAT(matched_params,
              UnorderedElementsAre(Pair("c20", "c20"), Pair("c21", "c21"),
                                   Pair("c22", "c22"), Pair("c23", "c23"),
                                   Pair("c24", "c24"), Pair("c25", "c25"),
                                   Pair("c26", "c26")));
}

TEST_F(BipartiteMatcherUtilsTest, MatchLeafInstructions) {
  const char* hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  iota = s32[4,3,5]{2,1,0} iota(), iota_dimension=0
  bitcast.1 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} bitcast(iota)
  p1 = bf16[2]{0} parameter(0)
  c1 = bf16[2]{0} constant({1.1, 2.2})
  add1 = bf16[2]{0} add(p1, c1)
  ROOT tuple = (s32[1,1,1,4,3,5]{5,4,3,2,1,0}, bf16[2]{0}) tuple(bitcast.1, add1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> left_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> right_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> left_gumgraph,
                          HloGumgraph::Create(left_module.get()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> right_gumgraph,
                          HloGumgraph::Create(right_module.get()));
  auto mappings = std::make_unique<HloGumgraphMappings>();
  const CallGraphNode& left_entry_computation =
      left_gumgraph->GetCallGraph().GetNode(left_module->entry_computation());
  const CallGraphNode& right_entry_computation =
      right_gumgraph->GetCallGraph().GetNode(right_module->entry_computation());

  mappings->MapComputationsIfAbsent(left_entry_computation,
                                    right_entry_computation,
                                    ComputationMatchType::kSignature);
  std::vector<HloInstructionNode*> left_instructions, right_instructions;
  for (const HloInstruction* instruction :
       left_entry_computation.computation()->instructions()) {
    if (auto node = left_gumgraph->GetNode(instruction);
        node->children.empty() ||
        instruction->opcode() == HloOpcode::kParameter) {
      left_instructions.push_back(node);
    }
  }
  for (const HloInstruction* instruction :
       right_entry_computation.computation()->instructions()) {
    if (auto node = right_gumgraph->GetNode(instruction);
        node->children.empty() ||
        instruction->opcode() == HloOpcode::kParameter) {
      right_instructions.push_back(node);
    }
  }

  MatchLeafInstructions(*left_gumgraph, *right_gumgraph, left_instructions,
                        right_instructions, *mappings,
                        MatcherType::kComputationGraphExactSignatureMatcher,
                        /*map_by_position=*/true);

  auto matched_params = ExtractMappedInstructionNames(*mappings);
  EXPECT_THAT(matched_params,
              UnorderedElementsAre(Pair("iota", "iota"), Pair("p1", "p1"),
                                   Pair("c1", "c1")));
}
}  // namespace
}  // namespace xla::hlo_diff
