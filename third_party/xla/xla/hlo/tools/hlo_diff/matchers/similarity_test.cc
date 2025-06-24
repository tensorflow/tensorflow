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

#include "xla/hlo/tools/hlo_diff/matchers/similarity.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/utils/test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace hlo_diff {
namespace {

using ::testing::DoubleEq;

class HloSimilarityTest : public HloHardwareIndependentTestBase {};

TEST_F(HloSimilarityTest, NodeAttributesSimilarity) {
  // Create left module with entry computation containing the following
  // structure:
  // [Param foo_L] ------> ┌-------┐
  //                       | add_1 | ---> ┌-------┐      ┌------┐
  // [Constant bar_L] ---> └-------┘      | add_0 | ---> | ROOT |
  // [Param baz_L] ---------------------> └-------┘      └------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(R"(
  HloModule module, is_scheduled=true
  
  ENTRY entry {
    foo_L = f32[8,2048]{1,0:T(8,128)} parameter(0)
    bar_L = f32[8,2048]{1,0:T(8,128)} constant(0)
    baz_L = f32[8,2048]{1,0:T(8,128)} parameter(1)
    add_1 = f32[8,2048]{1,0:T(8,128)} add(foo_L, bar_L)
    add_0 = f32[8,2048]{1,0:T(8,128)} add(add_1, baz_L)
  }
  )"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_l,
                          HloGumgraph::Create(module_l.get()));

  // Create right module with entry computation containing the following
  // structure:
  // [Param foo_R] ------> ┌-------┐
  //                       | add_1 | ---> ┌------------┐      ┌------┐
  // [Constant bar_R] ---> └-------┘      | subtract_0 | ---> | ROOT |
  // [Param baz_R] ---------------------> └------------┘      └------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(R"(
  HloModule module, is_scheduled=true
  
  ENTRY entry {
    foo_R = f32[8,2048]{1,0:T(8,128)} parameter(0)
    bar_R = f32[8,2048]{1,0:T(8,128)} constant(0)
    baz_R = f32[8,2048]{1,0:T(8,128)} parameter(1)
    add_1 = f32[8,2048]{1,0:T(8,128)} add(foo_R, bar_R)
    subtract_0 = f32[8,2048]{1,0:T(8,128)} subtract(add_1, baz_R)
  }
  )"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_r,
                          HloGumgraph::Create(module_r.get()));
  EXPECT_GT(NodeAttributesSimilarity(GetNodeByName(*graph_l, "foo_L"),
                                     GetNodeByName(*graph_r, "foo_R")),
            NodeAttributesSimilarity(GetNodeByName(*graph_l, "add_1"),
                                     GetNodeByName(*graph_r, "foo_R")));
}

TEST_F(HloSimilarityTest, AncestorSubGraphLcsSimilarity) {
  // Create left module with entry computation containing the following
  // structure:
  // [Param foo_L] ------> ┌-------┐
  //                       | add_1 | ---> ┌-------┐      ┌------┐
  // [Constant bar_L] ---> └-------┘      | add_0 | ---> | ROOT |
  // [Param baz_L] ---------------------> └-------┘      └------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(R"(
  HloModule module, is_scheduled=true
  
  ENTRY entry {
    foo_L = f32[8,2048]{1,0:T(8,128)} parameter(0)
    bar_L = f32[8,2048]{1,0:T(8,128)} constant(0)
    baz_L = f32[8,2048]{1,0:T(8,128)} parameter(1)
    add_1 = f32[8,2048]{1,0:T(8,128)} add(foo_L, bar_L)
    add_0 = f32[8,2048]{1,0:T(8,128)} add(add_1, baz_L)
  }
  )"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_l,
                          HloGumgraph::Create(module_l.get()));

  // Create right module with entry computation containing the following
  // structure:
  // [Param foo_R] ------> ┌-------┐
  //                       | add_1 | ---> ┌------------┐      ┌------┐
  // [Constant bar_R] ---> └-------┘      | subtract_0 | ---> | ROOT |
  // [Param baz_R] ---------------------> └------------┘      └------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(R"(
  HloModule module, is_scheduled=true
  
  ENTRY entry {
    foo_R = f32[8,2048]{1,0:T(8,128)} parameter(0)
    bar_R = f32[8,2048]{1,0:T(8,128)} constant(0)
    baz_R = f32[8,2048]{1,0:T(8,128)} parameter(1)
    add_1 = f32[8,2048]{1,0:T(8,128)} add(foo_R, bar_R)
    subtract_0 = f32[8,2048]{1,0:T(8,128)} subtract(add_1, baz_R)
  }
  )"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_r,
                          HloGumgraph::Create(module_r.get()));
  const HloInstructionNode* left_p1 = GetNodeByName(*graph_l, "foo_L");
  const HloInstructionNode* right_p1 = GetNodeByName(*graph_r, "foo_R");
  const HloInstructionNode* right_p2 = GetNodeByName(*graph_r, "baz_R");
  double sim_score_11 = AncestorSubGraphLcsSimilarity(left_p1, right_p1, 3, 1,
                                                      graph_l->GetNodeCount(),
                                                      graph_r->GetNodeCount());
  double sim_score_12 = AncestorSubGraphLcsSimilarity(left_p1, right_p2, 3, 1,
                                                      graph_l->GetNodeCount(),
                                                      graph_r->GetNodeCount());
  EXPECT_THAT(sim_score_11,
              DoubleEq(2.0 * 2.0 / (3 + 3)));  // LCS(paa, pas) = 2
  EXPECT_THAT(sim_score_12, DoubleEq(2.0 * 1.0 / (3 + 2)));  // LCS(paa, ps) = 1
}

TEST_F(HloSimilarityTest, ParamPropertySimilarity) {
  // Create left module with entry computation containing the following
  // structure:
  // [Param foo_L] ------> ┌-------┐
  //                       | add_1 | ---> ┌-------┐      ┌------┐
  // [Constant bar_L] ---> └-------┘      | add_0 | ---> | ROOT |
  // [Param baz_L] ---------------------> └-------┘      └------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(R"(
  HloModule module, is_scheduled=true
  
  ENTRY entry {
    foo_L = f32[8,2048]{1,0:T(8,128)} parameter(0)
    bar_L = f32[8,2048]{1,0:T(8,128)} constant(0)
    baz_L = f32[8,2048]{1,0:T(8,128)} parameter(1)
    add_1 = f32[8,2048]{1,0:T(8,128)} add(foo_L, bar_L)
    add_0 = f32[8,2048]{1,0:T(8,128)} add(add_1, baz_L)
  }
  )"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_l,
                          HloGumgraph::Create(module_l.get()));

  // Create right module with entry computation containing the following
  // structure:
  // [Param foo_R] ------> ┌-------┐
  //                       | add_1 | ---> ┌------------┐      ┌------┐
  // [Constant bar_R] ---> └-------┘      | subtract_0 | ---> | ROOT |
  // [Param baz_R] ---------------------> └------------┘      └------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(R"(
  HloModule module, is_scheduled=true
  
  ENTRY entry {
    foo_R = f32[8,2048]{1,0:T(8,128)} parameter(0)
    bar_R = f32[8,2048]{1,0:T(8,128)} constant(0)
    baz_R = f32[8,2048]{1,0:T(8,128)} parameter(1)
    add_1 = f32[8,2048]{1,0:T(8,128)} add(foo_R, bar_R)
    subtract_0 = f32[8,2048]{1,0:T(8,128)} subtract(add_1, baz_R)
  }
  )"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_r,
                          HloGumgraph::Create(module_r.get()));
  const HloInstructionNode* left_p1 = GetNodeByName(*graph_l, "foo_L");
  const HloInstructionNode* left_p2 = GetNodeByName(*graph_l, "baz_L");
  const HloInstructionNode* right_p1 = GetNodeByName(*graph_r, "foo_R");
  const HloInstructionNode* right_p2 = GetNodeByName(*graph_r, "baz_R");
  double sim_score_11 = ParamPropertySimilarity(left_p1, right_p1);
  double sim_score_12 = ParamPropertySimilarity(left_p1, right_p2);
  double sim_score_21 = ParamPropertySimilarity(left_p2, right_p1);
  double sim_score_22 = ParamPropertySimilarity(left_p2, right_p2);
  EXPECT_GE(sim_score_11, sim_score_12);
  EXPECT_GE(sim_score_22, sim_score_21);
}

TEST_F(HloSimilarityTest, ConstantPropertySimilarity) {
  // Create left module with entry computation containing the following
  // structure:
  // [Const 0] ---> ┌-------┐
  //                | add_0 | --------> ┌-------┐
  // [Const 1] ---> └-------┘           |       |      ┌-------┐
  //                                    | add_3 | ---> |       |
  // [Const 2] ---> ┌------------┐      |       |      |       |      ┌------┐
  //                | subtract_1 | ---> └-------┘      | add_4 | ---> | ROOT |
  // [Const 3] ---> └------------┘                     |       |      └------┘
  //                                                   |       |
  // [Const 4] --------------------------------------> └-------┘
  //
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    constant.0 = f32[] constant(0)
    constant.1 = f32[] constant(0)
    constant.2 = f32[] constant(0)
    constant.3 = f32[] constant(0)
    constant.4 = f32[] constant(0)
    add.0 = f32[] add(constant.0, constant.1)
    subtract.1 = f32[] subtract(constant.2, constant.3)
    add.3 = f32[] add(add.0, subtract.1)
    add.4 = f32[] add(add.3, constant.4)
  }
  )"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_l,
                          HloGumgraph::Create(module_l.get()));

  // Create right module with entry computation containing the following
  // structure:
  // [Const 0] ---> ┌-------┐
  //                | add_0 | ---> ┌-------┐
  // [Const 1] ---> └-------┘      |       |      ┌-------┐
  //                               | add_3 | ---> |       |
  // [Const 2] ---> ┌-------┐      |       |      |       |      ┌------┐
  //                | add_1 | ---> └-------┘      | add_4 | ---> | ROOT |
  // [Const 3] ---> └-------┘                     |       |      └------┘
  //                                              |       |
  // [Const 4] ---------------------------------> └-------┘
  //
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    constant.0 = f32[] constant(0)
    constant.1 = f32[] constant(0)
    constant.2 = f32[] constant(0)
    constant.3 = f32[] constant(0)
    constant.4 = f32[] constant(0)
    add.0 = f32[] add(constant.0, constant.1)
    add.1 = f32[] add(constant.2, constant.3)
    add.3 = f32[] add(add.0, add.1)
    add.4 = f32[] add(add.3, constant.4)
  }
  )"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_r,
                          HloGumgraph::Create(module_r.get()));
  const HloInstructionNode* left_c1 = GetNodeByName(*graph_l, "constant.1");
  const HloInstructionNode* left_c2 = GetNodeByName(*graph_l, "constant.2");
  const HloInstructionNode* right_c1 = GetNodeByName(*graph_r, "constant.1");
  const HloInstructionNode* right_c2 = GetNodeByName(*graph_r, "constant.2");
  double sim_score_11 = ConstantPropertySimilarity(left_c1, right_c1);
  double sim_score_12 = ConstantPropertySimilarity(left_c1, right_c2);
  double sim_score_21 = ConstantPropertySimilarity(left_c2, right_c1);
  double sim_score_22 = ConstantPropertySimilarity(left_c2, right_c2);
  EXPECT_GE(sim_score_11, sim_score_12);
  EXPECT_GE(sim_score_22, sim_score_21);
}

}  // namespace
}  // namespace hlo_diff
}  // namespace xla
