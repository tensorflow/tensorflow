// Copyright 2025 The OpenXLA Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/hlo/tools/hlo_diff/hlo_diff_eval.h"

#include <memory>

#include <gtest/gtest.h>
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_summary.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/utils/test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace hlo_diff {
namespace {

class HloDiffTest : public HloHardwareIndependentTestBase {};

TEST_F(HloDiffTest, SplitAllegianceWorks) {
  // Create two similar modules with entry computation containing the following
  // structure:
  // [Param p0]->[Param p2]->┌-------┐  ┌----------┐
  //                         | add.1 |->| fusion.1 |->┌-------┐
  // [Param p1]->[Param p3]->└-------┘  └----------┘  |       |  ┌------┐
  //                                                  | add.3 |->| ROOT |
  // [Param p4]->[Param p6]->┌-------┐  ┌----------┐  |       |  └------┘
  //                         | add.2 |->| fusion.2 |->└-------┘
  // [Param p5]->[Param p7]->└-------┘  └----------┘
  const char* hlo_string = R"(
  HloModule module, is_scheduled=true
  
  fused_computation.1 {
    p2 = s32[32,16]{0,1:T(1,128)} parameter(0)
    p3 = s32[32,16]{0,1:T(1,128)} parameter(1)
    add.1 = s32[32,16]{0,1:T(1,128)} add(p2, p3)
  }
  
  fused_computation.2 {
    p6 = s32[32,16]{0,1:T(1,128)} parameter(0)
    p7 = s32[32,16]{0,1:T(1,128)} parameter(1)
    add.2 = s32[32,16]{0,1:T(1,128)} add(p6, p7)
  }
  
  ENTRY entry {
    p0 = s32[32,16]{0, 1:T(1,128)} parameter(0)
    p1 = s32[32,16]{0,1:T(1,128)} parameter(1)
    p4 = s32[32,16]{0, 1:T(1,128)} parameter(2)
    p5 = s32[32,16]{0,1:T(1,128)} parameter(3)
    fusion.1 = s32[32,16]{0,1:T(1,128)} fusion(p0,p1), kind=kLoop, calls=fused_computation.1
    fusion.2 = s32[32,16]{0,1:T(1,128)} fusion(p4,p5), kind=kLoop, calls=fused_computation.2
    ROOT add.3 = s32[32,16]{0,1:T(1,128)} add(fusion.1, fusion.2)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_l,
                          HloGumgraph::Create(module_l.get()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_r,
                          HloGumgraph::Create(module_r.get()));
  HloGumgraphMappings mappings;
  // Map all nodes with the same name and then switch the mappings for add.1 and
  // add.2.
  mappings.MapInstructionsIfAbsent(&graph_l->GetRoot(), &graph_r->GetRoot(),
                                   MatcherType::kManual);
  MatchAllNodesByName(*graph_l, *graph_r, mappings);
  ASSERT_NO_FATAL_FAILURE(
      OverwriteMapInstructions(GetNodeByName(*graph_l, "add.1"),
                               GetNodeByName(*graph_r, "add.2"), mappings));
  ASSERT_NO_FATAL_FAILURE(
      OverwriteMapInstructions(GetNodeByName(*graph_l, "add.2"),
                               GetNodeByName(*graph_r, "add.1"), mappings));
  // Construct the diff eval from the manually mapped node mappings.
  std::unique_ptr<const DiffResult> diff_result =
      ConstructDiffResult(*graph_l, *graph_r, mappings);
  std::unique_ptr<const DiffSummary> diff_summary =
      ConstructDiffSummary(*graph_l, *graph_r, mappings, *diff_result);
  std::unique_ptr<const DiffEval> diff_eval = ComputeDiffEval(
      *graph_l, *graph_r, mappings, *diff_result, *diff_summary);

  EXPECT_EQ(diff_eval->num_split_allegiance_computation, 2);
  EXPECT_EQ(diff_eval->num_split_allegiance_instruction, 2);
  // The following pairs are split allegiance parental: parent are
  // matched but children are not. (add.1 is matched to add.2)
  // fusion.1
  //   add.1 -> add.1
  // fusion.2
  //   add.2 -> add.2
  // add.1
  //   param2 -> param6
  //   param3 -> param7
  // add.2
  //   param6 -> param2
  //   param7 -> param3
  EXPECT_EQ(diff_eval->num_split_allegiance_parental, 6);
}

TEST_F(HloDiffTest, GraphNodeCountsWork) {
  // Create a module with entry computation containing the following structure:
  // [Param p0]->[Param p2]->┌-------┐  ┌----------┐
  //                         | add.1 |->| fusion.1 |->┌-------┐
  // [Param p1]->[Param p3]->└-------┘  └----------┘  |       |  ┌------┐
  //                                                  | add.3 |->| ROOT |
  // [Param p4]->[Param p6]->┌-------┐  ┌----------┐  |       |  └------┘
  //                         | add.2 |->| fusion.2 |->└-------┘
  // [Param p5]->[Param p7]->└-------┘  └----------┘
  const char* hlo_string = R"(
  HloModule module, is_scheduled=true
  
  fused_computation.1 {
    p2 = s32[32,16]{0,1:T(1,128)} parameter(0)
    p3 = s32[32,16]{0,1:T(1,128)} parameter(1)
    add.1 = s32[32,16]{0,1:T(1,128)} add(p2, p3)
  }
  
  fused_computation.2 {
    p6 = s32[32,16]{0,1:T(1,128)} parameter(0)
    p7 = s32[32,16]{0,1:T(1,128)} parameter(1)
    add.2 = s32[32,16]{0,1:T(1,128)} add(p6, p7)
  }
  
  ENTRY entry {
    p0 = s32[32,16]{0, 1:T(1,128)} parameter(0)
    p1 = s32[32,16]{0,1:T(1,128)} parameter(1)
    p4 = s32[32,16]{0, 1:T(1,128)} parameter(2)
    p5 = s32[32,16]{0,1:T(1,128)} parameter(3)
    fusion.1 = s32[32,16]{0,1:T(1,128)} fusion(p0,p1), kind=kLoop, calls=fused_computation.1
    fusion.2 = s32[32,16]{0,1:T(1,128)} fusion(p4,p5), kind=kLoop, calls=fused_computation.2
    ROOT add.3 = s32[32,16]{0,1:T(1,128)} add(fusion.1, fusion.2)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_l,
                          HloGumgraph::Create(module_l.get()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_r,
                          HloGumgraph::Create(module_r.get()));
  std::unique_ptr<const DiffEval> diff_eval =
      ComputeDiffEval(*graph_l, *graph_r, {}, {}, {});

  EXPECT_EQ(diff_eval->left_node_count, 14);
  EXPECT_EQ(diff_eval->right_node_count, 14);
}

TEST_F(HloDiffTest, DiffSizeWorks) {
  // Create a module with entry computation containing the following structure:
  // [Param p0]->┌-------┐
  //             | add.1 |
  // [Param p1]->└-------┘
  const char* hlo_string = R"(
  HloModule module, is_scheduled=true
  
  ENTRY entry {
    p0 = s32[32,16]{0, 1:T(1,128)} parameter(0)
    p1 = s32[32,16]{0,1:T(1,128)} parameter(1)
    ROOT add.1 = s32[32,16]{0,1:T(1,128)} add(p0, p1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_l,
                          HloGumgraph::Create(module_l.get()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_r,
                          HloGumgraph::Create(module_r.get()));
  DiffResult diff_result;
  diff_result.left_module_unmatched_instructions.insert(
      graph_l->GetRoot().instruction);
  diff_result.right_module_unmatched_instructions.insert(
      graph_r->GetRoot().instruction);
  diff_result.changed_instructions.insert(
      {graph_l->GetRoot().instruction, graph_r->GetRoot().instruction});
  diff_result.unchanged_instructions.insert(
      {graph_l->GetRoot().instruction, graph_r->GetRoot().instruction});
  std::unique_ptr<const DiffEval> diff_eval =
      ComputeDiffEval(*graph_l, *graph_r, {}, diff_result, {});

  EXPECT_EQ(diff_eval->len_left_unmatched, 1);
  EXPECT_EQ(diff_eval->len_right_unmatched, 1);
  EXPECT_EQ(diff_eval->len_changed, 1);
  EXPECT_EQ(diff_eval->len_unchanged, 1);
}

}  // namespace
}  // namespace hlo_diff
}  // namespace xla
