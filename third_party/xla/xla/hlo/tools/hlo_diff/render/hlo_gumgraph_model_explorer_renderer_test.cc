/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/tools/hlo_diff/render/hlo_gumgraph_model_explorer_renderer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/json/include/nlohmann/json_fwd.hpp"
#include "third_party/json/src/json.hpp"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_summary.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/proto/diff_result.pb.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/direct_hlo_to_json_graph_convert.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/schema_structs.h"
#include "xla/hlo/tools/hlo_diff/utils/hlo_diff_util.h"
#include "xla/hlo/tools/hlo_diff/utils/test_util.h"

namespace xla {
namespace hlo_diff {
namespace {

using ::nlohmann::json;
using ::tooling::visualization_client::Attribute;
using ::tooling::visualization_client::GetInstructionId;
using ::tooling::visualization_client::Graph;
using ::tooling::visualization_client::GraphNode;
using ::tooling::visualization_client::Subgraph;

class HloDiffRendererTest : public HloHardwareIndependentTestBase {};

absl::StatusOr<std::string> GetSubgraphId(const Graph& graph) {
  if (graph.subgraphs.size() != 1) {
    return absl::InvalidArgumentError("graph missing subgraphs");
  }
  return graph.subgraphs.at(0).subgraph_id;
}

absl::StatusOr<GraphNode> GetNodeById(const Graph& graph,
                                      absl::string_view id) {
  if (graph.subgraphs.size() != 1) {
    return absl::InvalidArgumentError("graph missing subgraphs");
  }
  const Subgraph& subgraph = graph.subgraphs.at(0);
  for (const GraphNode& node : subgraph.nodes) {
    if (node.node_id == id) {
      return node;
    }
  }
  return absl::InvalidArgumentError("node not found");
}

absl::StatusOr<std::string> GetNodeAttr(const GraphNode& node,
                                        absl::string_view attr_name) {
  for (const Attribute& attr : node.node_attrs) {
    if (attr.key == attr_name) {
      return attr.value;
    }
  }
  return absl::InvalidArgumentError("attr not found");
}

TEST_F(HloDiffRendererTest, MatchedDifferentFingerprintMarkAsChanged) {
  // Create left module with entry computation containing the following
  // structure:
  // [Param 0] ---> ┌-------┐
  //                | add_0 |
  // [Param 1] ---> └-------┘
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                       ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  parameter.0 = f32[] parameter(0)
  const.1 = f32[] constant(1)
  add.0 = f32[] add(parameter.0, const.1)
}
)"));

  // Create right module with entry computation containing the following
  // structure:
  // [Param 1] ---> ┌------------┐
  //                | subtract_0 |
  // [Param 0] ---> └------------┘
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                       ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  parameter.0 = f32[] parameter(0)
  const.1 = f32[] constant(2)
  subtract.0 = f32[] subtract(const.1, parameter.0)
}
)"));

  DiffResult diff_result;

  HloInstruction* left_p0 =
      module_l->entry_computation()->parameter_instruction(0);
  HloInstruction* right_p0 =
      module_r->entry_computation()->parameter_instruction(0);
  diff_result.unchanged_instructions[left_p0] = right_p0;

  ASSERT_OK_AND_ASSIGN(HloInstruction * left_c1,
                       GetInstructionByName(*module_l, "const.1"));
  ASSERT_OK_AND_ASSIGN(HloInstruction * right_c1,
                       GetInstructionByName(*module_r, "const.1"));
  diff_result.changed_instructions[left_c1] = right_c1;

  HloInstruction* left_add = module_l->entry_computation()->root_instruction();
  diff_result.left_module_unmatched_instructions.insert(left_add);

  HloInstruction* right_sub = module_r->entry_computation()->root_instruction();
  diff_result.right_module_unmatched_instructions.insert(right_sub);

  ASSERT_OK_AND_ASSIGN(MeJson result,
                       RenderMe(*module_l, *module_r, diff_result));

  EXPECT_EQ(result.graph_collection.graphs.size(), 2);

  const Graph& me_graph_l = result.graph_collection.graphs.at(0);
  EXPECT_EQ(me_graph_l.label, "left");
  ASSERT_OK_AND_ASSIGN(std::string subgraph_id_l, GetSubgraphId(me_graph_l));
  EXPECT_EQ(subgraph_id_l, "entry_left");

  const Graph& me_graph_r = result.graph_collection.graphs.at(1);
  EXPECT_EQ(me_graph_r.label, "right");
  ASSERT_OK_AND_ASSIGN(std::string subgraph_id_r, GetSubgraphId(me_graph_r));
  EXPECT_EQ(subgraph_id_r, "entry_right");

  // left parameter 0 mapped to right parameter 0
  ASSERT_OK_AND_ASSIGN(const GraphNode& left_p0_node,
                       GetNodeById(me_graph_l, "parameter.0"));
  ASSERT_OK_AND_ASSIGN(std::string left_p0_diff_type,
                       GetNodeAttr(left_p0_node, "diff_type"));
  EXPECT_EQ(left_p0_diff_type, "kUnchanged");
  ASSERT_OK_AND_ASSIGN(std::string left_p0_mapped_node_id,
                       GetNodeAttr(left_p0_node, "mapped_node_id"));
  EXPECT_EQ(left_p0_mapped_node_id, absl::StrCat(GetInstructionId(right_p0)));

  // right parameter 0 mapped to left parameter 0
  ASSERT_OK_AND_ASSIGN(const GraphNode& right_p0_node,
                       GetNodeById(me_graph_r, "parameter.0"));
  ASSERT_OK_AND_ASSIGN(std::string right_p0_diff_type,
                       GetNodeAttr(right_p0_node, "diff_type"));
  EXPECT_EQ(right_p0_diff_type, "kUnchanged");
  ASSERT_OK_AND_ASSIGN(std::string right_p0_mapped_node_id,
                       GetNodeAttr(right_p0_node, "mapped_node_id"));
  EXPECT_EQ(right_p0_mapped_node_id, absl::StrCat(GetInstructionId(left_p0)));

  // left const 1 mapped to right const 1
  ASSERT_OK_AND_ASSIGN(const GraphNode& left_p1_node,
                       GetNodeById(me_graph_l, "const.1"));
  ASSERT_OK_AND_ASSIGN(std::string left_p1_diff_type,
                       GetNodeAttr(left_p1_node, "diff_type"));
  ASSERT_OK_AND_ASSIGN(std::string left_p1_changed_diff_type,
                       GetNodeAttr(left_p1_node, "changed_diff_types"));
  EXPECT_EQ(left_p1_diff_type, "kUpdated");
  EXPECT_EQ(left_p1_changed_diff_type, "kConstantLiteralChanged");
  ASSERT_OK_AND_ASSIGN(std::string left_p1_mapped_node_id,
                       GetNodeAttr(left_p1_node, "mapped_node_id"));
  EXPECT_EQ(left_p1_mapped_node_id, absl::StrCat(GetInstructionId(right_c1)));

  // right const 1 mapped to left const 1
  ASSERT_OK_AND_ASSIGN(const GraphNode& right_p1_node,
                       GetNodeById(me_graph_r, "const.1"));
  ASSERT_OK_AND_ASSIGN(std::string right_p1_diff_type,
                       GetNodeAttr(right_p1_node, "diff_type"));
  ASSERT_OK_AND_ASSIGN(std::string right_p1_changed_diff_type,
                       GetNodeAttr(right_p1_node, "changed_diff_types"));
  EXPECT_EQ(right_p1_diff_type, "kUpdated");
  EXPECT_EQ(right_p1_changed_diff_type, "kConstantLiteralChanged");
  ASSERT_OK_AND_ASSIGN(std::string right_p1_mapped_node_id,
                       GetNodeAttr(right_p1_node, "mapped_node_id"));
  EXPECT_EQ(right_p1_mapped_node_id, absl::StrCat(GetInstructionId(left_c1)));

  // left unmatched add
  ASSERT_OK_AND_ASSIGN(const GraphNode& left_add_node,
                       GetNodeById(me_graph_l, "add.0"));
  ASSERT_OK_AND_ASSIGN(std::string left_add_diff_type,
                       GetNodeAttr(left_add_node, "diff_type"));
  EXPECT_EQ(left_add_diff_type, "kLeftUnmatched");

  // right unmatched subtract
  ASSERT_OK_AND_ASSIGN(const GraphNode& right_sub_node,
                       GetNodeById(me_graph_r, "subtract.0"));
  ASSERT_OK_AND_ASSIGN(std::string right_sub_diff_type,
                       GetNodeAttr(right_sub_node, "diff_type"));
  EXPECT_EQ(right_sub_diff_type, "kRightUnmatched");
}

TEST_F(HloDiffRendererTest, DebugModeWorks) {
  // Create left module with entry computation containing the following
  // structure:
  // [Param 0]
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                       ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  parameter.0 = f32[] parameter(0)
}
)"));

  // Create right module with entry computation containing the following
  // structure:
  // [Param 1]
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                       ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  parameter.0 = f32[] parameter(0)
}
)"));

  DiffResult diff_result;

  HloInstruction* left_p0 =
      module_l->entry_computation()->parameter_instruction(0);
  uint64_t left_fp_p0 = GetHloInstructionFingerprint(left_p0);
  HloInstruction* right_p0 =
      module_r->entry_computation()->parameter_instruction(0);
  uint64_t right_fp_p0 = GetHloInstructionFingerprint(right_p0);
  diff_result.unchanged_instructions[left_p0] = right_p0;
  diff_result.map_by[std::make_pair(left_p0, right_p0)] =
      MatcherType::kGreedySubGraphExactMatcher;
  diff_result.node_props_left.emplace(left_p0, HloInstructionNodeProps{
                                                   0,
                                                   1,
                                                   left_fp_p0,
                                                   left_fp_p0,
                                               });
  diff_result.node_props_right.emplace(right_p0, HloInstructionNodeProps{
                                                     0,
                                                     1,
                                                     right_fp_p0,
                                                     right_fp_p0,
                                                 });

  MeRenderOptions options;
  options.debug_mode = true;
  ASSERT_OK_AND_ASSIGN(MeJson result,
                       RenderMe(*module_l, *module_r, diff_result,
                                /*diff_summary=*/nullptr, options));

  EXPECT_EQ(result.graph_collection.graphs.size(), 2);

  const Graph& me_graph_l = result.graph_collection.graphs.at(0);
  const Graph& me_graph_r = result.graph_collection.graphs.at(1);

  // left parameter 0 mapped to right parameter 0
  ASSERT_OK_AND_ASSIGN(const GraphNode& left_p0_node,
                       GetNodeById(me_graph_l, "parameter.0"));
  ASSERT_OK_AND_ASSIGN(std::string left_p0_matcher_type,
                       GetNodeAttr(left_p0_node, "matcher_type"));
  EXPECT_EQ(left_p0_matcher_type, "kGreedySubGraphExactMatcher");
  ASSERT_OK_AND_ASSIGN(std::string left_p0_fingerprint,
                       GetNodeAttr(left_p0_node, "fingerprint"));
  EXPECT_EQ(left_p0_fingerprint, absl::StrCat(left_fp_p0));
  ASSERT_OK_AND_ASSIGN(std::string left_p0_subgraph_fingerprint,
                       GetNodeAttr(left_p0_node, "subgraph_fingerprint"));
  EXPECT_EQ(left_p0_subgraph_fingerprint, absl::StrCat(left_fp_p0));
  ASSERT_OK_AND_ASSIGN(std::string left_p0_height,
                       GetNodeAttr(left_p0_node, "height"));
  EXPECT_EQ(left_p0_height, "1");
  ASSERT_OK_AND_ASSIGN(std::string left_p0_generation,
                       GetNodeAttr(left_p0_node, "generation"));
  EXPECT_EQ(left_p0_generation, "0");

  // right parameter 0 mapped to left parameter 0
  ASSERT_OK_AND_ASSIGN(const GraphNode& right_p0_node,
                       GetNodeById(me_graph_r, "parameter.0"));
  ASSERT_OK_AND_ASSIGN(std::string right_p0_matcher_type,
                       GetNodeAttr(right_p0_node, "matcher_type"));
  EXPECT_EQ(right_p0_matcher_type, "kGreedySubGraphExactMatcher");
  ASSERT_OK_AND_ASSIGN(std::string right_p0_fingerprint,
                       GetNodeAttr(right_p0_node, "fingerprint"));
  EXPECT_EQ(right_p0_fingerprint, absl::StrCat(right_fp_p0));
  ASSERT_OK_AND_ASSIGN(std::string right_p0_subgraph_fingerprint,
                       GetNodeAttr(right_p0_node, "subgraph_fingerprint"));
  EXPECT_EQ(right_p0_subgraph_fingerprint, absl::StrCat(right_fp_p0));
  ASSERT_OK_AND_ASSIGN(std::string right_p0_height,
                       GetNodeAttr(right_p0_node, "height"));
  EXPECT_EQ(right_p0_height, "1");
  ASSERT_OK_AND_ASSIGN(std::string right_p0_generation,
                       GetNodeAttr(right_p0_node, "generation"));
  EXPECT_EQ(right_p0_generation, "0");
}

TEST_F(HloDiffRendererTest, CreateSyncNavMapping) {
  // Create left module with entry computation containing the following
  // structure:
  // [Param 0] ---> ┌-------┐
  //                | add_0 |
  // [Param 1] ---> └-------┘
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                       ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  parameter.0 = f32[] parameter(0)
  parameter.1 = f32[] parameter(1)
  add.0 = f32[] add(parameter.0, parameter.1)
}
)"));

  // Create right module with entry computation containing the following
  // structure:
  // [Param 1] ---> ┌------------┐
  //                | subtract_0 |
  // [Param 0] ---> └------------┘
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                       ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  parameter.0 = f32[] parameter(0)
  parameter.1 = f32[] parameter(1)
  subtract.0 = f32[] subtract(parameter.1, parameter.0)
}
)"));

  DiffResult diff_result;

  HloInstruction* left_p0 =
      module_l->entry_computation()->parameter_instruction(0);
  HloInstruction* right_p0 =
      module_r->entry_computation()->parameter_instruction(0);
  diff_result.unchanged_instructions[left_p0] = right_p0;

  HloInstruction* left_p1 =
      module_l->entry_computation()->parameter_instruction(1);
  HloInstruction* right_p1 =
      module_r->entry_computation()->parameter_instruction(1);
  diff_result.changed_instructions[left_p1] = right_p1;

  ASSERT_OK_AND_ASSIGN(MeJson result,
                       RenderMe(*module_l, *module_r, diff_result));

  EXPECT_EQ(result.sync_nav_mapping["type"], "sync_navigation");
  EXPECT_TRUE(result.sync_nav_mapping["disableMappingFallback"]);
  EXPECT_EQ(result.sync_nav_mapping["mapping"]
                                   [absl::StrCat(GetInstructionId(left_p0))],
            absl::StrCat(GetInstructionId(right_p0)));
  EXPECT_EQ(result.sync_nav_mapping["mapping"]
                                   [absl::StrCat(GetInstructionId(left_p1))],
            absl::StrCat(GetInstructionId(right_p1)));
}

TEST_F(HloDiffRendererTest, GenerateMeJsonWithDiffResultAndProtoEquals) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                       ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true
ENTRY entry {
  parameter.0 = f32[] parameter(0)
  parameter.1 = f32[] parameter(1)
  add.0 = f32[] add(parameter.0, parameter.1)
}
)"));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                       ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true
ENTRY entry {
  parameter.0 = f32[] parameter(0)
  parameter.1 = f32[] parameter(1)
  add.0 = f32[] add(parameter.0, parameter.1)
}
)"));
  DiffResult diff_result;
  HloInstruction* left_p0 =
      module_l->entry_computation()->parameter_instruction(0);
  HloInstruction* right_p0 =
      module_r->entry_computation()->parameter_instruction(0);
  HloInstruction* left_p1 =
      module_l->entry_computation()->parameter_instruction(1);
  HloInstruction* right_p1 =
      module_r->entry_computation()->parameter_instruction(1);
  diff_result.left_module_unmatched_instructions.insert(left_p0);
  diff_result.right_module_unmatched_instructions.insert(right_p0);
  diff_result.unchanged_instructions[left_p1] = right_p1;
  DiffResultProto diff_result_proto = diff_result.ToProto();
  DiffResult diff_result_from_proto =
      DiffResult::FromProto(diff_result_proto, *module_l, *module_r);

  // Check graph collection is the same.
  ASSERT_OK_AND_ASSIGN(MeJson result,
                       RenderMe(*module_l, *module_r, diff_result));
  ASSERT_OK_AND_ASSIGN(MeJson result_from_proto,
                       RenderMe(*module_l, *module_r, diff_result_from_proto));
  EXPECT_EQ(result.DumpGraphCollectionJson(),
            result_from_proto.DumpGraphCollectionJson());

  // Check node data is the same.
  EXPECT_EQ(result.node_data.dump(), result_from_proto.node_data.dump());

  // Check sync nav mapping is the same.
  EXPECT_EQ(result.sync_nav_mapping.dump(),
            result_from_proto.sync_nav_mapping.dump());
}

TEST_F(HloDiffRendererTest, GenerateMeJsonWithDiffSummary) {
  std::string module_str = R"hlo(
HloModule module, is_scheduled=true

fused_computation {
  parameter.0 = f32[] parameter(0)
  parameter.1 = f32[] parameter(1)
  ROOT add.0 = f32[] add(parameter.0, parameter.1)
}

ENTRY entry {
  parameter.0 = f32[] parameter(0)
  parameter.1 = f32[] parameter(1)
  fusion.0 = f32[] fusion(parameter.0, parameter.1), kind=kLoop, calls=fused_computation
  ROOT add.0 = f32[] add(fusion.0, parameter.1)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                       ParseAndReturnVerifiedModule(module_str));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                       ParseAndReturnVerifiedModule(module_str));
  HloComputation* fusion_l =
      module_l->GetComputationWithName("fused_computation");
  HloComputation* fusion_r =
      module_r->GetComputationWithName("fused_computation");
  DiffResult diff_result;
  DiffSummary diff_summary;
  diff_summary.computation_summary = {
      {fusion_l, ComputationSummary{DiffSide::kLeft, nullptr, 0, 0, 0, false}},
      {fusion_r, ComputationSummary{DiffSide::kRight, nullptr, 0, 0, 0, true}}};

  ASSERT_OK_AND_ASSIGN(MeJson result, RenderMe(*module_l, *module_r,
                                               diff_result, &diff_summary));

  EXPECT_EQ(result.graph_collection.graphs.size(), 2);

  const Graph& me_graph_l = result.graph_collection.graphs.at(0);
  const Graph& me_graph_r = result.graph_collection.graphs.at(1);

  // Check the number of nodes in each graph.
  EXPECT_EQ(me_graph_l.subgraphs.at(0).nodes.size(), 8);
  EXPECT_EQ(me_graph_r.subgraphs.at(0).nodes.size(), 5);
}

TEST_F(HloDiffRendererTest,
       GenerateMeJsonWithDiffSummaryNotCollapseComputationWhenCallerHasDiff) {
  std::string module_str = R"hlo(
HloModule module, is_scheduled=true

fused_computation {
  parameter.0 = f32[] parameter(0)
  parameter.1 = f32[] parameter(1)
  ROOT add.0 = f32[] add(parameter.0, parameter.1)
}

ENTRY entry {
  parameter.0 = f32[] parameter(0)
  parameter.1 = f32[] parameter(1)
  fusion.0 = f32[] fusion(parameter.0, parameter.1), kind=kLoop, calls=fused_computation
  ROOT add.0 = f32[] add(fusion.0, parameter.1)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                       ParseAndReturnVerifiedModule(module_str));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                       ParseAndReturnVerifiedModule(module_str));
  HloComputation* fusion_l =
      module_l->GetComputationWithName("fused_computation");
  HloComputation* fusion_r =
      module_r->GetComputationWithName("fused_computation");
  ASSERT_OK_AND_ASSIGN(const HloInstruction* fusion_r_caller,
                       GetInstructionByName(*module_r.get(), "fusion.0"));
  DiffResult diff_result;
  diff_result.right_module_unmatched_instructions.insert(fusion_r_caller);
  DiffSummary diff_summary;
  diff_summary.computation_summary = {
      {fusion_l, ComputationSummary{DiffSide::kLeft, nullptr, 0, 0, 0, false}},
      {fusion_r, ComputationSummary{DiffSide::kRight, nullptr, 0, 0, 0, true}}};

  ASSERT_OK_AND_ASSIGN(MeJson result, RenderMe(*module_l, *module_r,
                                               diff_result, &diff_summary));

  EXPECT_EQ(result.graph_collection.graphs.size(), 2);

  const Graph& me_graph_l = result.graph_collection.graphs.at(0);
  const Graph& me_graph_r = result.graph_collection.graphs.at(1);

  // Check the number of nodes in each graph.
  EXPECT_EQ(me_graph_l.subgraphs.at(0).nodes.size(), 8);
  EXPECT_EQ(me_graph_r.subgraphs.at(0).nodes.size(), 8);
}

TEST_F(HloDiffRendererTest,
       GenerateMeJsonWithDiffSummaryForceExpandComputation) {
  std::string module_str = R"hlo(
HloModule module, is_scheduled=true

fused_computation {
  parameter.0 = f32[] parameter(0)
  parameter.1 = f32[] parameter(1)
  ROOT add.0 = f32[] add(parameter.0, parameter.1)
}

ENTRY entry {
  parameter.0 = f32[] parameter(0)
  parameter.1 = f32[] parameter(1)
  fusion.0 = f32[] fusion(parameter.0, parameter.1), kind=kLoop, calls=fused_computation
  ROOT add.0 = f32[] add(fusion.0, parameter.1)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                       ParseAndReturnVerifiedModule(module_str));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                       ParseAndReturnVerifiedModule(module_str));
  HloComputation* fusion_l =
      module_l->GetComputationWithName("fused_computation");
  HloComputation* fusion_r =
      module_r->GetComputationWithName("fused_computation");
  DiffResult diff_result;
  DiffSummary diff_summary;
  diff_summary.computation_summary = {
      {fusion_l, ComputationSummary{DiffSide::kLeft, nullptr, 0, 0, 0, false}},
      {fusion_r, ComputationSummary{DiffSide::kRight, nullptr, 0, 0, 0, true}}};

  MeRenderOptions options;
  options.collapse_unchanged_computations = false;
  ASSERT_OK_AND_ASSIGN(
      MeJson result,
      RenderMe(*module_l, *module_r, diff_result, &diff_summary, options));

  EXPECT_EQ(result.graph_collection.graphs.size(), 2);

  const Graph& me_graph_l = result.graph_collection.graphs.at(0);
  const Graph& me_graph_r = result.graph_collection.graphs.at(1);

  // Check the number of nodes in each graph.
  EXPECT_EQ(me_graph_l.subgraphs.at(0).nodes.size(), 8);
  EXPECT_EQ(me_graph_r.subgraphs.at(0).nodes.size(), 8);
}

TEST_F(HloDiffRendererTest, GenerateMeJsonWithHideUnchangedSubgraph) {
  std::string module_str = R"hlo(
HloModule module, is_scheduled=true
ENTRY entry {
  parameter.0 = f32[] parameter(0)
  parameter.1 = f32[] parameter(1)
  add0 = f32[] add(parameter.0, parameter.1)
  neg0 = f32[] negate(add0)
  abs0 = f32[] abs(parameter.1)
  ROOT tuple = tuple(neg0, abs0)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                       ParseAndReturnVerifiedModule(module_str));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                       ParseAndReturnVerifiedModule(module_str));

  DiffResult diff_result;
  const HloInstruction* l_p0 =
      module_l->entry_computation()->parameter_instruction(0);
  const HloInstruction* l_p1 =
      module_l->entry_computation()->parameter_instruction(1);
  ASSERT_OK_AND_ASSIGN(const HloInstruction* l_add0,
                       GetInstructionByName(*module_l, "add0"));
  ASSERT_OK_AND_ASSIGN(const HloInstruction* l_neg0,
                       GetInstructionByName(*module_l, "neg0"));
  ASSERT_OK_AND_ASSIGN(const HloInstruction* l_abs0,
                       GetInstructionByName(*module_l, "abs0"));
  ASSERT_OK_AND_ASSIGN(const HloInstruction* l_tuple,
                       GetInstructionByName(*module_l, "tuple"));

  const HloInstruction* r_p0 =
      module_r->entry_computation()->parameter_instruction(0);
  const HloInstruction* r_p1 =
      module_r->entry_computation()->parameter_instruction(1);
  ASSERT_OK_AND_ASSIGN(const HloInstruction* r_add0,
                       GetInstructionByName(*module_r, "add0"));
  ASSERT_OK_AND_ASSIGN(const HloInstruction* r_neg0,
                       GetInstructionByName(*module_r, "neg0"));
  ASSERT_OK_AND_ASSIGN(const HloInstruction* r_abs0,
                       GetInstructionByName(*module_r, "abs0"));
  ASSERT_OK_AND_ASSIGN(const HloInstruction* r_tuple,
                       GetInstructionByName(*module_r, "tuple"));

  diff_result.AddChangedInstruction(l_abs0, r_abs0);
  diff_result.AddUnchangedInstruction(l_p0, r_p0);
  diff_result.AddUnchangedInstruction(l_p1, r_p1);
  diff_result.AddUnchangedInstruction(l_add0, r_add0);
  diff_result.AddUnchangedInstruction(l_neg0, r_neg0);
  diff_result.AddUnchangedInstruction(l_tuple, r_tuple);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<const DiffSummary> diff_summary,
                       ConstructDiffSummary(*module_l, *module_r, diff_result));

  MeRenderOptions options;
  options.collapse_unchanged_computations = false;
  options.hide_unchanged_subgraphs = true;
  ASSERT_OK_AND_ASSIGN(
      MeJson result,
      RenderMe(*module_l, *module_r, diff_result, diff_summary.get(), options));

  EXPECT_EQ(result.graph_collection.graphs.size(), 2);

  const Graph& me_graph_l = result.graph_collection.graphs.at(0);
  const Graph& me_graph_r = result.graph_collection.graphs.at(1);

  // Check the number of nodes in each graph.
  // Only tuple and abs0 should remain.
  EXPECT_EQ(me_graph_l.subgraphs.at(0).nodes.size(), 2);
  EXPECT_EQ(me_graph_r.subgraphs.at(0).nodes.size(), 2);

  EXPECT_OK(GetNodeById(me_graph_l, "abs0"));
  EXPECT_OK(GetNodeById(me_graph_l, "tuple"));
  EXPECT_FALSE(GetNodeById(me_graph_l, "parameter.1").ok());
  EXPECT_FALSE(GetNodeById(me_graph_l, "parameter.0").ok());
  EXPECT_FALSE(GetNodeById(me_graph_l, "add0").ok());
  EXPECT_FALSE(GetNodeById(me_graph_l, "neg0").ok());
}

}  // namespace
}  // namespace hlo_diff
}  // namespace xla
