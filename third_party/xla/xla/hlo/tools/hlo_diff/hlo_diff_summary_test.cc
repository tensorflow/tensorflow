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

#include "xla/hlo/tools/hlo_diff/hlo_diff_summary.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/proto/diff_result.pb.h"
#include "xla/hlo/tools/hlo_diff/utils/test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace hlo_diff {
namespace {

using ::testing::ExplainMatchResult;
using ::testing::FieldsAre;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::Pointee;
using ::testing::Property;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedPointwise;

class HloDiffTest : public HloHardwareIndependentTestBase {};

TEST_F(HloDiffTest, FindMainMatchedComputationWorks) {
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
  HloGumgraphMappings mappings;
  // Root nodes are matched by default before the matcher is called.
  mappings.MapInstructionsIfAbsent(&graph_l->GetRoot(), &graph_r->GetRoot(),
                                   MatcherType::kManual);
  MatchAllNodesByName(*graph_l, *graph_r, mappings);
  ASSERT_NO_FATAL_FAILURE(
      OverwriteMapInstructions(GetNodeByName(*graph_l, "add.1"),
                               GetNodeByName(*graph_r, "add.2"), mappings,
                               /*position_unchanged=*/true));
  ASSERT_NO_FATAL_FAILURE(
      OverwriteMapInstructions(GetNodeByName(*graph_l, "add.2"),
                               GetNodeByName(*graph_r, "add.1"), mappings,
                               /*position_unchanged=*/true));
  std::unique_ptr<const DiffResult> diff_result =
      ConstructDiffResult(*graph_l, *graph_r, mappings);
  std::unique_ptr<const DiffSummary> diff_summary =
      ConstructDiffSummary(*module_l, *module_r, *diff_result);
  absl::flat_hash_map<const HloComputation*, ComputationSummary>
      left_computation_summary;
  for (const auto& [computation, _] : graph_l->AllComputationProps()) {
    if (auto it = diff_summary->computation_summary.find(computation);
        it != diff_summary->computation_summary.end()) {
      left_computation_summary[computation] = it->second;
    }
  }
  absl::flat_hash_map<const HloComputation*, ComputationSummary>
      right_computation_summary;
  for (const auto& [computation, _] : graph_r->AllComputationProps()) {
    if (auto it = diff_summary->computation_summary.find(computation);
        it != diff_summary->computation_summary.end()) {
      right_computation_summary[computation] = it->second;
    }
  }

  EXPECT_THAT(
      left_computation_summary,
      UnorderedElementsAre(
          Pair(Pointee(Property(&HloComputation::name, "entry")),
               FieldsAre(/*side=*/DiffSide::kLeft,
                         /*main_matched_computation=*/
                         Pointee(Property(&HloComputation::name, "entry")),
                         /*max_matched_instruction_count=*/7,
                         /*split_allegiance_instruction=*/0,
                         /*diff_fingerprint=*/3570884195340145402U,
                         /*all_unchanged=*/true)),
          Pair(Pointee(Property(&HloComputation::name, "fused_computation.1")),
               FieldsAre(/*side=*/DiffSide::kLeft,
                         /*main_matched_computation=*/
                         Pointee(Property(&HloComputation::name,
                                          "fused_computation.1")),
                         /*max_matched_instruction_count=*/2,
                         /*split_allegiance_instruction=*/1,
                         /*diff_fingerprint=*/2604941079081458563U,
                         /*all_unchanged=*/true)),
          Pair(Pointee(Property(&HloComputation::name, "fused_computation.2")),
               FieldsAre(/*side=*/DiffSide::kLeft,
                         /*main_matched_computation=*/
                         Pointee(Property(&HloComputation::name,
                                          "fused_computation.2")),
                         /*max_matched_instruction_count=*/2,
                         /*split_allegiance_instruction=*/1,
                         /*diff_fingerprint=*/2604941079081458563U,
                         /*all_unchanged=*/true))));
  EXPECT_THAT(
      right_computation_summary,
      UnorderedElementsAre(
          Pair(Pointee(Property(&HloComputation::name, "entry")),
               FieldsAre(/*side=*/DiffSide::kRight,
                         /*main_matched_computation=*/
                         Pointee(Property(&HloComputation::name, "entry")),
                         /*max_matched_instruction_count=*/7,
                         /*split_allegiance_instruction=*/0,
                         /*diff_fingerprint=*/3570884195340145402U,
                         /*all_unchanged=*/true)),
          Pair(Pointee(Property(&HloComputation::name, "fused_computation.1")),
               FieldsAre(/*side=*/DiffSide::kRight,
                         /*main_matched_computation=*/
                         Pointee(Property(&HloComputation::name,
                                          "fused_computation.1")),
                         /*max_matched_instruction_count=*/2,
                         /*split_allegiance_instruction=*/1,
                         /*diff_fingerprint=*/2604941079081458563U,
                         /*all_unchanged=*/true)),
          Pair(Pointee(Property(&HloComputation::name, "fused_computation.2")),
               FieldsAre(/*side=*/DiffSide::kRight,
                         /*main_matched_computation=*/
                         Pointee(Property(&HloComputation::name,
                                          "fused_computation.2")),
                         /*max_matched_instruction_count=*/2,
                         /*split_allegiance_instruction=*/1,
                         /*diff_fingerprint=*/2604941079081458563U,
                         /*all_unchanged=*/true))));
}

TEST_F(HloDiffTest, ComputationDiffFingerprintWorks) {
  // Create left module with entry computation containing the following
  // structure:
  // [Param 0] ---> ┌-------┐
  //                | add_0 |
  // [Param 1] ---> └-------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(R"(
  HloModule module, is_scheduled=true
  
  ENTRY entry {
    parameter.0 = f32[] parameter(0)
    parameter.1 = f32[] parameter(1)
    add.0 = f32[] add(parameter.0, parameter.1)
  }
  )"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_l,
                          HloGumgraph::Create(module_l.get()));

  // Create right module with entry computation containing the following
  // structure:
  // [Param 1] ---> ┌-------┐
  //                | add_0 |
  // [Param 0] ---> └-------┘

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(R"(
  HloModule module, is_scheduled=true
  
  ENTRY entry {
    parameter.0 = f32[] parameter(0)
    parameter.1 = f32[] parameter(1)
    add.0 = f32[] add(parameter.1, parameter.0)
  }
  )"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_r,
                          HloGumgraph::Create(module_r.get()));
  HloGumgraphMappings mappings;
  ASSERT_NO_FATAL_FAILURE(OverwriteMapInstructions(
      GetNodeByName(*graph_l, "add.0"), GetNodeByName(*graph_r, "add.0"),
      mappings, true));
  std::unique_ptr<const DiffResult> diff_result =
      ConstructDiffResult(*graph_l, *graph_r, mappings);
  std::unique_ptr<const DiffSummary> diff_summary =
      ConstructDiffSummary(*module_l, *module_r, *diff_result);
  absl::flat_hash_map<const HloComputation*, ComputationSummary>
      left_computation_summary;
  for (const auto& [computation, _] : graph_l->AllComputationProps()) {
    if (auto it = diff_summary->computation_summary.find(computation);
        it != diff_summary->computation_summary.end()) {
      left_computation_summary[computation] = it->second;
    }
  }
  absl::flat_hash_map<const HloComputation*, ComputationSummary>
      right_computation_summary;
  for (const auto& [computation, _] : graph_r->AllComputationProps()) {
    if (auto it = diff_summary->computation_summary.find(computation);
        it != diff_summary->computation_summary.end()) {
      right_computation_summary[computation] = it->second;
    }
  }
  EXPECT_THAT(left_computation_summary,
              UnorderedElementsAre(Pair(
                  Pointee(Property(&HloComputation::name, "entry")),
                  FieldsAre(/*side=*/DiffSide::kLeft,
                            /*main_matched_computation=*/
                            Pointee(Property(&HloComputation::name, "entry")),
                            /*max_matched_instruction_count=*/1,
                            /*split_allegiance_instruction=*/0,
                            /*diff_fingerprint=*/13464792036913846758U,
                            /*all_unchanged=*/false))));
  EXPECT_THAT(right_computation_summary,
              UnorderedElementsAre(Pair(
                  Pointee(Property(&HloComputation::name, "entry")),
                  FieldsAre(/*side=*/DiffSide::kRight,
                            /*main_matched_computation=*/
                            Pointee(Property(&HloComputation::name, "entry")),
                            /*max_matched_instruction_count=*/1,
                            /*split_allegiance_instruction=*/0,
                            /*diff_fingerprint=*/13464792036913846758U,
                            /*all_unchanged=*/false))));
  EXPECT_THAT(diff_summary->computation_diff_patterns,
              UnorderedElementsAre(FieldsAre(
                  /*fingerprint=*/2864899211444957078U,
                  /*computation_groups=*/
                  UnorderedElementsAre(FieldsAre(
                      /*left_computations=*/UnorderedElementsAre(
                          Pointee(Property(&HloComputation::name, "entry"))),
                      /*right_computations=*/UnorderedElementsAre(
                          Pointee(Property(&HloComputation::name, "entry"))))),
                  /*diff_metrics=*/
                  FieldsAre(/*changed_instruction_count=*/0,
                            /*left_unmatched_instruction_count=*/2,
                            /*right_unmatched_instruction_count=*/2))));
}

TEST_F(HloDiffTest, FindConnectedComponentsWorks) {
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
  auto mappings = std::make_unique<HloGumgraphMappings>();
  // Root nodes are matched by default before the matcher is called.
  mappings->MapInstructionsIfAbsent(&graph_l->GetRoot(), &graph_r->GetRoot(),
                                    MatcherType::kManual);
  ASSERT_NO_FATAL_FAILURE(
      OverwriteMapInstructions(GetNodeByName(*graph_l, "add.1"),
                               GetNodeByName(*graph_r, "add.2"), *mappings,
                               /*position_unchanged=*/true));
  ASSERT_NO_FATAL_FAILURE(
      OverwriteMapInstructions(GetNodeByName(*graph_l, "add.2"),
                               GetNodeByName(*graph_r, "add.1"), *mappings,
                               /*position_unchanged=*/true));
  ASSERT_NO_FATAL_FAILURE(
      OverwriteMapInstructions(GetNodeByName(*graph_l, "add.3"),
                               GetNodeByName(*graph_r, "add.3"), *mappings,
                               /*position_unchanged=*/true));
  std::unique_ptr<const DiffResult> diff_result =
      ConstructDiffResult(*graph_l, *graph_r, *mappings);
  std::unique_ptr<const DiffSummary> diff_summary =
      ConstructDiffSummary(*module_l, *module_r, *diff_result);
  EXPECT_THAT(
      diff_summary->computation_diff_patterns,
      UnorderedElementsAre(
          FieldsAre(
              /*fingerprint=*/2864899211444957078U,
              /*computation_groups=*/
              UnorderedElementsAre(
                  FieldsAre(/*left_computations=*/UnorderedElementsAre(
                                Pointee(Property(&HloComputation::name,
                                                 "fused_computation.1"))),
                            /*right_computations=*/UnorderedElementsAre(
                                Pointee(Property(&HloComputation::name,
                                                 "fused_computation.2")))),
                  FieldsAre(/*left_computations=*/UnorderedElementsAre(
                                Pointee(Property(&HloComputation::name,
                                                 "fused_computation.2"))),
                            /*right_computations=*/UnorderedElementsAre(
                                Pointee(Property(&HloComputation::name,
                                                 "fused_computation.1"))))),
              /*diff_metrics=*/
              FieldsAre(/*changed_instruction_count=*/0,
                        /*left_unmatched_instruction_count=*/2,
                        /*right_unmatched_instruction_count=*/2)),
          FieldsAre(/*fingerprint=*/15473561031564762362U,
                    /*computation_groups=*/
                    UnorderedElementsAre(FieldsAre(
                        /*left_computations=*/UnorderedElementsAre(
                            Pointee(Property(&HloComputation::name, "entry"))),
                        /*right_computations=*/UnorderedElementsAre(Pointee(
                            Property(&HloComputation::name, "entry"))))),
                    /*diff_metrics=*/
                    FieldsAre(/*changed_instruction_count=*/0,
                              /*left_unmatched_instruction_count=*/6,
                              /*right_unmatched_instruction_count=*/6))));
}

TEST_F(HloDiffTest, FindConnectedComponentsWorksForIsolatedComputations) {
  // Create left module with entry computation containing the following
  // structure:
  // [Param 0] ---> ┌-------┐
  //                | add_0 |
  // [Param 1] ---> └-------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
parameter.0 = f32[] parameter(0)
parameter.1 = f32[] parameter(1)
add.0 = f32[] add(parameter.0, parameter.1)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_l,
                          HloGumgraph::Create(module_l.get()));
  // Create left module with entry computation containing the following
  // structure:
  // [Const 0] ---> ┌-------┐
  //                | sub_0 |
  // [Const 1] ---> └-------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
constant.0 = f32[] constant(0)
constant.1 = f32[] constant(1)
subtract.0 = f32[] subtract(constant.0, constant.1)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_r,
                          HloGumgraph::Create(module_r.get()));
  auto mappings = std::make_unique<HloGumgraphMappings>();
  // Root nodes are matched by default before the matcher is called.
  mappings->MapInstructionsIfAbsent(&graph_l->GetRoot(), &graph_r->GetRoot(),
                                    MatcherType::kManual);
  std::unique_ptr<const DiffResult> diff_result =
      ConstructDiffResult(*graph_l, *graph_r, *mappings);
  std::unique_ptr<const DiffSummary> diff_summary =
      ConstructDiffSummary(*module_l, *module_r, *diff_result);
  EXPECT_THAT(diff_summary->computation_diff_patterns,
              UnorderedElementsAre(
                  FieldsAre(
                      /*fingerprint=*/619838372110990418U,
                      /*computation_groups=*/
                      UnorderedElementsAre(FieldsAre(
                          /*left_computations=*/UnorderedElementsAre(Pointee(
                              Property(&HloComputation::name, "entry"))),
                          /*right_computations=*/IsEmpty())),
                      /*diff_metrics=*/
                      FieldsAre(/*changed_instruction_count=*/0,
                                /*left_unmatched_instruction_count=*/3,
                                /*right_unmatched_instruction_count=*/0)),
                  FieldsAre(
                      /*fingerprint=*/591642684880638740U,
                      /*computation_groups=*/
                      UnorderedElementsAre(FieldsAre(
                          /*left_computations=*/IsEmpty(),
                          /*right_computations=*/UnorderedElementsAre(Pointee(
                              Property(&HloComputation::name, "entry"))))),
                      /*diff_metrics=*/
                      FieldsAre(/*changed_instruction_count=*/0,
                                /*left_unmatched_instruction_count=*/0,
                                /*right_unmatched_instruction_count=*/3))));
}

MATCHER(EqualsComputationGroup, "") {
  const ComputationGroup& a = std::get<0>(arg);
  const ComputationGroup& b = std::get<1>(arg);
  return ExplainMatchResult(a.left_computations, b.left_computations,
                            result_listener) &&
         ExplainMatchResult(a.right_computations, b.right_computations,
                            result_listener);
}

MATCHER_P(EqualsDiffMetrics, a, "") {
  const DiffMetrics& b = arg;
  return ExplainMatchResult(a.changed_instruction_count,
                            b.changed_instruction_count, result_listener) &&
         ExplainMatchResult(a.left_unmatched_instruction_count,
                            b.left_unmatched_instruction_count,
                            result_listener) &&
         ExplainMatchResult(a.right_unmatched_instruction_count,
                            b.right_unmatched_instruction_count,
                            result_listener);
}

MATCHER(EqualsComputationDiffPattern, "") {
  const ComputationDiffPattern& a = std::get<0>(arg);
  const ComputationDiffPattern& b = std::get<1>(arg);
  return ExplainMatchResult(a.fingerprint, b.fingerprint, result_listener) &&
         ExplainMatchResult(
             UnorderedPointwise(EqualsComputationGroup(), a.computation_groups),
             b.computation_groups, result_listener) &&
         ExplainMatchResult(EqualsDiffMetrics(a.diff_metrics), b.diff_metrics,
                            result_listener);
}

MATCHER_P(EqualsComputationSummary, a, "") {
  const ComputationSummary& b = arg;
  return ExplainMatchResult(a.side, b.side, result_listener) &&
         ExplainMatchResult(a.main_matched_computation,
                            b.main_matched_computation, result_listener) &&
         ExplainMatchResult(a.max_matched_instruction_count,
                            b.max_matched_instruction_count, result_listener) &&
         ExplainMatchResult(a.split_allegiance_instruction_count,
                            b.split_allegiance_instruction_count,
                            result_listener) &&
         ExplainMatchResult(a.diff_fingerprint, b.diff_fingerprint,
                            result_listener) &&
         ExplainMatchResult(a.all_unchanged, b.all_unchanged, result_listener);
}

MATCHER(EqualsComputationSummaryMapElement, "") {
  const auto& a = std::get<0>(arg);
  const auto& b = std::get<1>(arg);
  return ExplainMatchResult(a.first, b.first, result_listener) &&
         ExplainMatchResult(EqualsComputationSummary(a.second), b.second,
                            result_listener);
}

TEST_F(HloDiffTest, DiffSummaryFromDiffResultProtoWorks) {
  DiffResult diff_result;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(R"(
  HloModule module, is_scheduled=true
  
  ENTRY entry {
    parameter.0 = f32[] parameter(0)
    parameter.1 = f32[] parameter(1)
    add.0 = f32[] add(parameter.0, parameter.1)
  }
  )"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(R"(
  HloModule module, is_scheduled=true
  
  ENTRY entry {
    parameter.0 = f32[] parameter(0)
    parameter.1 = f32[] parameter(1)
    add.0 = f32[] add(parameter.1, parameter.0)
  }
  )"));
  diff_result.unchanged_instructions.insert(
      {module_l->entry_computation()->root_instruction(),
       module_r->entry_computation()->root_instruction()});
  diff_result.changed_instructions.insert(
      {module_l->entry_computation()->parameter_instruction(0),
       module_r->entry_computation()->parameter_instruction(1)});
  diff_result.left_module_unmatched_instructions.insert(
      module_l->entry_computation()->parameter_instruction(1));
  diff_result.right_module_unmatched_instructions.insert(
      module_r->entry_computation()->parameter_instruction(0));

  DiffResultProto proto = diff_result.ToProto();
  DiffResult diff_result_from_proto =
      DiffResult::FromProto(proto, *module_l, *module_r);
  std::unique_ptr<const DiffSummary> diff_summary =
      ConstructDiffSummary(*module_l, *module_r, diff_result_from_proto);
  std::unique_ptr<const DiffSummary> expected_diff_summary =
      ConstructDiffSummary(*module_l, *module_r, diff_result);
  EXPECT_THAT(diff_summary->computation_summary,
              UnorderedPointwise(EqualsComputationSummaryMapElement(),
                                 expected_diff_summary->computation_summary));
  EXPECT_THAT(
      diff_summary->computation_diff_patterns,
      UnorderedPointwise(EqualsComputationDiffPattern(),
                         expected_diff_summary->computation_diff_patterns));
}

}  // namespace
}  // namespace hlo_diff
}  // namespace xla
