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

#include "xla/hlo/tools/hlo_diff/matchers/bottom_up_matcher.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/utils/test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::hlo_diff {
namespace {

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

class BottomUpMatcherTest : public HloHardwareIndependentTestBase {};

TEST_F(BottomUpMatcherTest, GreedyLimitedCandidatesBottomUpMatcher) {
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
  auto mappings = std::make_unique<HloGumgraphMappings>();
  ASSERT_NO_FATAL_FAILURE(OverwriteMapInstructions(
      GetNodeByName(*graph_l, "constant.0"),
      GetNodeByName(*graph_r, "constant.0"), *mappings));
  ASSERT_NO_FATAL_FAILURE(OverwriteMapInstructions(
      GetNodeByName(*graph_l, "constant.1"),
      GetNodeByName(*graph_r, "constant.1"), *mappings));
  ASSERT_NO_FATAL_FAILURE(OverwriteMapInstructions(
      GetNodeByName(*graph_l, "constant.2"),
      GetNodeByName(*graph_r, "constant.2"), *mappings));
  ASSERT_NO_FATAL_FAILURE(OverwriteMapInstructions(
      GetNodeByName(*graph_l, "constant.3"),
      GetNodeByName(*graph_r, "constant.3"), *mappings));
  ASSERT_NO_FATAL_FAILURE(OverwriteMapInstructions(
      GetNodeByName(*graph_l, "constant.4"),
      GetNodeByName(*graph_r, "constant.4"), *mappings));
  ASSERT_NO_FATAL_FAILURE(
      OverwriteMapInstructions(GetNodeByName(*graph_l, "add.0"),
                               GetNodeByName(*graph_r, "add.0"), *mappings));
  auto matcher = std::make_unique<GreedyLimitedCandidatesBottomUpMatcher>(
      graph_l.get(), graph_r.get());
  // Root nodes are matched by default before the matcher is called.
  mappings->MapInstructionsIfAbsent(&graph_l->GetRoot(), &graph_r->GetRoot(),
                                    MatcherType::kManual);
  matcher->Match(*mappings);
  auto mapped_nodes = ExtractMappedInstructionNames(*mappings);

  EXPECT_THAT(
      mapped_nodes,
      UnorderedElementsAre(
          Pair("constant.0", "constant.0"), Pair("constant.1", "constant.1"),
          Pair("constant.2", "constant.2"), Pair("constant.3", "constant.3"),
          Pair("add.0", "add.0"), Pair("add.3", "add.3"),
          Pair("constant.4", "constant.4"), Pair("add.4", "add.4"),
          Pair("root_L", "root_R")));
}

TEST_F(BottomUpMatcherTest,
       GreedyLimitedCandidatesBottomUpMatcherAmbiguousMatch) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  constant.0 = f32[] constant(0)
  constant.1 = f32[] constant(0)
  add.0 = f32[] add(constant.0, constant.1)
  add.1 = f32[] add(constant.0, constant.1)
  add.2 = f32[] add(add.0, constant.0)
  subtract.1 = f32[] subtract(add.1, add.2)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_l,
                          HloGumgraph::Create(module_l.get()));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  constant.0 = f32[] constant(0)
  constant.1 = f32[] constant(0)
  add.10 = f32[] add(constant.0, constant.1)
  add.11 = f32[] add(constant.0, constant.1)
  add.12 = f32[] add(add.10, constant.0)
  subtract.1 = f32[] subtract(add.11, add.12)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_r,
                          HloGumgraph::Create(module_r.get()));
  auto mappings = std::make_unique<HloGumgraphMappings>();
  ASSERT_NO_FATAL_FAILURE(OverwriteMapInstructions(
      GetNodeByName(*graph_l, "constant.0"),
      GetNodeByName(*graph_r, "constant.0"), *mappings));
  ASSERT_NO_FATAL_FAILURE(OverwriteMapInstructions(
      GetNodeByName(*graph_l, "constant.1"),
      GetNodeByName(*graph_r, "constant.1"), *mappings));
  auto matcher = std::make_unique<GreedyLimitedCandidatesBottomUpMatcher>(
      graph_l.get(), graph_r.get());
  // Root nodes are matched by default before the matcher is called.
  mappings->MapInstructionsIfAbsent(&graph_l->GetRoot(), &graph_r->GetRoot(),
                                    MatcherType::kManual);
  matcher->Match(*mappings);
  auto mapped_nodes = ExtractMappedInstructionNames(*mappings);

  EXPECT_THAT(mapped_nodes,
              UnorderedElementsAre(
                  Pair("constant.0", "constant.0"),
                  Pair("constant.1", "constant.1"), Pair("add.0", "add.10"),
                  Pair("add.1", "add.11"), Pair("add.2", "add.12"),
                  Pair("subtract.1", "subtract.1"), Pair("root_L", "root_R")));
}

TEST_F(BottomUpMatcherTest,
       GreedyLimitedCandidatesBottomUpMatcherHloValueTraced) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

fused_computation0 {
  param_0 = f32[] parameter(0)
  ROOT negate.0 = f32[] negate(param_0)
}

fused_computation1 {
  param_1 = f32[] parameter(0)
  ROOT abs.0 = f32[] abs(param_1)
}

ENTRY entry {
  constant.0 = f32[] constant(0)
  bitcast.0 = f32[] bitcast(constant.0)
  copy.0 = f32[] copy(bitcast.0)
  fusion.0 = f32[] fusion(bitcast.0), kind=kLoop, calls=fused_computation0
  fusion.1 = f32[] fusion(copy.0), kind=kLoop, calls=fused_computation1
  ROOT add.0 = f32[] add(fusion.0, fusion.1)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_l,
                          HloGumgraph::Create(module_l.get()));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

fused_computation0 {
  param_0 = f32[] parameter(0)
  ROOT negate.0 = f32[] negate(param_0)
}

fused_computation1 {
  param_1 = f32[] parameter(0)
  ROOT abs.0 = f32[] abs(param_1)
}

ENTRY entry {
  constant.0 = f32[] constant(0)
  bitcast.0 = f32[] bitcast(constant.0)
  copy.0 = f32[] copy(bitcast.0)
  fusion.0 = f32[] fusion(bitcast.0), kind=kLoop, calls=fused_computation1
  fusion.1 = f32[] fusion(copy.0), kind=kLoop, calls=fused_computation0
  ROOT add.0 = f32[] add(fusion.0, fusion.1)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_r,
                          HloGumgraph::Create(module_r.get()));
  auto mappings = std::make_unique<HloGumgraphMappings>();
  ASSERT_NO_FATAL_FAILURE(OverwriteMapInstructions(
      GetNodeByName(*graph_l, "constant.0"),
      GetNodeByName(*graph_r, "constant.0"), *mappings));
  auto matcher = std::make_unique<GreedyLimitedCandidatesBottomUpMatcher>(
      graph_l.get(), graph_r.get());
  // Root nodes are matched by default before the matcher is called.
  mappings->MapInstructionsIfAbsent(&graph_l->GetRoot(), &graph_r->GetRoot(),
                                    MatcherType::kManual);
  matcher->Match(*mappings);
  auto mapped_nodes = ExtractMappedInstructionNames(*mappings);

  EXPECT_THAT(mapped_nodes,
              UnorderedElementsAre(
                  Pair("constant.0", "constant.0"),
                  Pair("bitcast.0", "bitcast.0"), Pair("copy.0", "copy.0"),
                  Pair("fusion.0", "fusion.1"), Pair("fusion.1", "fusion.0"),
                  Pair("add.0", "add.0"), Pair("negate.0", "negate.0"),
                  Pair("abs.0", "abs.0"), Pair("param_0", "param_0"),
                  Pair("param_1", "param_1"), Pair("root_L", "root_R")));
}

}  // namespace
}  // namespace xla::hlo_diff
