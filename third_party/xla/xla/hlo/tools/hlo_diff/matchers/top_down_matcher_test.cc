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

#include "xla/hlo/tools/hlo_diff/matchers/top_down_matcher.h"

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

class TopDownMatcherTest : public HloHardwareIndependentTestBase {};

TEST_F(TopDownMatcherTest, GreedyTopDownMatcherStopAtUnmatchedType) {
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
  auto matcher =
      std::make_unique<GreedyTopDownMatcher>(graph_l.get(), graph_r.get());
  // Root nodes are matched by default before the matcher is called.
  mappings->MapInstructionsIfAbsent(&graph_l->GetRoot(), &graph_r->GetRoot(),
                                    MatcherType::kManual);
  matcher->Match(*mappings);
  auto mapped_nodes = ExtractMappedInstructionNames(*mappings);

  EXPECT_THAT(mapped_nodes,
              UnorderedElementsAre(
                  Pair("constant.0", "constant.0"),
                  Pair("constant.1", "constant.1"), Pair("add.0", "add.0"),
                  Pair("add.3", "add.3"), Pair("constant.4", "constant.4"),
                  Pair("add.4", "add.4"), Pair("root_L", "root_R")));
}

TEST_F(TopDownMatcherTest, GreedyTopDownMatcherStopAtMappedNode) {
  // Create left module with entry computation containing the following
  // structure:
  // [const.0] ---> ┌-------┐
  //                | add.0 | ---> ┌-------┐
  // [const.1] ---> └-------┘      |       |      ┌-------┐
  //                               | add.3 | ---> |       |
  // [const.2] ---> ┌-------┐      |       |      |       |      ┌------┐
  //                | add.1 | ---> └-------┘      | add.4 | ---> | ROOT |
  // [const.3] ---> └-------┘                     |       |      └------┘
  //                                              |       |
  // [const.4] ---------------------------------> └-------┘
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
  add.1 = f32[] add(constant.2, constant.3)
  add.3 = f32[] add(add.0, add.1)
  add.4 = f32[] add(add.3, constant.4)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_l,
                          HloGumgraph::Create(module_l.get()));

  // Create right module with entry computation containing the following
  // structure:
  // [const.0] ---> ┌-------┐
  //                | add.0 | ---> ┌-------┐
  // [const.1] ---> └-------┘      |       |      ┌-------┐
  //                               | add.3 | ---> |       |
  // [const.2] ---> ┌-------┐      |       |      |       |
  //                | add.1 | ---> └-------┘      |       |      ┌------┐
  // [const.3] ---> └-------┘                     | add.4 | ---> | ROOT |
  //                                              |       |      └------┘
  // [const.4] ---> ┌-------┐                     |       |
  //                | add.2 | ------------------> |       |
  // [const.5] ---> └-------┘                     └-------┘
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
  constant.5 = f32[] constant(0)
  add.0 = f32[] add(constant.0, constant.1)
  add.1 = f32[] add(constant.2, constant.3)
  add.2 = f32[] add(constant.4, constant.5)
  add.3 = f32[] add(add.0, add.1)
  add.4 = f32[] add(add.3, add.2)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_r,
                          HloGumgraph::Create(module_r.get()));
  auto mappings = std::make_unique<HloGumgraphMappings>();
  ASSERT_NO_FATAL_FAILURE(
      OverwriteMapInstructions(GetNodeByName(*graph_l, "add.4"),
                               GetNodeByName(*graph_r, "add.4"), *mappings));
  ASSERT_NO_FATAL_FAILURE(
      OverwriteMapInstructions(GetNodeByName(*graph_l, "add.1"),
                               GetNodeByName(*graph_r, "add.2"), *mappings));
  auto matcher =
      std::make_unique<GreedyTopDownMatcher>(graph_l.get(), graph_r.get());
  // Root nodes are matched by default before the matcher is called.
  mappings->MapInstructionsIfAbsent(&graph_l->GetRoot(), &graph_r->GetRoot(),
                                    MatcherType::kManual);
  matcher->Match(*mappings);
  auto mapped_nodes = ExtractMappedInstructionNames(*mappings);

  EXPECT_THAT(
      mapped_nodes,
      UnorderedElementsAre(
          Pair("constant.0", "constant.0"), Pair("constant.1", "constant.1"),
          Pair("add.0", "add.0"), Pair("constant.2", "constant.4"),
          Pair("constant.3", "constant.5"), Pair("add.1", "add.2"),
          Pair("add.3", "add.3"), Pair("add.4", "add.4"),
          Pair("root_L", "root_R")));
}

TEST_F(TopDownMatcherTest, GreedyTopDownMatcherStopAtDifferentChildren) {
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
  auto matcher = std::make_unique<GreedyTopDownMatcher>(
      graph_l.get(), graph_r.get(), /*require_same_children=*/true);
  // Root nodes are matched by default before the matcher is called.
  mappings->MapInstructionsIfAbsent(&graph_l->GetRoot(), &graph_r->GetRoot(),
                                    MatcherType::kManual);
  matcher->Match(*mappings);
  auto mapped_nodes = ExtractMappedInstructionNames(*mappings);

  EXPECT_THAT(mapped_nodes,
              UnorderedElementsAre(
                  Pair("add.3", "add.3"), Pair("constant.4", "constant.4"),
                  Pair("add.4", "add.4"), Pair("root_L", "root_R")));
}

}  // namespace
}  // namespace xla::hlo_diff
