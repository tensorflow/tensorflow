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

#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/utils/test_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace hlo_diff {
namespace {

using ::testing::Pair;
using ::testing::Pointee;
using ::testing::Property;
using ::testing::UnorderedElementsAre;

class HloDiffTest : public HloTestBase {};

TEST_F(HloDiffTest, MatchedDifferentShapeMarkAsChanged) {
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
  // [Param 0] ---> ┌-------┐
  //                | add_0 |
  // [Param 1] ---> └-------┘

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  parameter.0 = f64[] parameter(0)
  parameter.1 = f32[] parameter(1)
  add.0 = f32[] add(parameter.0, parameter.1)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_r,
                          HloGumgraph::Create(module_r.get()));
  auto mappings = std::make_unique<HloGumgraphMappings>();
  ASSERT_NO_FATAL_FAILURE(OverwriteMapInstructions(
      GetNodeByName(*graph_l, "add.0"), GetNodeByName(*graph_r, "add.0"),
      *mappings, /*position_unchanged=*/true));
  ASSERT_NO_FATAL_FAILURE(
      OverwriteMapInstructions(GetNodeByName(*graph_l, "parameter.0"),
                               GetNodeByName(*graph_r, "parameter.0"),
                               *mappings, /*position_unchanged=*/true));
  ASSERT_NO_FATAL_FAILURE(
      OverwriteMapInstructions(GetNodeByName(*graph_l, "parameter.1"),
                               GetNodeByName(*graph_r, "parameter.1"),
                               *mappings, /*position_unchanged=*/true));
  auto diff_result = ConstructDiffResult(*graph_l, *graph_r, *mappings);

  EXPECT_THAT(diff_result->changed_instructions,
              UnorderedElementsAre(
                  Pair(Pointee(Property(&HloInstruction::name, "parameter.0")),
                       Pointee(Property(&HloInstruction::name, "parameter.0"))),
                  Pair(Pointee(Property(&HloInstruction::name, "add.0")),
                       Pointee(Property(&HloInstruction::name, "add.0")))));
  EXPECT_THAT(diff_result->unchanged_instructions,
              UnorderedElementsAre(Pair(
                  Pointee(Property(&HloInstruction::name, "parameter.1")),
                  Pointee(Property(&HloInstruction::name, "parameter.1")))));
}

TEST_F(HloDiffTest, MatchedDifferentFingerprintMarkAsChanged) {
  // Create left module with entry computation containing the following
  // structure:
  // [Param 0] ---> ┌-------┐      ┌------┐
  //                | add_0 | ---> | ROOT |
  // [Param 1] ---> └-------┘      └------┘
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
  // [Param 1] ---> ┌-------┐      ┌------┐
  //                | add_0 | ---> | ROOT |
  // [Param 0] ---> └-------┘      └------┘

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
  auto mappings = std::make_unique<HloGumgraphMappings>();
  ASSERT_NO_FATAL_FAILURE(OverwriteMapInstructions(
      GetNodeByName(*graph_l, "add.0"), GetNodeByName(*graph_r, "add.0"),
      *mappings, /*position_unchanged=*/true));
  ASSERT_NO_FATAL_FAILURE(
      OverwriteMapInstructions(GetNodeByName(*graph_l, "parameter.0"),
                               GetNodeByName(*graph_r, "parameter.1"),
                               *mappings, /*position_unchanged=*/true));
  ASSERT_NO_FATAL_FAILURE(
      OverwriteMapInstructions(GetNodeByName(*graph_l, "parameter.1"),
                               GetNodeByName(*graph_r, "parameter.0"),
                               *mappings, /*position_unchanged=*/true));
  auto diff_result = ConstructDiffResult(*graph_l, *graph_r, *mappings);

  EXPECT_THAT(
      diff_result->changed_instructions,
      UnorderedElementsAre(
          Pair(Pointee(Property(&HloInstruction::name, "parameter.0")),
               Pointee(Property(&HloInstruction::name, "parameter.1"))),
          Pair(Pointee(Property(&HloInstruction::name, "parameter.1")),
               Pointee(Property(&HloInstruction::name, "parameter.0")))));
  EXPECT_THAT(diff_result->unchanged_instructions,
              UnorderedElementsAre(
                  Pair(Pointee(Property(&HloInstruction::name, "add.0")),
                       Pointee(Property(&HloInstruction::name, "add.0")))));
}

TEST_F(HloDiffTest, UnmatchedInstructionsMarkAsUnmatched) {
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
  auto mappings = std::make_unique<HloGumgraphMappings>();
  ASSERT_NO_FATAL_FAILURE(OverwriteMapInstructions(
      GetNodeByName(*graph_l, "add.0"), GetNodeByName(*graph_r, "add.0"),
      *mappings, /*position_unchanged=*/true));
  auto diff_result = ConstructDiffResult(*graph_l, *graph_r, *mappings);

  EXPECT_THAT(diff_result->unchanged_instructions,
              UnorderedElementsAre(
                  Pair(Pointee(Property(&HloInstruction::name, "add.0")),
                       Pointee(Property(&HloInstruction::name, "add.0")))));
  EXPECT_THAT(diff_result->left_module_unmatched_instructions,
              UnorderedElementsAre(
                  Pointee(Property(&HloInstruction::name, "parameter.0")),
                  Pointee(Property(&HloInstruction::name, "parameter.1"))));
  EXPECT_THAT(diff_result->right_module_unmatched_instructions,
              UnorderedElementsAre(
                  Pointee(Property(&HloInstruction::name, "parameter.0")),
                  Pointee(Property(&HloInstruction::name, "parameter.1"))));
}

TEST_F(HloDiffTest, ShortFormConstantsMatched) {
  // Create left module with entry computation containing the following
  // structure:
  // [Param 0]    ---> ┌-------┐
  //                   | add_0 |
  // [Const 2958] ---> └-------┘
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> module_l,
      ParseAndReturnUnverifiedModule(
          R"(
HloModule module, is_scheduled=true

ENTRY entry {
  parameter.0 = s32[301]{0:T(512)} parameter(0)
  constant.2958 = s32[301]{0:T(512)} constant({...})
  add.0 = s32[301]{0:T(512)} add(parameter.0, constant.2958)
}
)",
          HloModuleConfig(),
          HloParserOptions().set_fill_shortform_constants_with_random_values(
              false)));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_l,
                          HloGumgraph::Create(module_l.get()));

  // Create right module with entry computation containing the following
  // structure:
  // [Param 0]    ---> ┌-------┐
  //                   | add_0 |
  // [Const 2958] ---> └-------┘

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> module_r,
      ParseAndReturnUnverifiedModule(
          R"(
HloModule module, is_scheduled=true

ENTRY entry {
  parameter.0 = s32[301]{0:T(512)} parameter(0)
  constant.2958 = s32[301]{0:T(512)} constant({...})
  add.0 = s32[301]{0:T(512)} add(parameter.0, constant.2958)
}
)",
          HloModuleConfig(),
          HloParserOptions().set_fill_shortform_constants_with_random_values(
              false)));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_r,
                          HloGumgraph::Create(module_r.get()));
  auto mappings = std::make_unique<HloGumgraphMappings>();
  ASSERT_NO_FATAL_FAILURE(OverwriteMapInstructions(
      GetNodeByName(*graph_l, "add.0"), GetNodeByName(*graph_r, "add.0"),
      *mappings, /*position_unchanged=*/true));
  ASSERT_NO_FATAL_FAILURE(
      OverwriteMapInstructions(GetNodeByName(*graph_l, "parameter.0"),
                               GetNodeByName(*graph_r, "parameter.0"),
                               *mappings, /*position_unchanged=*/true));
  ASSERT_NO_FATAL_FAILURE(
      OverwriteMapInstructions(GetNodeByName(*graph_l, "constant.2958"),
                               GetNodeByName(*graph_r, "constant.2958"),
                               *mappings, /*position_unchanged=*/true));
  auto diff_result = ConstructDiffResult(*graph_l, *graph_r, *mappings);

  EXPECT_THAT(
      diff_result->unchanged_instructions,
      UnorderedElementsAre(
          Pair(Pointee(Property(&HloInstruction::name, "constant.2958")),
               Pointee(Property(&HloInstruction::name, "constant.2958"))),
          Pair(Pointee(Property(&HloInstruction::name, "parameter.0")),
               Pointee(Property(&HloInstruction::name, "parameter.0"))),
          Pair(Pointee(Property(&HloInstruction::name, "add.0")),
               Pointee(Property(&HloInstruction::name, "add.0")))));
}

}  // namespace
}  // namespace hlo_diff
}  // namespace xla
