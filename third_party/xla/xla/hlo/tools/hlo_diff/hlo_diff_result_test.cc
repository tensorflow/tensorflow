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
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/proto/diff_result.pb.h"
#include "xla/hlo/tools/hlo_diff/utils/test_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace hlo_diff {
namespace {

using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::IsSupersetOf;
using ::testing::Not;
using ::testing::Pair;
using ::testing::Pointee;
using ::testing::Property;
using ::testing::UnorderedElementsAre;

class HloDiffTest : public HloHardwareIndependentTestBase {};

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

  EXPECT_THAT(diff_result->left_diff_codes,
              UnorderedElementsAre(
                  Pair(Pointee(Property(&HloInstruction::name, "parameter.0")),
                       DiffType::kChanged),
                  Pair(Pointee(Property(&HloInstruction::name, "parameter.1")),
                       DiffType::kUnchanged),
                  Pair(Pointee(Property(&HloInstruction::name, "add.0")),
                       DiffType::kChanged)));
  EXPECT_THAT(diff_result->right_diff_codes,
              UnorderedElementsAre(
                  Pair(Pointee(Property(&HloInstruction::name, "parameter.0")),
                       DiffType::kChanged),
                  Pair(Pointee(Property(&HloInstruction::name, "parameter.1")),
                       DiffType::kUnchanged),
                  Pair(Pointee(Property(&HloInstruction::name, "add.0")),
                       DiffType::kChanged)));
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
  parameter.0 = bf16[32,22,32]{0,2,1:T(8,128)(2,1)S(1)} parameter(0)
  parameter.1 = bf16[32,22,32]{0,2,1:T(8,128)(2,1)} parameter(1)
  add.0 = bf16[32,22,32]{0,2,1:T(8,128)(2,1)} add(parameter.0, parameter.1)
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
  parameter.0 = bf16[32,22,32]{0,2,1:T(8,128)(2,1)S(1)} parameter(0)
  parameter.1 = bf16[32,22,32]{0,2,1:T(8,128)(2,1)} parameter(1)
  add.0 = bf16[32,22,32]{0,2,1:T(8,128)(2,1)} add(parameter.1, parameter.0)
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

  EXPECT_THAT(diff_result->changed_instructions,
              UnorderedElementsAre(
                  Pair(Pointee(Property(&HloInstruction::name, "parameter.0")),
                       Pointee(Property(&HloInstruction::name, "parameter.1"))),
                  Pair(Pointee(Property(&HloInstruction::name, "parameter.1")),
                       Pointee(Property(&HloInstruction::name, "parameter.0"))),
                  Pair(Pointee(Property(&HloInstruction::name, "add.0")),
                       Pointee(Property(&HloInstruction::name, "add.0")))));
  // TODO(b/428249958): Should not mark as changed if the operands order has
  // changed.
  // EXPECT_THAT(diff_result->unchanged_instructions,
  //             UnorderedElementsAre(
  //                 Pair(Pointee(Property(&HloInstruction::name, "add.0")),
  //                      Pointee(Property(&HloInstruction::name, "add.0")))));

  EXPECT_THAT(diff_result->left_diff_codes,
              UnorderedElementsAre(
                  Pair(Pointee(Property(&HloInstruction::name, "parameter.0")),
                       DiffType::kChanged),
                  Pair(Pointee(Property(&HloInstruction::name, "parameter.1")),
                       DiffType::kChanged),
                  Pair(Pointee(Property(&HloInstruction::name, "add.0")),
                       DiffType::kChanged)));
  EXPECT_THAT(diff_result->right_diff_codes,
              UnorderedElementsAre(
                  Pair(Pointee(Property(&HloInstruction::name, "parameter.0")),
                       DiffType::kChanged),
                  Pair(Pointee(Property(&HloInstruction::name, "parameter.1")),
                       DiffType::kChanged),
                  Pair(Pointee(Property(&HloInstruction::name, "add.0")),
                       DiffType::kChanged)));
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

  EXPECT_THAT(diff_result->left_diff_codes,
              UnorderedElementsAre(
                  Pair(Pointee(Property(&HloInstruction::name, "parameter.0")),
                       DiffType::kUnmatched),
                  Pair(Pointee(Property(&HloInstruction::name, "parameter.1")),
                       DiffType::kUnmatched),
                  Pair(Pointee(Property(&HloInstruction::name, "add.0")),
                       DiffType::kUnchanged)));
  EXPECT_THAT(diff_result->right_diff_codes,
              UnorderedElementsAre(
                  Pair(Pointee(Property(&HloInstruction::name, "parameter.0")),
                       DiffType::kUnmatched),
                  Pair(Pointee(Property(&HloInstruction::name, "parameter.1")),
                       DiffType::kUnmatched),
                  Pair(Pointee(Property(&HloInstruction::name, "add.0")),
                       DiffType::kUnchanged)));
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

  EXPECT_THAT(
      diff_result->left_diff_codes,
      UnorderedElementsAre(
          Pair(Pointee(Property(&HloInstruction::name, "constant.2958")),
               DiffType::kUnchanged),
          Pair(Pointee(Property(&HloInstruction::name, "parameter.0")),
               DiffType::kUnchanged),
          Pair(Pointee(Property(&HloInstruction::name, "add.0")),
               DiffType::kUnchanged)));
  EXPECT_THAT(
      diff_result->right_diff_codes,
      UnorderedElementsAre(
          Pair(Pointee(Property(&HloInstruction::name, "constant.2958")),
               DiffType::kUnchanged),
          Pair(Pointee(Property(&HloInstruction::name, "parameter.0")),
               DiffType::kUnchanged),
          Pair(Pointee(Property(&HloInstruction::name, "add.0")),
               DiffType::kUnchanged)));
}

TEST_F(HloDiffTest, LargeConstantsWithDifferentValuesMarkAsChanged) {
  // Create left module with entry computation containing the following
  // structure:
  // [Param 0] ---> ┌-------┐
  //                | add_0 |
  // [Const 0] ---> └-------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  parameter.0 = s32[16]{0} parameter(0)
  constant.0 = s32[16]{0} constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
  add.0 = s32[16]{0} add(parameter.0, constant.0)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_l,
                          HloGumgraph::Create(module_l.get()));

  // Create right module with entry computation containing the following
  // structure:
  // [Param 0] ---> ┌-------┐
  //                | add_0 |
  // [Const 0] ---> └-------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  parameter.0 = s32[16]{0} parameter(0)
  constant.0 = s32[16]{0} constant({99, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
  add.0 = s32[16]{0} add(parameter.0, constant.0)
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
      OverwriteMapInstructions(GetNodeByName(*graph_l, "constant.0"),
                               GetNodeByName(*graph_r, "constant.0"), *mappings,
                               /*position_unchanged=*/true));
  auto diff_result = ConstructDiffResult(*graph_l, *graph_r, *mappings);

  EXPECT_THAT(diff_result->changed_instructions,
              UnorderedElementsAre(Pair(
                  Pointee(Property(&HloInstruction::name, "constant.0")),
                  Pointee(Property(&HloInstruction::name, "constant.0")))));
  EXPECT_THAT(diff_result->unchanged_instructions,
              UnorderedElementsAre(
                  Pair(Pointee(Property(&HloInstruction::name, "parameter.0")),
                       Pointee(Property(&HloInstruction::name, "parameter.0"))),
                  Pair(Pointee(Property(&HloInstruction::name, "add.0")),
                       Pointee(Property(&HloInstruction::name, "add.0")))));

  EXPECT_THAT(diff_result->left_diff_codes,
              UnorderedElementsAre(
                  Pair(Pointee(Property(&HloInstruction::name, "constant.0")),
                       DiffType::kChanged),
                  Pair(Pointee(Property(&HloInstruction::name, "parameter.0")),
                       DiffType::kUnchanged),
                  Pair(Pointee(Property(&HloInstruction::name, "add.0")),
                       DiffType::kUnchanged)));
  EXPECT_THAT(diff_result->right_diff_codes,
              UnorderedElementsAre(
                  Pair(Pointee(Property(&HloInstruction::name, "constant.0")),
                       DiffType::kChanged),
                  Pair(Pointee(Property(&HloInstruction::name, "parameter.0")),
                       DiffType::kUnchanged),
                  Pair(Pointee(Property(&HloInstruction::name, "add.0")),
                       DiffType::kUnchanged)));
}

TEST_F(HloDiffTest, DiffResultToAndFromProtoWorks) {
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

  EXPECT_EQ(proto.unchanged_instructions_size(), 1);
  EXPECT_EQ(proto.unchanged_instructions(0).left(), "add.0");
  EXPECT_EQ(proto.unchanged_instructions(0).right(), "add.0");
  EXPECT_EQ(proto.changed_instructions_size(), 1);
  EXPECT_EQ(proto.changed_instructions(0).left(), "parameter.0");
  EXPECT_EQ(proto.changed_instructions(0).right(), "parameter.1");
  EXPECT_EQ(proto.left_unmatched_instructions_size(), 1);
  EXPECT_EQ(proto.left_unmatched_instructions(0), "parameter.1");
  EXPECT_EQ(proto.right_unmatched_instructions_size(), 1);
  EXPECT_EQ(proto.right_unmatched_instructions(0), "parameter.0");

  DiffResult diff_result_from_proto =
      DiffResult::FromProto(proto, *module_l, *module_r);
  EXPECT_EQ(diff_result_from_proto.unchanged_instructions.size(), 1);
  EXPECT_EQ(diff_result_from_proto.changed_instructions.size(), 1);
  EXPECT_EQ(diff_result_from_proto.left_module_unmatched_instructions.size(),
            1);
  EXPECT_EQ(diff_result_from_proto.right_module_unmatched_instructions.size(),
            1);
}

TEST_F(HloDiffTest, MovedInstructionsToAndFromProtoWork) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(R"(
HloModule module

ENTRY entry {
  x = f32[] parameter(0)
  sine = f32[] sine(x)
  cosine = f32[] cosine(x)
  ROOT subtract = f32[] subtract(sine, cosine)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(R"(
HloModule module

ENTRY entry {
  x = f32[] parameter(0)
  sine = f32[] sine(x)
  cosine = f32[] cosine(x)
  ROOT subtract = f32[] subtract(cosine, sine)
}
)"));
  HloComputation* left = module_l->entry_computation();
  HloComputation* right = module_r->entry_computation();
  DiffResult diff_result;
  diff_result.AddMovedInstruction(left->GetInstructionWithName("sine"),
                                  right->GetInstructionWithName("sine"));

  EXPECT_EQ(
      diff_result.left_diff_codes.at(left->GetInstructionWithName("sine")),
      DiffType::kMoved);
  EXPECT_EQ(
      diff_result.right_diff_codes.at(right->GetInstructionWithName("sine")),
      DiffType::kMoved);

  DiffResultProto proto = diff_result.ToProto();
  ASSERT_EQ(proto.moved_instructions_size(), 1);
  EXPECT_EQ(proto.moved_instructions(0).left(), "sine");
  EXPECT_EQ(proto.moved_instructions(0).right(), "sine");

  DiffResult diff_result_from_proto =
      DiffResult::FromProto(proto, *module_l, *module_r);
  EXPECT_THAT(diff_result_from_proto.moved_instructions,
              UnorderedElementsAre(
                  Pair(Pointee(Property(&HloInstruction::name, "sine")),
                       Pointee(Property(&HloInstruction::name, "sine")))));
}

TEST_F(HloDiffTest, DebugInfoIsAttached) {
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
      *mappings, /*position_unchanged=*/true,
      /*matcher_debug_info=*/"debug_info_add"));
  ASSERT_NO_FATAL_FAILURE(OverwriteMapInstructions(
      GetNodeByName(*graph_l, "parameter.0"),
      GetNodeByName(*graph_r, "parameter.0"), *mappings,
      /*position_unchanged=*/true, /*matcher_debug_info=*/"debug_info_param0"));
  ASSERT_NO_FATAL_FAILURE(OverwriteMapInstructions(
      GetNodeByName(*graph_l, "parameter.1"),
      GetNodeByName(*graph_r, "parameter.1"), *mappings,
      /*position_unchanged=*/true, /*matcher_debug_info=*/"debug_info_param1"));
  auto diff_result = ConstructDiffResult(*graph_l, *graph_r, *mappings);

  EXPECT_THAT(
      diff_result->map_by,
      UnorderedElementsAre(
          Pair(Pair(Pointee(Property(&HloInstruction::name, "parameter.0")),
                    Pointee(Property(&HloInstruction::name, "parameter.0"))),
               MatcherType::kManual),
          Pair(Pair(Pointee(Property(&HloInstruction::name, "add.0")),
                    Pointee(Property(&HloInstruction::name, "add.0"))),
               MatcherType::kManual),
          Pair(Pair(Pointee(Property(&HloInstruction::name, "parameter.1")),
                    Pointee(Property(&HloInstruction::name, "parameter.1"))),
               MatcherType::kManual)));

  EXPECT_THAT(
      diff_result->matcher_debug_info,
      UnorderedElementsAre(
          Pair(Pair(Pointee(Property(&HloInstruction::name, "parameter.0")),
                    Pointee(Property(&HloInstruction::name, "parameter.0"))),
               "debug_info_param0"),
          Pair(Pair(Pointee(Property(&HloInstruction::name, "add.0")),
                    Pointee(Property(&HloInstruction::name, "add.0"))),
               "debug_info_add"),
          Pair(Pair(Pointee(Property(&HloInstruction::name, "parameter.1")),
                    Pointee(Property(&HloInstruction::name, "parameter.1"))),
               "debug_info_param1")));
}

// Names of the left instructions reported as changed and moved.
struct MovedTestResult {
  std::vector<std::string> changed;
  std::vector<std::string> moved;
};

// Diffs `left_hlo` against `right_hlo`. The `cross_pairs` are matched with
// `cross_matcher_type`, all other instructions are matched by name.
absl::StatusOr<MovedTestResult> DiffMatchedByName(
    absl::string_view left_hlo, absl::string_view right_hlo,
    absl::Span<const std::pair<absl::string_view, absl::string_view>>
        cross_pairs = {},
    MatcherType cross_matcher_type = MatcherType::kManual) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module_l,
                   ParseAndReturnUnverifiedModule(left_hlo));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module_r,
                   ParseAndReturnUnverifiedModule(right_hlo));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<const HloGumgraph> left_graph,
                   HloGumgraph::Create(module_l.get()));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<const HloGumgraph> right_graph,
                   HloGumgraph::Create(module_r.get()));
  HloGumgraphMappings mappings;
  for (const auto& [left, right] : cross_pairs) {
    mappings.MapInstructionsIfAbsent(GetNodeByName(*left_graph, left),
                                     GetNodeByName(*right_graph, right),
                                     cross_matcher_type);
  }
  MatchAllNodesByName(*left_graph, *right_graph, mappings);
  std::unique_ptr<const DiffResult> diff_result =
      ConstructDiffResult(*left_graph, *right_graph, mappings);
  return MovedTestResult{
      GetSortedLeftInstructionNames(diff_result->changed_instructions),
      GetSortedLeftInstructionNames(diff_result->moved_instructions)};
}

TEST_F(HloDiffTest, SwappedOperandsAreMoved) {
  TF_ASSERT_OK_AND_ASSIGN(MovedTestResult result, DiffMatchedByName(R"(
HloModule module

ENTRY entry {
  x = f32[] parameter(0)
  sine = f32[] sine(x)
  cosine = f32[] cosine(x)
  ROOT subtract = f32[] subtract(sine, cosine)
}
)",
                                                                    R"(
HloModule module

ENTRY entry {
  x = f32[] parameter(0)
  sine = f32[] sine(x)
  cosine = f32[] cosine(x)
  ROOT subtract = f32[] subtract(cosine, sine)
}
)"));
  EXPECT_THAT(result.changed, IsEmpty());
  EXPECT_THAT(result.moved, ElementsAre("cosine", "sine"));
}

TEST_F(HloDiffTest, SwappedCommutativeOperandsAreNotMoved) {
  TF_ASSERT_OK_AND_ASSIGN(MovedTestResult result, DiffMatchedByName(R"(
HloModule module

ENTRY entry {
  x = f32[] parameter(0)
  sine = f32[] sine(x)
  cosine = f32[] cosine(x)
  ROOT add = f32[] add(sine, cosine)
}
)",
                                                                    R"(
HloModule module

ENTRY entry {
  x = f32[] parameter(0)
  sine = f32[] sine(x)
  cosine = f32[] cosine(x)
  ROOT add = f32[] add(cosine, sine)
}
)"));
  EXPECT_THAT(result.changed, IsEmpty());
  EXPECT_THAT(result.moved, IsEmpty());
}

TEST_F(HloDiffTest, OperandsUsedByOtherUsersAreMoved) {
  TF_ASSERT_OK_AND_ASSIGN(MovedTestResult result, DiffMatchedByName(R"(
HloModule module

ENTRY entry {
  x = f32[] parameter(0)
  sine = f32[] sine(x)
  negate = f32[] negate(sine)
  abs = f32[] abs(x)
  ROOT tuple = (f32[], f32[]) tuple(negate, abs)
}
)",
                                                                    R"(
HloModule module

ENTRY entry {
  x = f32[] parameter(0)
  sine = f32[] sine(x)
  negate = f32[] negate(x)
  abs = f32[] abs(sine)
  ROOT tuple = (f32[], f32[]) tuple(negate, abs)
}
)"));
  EXPECT_THAT(result.changed, IsEmpty());
  EXPECT_THAT(result.moved, ElementsAre("sine", "x"));
}

TEST_F(HloDiffTest, CrossMatchedIdenticalOperandsAreNotMoved) {
  constexpr absl::string_view kHlo = R"(
HloModule module

ENTRY entry {
  x = f32[] parameter(0)
  sine.0 = f32[] sine(x)
  sine.1 = f32[] sine(x)
  ROOT subtract = f32[] subtract(sine.0, sine.1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      MovedTestResult result,
      DiffMatchedByName(kHlo, kHlo,
                        {{"sine.0", "sine.1"}, {"sine.1", "sine.0"}}));
  EXPECT_THAT(result.changed, IsEmpty());
  EXPECT_THAT(result.moved, IsEmpty());
}

TEST_F(HloDiffTest, OperandReplacedByUnmatchedInstructionIsNotMoved) {
  TF_ASSERT_OK_AND_ASSIGN(MovedTestResult result, DiffMatchedByName(R"(
HloModule module

ENTRY entry {
  x = f32[] parameter(0)
  sine = f32[] sine(x)
  cosine = f32[] cosine(x)
  subtract = f32[] subtract(sine, cosine)
  ROOT tuple = (f32[], f32[]) tuple(subtract, cosine)
}
)",
                                                                    R"(
HloModule module

ENTRY entry {
  x = f32[] parameter(0)
  sine = f32[] sine(x)
  cosine = f32[] cosine(x)
  tanh = f32[] tanh(x)
  subtract = f32[] subtract(sine, tanh)
  ROOT tuple = (f32[], f32[]) tuple(subtract, cosine)
}
)"));
  // The new operand `tanh` is already reported as unmatched.
  EXPECT_THAT(result.changed, IsEmpty());
  EXPECT_THAT(result.moved, IsEmpty());
}

constexpr absl::string_view kLookAlikeLeftHlo = R"(
HloModule module

ENTRY entry {
  p0 = f32[] parameter(0)
  cosine = f32[] cosine(p0)
  sine.0 = f32[] sine(p0)
  sine.1 = f32[] sine(cosine)
  negate.0 = f32[] negate(sine.0)
  negate.1 = f32[] negate(sine.1)
  ROOT tuple = (f32[], f32[]) tuple(negate.0, negate.1)
}
)";

constexpr absl::string_view kLookAlikeRightHlo = R"(
HloModule module

ENTRY entry {
  p0 = f32[] parameter(0)
  cosine = f32[] cosine(p0)
  exponential = f32[] exponential(p0)
  sine.0 = f32[] sine(exponential)
  sine.1 = f32[] sine(cosine)
  negate.0 = f32[] negate(sine.0)
  negate.1 = f32[] negate(sine.1)
  ROOT tuple = (f32[], f32[]) tuple(negate.0, negate.1)
}
)";

TEST_F(HloDiffTest, HeuristicallyCrossMatchedLookAlikesAreNotMoved) {
  TF_ASSERT_OK_AND_ASSIGN(
      MovedTestResult result,
      DiffMatchedByName(kLookAlikeLeftHlo, kLookAlikeRightHlo,
                        {{"sine.0", "sine.1"}, {"sine.1", "sine.0"}},
                        MatcherType::kGreedyLimitedCandidatesBottomUpMatcher));
  // The bottom-up matcher likely picked the wrong one of the two look-alike
  // sines, so they are not reported as moved.
  EXPECT_THAT(result.moved, Not(Contains("sine.0")));
  EXPECT_THAT(result.moved, Not(Contains("sine.1")));
}

TEST_F(HloDiffTest, ExactlyCrossMatchedLookAlikesAreMoved) {
  TF_ASSERT_OK_AND_ASSIGN(
      MovedTestResult result,
      DiffMatchedByName(kLookAlikeLeftHlo, kLookAlikeRightHlo,
                        {{"sine.0", "sine.1"}, {"sine.1", "sine.0"}}));
  EXPECT_THAT(result.moved, IsSupersetOf({"sine.0", "sine.1"}));
}

TEST_F(HloDiffTest, HeuristicallyMatchedSwappedLookAlikesAreMoved) {
  TF_ASSERT_OK_AND_ASSIGN(
      MovedTestResult result,
      DiffMatchedByName(
          R"(
HloModule module

ENTRY entry {
  p0 = f32[] parameter(0)
  cosine = f32[] cosine(p0)
  sine.0 = f32[] sine(p0)
  sine.1 = f32[] sine(cosine)
  ROOT subtract = f32[] subtract(sine.0, sine.1)
}
)",
          R"(
HloModule module

ENTRY entry {
  p0 = f32[] parameter(0)
  cosine = f32[] cosine(p0)
  sine.0 = f32[] sine(p0)
  sine.1 = f32[] sine(cosine)
  ROOT subtract = f32[] subtract(sine.1, sine.0)
}
)",
          {{"sine.0", "sine.0"}, {"sine.1", "sine.1"}},
          MatcherType::kGreedyLimitedCandidatesBottomUpMatcher));
  // The operands are permuted within the subtract, so the move is trusted even
  // though the sines look alike and were matched heuristically.
  EXPECT_THAT(result.moved, ElementsAre("sine.0", "sine.1"));
}

TEST_F(HloDiffTest, ChangedTakesPrecedenceOverMoved) {
  TF_ASSERT_OK_AND_ASSIGN(MovedTestResult result, DiffMatchedByName(R"(
HloModule module

ENTRY entry {
  x = f32[] parameter(0)
  sine = f32[] sine(x)
  cosine = f32[] cosine(x)
  ROOT subtract = f32[] subtract(sine, cosine)
}
)",
                                                                    R"(
HloModule module

ENTRY entry {
  x = f32[] parameter(0)
  sine = f32[] exponential(x)
  cosine = f32[] cosine(x)
  ROOT subtract = f32[] subtract(cosine, sine)
}
)"));
  // `sine` is both changed and moved, and is only reported as changed.
  EXPECT_THAT(result.changed, ElementsAre("sine"));
  EXPECT_THAT(result.moved, ElementsAre("cosine"));
}

TEST_F(HloDiffTest, UnmatchedUsersAreIgnored) {
  TF_ASSERT_OK_AND_ASSIGN(MovedTestResult result, DiffMatchedByName(R"(
HloModule module

ENTRY entry {
  x = f32[] parameter(0)
  sine = f32[] sine(x)
  negate = f32[] negate(sine)
  ROOT tuple = (f32[]) tuple(negate)
}
)",
                                                                    R"(
HloModule module

ENTRY entry {
  x = f32[] parameter(0)
  sine = f32[] sine(x)
  negate = f32[] negate(sine)
  abs = f32[] abs(sine)
  ROOT tuple = (f32[], f32[]) tuple(negate, abs)
}
)"));
  // The new user `abs` is already reported as unmatched.
  EXPECT_THAT(result.changed, ElementsAre("tuple"));
  EXPECT_THAT(result.moved, IsEmpty());
}

TEST_F(HloDiffTest, OperandsMovedToOtherCommutativeUsersAreMoved) {
  TF_ASSERT_OK_AND_ASSIGN(MovedTestResult result, DiffMatchedByName(R"(
HloModule module

ENTRY entry {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  sine = f32[] sine(x)
  add.0 = f32[] add(sine, y)
  add.1 = f32[] add(x, y)
  ROOT tuple = (f32[], f32[]) tuple(add.0, add.1)
}
)",
                                                                    R"(
HloModule module

ENTRY entry {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  sine = f32[] sine(x)
  add.0 = f32[] add(x, y)
  add.1 = f32[] add(sine, y)
  ROOT tuple = (f32[], f32[]) tuple(add.0, add.1)
}
)"));
  // `y` is used by both adds on both sides, so it isn't moved.
  EXPECT_THAT(result.changed, IsEmpty());
  EXPECT_THAT(result.moved, ElementsAre("sine", "x"));
}

}  // namespace
}  // namespace hlo_diff
}  // namespace xla
