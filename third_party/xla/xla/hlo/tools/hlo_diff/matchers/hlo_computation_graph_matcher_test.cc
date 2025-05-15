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

#include "xla/hlo/tools/hlo_diff/matchers/hlo_computation_graph_matcher.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/utils/test_util.h"
#include "xla/service/call_graph.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::hlo_diff {
namespace {

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

class HloComputationGraphMatcherTest : public HloHardwareIndependentTestBase {};

TEST_F(HloComputationGraphMatcherTest, MatchSingleParameterOrConstant) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> left_module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p1 = bf16[2]{0} parameter(0), metadata={op_name="first-phase"}
  c1 = bf16[2]{0} constant({1.1, 2.2})

  ROOT add1 = bf16[2]{0} add(p1, c1)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> right_module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p2 = bf16[3]{0} parameter(0), metadata={op_name="first-phase.modify"}
  c2 = bf16[3]{0} constant({1.1, 2.2, 3.3})

  ROOT add2 = bf16[3]{0} add(p2, c2)
}
)"));
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
  MatchComputationGraphs(*left_gumgraph, *right_gumgraph,
                         left_entry_computation, right_entry_computation,
                         *mappings);

  auto matched_params = ExtractMappedInstructionNames(*mappings);
  EXPECT_THAT(matched_params,
              UnorderedElementsAre(Pair("p1", "p2"), Pair("c1", "c2"),
                                   Pair("add1", "add2")));
}

TEST_F(HloComputationGraphMatcherTest, MatchComputationParams) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> left_module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p21 = f32[10]{0} parameter(0), metadata={op_name="first-phase"}
  p22 = f32[10]{0:T(128)} parameter(1), metadata={op_name="first-phase.multiple-matches", source_file="test.cc", source_line=43}
  p23 = f32[20]{0} parameter(2)
  p24 = f32[10]{0} parameter(3), metadata={source_file="test.cc", source_line=42}
  p25 = f32[10]{0} parameter(4), sharding={maximal device=1}
  p26 = f32[30]{0} parameter(5), sharding={maximal device=1}

  add21 = f32[10]{0} add(p21, p22)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> right_module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p11 = f32[10]{0} parameter(0), metadata={op_name="first-phase"}
  p12 = f32[10]{0} parameter(1), metadata={op_name="first-phase.multiple-matches"}
  p13 = f32[10]{0} parameter(2), metadata={op_name="first-phase.multiple-matches"}
  p14 = f32[20]{0} parameter(3)
  p15 = f32[10]{0} parameter(4), metadata={source_file="test.cc", source_line=42}
  p16 = f32[10]{0} parameter(5), sharding={maximal device=1}
  p17 = f32[30]{0} parameter(6)
  p18 = f32[10]{0:T(128)} parameter(7), metadata={source_file="test.cc", source_line=43}

  ROOT add22 = f32[10]{0} add(p11, p18)
}
)"));
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
  MatchComputationGraphs(
      *left_gumgraph, *right_gumgraph,
      left_gumgraph->GetCallGraph().GetNode(left_module->entry_computation()),
      right_gumgraph->GetCallGraph().GetNode(right_module->entry_computation()),
      *mappings);

  auto matched_params = ExtractMappedInstructionNames(*mappings);
  EXPECT_THAT(matched_params,
              UnorderedElementsAre(Pair("p21", "p11"), Pair("p22", "p18"),
                                   Pair("p23", "p14"), Pair("p24", "p15"),
                                   Pair("p25", "p16"), Pair("p26", "p17"),
                                   Pair("add21", "add22")));
}

TEST_F(HloComputationGraphMatcherTest, MatchComputationConstants) {
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
  MatchComputationGraphs(*left_gumgraph, *right_gumgraph,
                         left_entry_computation, right_entry_computation,
                         *mappings);

  auto matched_params = ExtractMappedInstructionNames(*mappings);
  EXPECT_THAT(matched_params,
              UnorderedElementsAre(Pair("c22", "c22"), Pair("c23", "c23"),
                                   Pair("c24", "c24"), Pair("c25", "c25"),
                                   Pair("c26", "c26"), Pair("add23", "add23")));
}

TEST_F(HloComputationGraphMatcherTest,
       ExactMatchComputationsInstructionsExactlyMatched) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> left_module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

fused_computation.1 {
  p2 = s32[32,16]{0,1:T(1,128)}  parameter(0)
  p3 = s32[32,16]{0,1:T(1,128)} parameter(1)
  add.1 = s32[32,16]{0,1:T(1,128)} add(p2, p3)
}

ENTRY entry {
  p0 = s32[32,16]{0, 1:T(1,128)} parameter(0)
  p1 = s32[32,16]{0,1:T(1,128)} parameter(1)
  ROOT fusion.1 = s32[32,16]{0,1:T(1,128)} fusion(p0,p1), kind=kLoop, calls=fused_computation.1
}
)"));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> right_module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

fused_computation.11 {
  p21 = s32[32,16]{0,1:T(1,128)}  parameter(0)
  p31 = s32[32,16]{0,1:T(1,128)} parameter(1)
  add.11 = s32[32,16]{0,1:T(1,128)} add(p21, p31)
}

ENTRY entry {
  p01 = s32[32,16]{0, 1:T(1,128)} parameter(0)
  p11 = s32[32,16]{0,1:T(1,128)} parameter(1)
  ROOT fusion.11 = s32[32,16]{0,1:T(1,128)} fusion(p01,p11), kind=kLoop, calls=fused_computation.11
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> left_gumgraph,
                          HloGumgraph::Create(left_module.get()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> right_gumgraph,
                          HloGumgraph::Create(right_module.get()));
  auto mappings = std::make_unique<HloGumgraphMappings>();
  const CallGraphNode& left_fused_computation =
      left_gumgraph->GetCallGraph().GetNode(
          left_module->GetComputationWithName("fused_computation.1"));
  const CallGraphNode& right_fused_computation =
      right_gumgraph->GetCallGraph().GetNode(
          right_module->GetComputationWithName("fused_computation.11"));

  mappings->MapComputationsIfAbsent(left_fused_computation,
                                    right_fused_computation,
                                    ComputationMatchType::kExact);
  MatchComputationGraphs(*left_gumgraph, *right_gumgraph,
                         left_fused_computation, right_fused_computation,
                         *mappings);

  auto matched_params = ExtractMappedInstructionNames(*mappings);
  EXPECT_THAT(matched_params,
              UnorderedElementsAre(Pair("p2", "p21"), Pair("p3", "p31"),
                                   Pair("add.1", "add.11"),
                                   Pair("fusion.1", "fusion.11")));
}

}  // namespace
}  // namespace xla::hlo_diff
