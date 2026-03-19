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

#include "xla/hlo/tools/hlo_diff/matchers/hlo_call_graph_matcher.h"

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

class HloCallGraphMatcherTest : public HloHardwareIndependentTestBase {};

TEST_F(HloCallGraphMatcherTest, ExactFingerprintMatches) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> left_module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

fused_computation.1 {
  p2 = s32[32,16]{0,1:T(1,128)}  parameter(0)
  p3 = s32[32,16]{0,1:T(1,128)} parameter(1)
  add.1 = s32[32,16]{0,1:T(1,128)} add(p2, p3)
}

fused_computation.2 {
  p4 = s32[32,16]{0,1:T(1,128)}  parameter(0)
  p5 = s32[32,16]{0,1:T(1,128)} parameter(1)
  add.2 = s32[32,16]{0,1:T(1,128)} add(p4, p5)
}

ENTRY entry {
  p0 = s32[32,16]{0, 1:T(1,128)} parameter(0)
  p1 = s32[32,16]{0,1:T(1,128)} parameter(1)
  fusion.1 = s32[32,16]{0,1:T(1,128)} fusion(p0,p1), kind=kLoop, calls=fused_computation.1
  fusion.2 = s32[32,16]{0,1:T(1,128)} fusion(p0,p1), kind=kLoop, calls=fused_computation.2
  ROOT add = s32[32,16]{0,1:T(1,128)} add(fusion.1, fusion.2)
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

fused_computation.21 {
  p41 = s32[32,16]{0,1:T(1,128)}  parameter(0)
  p51 = s32[32,16]{0,1:T(1,128)} parameter(1)
  add.21 = s32[32,16]{0,1:T(1,128)} add(p41, p51)
}

ENTRY entry {
  p01 = s32[32,16]{0, 1:T(1,128)} parameter(0)
  p11 = s32[32,16]{0,1:T(1,128)} parameter(1)
  fusion.11 = s32[32,16]{0,1:T(1,128)} fusion(p01,p11), kind=kLoop, calls=fused_computation.11
  fusion.21 = s32[32,16]{0,1:T(1,128)} fusion(p01,p11), kind=kLoop, calls=fused_computation.21
  ROOT add11 = s32[32,16]{0,1:T(1,128)} add(fusion.11, fusion.21)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> left_gumgraph,
                          HloGumgraph::Create(left_module.get()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> right_gumgraph,
                          HloGumgraph::Create(right_module.get()));
  auto mappings = std::make_unique<HloGumgraphMappings>();

  MatchCallGraphs(*left_gumgraph, *right_gumgraph, *mappings);

  auto matched_computations = ExtractMappedComputationNames(*mappings);
  auto match_type = ExtractComputationMatchType(*mappings);
  EXPECT_THAT(
      matched_computations,
      UnorderedElementsAre(Pair("fused_computation.1", "fused_computation.11"),
                           Pair("fused_computation.2", "fused_computation.21"),
                           Pair("entry", "entry")));
  EXPECT_THAT(match_type,
              UnorderedElementsAre(
                  Pair("fused_computation.1", ComputationMatchType::kExact),
                  Pair("fused_computation.2", ComputationMatchType::kExact),
                  Pair("entry", ComputationMatchType::kExact)));
}

TEST_F(HloCallGraphMatcherTest, UnequalFingerprintMatchesNotMatched) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> left_module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

fused_computation.1 {
  p2 = s32[32,16]{0,1:T(1,128)}  parameter(0)
  p3 = s32[32,16]{0,1:T(1,128)} parameter(1)
  add.1 = s32[32,16]{0,1:T(1,128)} add(p2, p3)
}

fused_computation.2 {
  p4 = s32[32,16]{0,1:T(1,128)}  parameter(0)
  p5 = s32[32,16]{0,1:T(1,128)} parameter(1)
  add.2 = s32[32,16]{0,1:T(1,128)} add(p4, p5)
}

ENTRY entry {
  p0 = s32[32,16]{0, 1:T(1,128)} parameter(0)
  p1 = s32[32,16]{0,1:T(1,128)} parameter(1)
  fusion.1 = s32[32,16]{0,1:T(1,128)} fusion(p0,p1), kind=kLoop, calls=fused_computation.1
  fusion.2 = s32[32,16]{0,1:T(1,128)} fusion(p0,p1), kind=kLoop, calls=fused_computation.2
  ROOT add = s32[32,16]{0,1:T(1,128)} add(fusion.1, fusion.2)
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

  MatchCallGraphs(*left_gumgraph, *right_gumgraph, *mappings);

  auto matched_computations = ExtractMappedComputationNames(*mappings);
  auto match_type = ExtractComputationMatchType(*mappings);
  EXPECT_THAT(matched_computations,
              UnorderedElementsAre(Pair("entry", "entry")));
  EXPECT_THAT(match_type, UnorderedElementsAre(
                              Pair("entry", ComputationMatchType::kSignature)));
}

TEST_F(HloCallGraphMatcherTest, MultipleWhileInstructionsMatched) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> left_module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

body_1 {
  prev_1 = s32[2] parameter(0)
  all-gather_1 = s32[4] all-gather(s32[2] prev_1), replica_groups={}, dimensions={0}, backend_config="{}"
  ROOT slice_1 = s32[2] slice(all-gather_1), slice={[0:2]}
}

condition_1 {
  prev_1 = s32[2] parameter(0)
  constant_1 = pred[] constant(true)
  ROOT copy_1 = pred[] copy(constant_1)
}

body_2 {
  prev_2 = s32[2] parameter(0)
  all-gather_2 = s32[4] all-gather(s32[2] prev_2), replica_groups={}, dimensions={0}, backend_config="{}"
  ROOT slice_2 = s32[2] slice(all-gather_2), slice={[0:2]}
}

condition_2 {
  prev_2 = s32[2] parameter(0)
  constant_2 = pred[] constant(true)
  ROOT copy_2 = pred[] copy(constant_2)
}

body_3 {
  prev_3 = s32[2] parameter(0)
  all-gather_3 = s32[4] all-gather(s32[2] prev_3), replica_groups={}, dimensions={0}, backend_config="{}"
  ROOT slice_3 = s32[2] slice(all-gather_3), slice={[0:2]}
}

condition_3 {
  prev_3 = s32[2] parameter(0)
  constant_3 = pred[] constant(true)
  ROOT copy_3 = pred[] copy(constant_3)
}

ENTRY entry {
  constant = s32[2] constant({0,0})
  while.1 = s32[2] while(s32[2] constant), condition=condition_1, body=body_1
  while.2 = s32[2] while(s32[2] constant), condition=condition_2, body=body_2, metadata={op_name="while-activations"}
  while.3 = s32[2] while(s32[2] constant), condition=condition_3, body=body_3
  add.1 = s32[2] add(while.1, while.2)
  ROOT add.2 = s32[2] add(add.1, while.3)
}
)"));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> right_module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true
body_1 {
  prev_1 = s32[2] parameter(0)
  all-gather_1 = s32[4] all-gather(s32[2] prev_1), replica_groups={}, dimensions={0}, backend_config="{}"
  ROOT slice_1 = s32[2] slice(all-gather_1), slice={[0:2]}
}

condition_1 {
  prev_1 = s32[2] parameter(0)
  constant_1 = pred[] constant(true)
  ROOT copy_1 = pred[] copy(constant_1)
}

body_2 {
  prev_2 = s32[2] parameter(0)
  all-gather_2 = s32[4] all-gather(s32[2] prev_2), replica_groups={}, dimensions={0}, backend_config="{}"
  ROOT slice_2 = s32[2] slice(all-gather_2), slice={[0:2]}
}

condition_2 {
  prev_2 = s32[2] parameter(0)
  constant_2 = pred[] constant(true)
  ROOT copy_2 = pred[] copy(constant_2)
}

body_3 {
  prev_3 = s32[2] parameter(0)
  all-gather_3 = s32[4] all-gather(s32[2] prev_3), replica_groups={}, dimensions={0}, backend_config="{}"
  ROOT slice_3 = s32[2] slice(all-gather_3), slice={[0:2]}
}

condition_3 {
  prev_3 = s32[2] parameter(0)
  constant_3 = pred[] constant(true)
  ROOT copy_3 = pred[] copy(constant_3)
}

ENTRY entry {
  constant = s32[2] constant({0,0})
  while.1 = s32[2] while(s32[2] constant), condition=condition_1, body=body_1, metadata={op_name="while-activations"} 
  while.2 = s32[2] while(s32[2] constant), condition=condition_2, body=body_2
  while.3 = s32[2] while(s32[2] constant), condition=condition_3, body=body_3
  add.1 = s32[2] add(while.1, while.2)
  ROOT add.2 = s32[2] add(add.1, while.3)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> left_gumgraph,
                          HloGumgraph::Create(left_module.get()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> right_gumgraph,
                          HloGumgraph::Create(right_module.get()));
  auto mappings = std::make_unique<HloGumgraphMappings>();

  MatchCallGraphs(*left_gumgraph, *right_gumgraph, *mappings);

  auto matched_computations = ExtractMappedComputationNames(*mappings);
  auto match_type = ExtractComputationMatchType(*mappings);
  EXPECT_THAT(matched_computations,
              UnorderedElementsAre(
                  Pair("body_1", "body_2"), Pair("body_2", "body_1"),
                  Pair("body_3", "body_3"), Pair("condition_1", "condition_2"),
                  Pair("condition_2", "condition_1"),
                  Pair("condition_3", "condition_3"), Pair("entry", "entry")));
  EXPECT_THAT(match_type,
              UnorderedElementsAre(
                  Pair("body_1", ComputationMatchType::kSignature),
                  Pair("body_2", ComputationMatchType::kSignature),
                  Pair("body_3", ComputationMatchType::kSignature),
                  Pair("condition_1", ComputationMatchType::kSignature),
                  Pair("condition_2", ComputationMatchType::kSignature),
                  Pair("condition_3", ComputationMatchType::kSignature),
                  Pair("entry", ComputationMatchType::kExact)));
}

TEST_F(HloCallGraphMatcherTest, ExactSignatureMatches) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> left_module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

fused_computation.1 {
  p.2 = s32[32,16]{0,1:T(1,128)}  parameter(0)
  p.3 = s32[32,16]{0,1:T(1,128)} parameter(1)
  add.1 = s32[32,16]{0,1:T(1,128)} add(p.2, p.3)
}

fused_computation.2 {
  p.4 = s32[]  parameter(0)
  p.5 = s32[] parameter(1)
  add.2 = s32[] add(p.4, p.5)
}

fused_computation.3 {
  p.6 = s32[32,16]{0,1:T(1,128)}  parameter(0)
  p.7 = s32[32,16]{0,1:T(1,128)} parameter(1)
  add.3 = s32[32,16]{0,1:T(1,128)} add(p.6, p.7)
}

fused_computation.4 {
  p.8 = s32[]  parameter(0)
  p.9 = s32[] parameter(1)
  add.4 = s32[] add(p.8, p.9)
}

ENTRY entry {
  p.0 = s32[32,16]{0, 1:T(1,128)} parameter(0)
  p.1 = s32[32,16]{0,1:T(1,128)} parameter(1)
  p.2 = s32[] parameter(2)
  p.3 = s32[] parameter(3)
  fusion.1 = s32[32,16]{0,1:T(1,128)} fusion(p.0, p.1), kind=kLoop, calls=fused_computation.1
  fusion.2 = s32[] fusion(p.2, p.3), kind=kLoop, calls=fused_computation.2
  fusion.3 = s32[32,16]{0,1:T(1,128)} fusion(p.0, p.1), kind=kLoop, calls=fused_computation.3, metadata={op_name="add_fusion"}
  fusion.4 = s32[] fusion(p.2, p.3), kind=kLoop, calls=fused_computation.4
}
)"));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> right_module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

fused_computation.11 {
  p.21 = s32[32,16]{0,1:T(1,128)}  parameter(0)
  p.31 = s32[32,16]{0,1:T(1,128)} parameter(1)
  subtract.11 = s32[32,16]{0,1:T(1,128)} subtract(p.21, p.31)
}

fused_computation.21 {
  p.41 = s32[]  parameter(0)
  p.51 = s32[] parameter(1)
  subtract.21 = s32[] subtract(p.41, p.51)
}

fused_computation.31 {
  p.21 = s32[32,16]{0,1:T(1,128)}  parameter(0)
  p.31 = s32[32,16]{0,1:T(1,128)} parameter(1)
  subtract.11 = s32[32,16]{0,1:T(1,128)} subtract(p.21, p.31)
}

fused_computation.41 {
  p.41 = s32[]  parameter(0)
  p.51 = s32[] parameter(1)
  subtract.21 = s32[] subtract(p.41, p.51)
}

ENTRY entry {
  p.01 = s32[32,16]{0, 1:T(1,128)} parameter(0)
  p.11 = s32[32,16]{0,1:T(1,128)} parameter(1)
  p.21 = s32[] parameter(2)
  p.31 = s32[] parameter(3)
  fusion.11 = s32[32,16]{0,1:T(1,128)} fusion(p.01,p.11), kind=kLoop, calls=fused_computation.11
  fusion.21 = s32[] fusion(p.21, p.31), kind=kLoop, calls=fused_computation.21
  fusion.31 = s32[32,16]{0,1:T(1,128)} fusion(p.01,p.11), kind=kLoop, calls=fused_computation.31, metadata={op_name="add_fusion"}
  fusion.41 = s32[] fusion(p.21, p.31), kind=kLoop, calls=fused_computation.41
  
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> left_gumgraph,
                          HloGumgraph::Create(left_module.get()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> right_gumgraph,
                          HloGumgraph::Create(right_module.get()));
  auto mappings = std::make_unique<HloGumgraphMappings>();

  MatchCallGraphs(*left_gumgraph, *right_gumgraph, *mappings);

  auto matched_computations = ExtractMappedComputationNames(*mappings);
  auto match_type = ExtractComputationMatchType(*mappings);
  EXPECT_THAT(
      matched_computations,
      UnorderedElementsAre(Pair("fused_computation.1", "fused_computation.11"),
                           Pair("fused_computation.2", "fused_computation.21"),
                           Pair("fused_computation.3", "fused_computation.31"),
                           Pair("fused_computation.4", "fused_computation.41"),
                           Pair("entry", "entry")));
  EXPECT_THAT(match_type,
              UnorderedElementsAre(
                  Pair("fused_computation.1", ComputationMatchType::kSignature),
                  Pair("fused_computation.2", ComputationMatchType::kSignature),
                  Pair("fused_computation.3", ComputationMatchType::kSignature),
                  Pair("fused_computation.4", ComputationMatchType::kSignature),
                  Pair("entry", ComputationMatchType::kExact)));
}

}  // namespace
}  // namespace xla::hlo_diff
