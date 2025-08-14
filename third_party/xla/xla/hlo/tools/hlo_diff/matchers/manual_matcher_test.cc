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

#include "xla/hlo/tools/hlo_diff/matchers/manual_matcher.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/utils/test_util.h"
#include "xla/tsl/platform/statusor.h"
namespace xla {
namespace hlo_diff {
namespace {

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

class ManualMatcherTest : public HloHardwareIndependentTestBase {};

TEST_F(ManualMatcherTest, ManualMatcherWorks) {
  // Create module with entry computation containing the following
  // structure:
  // [Param foo] ------> ┌-------┐
  //                     | add_1 | ---> ┌-------┐      ┌------┐
  // [Constant bar] ---> └-------┘      | add_0 | ---> | ROOT |
  // [Param baz] ---------------------> └-------┘      └------┘
  std::string hlo_string = R"(
  HloModule module, is_scheduled=true
  
  ENTRY entry {
    foo = f32[8,2048]{1,0:T(8,128)} parameter(0)
    bar = f32[8,2048]{1,0:T(8,128)} constant(0)
    baz = f32[8,2048]{1,0:T(8,128)} parameter(1)
    add_1 = f32[8,2048]{1,0:T(8,128)} add(foo, bar)
    add_0 = f32[8,2048]{1,0:T(8,128)} add(add_1, baz)
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
  std::vector<std::pair<absl::string_view, absl::string_view>> manual_mappings =
      {std::make_pair("add_1", "add_1"), std::make_pair("foo", "foo"),
       std::make_pair("baz", "baz")};
  auto matcher = std::make_unique<ManualMatcher>(graph_l.get(), graph_r.get(),
                                                 manual_mappings);
  matcher->Match(*mappings);
  auto mapped_nodes = ExtractMappedInstructionNames(*mappings);

  EXPECT_THAT(mapped_nodes,
              UnorderedElementsAre(Pair("add_1", "add_1"), Pair("foo", "foo"),
                                   Pair("baz", "baz")));
}
}  // namespace
}  // namespace hlo_diff
}  // namespace xla
