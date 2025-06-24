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

#include "xla/hlo/tools/hlo_diff/graph/utils/hlo_gumgraph_dfs.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace hlo_diff {
namespace {

using ::testing::ElementsAre;

class HloGumgraphDfsTest : public HloHardwareIndependentTestBase {};

TEST_F(HloGumgraphDfsTest, DfsPreOrderWorks) {
  // Create a module with entry computation containing the following structure:
  // [Param foo] ------> ┌-------┐
  //                     | add_1 | ---> ┌-------┐      ┌------┐
  // [Constant bar] ---> └-------┘      | add_0 | ---> | ROOT |
  // [Param baz] ---------------------> └-------┘      └------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  foo = f32[8,2048]{1,0:T(8,128)} parameter(0)
  bar = f32[8,2048]{1,0:T(8,128)} constant(0)
  baz = f32[8,2048]{1,0:T(8,128)} parameter(1)
  add_1 = f32[8,2048]{1,0:T(8,128)} add(foo, bar)
  add_0 = f32[8,2048]{1,0:T(8,128)} add(add_1, baz)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph,
                          HloGumgraph::Create(module.get()));
  const auto root = graph->GetRoot();
  std::vector<absl::string_view> visited_nodes;
  HloGumgraphDfs(
      root,
      [&](const HloInstructionNode& node) {
        visited_nodes.push_back(node.GetName());
      },
      DfsTraversalOrder::kPreOrder, graph->GetNodeCount());

  EXPECT_THAT(visited_nodes,
              ElementsAre("root", "add_0", "add_1", "foo", "bar", "baz"));
}

TEST_F(HloGumgraphDfsTest, DfsPostOrderWorks) {
  // Create a module with entry computation containing the following structure:
  // [Param foo] ------> ┌-------┐
  //                     | add_1 | ---> ┌-------┐      ┌------┐
  // [Constant bar] ---> └-------┘      | add_0 | ---> | ROOT |
  // [Param baz] ---------------------> └-------┘      └------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  foo = f32[8,2048]{1,0:T(8,128)} parameter(0)
  bar = f32[8,2048]{1,0:T(8,128)} constant(0)
  baz = f32[8,2048]{1,0:T(8,128)} parameter(1)
  add_1 = f32[8,2048]{1,0:T(8,128)} add(foo, bar)
  add_0 = f32[8,2048]{1,0:T(8,128)} add(add_1, baz)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph,
                          HloGumgraph::Create(module.get()));
  const auto root = graph->GetRoot();
  std::vector<absl::string_view> visited_nodes;
  HloGumgraphDfs(
      root,
      [&](const HloInstructionNode& node) {
        visited_nodes.push_back(node.GetName());
      },
      DfsTraversalOrder::kPostOrder, graph->GetNodeCount());

  EXPECT_THAT(visited_nodes,
              ElementsAre("foo", "bar", "add_1", "baz", "add_0", "root"));
}

TEST_F(HloGumgraphDfsTest, DfsPostOrderWorksForMultiplePathsFromRoot) {
  // Create a module with entry computation containing the following structure:
  // [Param foo] ------> ┌-------┐      ┌------┐
  //     |               | add_1 | ---> | ROOT |
  // [copy_foo] -------> └-------┘      └------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  foo = f32[8,2048]{1,0:T(8,128)} parameter(0)
  copy_foo = f32[8,2048]{1,0:T(8,128)} copy(foo)
  add_1 = f32[8,2048]{1,0:T(8,128)} add(foo, copy_foo)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph,
                          HloGumgraph::Create(module.get()));
  const auto root = graph->GetRoot();
  std::vector<absl::string_view> visited_nodes;
  HloGumgraphDfs(
      root,
      [&](const HloInstructionNode& node) {
        visited_nodes.push_back(node.GetName());
      },
      DfsTraversalOrder::kPostOrder, graph->GetNodeCount());

  EXPECT_THAT(visited_nodes, ElementsAre("foo", "copy_foo", "add_1", "root"));
}

TEST_F(HloGumgraphDfsTest, GetAllNodesWorks) {
  // Create a module with entry computation containing the following structure:
  // [Param foo] ------> ┌-------┐
  //                     | add_1 | ---> ┌-------┐      ┌------┐
  // [Constant bar] ---> └-------┘      | add_0 | ---> | ROOT |
  // [Param baz] ---------------------> └-------┘      └------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  foo = f32[8,2048]{1,0:T(8,128)} parameter(0)
  bar = f32[8,2048]{1,0:T(8,128)} constant(0)
  baz = f32[8,2048]{1,0:T(8,128)} parameter(1)
  add_1 = f32[8,2048]{1,0:T(8,128)} add(foo, bar)
  add_0 = f32[8,2048]{1,0:T(8,128)} add(add_1, baz)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph,
                          HloGumgraph::Create(module.get()));
  const auto root = graph->GetRoot();
  std::vector<const HloInstructionNode*> visited_nodes = GetAllNodesInDfsOrder(
      root, DfsTraversalOrder::kPreOrder, graph->GetNodeCount());
  std::vector<absl::string_view> string_views;
  string_views.reserve(visited_nodes.size());
  for (const HloInstructionNode* node : visited_nodes) {
    string_views.push_back(node->GetName());
  }

  EXPECT_THAT(string_views,
              ElementsAre("root", "add_0", "add_1", "foo", "bar", "baz"));
}

TEST_F(HloGumgraphDfsTest, DfsPreOrderStopExpandingWorks) {
  // Create a module with entry computation containing the following structure:
  // [Param foo] ------> ┌-------┐
  //                     | add_1 | ---> ┌-------┐      ┌------┐
  // [Constant bar] ---> └-------┘      | add_0 | ---> | ROOT |
  // [Param baz] ---------------------> └-------┘      └------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  foo = f32[8,2048]{1,0:T(8,128)} parameter(0)
  bar = f32[8,2048]{1,0:T(8,128)} constant(0)
  baz = f32[8,2048]{1,0:T(8,128)} parameter(1)
  add_1 = f32[8,2048]{1,0:T(8,128)} add(foo, bar)
  add_0 = f32[8,2048]{1,0:T(8,128)} add(add_1, baz)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph,
                          HloGumgraph::Create(module.get()));

  std::vector<absl::string_view> visited_nodes;
  HloGumgraphDfs(
      graph->GetRoot(),
      [&](const HloInstructionNode& node) {
        visited_nodes.push_back(node.GetName());
      },
      DfsTraversalOrder::kPreOrder, 6,
      [](const HloInstructionNode& node) { return node.GetName() != "add_1"; });

  EXPECT_THAT(visited_nodes, ElementsAre("root", "add_0", "add_1", "baz"));
}

TEST_F(HloGumgraphDfsTest, DfsPostOrderStopExpandingWorks) {
  // Create a module with entry computation containing the following structure:
  // [Param foo] ------> ┌-------┐
  //                     | add_1 | ---> ┌-------┐      ┌------┐
  // [Constant bar] ---> └-------┘      | add_0 | ---> | ROOT |
  // [Param baz] ---------------------> └-------┘      └------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  foo = f32[8,2048]{1,0:T(8,128)} parameter(0)
  bar = f32[8,2048]{1,0:T(8,128)} constant(0)
  baz = f32[8,2048]{1,0:T(8,128)} parameter(1)
  add_1 = f32[8,2048]{1,0:T(8,128)} add(foo, bar)
  add_0 = f32[8,2048]{1,0:T(8,128)} add(add_1, baz)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph,
                          HloGumgraph::Create(module.get()));

  std::vector<absl::string_view> visited_nodes;
  HloGumgraphDfs(
      graph->GetRoot(),
      [&](const HloInstructionNode& node) {
        visited_nodes.push_back(node.GetName());
      },
      DfsTraversalOrder::kPostOrder, 6,
      [](const HloInstructionNode& node) { return node.GetName() != "add_1"; });

  EXPECT_THAT(visited_nodes, ElementsAre("add_1", "baz", "add_0", "root"));
}

}  // namespace
}  // namespace hlo_diff
}  // namespace xla
