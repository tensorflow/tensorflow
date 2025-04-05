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

#include "xla/hlo/tools/hlo_diff/graph/utils/hlo_gumgraph_bfs.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace hlo_diff {
namespace {

using ::testing::ElementsAre;

class HloGumgraphBfsTest : public HloTestBase {};

TEST_F(HloGumgraphBfsTest, BfsForwardWorks) {
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
  const auto root = graph->GetRoot();
  HloGumgraphBfs(
      root,
      [&visited_nodes](const HloInstructionNode& node) {
        visited_nodes.push_back(node.GetName());
        return true;
      },
      BfsTraversalDirection::kForward, graph->GetNodeCount());

  EXPECT_THAT(visited_nodes,
              ElementsAre("root", "add_0", "add_1", "baz", "foo", "bar"));
}

TEST_F(HloGumgraphBfsTest, BfsReverseWorks) {
  // Create a module with entry computation containing the following structure:
  // [Param foo] ------> ┌-------┐      ┌-------┐      ┌------┐
  //                     | add_0 | ---> | abs_0 | ---> | ROOT |
  // [Constant bar] ---> └-------┘      └-------┘      └------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
  HloModule module, is_scheduled=true
  
  ENTRY entry {
    foo = f32[8,2048]{1,0:T(8,128)} parameter(0)
    bar = f32[8,2048]{1,0:T(8,128)} constant(0)
    add_0 = f32[8,2048]{1,0:T(8,128)} add(foo, bar)
    abs_0 = f32[8,2048]{1,0:T(8,128)} abs(f32[8,2048]{1,0:T(8,128)} %add_0)
  }
  )"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph,
                          HloGumgraph::Create(module.get()));

  std::vector<absl::string_view> visited_nodes;
  const auto root = graph->GetRoot();
  HloGumgraphBfs(
      *root.children[0]->children[0]->children[0],
      [&visited_nodes](const HloInstructionNode& node) {
        visited_nodes.push_back(node.GetName());
        return true;
      },
      BfsTraversalDirection::kReverse, graph->GetNodeCount());

  EXPECT_THAT(visited_nodes, ElementsAre("foo", "add_0", "abs_0", "root"));
}

TEST_F(HloGumgraphBfsTest, GetAllNodesWorks) {
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
  std::vector<const HloInstructionNode*> visited_nodes =
      GetAllNodesInBfsOrder(root, BfsTraversalDirection::kForward);
  std::vector<absl::string_view> string_views;
  string_views.reserve(visited_nodes.size());
  for (const HloInstructionNode* node : visited_nodes) {
    string_views.push_back(node->GetName());
  }

  EXPECT_THAT(string_views,
              ElementsAre("root", "add_0", "add_1", "baz", "foo", "bar"));
}

TEST_F(HloGumgraphBfsTest, BfsFromMultipleNodesWorks) {
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
  const auto root = graph->GetRoot();
  HloGumgraphBfs(
      std::vector<const HloInstructionNode*>{root.children[0]->children[0],
                                             root.children[0]},
      [&visited_nodes](const HloInstructionNode& node) {
        visited_nodes.push_back(node.GetName());
        return true;
      },
      BfsTraversalDirection::kForward, graph->GetNodeCount());

  EXPECT_THAT(visited_nodes,
              ElementsAre("add_1", "add_0", "foo", "bar", "baz"));
}

TEST_F(HloGumgraphBfsTest, BfsStopExpandingWorks) {
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
  HloGumgraphBfs(
      graph->GetRoot(),
      [&](const HloInstructionNode& node) {
        visited_nodes.push_back(node.GetName());
        return true;
      },
      BfsTraversalDirection::kForward, 6,
      [&](const HloInstructionNode& node) {
        return node.GetName() != "add_1";
      });

  EXPECT_THAT(visited_nodes, ElementsAre("root", "add_0", "add_1", "baz"));
}

TEST_F(HloGumgraphBfsTest, BfsEarlyTerminationWorks) {
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
  // This is an example of how to use per_node_fn return value to limit a BFS
  // traversal to stop after already visiting 5 nodes.
  HloGumgraphBfs(
      graph->GetRoot(),
      [&](const HloInstructionNode& node) {
        visited_nodes.push_back(node.GetName());
        return visited_nodes.size() < 5;
      },
      BfsTraversalDirection::kForward, graph->GetNodeCount());

  EXPECT_THAT(visited_nodes,
              ElementsAre("root", "add_0", "add_1", "baz", "foo"));
}
}  // namespace
}  // namespace hlo_diff
}  // namespace xla
