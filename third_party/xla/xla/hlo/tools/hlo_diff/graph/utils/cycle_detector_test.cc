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

#include "xla/hlo/tools/hlo_diff/graph/utils/cycle_detector.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/shape_util.h"

namespace xla {
namespace hlo_diff {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

class DetectAndLogAllCyclesTest : public ::testing::Test {
 protected:
  std::vector<std::unique_ptr<HloInstruction>> owned_instructions_;
  std::vector<std::unique_ptr<HloInstructionNode>> owned_nodes_;

  HloInstructionNode* CreateTestNode(const std::string& name) {
    auto node = std::make_unique<HloInstructionNode>();
    auto instruction =
        HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), name);
    node->instruction = instruction.get();
    owned_instructions_.push_back(std::move(instruction));

    HloInstructionNode* raw_ptr = node.get();
    owned_nodes_.push_back(std::move(node));
    return raw_ptr;
  }

  // Converts a cycle path (vector of nodes) to a vector of test names from our
  // map.
  std::vector<absl::string_view> GetCycleNodeNamesFromMap(
      const std::vector<const HloInstructionNode*>& cycle_path) {
    std::vector<absl::string_view> names;
    for (const auto* node_ptr : cycle_path) {
      names.push_back(node_ptr->instruction->name());
    }

    return names;
  }

  // Converts a list of detected cycles into a canonical form for comparison.
  std::vector<std::vector<absl::string_view>> GetCanonicalCycleSetNamesFromMap(
      const std::vector<std::vector<const HloInstructionNode*>>& cycles) {
    std::vector<std::vector<absl::string_view>> cycle_sets_names;
    for (const auto& cycle_path : cycles) {
      std::vector<absl::string_view> current_cycle_names =
          GetCycleNodeNamesFromMap(cycle_path);
      if (!current_cycle_names.empty()) {
        auto min_it = std::min_element(current_cycle_names.begin(),
                                       current_cycle_names.end());
        std::rotate(current_cycle_names.begin(), min_it,
                    current_cycle_names.end());
      }
      cycle_sets_names.push_back(current_cycle_names);
    }
    // Sort the set of cycles for consistent order in UnorderedElementsAre.
    std::sort(cycle_sets_names.begin(), cycle_sets_names.end());
    return cycle_sets_names;
  }

  void TearDown() override {
    owned_instructions_.clear();
    owned_nodes_.clear();
  }
};

TEST_F(DetectAndLogAllCyclesTest, EmptyGraphNoCyclesDetected) {
  std::vector<HloInstructionNode*> graph_nodes;
  std::vector<std::vector<const HloInstructionNode*>> cycles =
      DetectAndLogAllCycles(graph_nodes);
  EXPECT_THAT(cycles, IsEmpty());
}

TEST_F(DetectAndLogAllCyclesTest, SimpleGraphNoCyclesDetected) {
  HloInstructionNode* node_a = CreateTestNode("A");
  HloInstructionNode* node_b = CreateTestNode("B");
  HloInstructionNode* node_c = CreateTestNode("C");

  node_a->children.push_back(node_b);  // A -> B
  node_b->children.push_back(node_c);  // B -> C

  std::vector<HloInstructionNode*> graph_nodes = {node_a, node_b, node_c};
  std::vector<std::vector<const HloInstructionNode*>> cycles =
      DetectAndLogAllCycles(graph_nodes);
  EXPECT_THAT(cycles, IsEmpty());
}

TEST_F(DetectAndLogAllCyclesTest, ThreeNodeCycleDetected) {
  HloInstructionNode* node_a = CreateTestNode("A");
  HloInstructionNode* node_b = CreateTestNode("B");
  HloInstructionNode* node_c = CreateTestNode("C");

  node_a->children.push_back(node_b);  // A -> B
  node_b->children.push_back(node_c);  // B -> C
  node_c->children.push_back(node_a);  // C -> A

  std::vector<HloInstructionNode*> graph_nodes = {node_a, node_b, node_c};
  std::vector<std::vector<const HloInstructionNode*>> cycles =
      DetectAndLogAllCycles(graph_nodes);

  ASSERT_THAT(cycles, SizeIs(1));
  std::vector<std::vector<absl::string_view>> cycle_names =
      GetCanonicalCycleSetNamesFromMap(cycles);
  EXPECT_THAT(cycle_names, ElementsAre(ElementsAre("A", "B", "C")));
}

TEST_F(DetectAndLogAllCyclesTest, MultipleDisjointCycles) {
  HloInstructionNode* node_a = CreateTestNode("A");
  HloInstructionNode* node_b = CreateTestNode("B");
  HloInstructionNode* node_x = CreateTestNode("X");
  HloInstructionNode* node_y = CreateTestNode("Y");
  HloInstructionNode* node_z = CreateTestNode("Z");

  // Cycle 1: A -> B -> A
  node_a->children.push_back(node_b);
  node_b->children.push_back(node_a);

  // Cycle 2: X -> Y -> Z -> X
  node_x->children.push_back(node_y);
  node_y->children.push_back(node_z);
  node_z->children.push_back(node_x);

  std::vector<HloInstructionNode*> graph_nodes = {node_a, node_b, node_x,
                                                  node_y, node_z};
  std::vector<std::vector<const HloInstructionNode*>> cycles =
      DetectAndLogAllCycles(graph_nodes);

  ASSERT_THAT(cycles, SizeIs(2));
  std::vector<std::vector<absl::string_view>> cycle_names =
      GetCanonicalCycleSetNamesFromMap(cycles);
  EXPECT_THAT(cycle_names, UnorderedElementsAre(ElementsAre("A", "B"),
                                                ElementsAre("X", "Y", "Z")));
}

TEST_F(DetectAndLogAllCyclesTest, OverlappingCycles) {
  HloInstructionNode* node_a = CreateTestNode("A");
  HloInstructionNode* node_b = CreateTestNode("B");
  HloInstructionNode* node_c = CreateTestNode("C");
  HloInstructionNode* node_d = CreateTestNode("D");

  // Cycle 1: A -> B -> C -> A
  node_a->children.push_back(node_b);
  node_b->children.push_back(node_c);
  node_c->children.push_back(node_a);

  // Cycle 2: B -> D -> D (shares B with cycle 1)
  node_b->children.push_back(node_d);
  node_d->children.push_back(node_b);

  std::vector<HloInstructionNode*> graph_nodes = {node_a, node_b, node_c,
                                                  node_d};
  std::vector<std::vector<const HloInstructionNode*>> cycles =
      DetectAndLogAllCycles(graph_nodes);

  ASSERT_THAT(cycles, SizeIs(2));
  EXPECT_THAT(
      GetCanonicalCycleSetNamesFromMap(cycles),
      UnorderedElementsAre(ElementsAre("A", "B", "C"), ElementsAre("B", "D")));
}

TEST_F(DetectAndLogAllCyclesTest, AcyclicGraphWithBranchAndJoin) {
  HloInstructionNode* node_a = CreateTestNode("A");
  HloInstructionNode* node_b = CreateTestNode("B");
  HloInstructionNode* node_c = CreateTestNode("C");
  HloInstructionNode* node_d = CreateTestNode("D");

  // Diamond shape graph: A -> B, A -> C, B -> D, C -> D
  node_a->children.push_back(node_b);
  node_a->children.push_back(node_c);
  node_b->children.push_back(node_d);
  node_c->children.push_back(node_d);

  std::vector<HloInstructionNode*> graph_nodes = {node_a, node_b, node_c,
                                                  node_d};
  std::vector<std::vector<const HloInstructionNode*>> cycles =
      DetectAndLogAllCycles(graph_nodes);

  EXPECT_THAT(cycles, IsEmpty());
}

}  // namespace
}  // namespace hlo_diff
}  // namespace xla
