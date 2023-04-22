/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/graph_analyzer/gen_node.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/core/grappler/graph_analyzer/test_tools.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {
namespace test {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Ne;

TEST(GenNodeTest, Port) {
  {
    GenNode::Port p(true, 100);
    EXPECT_THAT(p.IsInbound(), Eq(true));
    EXPECT_THAT(p.IsControl(), Eq(false));
    EXPECT_THAT(p.Id(), Eq(100));
    GenNode::Port p2 = GenNode::Port::Decode(p.Encoded());
    EXPECT_THAT(p2.IsInbound(), Eq(true));
    EXPECT_THAT(p2.IsControl(), Eq(false));
    EXPECT_THAT(p2.Id(), Eq(100));
  }
  {
    GenNode::Port p(false, 0);
    EXPECT_THAT(p.IsInbound(), Eq(false));
    EXPECT_THAT(p.IsControl(), Eq(false));
    EXPECT_THAT(p.Id(), Eq(0));
    GenNode::Port p2 = GenNode::Port::Decode(p.Encoded());
    EXPECT_THAT(p2.IsInbound(), Eq(false));
    EXPECT_THAT(p2.IsControl(), Eq(false));
    EXPECT_THAT(p2.Id(), Eq(0));
  }
  {
    GenNode::Port p(true, -100);
    EXPECT_THAT(p.IsInbound(), Eq(true));
    EXPECT_THAT(p.IsControl(), Eq(true));
    EXPECT_THAT(p.Id(), Eq(-100));
    GenNode::Port p2 = GenNode::Port::Decode(p.Encoded());
    EXPECT_THAT(p2.IsInbound(), Eq(true));
    EXPECT_THAT(p2.IsControl(), Eq(true));
    EXPECT_THAT(p2.Id(), Eq(-100));
  }
  {
    GenNode::Port p(false, -1);
    EXPECT_THAT(p.IsInbound(), Eq(false));
    EXPECT_THAT(p.IsControl(), Eq(true));
    EXPECT_THAT(p.Id(), Eq(-1));
    GenNode::Port p2 = GenNode::Port::Decode(p.Encoded());
    EXPECT_THAT(p2.IsInbound(), Eq(false));
    EXPECT_THAT(p2.IsControl(), Eq(true));
    EXPECT_THAT(p2.Id(), Eq(-1));
  }
}

TEST(GenNodeTest, ParseNodeNoInputs) {
  GenNodeMap map;
  NodeDef node1 = MakeNodeConst("node1");
  map["node1"] = absl::make_unique<GenNode>(&node1);

  auto gn1 = map["node1"].get();
  ASSERT_THAT(gn1->ParseInputs(&map), Eq(Status::OK()));
  EXPECT_THAT(DumpLinkMap(gn1->links()), ElementsAre());
}

// A general operation, and a control link.
TEST(GenNodeTest, ParseNodeWithControl) {
  GenNodeMap map;

  NodeDef node1 = MakeNodeConst("node1");
  map["node1"] = absl::make_unique<GenNode>(&node1);

  NodeDef node2 = MakeNodeConst("node2");
  map["node2"] = absl::make_unique<GenNode>(&node2);

  NodeDef node3 = MakeNodeSub("node3", "node1", "node2");
  node3.add_input("^node1");  // The control link.
  node3.add_input("^node2");  // The control link.
  map["node3"] = absl::make_unique<GenNode>(&node3);

  auto gn1 = map["node1"].get();
  auto gn2 = map["node2"].get();
  auto gn3 = map["node3"].get();
  ASSERT_THAT(gn3->ParseInputs(&map), Eq(Status::OK()));
  // clang-format off
  EXPECT_THAT(DumpLinkMap(gn1->links()), ElementsAre(
      "o0: node3[i0]",
      "oC: node3[iC]"
      ));
  EXPECT_THAT(DumpLinkMap(gn2->links()), ElementsAre(
      "o0: node3[i1]",
      "oC: node3[iC]"
      ));
  EXPECT_THAT(DumpLinkMap(gn3->links()), ElementsAre(
      "i0: node1[o0]",
      "i1: node2[o0]",
      "iC: node1[oC], node2[oC]"
      ));
  // clang-format on

  EXPECT_THAT(gn3->IsMultiInput(GenNode::Port(true, 0)), Eq(false));

  // This is a multi-control-input.
  EXPECT_THAT(gn3->IsMultiInput(GenNode::Port(true, -1)), Eq(true));

  EXPECT_FALSE(gn1->AllInputsOrNone());
  EXPECT_FALSE(gn2->AllInputsOrNone());
  EXPECT_FALSE(gn3->AllInputsOrNone());
}

// Commutative nodes are treated as having a single input,
// because their inputs are equivalent.
TEST(GenNodeTest, ParseNodeCommutative) {
  GenNodeMap map;

  NodeDef node1 = MakeNodeConst("node1");
  map["node1"] = absl::make_unique<GenNode>(&node1);

  NodeDef node2 = MakeNodeConst("node2");
  map["node2"] = absl::make_unique<GenNode>(&node2);

  // TODO(babkin): grappler::IsCommutative() should return true for Add but
  // apparently doesn't. So use Mul in the meantime.
  NodeDef node3 = MakeNodeMul("node3", "node1", "node2");
  map["node3"] = absl::make_unique<GenNode>(&node3);

  auto gn1 = map["node1"].get();
  auto gn2 = map["node2"].get();
  auto gn3 = map["node3"].get();
  ASSERT_THAT(gn3->ParseInputs(&map), Eq(Status::OK()));
  // clang-format off
  EXPECT_THAT(DumpLinkMap(gn1->links()), ElementsAre(
      "o0: node3[i0]"
      ));
  EXPECT_THAT(DumpLinkMap(gn2->links()), ElementsAre(
      "o0: node3[i0]"
      ));
  EXPECT_THAT(DumpLinkMap(gn3->links()), ElementsAre(
      "i0: node1[o0], node2[o0]"
      ));
  // clang-format on

  EXPECT_THAT(gn3->IsMultiInput(GenNode::Port(true, 0)), Eq(true));

  EXPECT_FALSE(gn3->AllInputsOrNone());
}

TEST(GenNodeTest, ParseNodeMultiInputCommutative) {
  GenNodeMap map;

  NodeDef node1 = MakeNodeConst("node1");
  map["node1"] = absl::make_unique<GenNode>(&node1);

  NodeDef node2 = MakeNodeConst("node2");
  map["node2"] = absl::make_unique<GenNode>(&node2);

  NodeDef node3 = MakeNodeAddN("node3", "node1", "node2");
  map["node3"] = absl::make_unique<GenNode>(&node3);

  auto gn1 = map["node1"].get();
  auto gn2 = map["node2"].get();
  auto gn3 = map["node3"].get();
  ASSERT_THAT(gn3->ParseInputs(&map), Eq(Status::OK()));
  // clang-format off
  EXPECT_THAT(DumpLinkMap(gn1->links()), ElementsAre(
      "o0: node3[i0]"
      ));
  EXPECT_THAT(DumpLinkMap(gn2->links()), ElementsAre(
      "o0: node3[i0]"
      ));
  EXPECT_THAT(DumpLinkMap(gn3->links()), ElementsAre(
      "i0: node1[o0], node2[o0]"
      ));
  // clang-format on

  // This is a multi-output.
  EXPECT_THAT(gn2->IsMultiInput(GenNode::Port(false, 0)), Eq(false));
  // This is a multi-input.
  EXPECT_THAT(gn3->IsMultiInput(GenNode::Port(true, 0)), Eq(true));

  EXPECT_FALSE(gn3->AllInputsOrNone());
}

TEST(GenNodeTest, ParseNodeMultiInputNotCommutative) {
  GenNodeMap map;

  NodeDef node1 = MakeNodeConst("node1");
  map["node1"] = absl::make_unique<GenNode>(&node1);

  NodeDef node2 = MakeNodeConst("node2");
  map["node2"] = absl::make_unique<GenNode>(&node2);

  NodeDef node3 = MakeNodeShapeN("node3", "node1", "node2");
  map["node3"] = absl::make_unique<GenNode>(&node3);

  auto gn1 = map["node1"].get();
  auto gn2 = map["node2"].get();
  auto gn3 = map["node3"].get();
  ASSERT_THAT(gn3->ParseInputs(&map), Eq(Status::OK()));
  // clang-format off
  EXPECT_THAT(DumpLinkMap(gn1->links()), ElementsAre(
      "o0: node3[i0]"
      ));
  EXPECT_THAT(DumpLinkMap(gn2->links()), ElementsAre(
      "o0: node3[i1]"
      ));
  EXPECT_THAT(DumpLinkMap(gn3->links()), ElementsAre(
      "i0: node1[o0]",
      "i1: node2[o0]"
      ));
  // clang-format on

  // Non-commutative multi-input doesn't count.
  EXPECT_THAT(gn3->IsMultiInput(GenNode::Port(true, 0)), Eq(false));
  EXPECT_TRUE(gn3->AllInputsOrNone());
}

TEST(GenNodeTest, ParseNodeMultiInputList) {
  GenNodeMap map;

  NodeDef node1 = MakeNodeConst("node1");
  map["node1"] = absl::make_unique<GenNode>(&node1);

  NodeDef node2 = MakeNodeConst("node2");
  map["node2"] = absl::make_unique<GenNode>(&node2);

  NodeDef node3 = MakeNodeIdentityN("node3", "node1", "node2");
  map["node3"] = absl::make_unique<GenNode>(&node3);

  auto gn1 = map["node1"].get();
  auto gn2 = map["node2"].get();
  auto gn3 = map["node3"].get();
  ASSERT_THAT(gn3->ParseInputs(&map), Eq(Status::OK()));
  // clang-format off
  EXPECT_THAT(DumpLinkMap(gn1->links()), ElementsAre(
      "o0: node3[i0]"
      ));
  EXPECT_THAT(DumpLinkMap(gn2->links()), ElementsAre(
      "o0: node3[i1]"
      ));
  EXPECT_THAT(DumpLinkMap(gn3->links()), ElementsAre(
      "i0: node1[o0]",
      "i1: node2[o0]"
      ));
  // clang-format on

  // Non-commutative multi-input doesn't count.
  EXPECT_THAT(gn3->IsMultiInput(GenNode::Port(true, 0)), Eq(false));
  EXPECT_TRUE(gn3->AllInputsOrNone());
}

TEST(GenNodeTest, ParseNodeMultiMultiInput) {
  GenNodeMap map;

  NodeDef node1 = MakeNodeConst("node1");
  map["node1"] = absl::make_unique<GenNode>(&node1);

  NodeDef node2 = MakeNodeConst("node2");
  map["node2"] = absl::make_unique<GenNode>(&node2);

  NodeDef node3 = MakeNodeConst("node3");
  map["node3"] = absl::make_unique<GenNode>(&node3);

  NodeDef node4 = MakeNodeConst("node4");
  map["node4"] = absl::make_unique<GenNode>(&node4);

  NodeDef node5 =
      MakeNodeQuantizedConcat("node5", "node1", "node2", "node3", "node4");
  map["node5"] = absl::make_unique<GenNode>(&node5);

  auto gn1 = map["node1"].get();
  auto gn2 = map["node2"].get();
  auto gn3 = map["node3"].get();
  auto gn4 = map["node4"].get();
  auto gn5 = map["node5"].get();
  ASSERT_THAT(gn5->ParseInputs(&map), Eq(Status::OK()));
  // clang-format off
  EXPECT_THAT(DumpLinkMap(gn1->links()), ElementsAre(
      "o0: node5[i0]"
      ));
  EXPECT_THAT(DumpLinkMap(gn2->links()), ElementsAre(
      "o0: node5[i1]"
      ));
  EXPECT_THAT(DumpLinkMap(gn3->links()), ElementsAre(
      "o0: node5[i2]"
      ));
  EXPECT_THAT(DumpLinkMap(gn4->links()), ElementsAre(
      "o0: node5[i3]"
      ));
  EXPECT_THAT(DumpLinkMap(gn5->links()), ElementsAre(
      "i0: node1[o0]",
      "i1: node2[o0]",
      "i2: node3[o0]",
      "i3: node4[o0]"
      ));
  // clang-format on

  // Non-commutative multi-input doesn't count.
  EXPECT_THAT(gn5->IsMultiInput(GenNode::Port(true, 1)), Eq(false));
  EXPECT_THAT(gn5->IsMultiInput(GenNode::Port(true, 2)), Eq(false));
  EXPECT_TRUE(gn5->AllInputsOrNone());
}

TEST(GenNodeTest, ParseNodeMultiOutput) {
  GenNodeMap map;

  NodeDef node1 = MakeNodeConst("node1");
  map["node1"] = absl::make_unique<GenNode>(&node1);

  NodeDef node2 = MakeNodeConst("node2");
  map["node2"] = absl::make_unique<GenNode>(&node2);

  NodeDef node3 = MakeNodeBroadcastGradientArgs("node3", "node1", "node2");
  map["node3"] = absl::make_unique<GenNode>(&node3);

  NodeDef node4 = MakeNodeSub("node4", "node3:1", "node3:0");
  map["node4"] = absl::make_unique<GenNode>(&node4);

  auto gn4 = map["node4"].get();
  ASSERT_THAT(gn4->ParseInputs(&map), Eq(Status::OK()));
  // clang-format off
  EXPECT_THAT(DumpLinkMap(gn4->links()), ElementsAre(
      "i0: node3[o1]",
      "i1: node3[o0]"
      ));
  // clang-format on
}

TEST(GenNodeTest, ParseNodeUndefinedOp) {
  GenNodeMap map;
  NodeDef node1;
  node1.set_name("node1");
  node1.set_op("Zzzx");

  map["node1"] = absl::make_unique<GenNode>(&node1);

  const OpDef* opdef;
  Status nested_error = OpRegistry::Global()->LookUpOpDef("Zzzx", &opdef);

  auto gn = map["node1"].get();
  ASSERT_THAT(
      gn->ParseInputs(&map),
      Eq(Status(error::INVALID_ARGUMENT,
                "Node 'node1' contains an undefined operation 'Zzzx': " +
                    nested_error.error_message())));
}

TEST(GenNodeTest, ParseNodeUnexpectedInputs) {
  GenNodeMap map;

  NodeDef node1 = MakeNodeConst("node1");
  map["node1"] = absl::make_unique<GenNode>(&node1);
  node1.add_input("node1");

  auto gn1 = map["node1"].get();
  EXPECT_THAT(gn1->ParseInputs(&map),
              Eq(Status(error::INVALID_ARGUMENT,
                        "Node 'node1' has a non-control "
                        "input from 'node1' at index 0 but its operation "
                        "'Const' defines only 0 inputs.")));

  NodeDef node2 = MakeNodeConst("node2");
  map["node2"] = absl::make_unique<GenNode>(&node2);

  NodeDef node3 = MakeNodeSub("node3", "node1", "node2");
  map["node3"] = absl::make_unique<GenNode>(&node3);
  node3.add_input("node1");

  auto gn3 = map["node3"].get();
  EXPECT_THAT(gn3->ParseInputs(&map),
              Eq(Status(error::INVALID_ARGUMENT,
                        "Node 'node3' has a non-control "
                        "input from 'node1' at index 2 but its operation "
                        "'Sub' defines only 2 inputs.")));
}

// Even if an opcode defines no inputs, the node may still accept the control
// inputs.
TEST(GenNodeTest, ParseNodeControlInputsAlwaysOk) {
  GenNodeMap map;
  NodeDef node1 = MakeNodeConst("node1");
  map["node1"] = absl::make_unique<GenNode>(&node1);
  node1.add_input("^node1");
  auto gn1 = map["node1"].get();
  ASSERT_THAT(gn1->ParseInputs(&map), Eq(Status::OK()));
  // clang-format off
  EXPECT_THAT(DumpLinkMap(gn1->links()), ElementsAre(
      "iC: node1[oC]",
      "oC: node1[iC]"
      ));
  // clang-format on
}

TEST(GenNodeTest, ParseNodeInvalidInput) {
  GenNodeMap map;
  NodeDef node1 = MakeNodeAddN("node1", "node2", "node3");
  map["node1"] = absl::make_unique<GenNode>(&node1);
  node1.add_input("node1");
  auto gn1 = map["node1"].get();
  ASSERT_THAT(
      gn1->ParseInputs(&map),
      Eq(Status(
          error::INVALID_ARGUMENT,
          "Node 'node1' input 0 refers to a non-existing node 'node2'.")));
}

TEST(GenNodeTest, BuildGraphInMap) {
  GraphDef graph;
  // A topology with a loop.
  (*graph.add_node()) = MakeNodeConst("node1");
  (*graph.add_node()) = MakeNodeSub("node2", "node3:1", "node3:0");
  (*graph.add_node()) =
      MakeNodeBroadcastGradientArgs("node3", "node1", "node2");

  GenNodeMap map;
  ASSERT_THAT(GenNode::BuildGraphInMap(graph, &map), Eq(Status::OK()));
  ASSERT_THAT(map.find("node1"), Ne(map.end()));
  ASSERT_THAT(map.find("node2"), Ne(map.end()));
  ASSERT_THAT(map.find("node3"), Ne(map.end()));

  EXPECT_THAT(map["node1"]->name(), Eq("node1"));
  EXPECT_THAT(map["node2"]->name(), Eq("node2"));
  EXPECT_THAT(map["node3"]->name(), Eq("node3"));

  // clang-format off
  EXPECT_THAT(DumpLinkMap(map["node1"]->links()), ElementsAre(
      "o0: node3[i0]"
      ));
  EXPECT_THAT(DumpLinkMap(map["node2"]->links()), ElementsAre(
      "i0: node3[o1]",
      "i1: node3[o0]",
      "o0: node3[i1]"
      ));
  EXPECT_THAT(DumpLinkMap(map["node3"]->links()), ElementsAre(
      "i0: node1[o0]",
      "i1: node2[o0]",
      "o0: node2[i1]",
      "o1: node2[i0]"
      ));
  // clang-format on
}

TEST(GenNodeTest, BuildGraphInMapDuplicateNode) {
  GraphDef graph;
  (*graph.add_node()) = MakeNodeConst("node1");
  (*graph.add_node()) = MakeNodeConst("node1");
  GenNodeMap map;
  ASSERT_THAT(
      GenNode::BuildGraphInMap(graph, &map),
      Eq(Status(error::INVALID_ARGUMENT, "Duplicate node name 'node1'.")));
}

TEST(GenNodeTest, BuildGraphInMapParseError) {
  GraphDef graph;
  // A topology with a loop.
  (*graph.add_node()) = MakeNodeConst("node1");
  (*graph.add_node()) = MakeNodeSub("node2", "node3:1", "node3:0");

  GenNodeMap map;
  ASSERT_THAT(
      GenNode::BuildGraphInMap(graph, &map),
      Eq(Status(
          error::INVALID_ARGUMENT,
          "Node 'node2' input 0 refers to a non-existing node 'node3'.")));
}

}  // end namespace
}  // end namespace test
}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow
