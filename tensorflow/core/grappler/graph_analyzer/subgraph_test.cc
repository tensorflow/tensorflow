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

#include "tensorflow/core/grappler/graph_analyzer/subgraph.h"

#include <algorithm>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/grappler/graph_analyzer/test_tools.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {
namespace test {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Ne;

TEST(SubgraphTest, Comparison) {
  GraphDef graph;
  // A topology with a loop.
  (*graph.add_node()) = MakeNodeConst("node1");
  (*graph.add_node()) = MakeNodeConst("node2");
  GenNodeMap map;
  ASSERT_THAT(GenNode::BuildGraphInMap(graph, &map), Eq(Status::OK()));
  auto gn1 = map["node1"].get();
  auto gn2 = map["node2"].get();
  ASSERT_THAT(gn1, Ne(nullptr));
  ASSERT_THAT(gn2, Ne(nullptr));

  Subgraph::Identity id1;
  Subgraph::Identity id2;

  id1.insert(gn1);
  id2.insert(gn2);

  Subgraph sg1(id1);
  Subgraph sg2(id2);

  EXPECT_TRUE(id1 == sg1.id());
  EXPECT_TRUE(id2 == sg2.id());

  EXPECT_THAT(sg1 < sg2, Eq(id1 < id2));
}

TEST(SubgraphTest, EmptyIteration) {
  NodeDef node1 = MakeNodeConst("node1");
  auto gn1 = absl::make_unique<GenNode>(&node1);
  Subgraph::Identity id1;
  id1.insert(gn1.get());
  Subgraph sg1(id1);
  SubgraphIterator sit(&sg1);

  EXPECT_TRUE(sit.AtEnd());
  EXPECT_FALSE(sit.Next());
  EXPECT_TRUE(sit.AtEnd());

  SubgraphIterator sit2(&sg1);
  EXPECT_TRUE(sit == sit2);
}

TEST(SubgraphTest, Iteration) {
  GraphDef graph;
  // A topology with a loop.
  (*graph.add_node()) = MakeNodeConst("node1");
  (*graph.add_node()) = MakeNodeSub("node2", "node3:1", "node3:0");
  auto node3 = graph.add_node();
  *node3 = MakeNodeBroadcastGradientArgs("node3", "node1", "node2");
  node3->add_input("^node3");  // The control link goes back to self.

  GenNodeMap map;
  ASSERT_THAT(GenNode::BuildGraphInMap(graph, &map), Eq(Status::OK()));
  ASSERT_THAT(map.find("node3"), Ne(map.end()));

  Subgraph::Identity id;
  id.insert(map["node3"].get());
  Subgraph sg(id);

  // node3 has 2 incoming data links, 2 outgoing data , 1 control incoming, 1
  // control outgoing = total of 6
  {
    SubgraphIterator sit(&sg);
    EXPECT_FALSE(sit.AtEnd());  // 1
    EXPECT_TRUE(sit.Next());
    EXPECT_FALSE(sit.AtEnd());  // 2
    EXPECT_TRUE(sit.Next());
    EXPECT_FALSE(sit.AtEnd());  // 3
    EXPECT_TRUE(sit.Next());
    EXPECT_FALSE(sit.AtEnd());  // 4
    EXPECT_TRUE(sit.Next());
    EXPECT_FALSE(sit.AtEnd());  // 5
    EXPECT_TRUE(sit.Next());
    EXPECT_FALSE(sit.AtEnd());  // 6
    EXPECT_FALSE(sit.Next());
    EXPECT_TRUE(sit.AtEnd());
  }

  // Now get the values out. And more equality testing along the way.
  {
    SubgraphIterator sit(&sg);
    SubgraphIterator sit2(&sg);
    std::vector<string> links;
    for (; !sit.AtEnd(); sit.Next()) {
      EXPECT_TRUE(sit == sit2);
      sit2.Next();
      EXPECT_FALSE(sit == sit2);

      links.push_back(absl::StrFormat("[%s,%s,%s]", string(sit.GetPort()),
                                      sit.GetNeighbor().node->name(),
                                      string(sit.GetNeighbor().port)));
    }
    EXPECT_TRUE(sit == sit2);

    std::sort(links.begin(), links.end());
    // clang-format off
    EXPECT_THAT(links, ElementsAre(
        "[i0,node1,o0]",
        "[i1,node2,o0]",
        "[iC,node3,oC]",
        "[o0,node2,i1]",
        "[o1,node2,i0]",
        "[oC,node3,iC]"
        ));
    // clang-format on
  }
}

TEST(SubgraphTest, IterationSamePort) {
  GraphDef graph;
  (*graph.add_node()) = MakeNodeConst("node1");
  (*graph.add_node()) = MakeNodeSub("node2", "node3", "node3");
  (*graph.add_node()) = MakeNodeAddN("node3", "node1", "node2");

  GenNodeMap map;
  ASSERT_THAT(GenNode::BuildGraphInMap(graph, &map), Eq(Status::OK()));
  ASSERT_THAT(map.find("node3"), Ne(map.end()));

  Subgraph::Identity id;
  id.insert(map["node3"].get());
  Subgraph sg(id);

  int total_links = 0;
  for (SubgraphIterator sit(&sg); !sit.AtEnd(); sit.Next()) {
    ++total_links;
  }

  // Initialize the port as control, which doesn't occur in this graph.
  GenNode::Port last_port(false, -1);
  int steps_total_same_port = 0;
  int steps_with_same_port = 0;
  for (SubgraphIterator sit(&sg); !sit.AtEnd(); sit.Next()) {
    GenNode::Port new_port = sit.GetPort();
    EXPECT_THAT(last_port.Encoded(), Ne(new_port.Encoded()))
        << "At step " << steps_total_same_port;
    last_port = new_port;

    ++steps_total_same_port;

    SubgraphIterator sit2(sit);
    sit2.SkipPort();

    while (sit.NextIfSamePort()) {
      new_port = sit.GetPort();
      EXPECT_THAT(last_port.Encoded(), Eq(new_port.Encoded()))
          << "At step " << steps_total_same_port;
      ++steps_total_same_port;
      ++steps_with_same_port;
    }

    EXPECT_TRUE(sit == sit2);
  }

  EXPECT_THAT(steps_total_same_port, Eq(total_links));
  // There is one 2-way input and one 2-way output.
  EXPECT_THAT(steps_with_same_port, Eq(2));
}

TEST(SubgraphTest, IterationSameNode) {
  GraphDef graph;
  (*graph.add_node()) = MakeNodeConst("node1");
  (*graph.add_node()) = MakeNodeSub("node2", "node3", "node3");
  (*graph.add_node()) = MakeNodeAddN("node3", "node1", "node2");

  GenNodeMap map;
  ASSERT_THAT(GenNode::BuildGraphInMap(graph, &map), Eq(Status::OK()));
  ASSERT_THAT(map.find("node3"), Ne(map.end()));

  Subgraph::Identity id;
  id.insert(map["node3"].get());
  Subgraph sg(id);

  const GenNode* last_node = nullptr;
  SubgraphIterator sit(&sg);
  while (!sit.AtEnd()) {
    const GenNode* new_node = sit.GetNode();

    EXPECT_THAT(new_node, Ne(last_node)) << "At node " << new_node->name();

    SubgraphIterator sit2(sit);
    sit2.SkipNode();

    ASSERT_FALSE(sit2.AtEnd());
    EXPECT_THAT(sit2.GetNode(), Eq(new_node))
        << "At expected node " << new_node->name() << ", got "
        << sit2.GetNode()->name();

    while (sit != sit2 && !sit.AtEnd()) {
      sit.Next();
    }

    ASSERT_FALSE(sit.AtEnd());
    EXPECT_THAT(sit.GetNode(), Eq(new_node))
        << "At expected node " << new_node->name() << ", got "
        << sit2.GetNode()->name();

    sit.Next();

    last_node = new_node;
  }

  // Check that it doesn't fail if already at end.
  sit.SkipNode();
  EXPECT_TRUE(sit.AtEnd());
}

TEST(SubgraphTest, ExtendSet) {
  GraphDef graph;
  // A topology with a loop.
  (*graph.add_node()) = MakeNodeConst("node1");
  (*graph.add_node()) = MakeNodeSub("node2", "node3:1", "node3:0");
  auto node3 = graph.add_node();
  *node3 = MakeNodeBroadcastGradientArgs("node3", "node1", "node2");
  node3->add_input("^node3");  // The control link goes back to self.

  GenNodeMap map;
  ASSERT_THAT(GenNode::BuildGraphInMap(graph, &map), Eq(Status::OK()));
  ASSERT_THAT(map.find("node2"), Ne(map.end()));
  ASSERT_THAT(map.find("node3"), Ne(map.end()));

  Subgraph::Identity id_empty;

  Subgraph::Identity id3;
  id3.insert(map["node3"].get());

  Subgraph::Identity id23 = id3;
  id23.insert(map["node2"].get());

  Subgraph* sg;
  SubgraphPtrSet set;

  // Extend an empty identity.
  sg = set.ExtendParent(id_empty, map["node3"].get());
  EXPECT_THAT(set.size(), Eq(1));
  ASSERT_THAT(sg, Ne(nullptr));
  EXPECT_TRUE(sg->id() == id3);

  // Extend with a node that is already in the parent.
  sg = set.ExtendParent(id3, map["node3"].get());
  EXPECT_THAT(set.size(), Eq(1));
  EXPECT_THAT(sg, Eq(nullptr));

  // Extend to a 2-node subgraph.
  sg = set.ExtendParent(id3, map["node2"].get());
  EXPECT_THAT(set.size(), Eq(2));
  ASSERT_THAT(sg, Ne(nullptr));
  EXPECT_TRUE(sg->id() == id23);

  // The second insert of the same node gets ignored.
  sg = set.ExtendParent(id3, map["node2"].get());
  EXPECT_THAT(set.size(), Eq(2));
  EXPECT_THAT(sg, Eq(nullptr));
}

TEST(SubgraphTest, ExtractForSignature) {
  GraphDef graph;
  (*graph.add_node()) = MakeNodeConst("node1");
  (*graph.add_node()) = MakeNodeSub("node2", "node3:1", "node3:0");
  auto node3 = graph.add_node();
  *node3 = MakeNodeBroadcastGradientArgs("node3", "node1", "node2");
  node3->add_input("^node1");
  node3->add_input("^node2");
  node3->add_input("^node3");  // The control link goes back to self.

  GenNodeMap map;
  ASSERT_THAT(GenNode::BuildGraphInMap(graph, &map), Eq(Status::OK()));
  ASSERT_THAT(map.find("node1"), Ne(map.end()));
  ASSERT_THAT(map.find("node2"), Ne(map.end()));
  ASSERT_THAT(map.find("node3"), Ne(map.end()));

  Subgraph::Identity id;
  id.insert(map["node1"].get());
  id.insert(map["node3"].get());

  Subgraph sg(id);

  SigNodeMap map2;
  sg.ExtractForSignature(&map2);
  ASSERT_THAT(map2.find("node1"), Ne(map2.end()));
  ASSERT_THAT(map2.find("node2"), Eq(map2.end()));
  ASSERT_THAT(map2.find("node3"), Ne(map2.end()));

  // clang-format off
  EXPECT_THAT(DumpLinkHashMap(map2["node1"]->hash_to_link()), ElementsAre(
      "oC:iC: node3",
      "o0:i0: node3"
      ));
  EXPECT_THAT(DumpHashedPeerVector(map2["node1"]->hashed_peers()), ElementsAre(
      "node3",
      "node3"
      ));
  EXPECT_THAT(DumpLinkHashMap(map2["node3"]->hash_to_link()), ElementsAre(
      "oC:iC: node3",
      "iC:oC: node1, node3",
      "i0:o0: node1"
      ));
  EXPECT_THAT(DumpHashedPeerVector(map2["node3"]->hashed_peers()), ElementsAre(
      "node3",
      "node1",
      "node3",
      "node1"
      ));
  // clang-format on
}

}  // end namespace
}  // end namespace test
}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow
