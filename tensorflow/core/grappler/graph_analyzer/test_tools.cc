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

#include "tensorflow/core/grappler/graph_analyzer/test_tools.h"

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {
namespace test {

//=== Helper methods to construct the nodes.

NodeDef MakeNodeConst(const string& name) {
  NodeDef n;
  n.set_name(name);
  n.set_op("Const");
  return n;
}

NodeDef MakeNode2Arg(const string& name, const string& opcode,
                     const string& arg1, const string& arg2) {
  NodeDef n;
  n.set_name(name);
  n.set_op(opcode);
  n.add_input(arg1);
  n.add_input(arg2);
  return n;
}

NodeDef MakeNode4Arg(const string& name, const string& opcode,
                     const string& arg1, const string& arg2, const string& arg3,
                     const string& arg4) {
  NodeDef n;
  n.set_name(name);
  n.set_op(opcode);
  n.add_input(arg1);
  n.add_input(arg2);
  n.add_input(arg3);
  n.add_input(arg4);
  return n;
}

// Not really a 2-argument but convenient to construct.
NodeDef MakeNodeShapeN(const string& name, const string& arg1,
                       const string& arg2) {
  // This opcode is multi-input but not commutative.
  return MakeNode2Arg(name, "ShapeN", arg1, arg2);
}

// Not really a 2-argument but convenient to construct.
NodeDef MakeNodeIdentityN(const string& name, const string& arg1,
                          const string& arg2) {
  // The argument is of a list type.
  return MakeNode2Arg(name, "IdentityN", arg1, arg2);
}

NodeDef MakeNodeQuantizedConcat(const string& name, const string& arg1,
                                const string& arg2, const string& arg3,
                                const string& arg4) {
  // This opcode has multiple multi-inputs.
  return MakeNode4Arg(name, "QuantizedConcat", arg1, arg2, arg3, arg4);
}

//=== Helper methods for analysing the structures.

std::vector<string> DumpLinkMap(const GenNode::LinkMap& link_map) {
  // This will order the entries first.
  std::map<string, string> ordered;
  for (const auto& link : link_map) {
    string key = string(link.first);

    // Order the other sides too. They may be repeating, so store them
    // in a multiset.
    std::multiset<string> others;
    for (const auto& other : link.second) {
      others.emplace(
          absl::StrFormat("%s[%s]", other.node->name(), string(other.port)));
    }
    ordered[key] = absl::StrJoin(others, ", ");
  }
  // Now dump the result in a predictable order.
  std::vector<string> result;
  result.reserve(ordered.size());
  for (const auto& link : ordered) {
    result.emplace_back(link.first + ": " + link.second);
  }
  return result;
}

std::vector<string> DumpLinkHashMap(const SigNode::LinkHashMap& link_hash_map) {
  // The entries in this map are ordered by hash value which might change
  // at any point. Re-order them by the link tag.
  std::map<SigNode::LinkTag, size_t> tags;
  for (const auto& entry : link_hash_map) {
    tags[entry.second.tag] = entry.first;
  }

  std::vector<string> result;
  for (const auto& id : tags) {
    // For predictability, the nodes need to be sorted.
    std::vector<string> nodes;
    for (const auto& peer : link_hash_map.at(id.second).peers) {
      nodes.emplace_back(peer->name());
    }
    std::sort(nodes.begin(), nodes.end());
    result.emplace_back(string(id.first.local) + ":" + string(id.first.remote) +
                        ": " + absl::StrJoin(nodes, ", "));
  }
  return result;
}

std::vector<string> DumpHashedPeerVector(
    const SigNode::HashedPeerVector& hashed_peers) {
  std::vector<string> result;

  // Each subset of nodes with the same hash has to be sorted by name.
  // Other than that, the vector is already ordered by full tags.
  size_t last_hash = 0;
  // Index, since iterators may get invalidated on append.
  size_t subset_start = 0;

  for (const auto& entry : hashed_peers) {
    if (entry.link_hash != last_hash) {
      std::sort(result.begin() + subset_start, result.end());
      subset_start = result.size();
    }
    result.emplace_back(entry.peer->name());
  }
  std::sort(result.begin() + subset_start, result.end());

  return result;
}

TestGraphs::TestGraphs() {
  {
    GraphDef& graph = graph_3n_self_control_;
    // The topology includes a loop and a link to self.
    (*graph.add_node()) = MakeNodeConst("node1");
    (*graph.add_node()) = MakeNodeSub("node2", "node3:1", "node3:0");
    auto node3 = graph.add_node();
    *node3 = MakeNodeBroadcastGradientArgs("node3", "node1", "node2");
    node3->add_input("^node3");  // The control link goes back to self.
  }
  {
    GraphDef& graph = graph_multi_input_;
    // The topology includes a loop and a link to self.
    (*graph.add_node()) = MakeNodeConst("const1_1");
    (*graph.add_node()) = MakeNodeConst("const1_2");
    (*graph.add_node()) = MakeNodeAddN("add1", "const1_1", "const1_2");

    (*graph.add_node()) = MakeNodeConst("const2_1");
    (*graph.add_node()) = MakeNodeConst("const2_2");
    (*graph.add_node()) = MakeNodeConst("const2_3");

    auto add2 = graph.add_node();
    *add2 = MakeNodeAddN("add2", "const2_1", "const2_2");
    // The 3rd node is connected twice, to 4 links total.
    add2->add_input("const2_3");
    add2->add_input("const2_3");

    (*graph.add_node()) = MakeNodeSub("sub", "add1", "add2");
  }
  {
    GraphDef& graph = graph_all_or_none_;
    // The topology includes a loop and a link to self.
    (*graph.add_node()) = MakeNodeConst("const1_1");
    (*graph.add_node()) = MakeNodeConst("const1_2");
    auto pass1 = graph.add_node();
    *pass1 = MakeNodeIdentityN("pass1", "const1_1", "const1_2");

    (*graph.add_node()) = MakeNodeConst("const2_1");
    (*graph.add_node()) = MakeNodeConst("const2_2");
    (*graph.add_node()) = MakeNodeConst("const2_3");

    auto pass2 = graph.add_node();
    *pass2 = MakeNodeIdentityN("pass2", "const2_1", "const2_2");
    // The 3rd node is connected twice, to 4 links total.
    pass2->add_input("const2_3");
    pass2->add_input("const2_3");

    // Add the control links, they get handled separately than the normal
    // links.
    pass1->add_input("^const2_1");
    pass1->add_input("^const2_2");
    pass1->add_input("^const2_3");

    (*graph.add_node()) = MakeNodeSub("sub", "pass1", "pass2");
  }
  {
    GraphDef& graph = graph_circular_onedir_;
    (*graph.add_node()) = MakeNodeMul("node1", "node5", "node5");
    (*graph.add_node()) = MakeNodeMul("node2", "node1", "node1");
    (*graph.add_node()) = MakeNodeMul("node3", "node2", "node2");
    (*graph.add_node()) = MakeNodeMul("node4", "node3", "node3");
    (*graph.add_node()) = MakeNodeMul("node5", "node4", "node4");
  }
  {
    GraphDef& graph = graph_circular_bidir_;
    // The left and right links are intentionally mixed up.
    (*graph.add_node()) = MakeNodeMul("node1", "node5", "node2");
    (*graph.add_node()) = MakeNodeMul("node2", "node3", "node1");
    (*graph.add_node()) = MakeNodeMul("node3", "node2", "node4");
    (*graph.add_node()) = MakeNodeMul("node4", "node5", "node3");
    (*graph.add_node()) = MakeNodeMul("node5", "node4", "node1");
  }
  {
    GraphDef& graph = graph_linear_;
    (*graph.add_node()) = MakeNodeConst("node1");
    (*graph.add_node()) = MakeNodeMul("node2", "node1", "node1");
    (*graph.add_node()) = MakeNodeMul("node3", "node2", "node2");
    (*graph.add_node()) = MakeNodeMul("node4", "node3", "node3");
    (*graph.add_node()) = MakeNodeMul("node5", "node4", "node4");
  }
  {
    GraphDef& graph = graph_cross_;
    (*graph.add_node()) = MakeNodeConst("node1");
    (*graph.add_node()) = MakeNodeMul("node2", "node1", "node1");
    (*graph.add_node()) = MakeNodeConst("node3");
    (*graph.add_node()) = MakeNodeMul("node4", "node3", "node3");
    (*graph.add_node()) = MakeNodeConst("node5");
    (*graph.add_node()) = MakeNodeMul("node6", "node5", "node5");
    (*graph.add_node()) = MakeNodeConst("node7");
    (*graph.add_node()) = MakeNodeMul("node8", "node7", "node7");

    auto center = graph.add_node();
    *center = MakeNodeMul("node9", "node2", "node4");
    center->add_input("node6");
    center->add_input("node8");
  }
  {
    GraphDef& graph = graph_small_cross_;
    (*graph.add_node()) = MakeNodeConst("node1");
    (*graph.add_node()) = MakeNodeConst("node2");
    (*graph.add_node()) = MakeNodeConst("node3");
    (*graph.add_node()) = MakeNodeConst("node4");

    auto center = graph.add_node();
    *center = MakeNodeMul("node5", "node1", "node2");
    center->add_input("node3");
    center->add_input("node4");
  }
  {
    GraphDef& graph = graph_for_link_order_;
    (*graph.add_node()) = MakeNodeConst("node1");
    (*graph.add_node()) = MakeNodeConst("node2");
    (*graph.add_node()) = MakeNodeConst("node3");
    (*graph.add_node()) = MakeNodeConst("node4");

    // One group of equivalent links.
    auto center = graph.add_node();
    *center = MakeNodeMul("node5", "node1", "node2");
    center->add_input("node3");
    center->add_input("node4");

    // Multiple groups, separated by unique links.
    auto center2 = graph.add_node();
    *center2 = MakeNodeMul("node6", "node1", "node2");
    center2->add_input("node2:1");
    center2->add_input("node3:2");
    center2->add_input("node4:2");
    center2->add_input("node4:3");
  }
  {
    GraphDef& graph = graph_sun_;
    (*graph.add_node()) = MakeNodeConst("node1");
    (*graph.add_node()) = MakeNodeConst("node2");
    (*graph.add_node()) = MakeNodeConst("node3");
    (*graph.add_node()) = MakeNodeConst("node4");
    (*graph.add_node()) = MakeNodeConst("node5");
    (*graph.add_node()) = MakeNodeSub("node6", "node1", "node10");
    (*graph.add_node()) = MakeNodeSub("node7", "node2", "node6");
    (*graph.add_node()) = MakeNodeSub("node8", "node3", "node7");
    (*graph.add_node()) = MakeNodeSub("node9", "node4", "node8");
    (*graph.add_node()) = MakeNodeSub("node10", "node5", "node9");
  }
}

}  // end namespace test
}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow
