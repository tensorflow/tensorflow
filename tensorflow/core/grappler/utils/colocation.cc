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

#include "tensorflow/core/grappler/utils/colocation.h"

#include <cstring>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

namespace {

// Find root node of the colocation group.
// The map is mapping from one node name to its parent. node_name is the
// starting node to search. By iteratively following the path from child to
// parent, we can find the root node for the colocation group that node_name
// belongs to.
string GetColocationGroupRoot(std::unordered_map<string, string>* map,
                              const string& node_name) {
  if (map->find(node_name) == map->end()) {
    // If node_name is not in the map, we create a new root node which points
    // to itself.
    map->insert({node_name, node_name});
    return node_name;
  }
  string cur = node_name;
  while ((*map)[cur] != cur) {
    // Backtracing the map until we reach the root node.
    cur = (*map)[cur];
  }
  return cur;
}

// Merge two colocation groups into one.
// left and right is the root node of two colocation groups respectively.
void MergeColocationGroup(std::unordered_map<string, string>* map,
                          const string& left, const string& right) {
  // Do nothing if left or right node is not in the map.
  if (map->find(left) == map->end() || map->find(right) == map->end()) {
    return;
  }
  if (left != right) {
    // Make the right node a child of the left node, which merges the two
    // groups.
    map->at(right) = left;
  }
}
}  // namespace

// Use of disjoint set algorithm to build the colocation groups from the input
// graph. The core data structure in use is a hash map from one node to its
// parent node. Whenever we see two nodes colocate with each other, we merge
// their colocation groups together. After we traverse all colocation pairs
// in the graph, we will have several disjoint sets. Then we pick the root node
// of each disjoint set as the representative node, and let all other nodes in
// the group colocate with the representative node.
void ReassignColocation(GraphDef* graph) {
  constexpr char kClassAttr[] = "_class";
  constexpr char kColocPrefix[] = "loc:@";

  // A hashmap that maps from a node name to its parent node name.
  std::unordered_map<string, string> coloc_groups;
  NodeMap node_map(graph);
  for (const auto& node : graph->node()) {
    auto iter = node.attr().find(kClassAttr);
    if (iter != node.attr().end() && iter->second.has_list()) {
      for (const auto& str : iter->second.list().s()) {
        size_t pos = str.find(kColocPrefix);
        if (pos == 0) {
          // After we find a colocation, update the colocation groups.
          string colocate_node = str.substr(pos + strlen(kColocPrefix));
          MergeColocationGroup(
              &coloc_groups, GetColocationGroupRoot(&coloc_groups, node.name()),
              GetColocationGroupRoot(&coloc_groups, colocate_node));
        }
      }
    }
  }

  // We use the root node of each colocation groups as its representative
  // node. For each node in one group, colocate with the representative node
  // if the node is in the graph.
  for (const auto& pair : coloc_groups) {
    if (pair.first != pair.second) {
      // This is a child node.
      NodeDef* node = node_map.GetNode(pair.first);
      if (node) {
        // Colocate this node with the root node.
        AttrValue new_value;
        new_value.mutable_list()->add_s(
            kColocPrefix + GetColocationGroupRoot(&coloc_groups, pair.first));
        node->mutable_attr()->erase(kClassAttr);
        node->mutable_attr()->insert({kClassAttr, new_value});
      }
    } else {
      // This is a root node. Clear the _class attribute.
      NodeDef* node = node_map.GetNode(pair.first);
      if (node) {  // root node should always exist in the graph as guaranteed
                   // by order of merging. Just put check here to ensure safety.
        node->mutable_attr()->erase(kClassAttr);
      }
    }
  }
}

}  // namespace grappler
}  // namespace tensorflow
