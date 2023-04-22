/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils/pattern_utils.h"

namespace tensorflow {
namespace grappler {
namespace utils {

// A subgraph pattern syntax implicitly defines a DAG having a single root. We
// traverse the syntax DAG in DFS manner. This function finds a match for
// current root of the pattern with the current node and recursively matches
// children subpatterns with the children of current node.
template <>
bool SubGraphMatcher<MatchingDirection::kFollowInputs>::DoesOpTypePatternMatch(
    const OpTypePattern& pattern, MutableNodeView* node_view,
    NodeViewMatch* match) {
  // Currently no control inputs and outputs are allowed.
  if (node_view->NumControllingFanins() > 0 ||
      node_view->NumControlledFanouts() > 0)
    return false;

  bool op_type_matched = false;
  if (pattern.op == "*") {
    op_type_matched = true;
  } else {
    // The op field string of current pattern might express an op among multiple
    // op types (mutually exclusive) separated by '|'.
    std::vector<string> op_list = str_util::Split(pattern.op, '|');
    for (const string& op : op_list) {
      if (node_view->node()->op() == op) {
        op_type_matched = true;
        break;
      }
    }
  }
  if (op_type_matched) {
    // If op type matches and current node is visited first time, insert current
    // node to node_label_to_index_ map with the current label as the key.
    // Multiple occurances of same label in the pattern syntax indicates that
    // the same node needs to be visited for each of such occurances. Hence
    // subsequent visits should find the corresponding label in the map as a key
    // and the current node should be the value for that key.
    if (node_label_to_index_.find(pattern.label) ==
        node_label_to_index_.end()) {
      node_label_to_index_[pattern.label] = node_view->node_index();
      // Bookkeeping
      matched_node_indices_.insert(node_view->node_index());
      if (pattern.node_status == NodeStatus::kRemove) {
        remove_node_indices_.insert(node_view->node_index());
      }
    } else if (node_label_to_index_[pattern.label] != node_view->node_index()) {
      return false;  // label constraint could not be satisfied.
    } else {
      DCHECK(node_label_to_index_[pattern.label] == node_view->node_index());
    }
  } else {
    return false;
  }
  // Current root of the pattern syntax is matched with the current node.
  match->node_view = node_view;

  // Go for matching child subpattern.
  if (!pattern.children.empty()) {
    // Currently only direction toward inputs is implemented.
    auto node_view_children = node_view->GetRegularFanins();
    if (node_view_children.size() != pattern.children.size()) {
      return false;
    } else {
      for (int i = 0; i < pattern.children.size(); ++i) {
        auto child_node_index = node_view_children[i].node_index();
        // TODO (mdfaijul): Is it guaranted that GetNode will reuturn non null
        // pointer.
        MutableNodeView* child_node_view =
            graph_view_->GetNode(child_node_index);
        const OpTypePattern& child_pattern = pattern.children[i];
        match->children.push_back(NodeViewMatch());
        NodeViewMatch* child_match = &(match->children.back());
        if (!DoesOpTypePatternMatch(child_pattern, child_node_view,
                                    child_match)) {
          return false;
        }
      }
    }
  }
  return true;
}

// Current implementation supports pattern maching toward node's inputs only.
template <>
bool SubGraphMatcher<MatchingDirection::kFollowInputs>::GetMatchedNodes(
    const OpTypePattern& pattern, MutableNodeView* node_view,
    std::map<string, int>* matched_nodes_map,
    std::set<int>* remove_node_indices) {
  bool found_match = false;
  match_.reset(new NodeViewMatch());
  if (DoesOpTypePatternMatch(pattern, node_view, match_.get())) {
    if (!HasRemoveNodeExternalDependents()) {
      found_match = true;
      matched_nodes_map->swap(this->node_label_to_index_);
      remove_node_indices->swap(this->remove_node_indices_);
    }
  } else {
    found_match = false;
    // Clear all bookkeeping data
    match_->Clear();
    match_.reset(nullptr);
    node_label_to_index_.clear();
    matched_node_indices_.clear();
    remove_node_indices_.clear();
  }
  return found_match;
}

}  // namespace utils
}  // namespace grappler
}  // namespace tensorflow
