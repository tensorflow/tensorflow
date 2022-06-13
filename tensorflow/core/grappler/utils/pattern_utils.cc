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

#include <algorithm>

#include "absl/container/flat_hash_set.h"

namespace tensorflow {
namespace grappler {
namespace utils {

const bool IsCommutativeOp(const string& op) {
  // TODO(intel-tf): Add more ops to this list if needed.
  std::vector<string> op_list = str_util::Split(op, '|');
  static const auto* commutative_ops = new absl::flat_hash_set<string>(
      {"Add", "AddV2", "Mul", "Maximum", "SquaredDifference"});
  for (const string& op_ : op_list) {
    if (commutative_ops->contains(op_)) return true;
  }
  return false;
}

// op1 is an op name in the pattern and it could be wildcard `*` or some
// registered op in tensorflow and may have multiple ops separated by '|'.
// op2 is an op name in the computation graph and
// is always one of the registered ops in tensorflow.
bool IsSame(string op1, string op2) {
  if (op1 == "*") return true;

  std::vector<string> op1_list = str_util::Split(op1, '|');
  for (const string& op_1 : op1_list) {
    if (op_1 == op2) return true;
  }

  return false;
}

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
    auto graph_children = node_view->GetRegularFanins();
    int num_children = graph_children.size();
    if (num_children != pattern.children.size()) {
      return false;
    } else {
      // A pattern is a graph that we would like to match with a subgraph of
      // a tensorflow computation graph. We travese both pattern-graph and the
      // given graph in DFS manner and try to find one-to-one mapping between
      // the nodes. However, commutative binary ops (e.g., Add, AddV2, Mul
      // etc.) in the computation graph can have their inputs in different order
      // than the pattern syntax graph. To allow such input permutation in a
      // limited manner, we employ a heuristic of looking one level ahead in
      // both graphs, whether visiting the right child of pattern is likely to
      // match left child of the given graph. In that case, we simply swap the
      // left subtree with right subtree of pattern syntax graph and continue
      // matching children of pattern with the children of given computation
      // graph. Note, we do not change anything in the computation graph during
      // pattern matching, only the pattern graph is changed. By looking ahead
      // one step in case of commutative ops, we keep the time comlexity of
      // pattern matching linear. Since it is only a heuristic and we look only
      // one step ahead it is not guranteed that all possible permutations will
      // be matched. For example, when both the input ops to the commutative op
      // are same, we cannot anticipate which of the permutation is likely to
      // match unless we look two level down the graphs.
      std::vector<int> pattern_child_indices(num_children);
      std::iota(pattern_child_indices.begin(), pattern_child_indices.end(), 0);
      string op_name = pattern.op;
      if (IsCommutativeOp(op_name) && num_children == 2) {
        MutableNodeView* graph_child0_node_view =
            graph_view_->GetNode(graph_children[0].node_index());
        MutableNodeView* graph_child1_node_view =
            graph_view_->GetNode(graph_children[1].node_index());
        if ((!IsSame(pattern.children[0].op, graph_child0_node_view->GetOp()) &&
             IsSame(pattern.children[1].op, graph_child0_node_view->GetOp())) ||
            (!IsSame(pattern.children[1].op, graph_child1_node_view->GetOp()) &&
             IsSame(pattern.children[0].op, graph_child1_node_view->GetOp())))
          std::swap(pattern_child_indices[0], pattern_child_indices[1]);
      }
      for (int i = 0; i < num_children; ++i) {
        auto child_node_index = graph_children[i].node_index();
        // TODO (mdfaijul): Is it guaranted that GetNode will reuturn non null
        // pointer.
        MutableNodeView* child_node_view =
            graph_view_->GetNode(child_node_index);
        const OpTypePattern& child_pattern =
            pattern.children[pattern_child_indices[i]];
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
    const OpTypePattern& pattern,
    const std::unordered_set<string>& nodes_to_preserve,
    MutableNodeView* node_view, std::map<string, int>* matched_nodes_map,
    std::set<int>* remove_node_indices) {
  bool found_match = false;
  match_.reset(new NodeViewMatch());
  if (DoesOpTypePatternMatch(pattern, node_view, match_.get())) {
    if (IsSafeNodesToRemove(nodes_to_preserve)) {
      found_match = true;
      *matched_nodes_map = this->node_label_to_index_;
      *remove_node_indices = this->remove_node_indices_;
    }
  } else {
    found_match = false;
  }

  // Clear all bookkeeping data
  match_->Clear();
  match_.reset(nullptr);
  matched_node_indices_.clear();
  node_label_to_index_.clear();
  remove_node_indices_.clear();

  return found_match;
}

}  // namespace utils
}  // namespace grappler
}  // namespace tensorflow
