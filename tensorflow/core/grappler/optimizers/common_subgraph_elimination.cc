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

#include "tensorflow/core/grappler/optimizers/common_subgraph_elimination.h"

#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/canonicalizer.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/grappler/utils/traversal.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/hash.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace grappler {
class Cluster;
}  // namespace grappler
}  // namespace tensorflow

using tensorflow::strings::StrCat;

namespace tensorflow {
namespace grappler {

class UniqueNodes {
 public:
  NodeDef* FindOrAddRepresentative(NodeDef* node) {
    uint64 sig = ComputeSignature(*node);
    std::vector<NodeDef*>& candidates = rep_[sig];
    for (auto& candidate : candidates) {
      if ((candidate == node) || SameNode(*candidate, *node)) {
        return candidate;
      }
    }
    candidates.push_back(node);
    return node;
  }

  void RemoveRepresentative(NodeDef* node) {
    auto it = memoized_signatures_.find(node);
    if (it == memoized_signatures_.end()) return;

    std::vector<NodeDef*>& candidates = rep_[it->second];
    for (int i = 0, end = candidates.size(); i < end; ++i) {
      if (candidates[i] == node) {
        std::swap(candidates[i], candidates[candidates.size() - 1]);
        candidates.resize(candidates.size() - 1);
        break;
      }
    }
    memoized_signatures_.erase(node);
  }

 private:
  uint64 ComputeSignature(const NodeDef& node);
  bool SameNode(const NodeDef& node1, const NodeDef& node2) const;

  absl::flat_hash_map<uint64, std::vector<NodeDef*>> rep_;
  absl::flat_hash_map<const NodeDef*, uint64> memoized_signatures_;
};

uint64 UniqueNodes::ComputeSignature(const NodeDef& node) {
  auto it = memoized_signatures_.find(&node);
  if (it != memoized_signatures_.end()) return it->second;

  uint64 h = Hash64(node.op());
  h = Hash64Combine(Hash64(node.device()), h);

  for (const auto& input : node.input()) {
    const TensorId input_tensor = ParseTensorName(input);
    uint64 input_hash = Hash64Combine(
        Hash64(input_tensor.node().data(), input_tensor.node().size()),
        std::hash<int>()(input_tensor.index()));
    h = Hash64CombineUnordered(input_hash, h);
  }
  for (const auto& attr : node.attr()) {
    uint64 attr_hash =
        Hash64Combine(Hash64(attr.first), FastAttrValueHash(attr.second));
    h = Hash64CombineUnordered(attr_hash, h);
  }
  memoized_signatures_.emplace(&node, h);
  return h;
}

// PRECONDITION:
//  Node input orders are assumed to be canonicalized, i.e. control inputs for
//  all nodes as well as regular inputs for commutative nodes must be sorted.
bool UniqueNodes::SameNode(const NodeDef& node1, const NodeDef& node2) const {
  if (node1.op() != node2.op()) {
    return false;
  }
  if (node1.device() != node2.device()) {
    return false;
  }
  if (node1.input_size() != node2.input_size()) {
    return false;
  }
  if (node1.attr_size() != node2.attr_size()) {
    return false;
  }

  // Compare inputs.
  auto it1 = node1.input().begin();
  auto it2 = node2.input().begin();
  for (; it1 != node1.input().end(); ++it1, ++it2) {
    if (*it1 != *it2) return false;
  }

  // Compare attributes.
  for (const auto& attr1 : node1.attr()) {
    auto it = node2.attr().find(attr1.first);
    if (it == node2.attr().end()) return false;
    if (!FastAreAttrValuesEqual(attr1.second, it->second)) return false;
  }

  return true;
}

bool CommonSubgraphElimination::CanDedup(const NodeDef& node) const {
  if (nodes_to_preserve_.find(node.name()) != nodes_to_preserve_.end()) {
    return false;
  }
  if (IsEnter(node) || IsExit(node)) {
    return false;
  }
  if (node.device().find("SPU") != string::npos) {
    return false;
  }
  // Workaround for Assert and Print mistakenly being labeled as stateful.
  if (IsAssert(node) || IsPrint(node)) {
    return true;
  }
  return IsFreeOfSideEffect(node);
}

Status CommonSubgraphElimination::DedupComputations(GraphDef* optimized_graph) {
  CanonicalizeGraph(optimized_graph);

  GraphTopologyView graph_view;
  if (!graph_view.InitializeFromGraph(*optimized_graph).ok()) {
    LOG(WARNING) << "Failed to initialize GraphTopologyView.";
    return Status::OK();
  }

  // If either node or rep feeds an inplace op, deduping them may cause data
  // races. For example: If we dedup nodes initializing two independent
  // inplace accumulations, they will write to the same buffer, clobbering
  // each other's results.
  absl::flat_hash_set<const NodeDef*> feeds_inplace_op;
  for (int i = 0; i < optimized_graph->node_size(); ++i) {
    const NodeDef& root = optimized_graph->node(i);
    if (feeds_inplace_op.find(&root) != feeds_inplace_op.end()) continue;
    if (ModifiesInputsInPlace(root)) {
      const auto is_continue_traversal = [&](const NodeDef* node) -> bool {
        return node->op() == root.op() || !NeverForwardsInputs(*node);
      };

      DfsTraversal(graph_view, {&root}, TraversalDirection::kFollowInputs,
                   DfsPredicates::Advance(is_continue_traversal),
                   DfsCallbacks::PreOrder([&](const NodeDef* node) {
                     feeds_inplace_op.insert(node);
                   }));
    }
  }

  std::vector<bool> can_dedup(optimized_graph->node_size());
  for (int i = 0; i < optimized_graph->node_size(); ++i) {
    const NodeDef& node = optimized_graph->node(i);
    can_dedup[i] = (feeds_inplace_op.find(&node) == feeds_inplace_op.end()) &&
                   CanDedup(node);
  }

  bool stop = true;
  std::set<int> duplicates;
  UniqueNodes nodes;
  NodeMap node_map(optimized_graph);
  do {
    stop = true;
    for (int i = 0; i < optimized_graph->node_size(); ++i) {
      if (!can_dedup[i] || duplicates.find(i) != duplicates.end()) {
        continue;
      }
      NodeDef* node = optimized_graph->mutable_node(i);
      NodeDef* rep = nodes.FindOrAddRepresentative(node);
      if (rep == node) {
        continue;
      }
      // Make a copy since we mutate the set below.
      const auto fanouts = node_map.GetOutputs(node->name());
      for (NodeDef* fanout : fanouts) {
        // Update consumers of node.
        bool updated_fanout = false;
        for (int i = 0; i < fanout->input_size(); ++i) {
          string* fanout_input = fanout->mutable_input(i);

          const int position =
              NodePositionIfSameNode(*fanout_input, node->name());
          // Update name in-place.
          if (position < -1) {
            continue;
          } else {
            if (!updated_fanout) {
              // The signature of the fanout node will change. Remove it from
              // nodes.
              nodes.RemoveRepresentative(fanout);
            }
            updated_fanout = true;
            if (position > 0) {
              *fanout_input = StrCat(rep->name(), ":", position);
            } else if (position == 0) {
              *fanout_input = rep->name();
            } else {
              *fanout_input = StrCat("^", rep->name());
            }
          }
        }
        if (updated_fanout) {
          node_map.UpdateInput(fanout->name(), node->name(), rep->name());
          CanonicalizeNode(fanout);
        }
      }
      if (fetch_nodes_known_) {
        node->Clear();
      }
      duplicates.insert(i);
      stop = false;
    }
  } while (!stop);

  // Delete duplicates
  if (fetch_nodes_known_ && !duplicates.empty()) {
    EraseNodesFromGraph(duplicates, optimized_graph);
  }

  return Status::OK();
}

Status CommonSubgraphElimination::Optimize(Cluster* /*cluster*/,
                                           const GrapplerItem& item,
                                           GraphDef* optimized_graph) {
  // Set up helper data structures.
  nodes_to_preserve_ = item.NodesToPreserve();
  fetch_nodes_known_ = !item.fetch.empty();
  *optimized_graph = item.graph;

  // Perform topological sort on the graph in order to help DedupComputations
  // optimize larger subgraphs starting from the roots with more inputs.
  TF_RETURN_IF_ERROR(TopologicalSort(optimized_graph));
  GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();

  return DedupComputations(optimized_graph);
}

void CommonSubgraphElimination::Feedback(Cluster* /*cluster*/,
                                         const GrapplerItem& /*item*/,
                                         const GraphDef& /*optimized_graph*/,
                                         double /*result*/) {
  // Nothing to do for ArithmeticOptimizer.
}

}  // namespace grappler
}  // namespace tensorflow
