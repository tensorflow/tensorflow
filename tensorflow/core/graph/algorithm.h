/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPH_ALGORITHM_H_
#define TENSORFLOW_CORE_GRAPH_ALGORITHM_H_

#include <functional>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

// Comparator for two nodes. This is used in order to get a stable ording.
using NodeComparator = std::function<bool(const Node*, const Node*)>;

using EdgeFilter = std::function<bool(const Edge&)>;

// Compares two node based on their ids.
struct NodeComparatorID {
  bool operator()(const Node* n1, const Node* n2) const {
    return n1->id() < n2->id();
  }
};

// Compare two nodes based on their names.
struct NodeComparatorName {
  bool operator()(const Node* n1, const Node* n2) const {
    return n1->name() < n2->name();
  }
};

// Perform a depth-first-search on g starting at the source node.
// If enter is not empty, calls enter(n) before visiting any children of n.
// If leave is not empty, calls leave(n) after visiting all children of n.
// If stable_comparator is set, a stable ordering of visit is achieved by
// sorting a node's neighbors first before visiting them.
// If edge_filter is set then ignores edges for which edge_filter returns false.
extern void DFS(const Graph& g, const std::function<void(Node*)>& enter,
                const std::function<void(Node*)>& leave,
                const NodeComparator& stable_comparator = {},
                const EdgeFilter& edge_filter = {});

// Perform a depth-first-search on g starting at the 'start' nodes.
// If enter is not empty, calls enter(n) before visiting any children of n.
// If leave is not empty, calls leave(n) after visiting all children of n.
// If stable_comparator is set, a stable ordering of visit is achieved by
// sorting a node's neighbors first before visiting them.
// If edge_filter is set then ignores edges for which edge_filter returns false.
extern void DFSFrom(const Graph& g, gtl::ArraySlice<Node*> start,
                    const std::function<void(Node*)>& enter,
                    const std::function<void(Node*)>& leave,
                    const NodeComparator& stable_comparator = {},
                    const EdgeFilter& edge_filter = {});
extern void DFSFrom(const Graph& g, gtl::ArraySlice<const Node*> start,
                    const std::function<void(const Node*)>& enter,
                    const std::function<void(const Node*)>& leave,
                    const NodeComparator& stable_comparator = {},
                    const EdgeFilter& edge_filter = {});

// Perform a reverse depth-first-search on g starting at the sink node.
// If enter is not empty, calls enter(n) before visiting any parents of n.
// If leave is not empty, calls leave(n) after visiting all parents of n.
// If stable_comparator is set, a stable ordering of visit is achieved by
// sorting a node's neighbors first before visiting them.
// If edge_filter is set then ignores edges for which edge_filter returns false.
extern void ReverseDFS(const Graph& g, const std::function<void(Node*)>& enter,
                       const std::function<void(Node*)>& leave,
                       const NodeComparator& stable_comparator = {},
                       const EdgeFilter& edge_filter = {});

// Perform a reverse depth-first-search on g starting at the 'start' nodes.
// If enter is not empty, calls enter(n) before visiting any parents of n.
// If leave is not empty, calls leave(n) after visiting all parents of n.
// If stable_comparator is set, a stable ordering of visit is achieved by
// sorting a node's neighbors first before visiting them.
// If edge_filter is set then ignores edges for which edge_filter returns false.
extern void ReverseDFSFrom(const Graph& g, gtl::ArraySlice<Node*> start,
                           const std::function<void(Node*)>& enter,
                           const std::function<void(Node*)>& leave,
                           const NodeComparator& stable_comparator = {},
                           const EdgeFilter& edge_filter = {});
extern void ReverseDFSFrom(const Graph& g, gtl::ArraySlice<const Node*> start,
                           const std::function<void(const Node*)>& enter,
                           const std::function<void(const Node*)>& leave,
                           const NodeComparator& stable_comparator = {},
                           const EdgeFilter& edge_filter = {});

// Stores in *order the post-order numbering of all nodes
// in graph found via a depth first search starting at the source node.
//
// Note that this is equivalent to reverse topological sorting when the
// graph does not have cycles.
//
// If stable_comparator is set, a stable ordering of visit is achieved by
// sorting a node's neighbors first before visiting them.
//
// If edge_filter is set then ignores edges for which edge_filter returns false.
//
// REQUIRES: order is not NULL.
void GetPostOrder(const Graph& g, std::vector<Node*>* order,
                  const NodeComparator& stable_comparator = {},
                  const EdgeFilter& edge_filter = {});

// Stores in *order the reverse post-order numbering of all nodes
// If stable_comparator is set, a stable ordering of visit is achieved by
// sorting a node's neighbors first before visiting them.
//
// If edge_filter is set then ignores edges for which edge_filter returns false.
void GetReversePostOrder(const Graph& g, std::vector<Node*>* order,
                         const NodeComparator& stable_comparator = {},
                         const EdgeFilter& edge_filter = {});

// Prune nodes in "g" that are not in some path from the source node
// to any node in 'nodes'. Returns true if changes were made to the graph.
// Does not fix up source and sink edges.
bool PruneForReverseReachability(Graph* g,
                                 std::unordered_set<const Node*> nodes);

// Connect all nodes with no incoming edges to source.
// Connect all nodes with no outgoing edges to sink.
//
// Returns true if and only if 'g' is mutated.
bool FixupSourceAndSinkEdges(Graph* g);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_ALGORITHM_H_
