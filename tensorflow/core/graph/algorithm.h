#ifndef TENSORFLOW_GRAPH_ALGORITHM_H_
#define TENSORFLOW_GRAPH_ALGORITHM_H_

#include <functional>
#include <unordered_set>

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// Perform a depth-first-search on g starting at the source node.
// If enter is not empty, calls enter(n) before visiting any children of n.
// If leave is not empty, calls leave(n) after visiting all children of n.
extern void DFS(const Graph& g, std::function<void(Node*)> enter,
                std::function<void(Node*)> leave);

// Stores in *order the post-order numbering of all nodes
// in graph found via a depth first search starting at the source node.
//
// Note that this is equivalent to topological sorting when the
// graph does not have cycles.
//
// REQUIRES: order is not NULL.
void GetPostOrder(const Graph& g, std::vector<Node*>* order);

// Stores in *order the reverse post-order numbering of all nodes
void GetReversePostOrder(const Graph& g, std::vector<Node*>* order);

// Prune nodes in "g" that are not in some path from the source node
// to any node in 'nodes'.
void PruneForReverseReachability(Graph* g,
                                 const std::unordered_set<const Node*>& nodes);

// Connect all nodes with no incoming edges to source.
// Connect all nodes with no outgoing edges to sink.
void FixupSourceAndSinkEdges(Graph* g);

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_ALGORITHM_H_
