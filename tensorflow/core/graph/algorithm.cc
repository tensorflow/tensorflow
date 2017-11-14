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

#include "tensorflow/core/graph/algorithm.h"

#include <algorithm>
#include <deque>
#include <vector>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

void DFS(const Graph& g, const std::function<void(Node*)>& enter,
         const std::function<void(Node*)>& leave,
         const NodeComparator& stable_comparator) {
  // Stack of work to do.
  struct Work {
    Node* node;
    bool leave;  // Are we entering or leaving n?
  };
  std::vector<Work> stack;
  stack.push_back(Work{g.source_node(), false});

  std::vector<bool> visited(g.num_node_ids(), false);
  while (!stack.empty()) {
    Work w = stack.back();
    stack.pop_back();

    Node* n = w.node;
    if (w.leave) {
      leave(n);
      continue;
    }

    if (visited[n->id()]) continue;
    visited[n->id()] = true;
    if (enter) enter(n);

    // Arrange to call leave(n) when all done with descendants.
    if (leave) stack.push_back(Work{n, true});

    gtl::iterator_range<NeighborIter> nodes = n->out_nodes();
    auto add_work = [&visited, &stack](Node* out) {
      if (!visited[out->id()]) {
        // Note; we must not mark as visited until we actually process it.
        stack.push_back(Work{out, false});
      }
    };

    if (stable_comparator) {
      std::vector<Node*> nodes_sorted;
      for (Node* out : nodes) {
        nodes_sorted.emplace_back(out);
      }
      std::sort(nodes_sorted.begin(), nodes_sorted.end(), stable_comparator);
      for (Node* out : nodes_sorted) {
        add_work(out);
      }
    } else {
      for (Node* out : nodes) {
        add_work(out);
      }
    }
  }
}

void ReverseDFS(const Graph& g, const std::function<void(Node*)>& enter,
                const std::function<void(Node*)>& leave,
                const NodeComparator& stable_comparator) {
  ReverseDFSFrom(g, {g.sink_node()}, enter, leave, stable_comparator);
}

void ReverseDFSFrom(const Graph& g, gtl::ArraySlice<Node*> start,
                    const std::function<void(Node*)>& enter,
                    const std::function<void(Node*)>& leave,
                    const NodeComparator& stable_comparator) {
  // Stack of work to do.
  struct Work {
    Node* node;
    bool leave;  // Are we entering or leaving n?
  };
  std::vector<Work> stack(start.size());
  for (int i = 0; i < start.size(); ++i) {
    stack[i] = Work{start[i], false};
  }

  std::vector<bool> visited(g.num_node_ids(), false);
  while (!stack.empty()) {
    Work w = stack.back();
    stack.pop_back();

    Node* n = w.node;
    if (w.leave) {
      leave(n);
      continue;
    }

    if (visited[n->id()]) continue;
    visited[n->id()] = true;
    if (enter) enter(n);

    // Arrange to call leave(n) when all done with descendants.
    if (leave) stack.push_back(Work{n, true});

    gtl::iterator_range<NeighborIter> nodes = n->in_nodes();

    auto add_work = [&visited, &stack](Node* out) {
      if (!visited[out->id()]) {
        // Note; we must not mark as visited until we actually process it.
        stack.push_back(Work{out, false});
      }
    };

    if (stable_comparator) {
      std::vector<Node*> nodes_sorted;
      for (Node* in : nodes) {
        nodes_sorted.emplace_back(in);
      }
      std::sort(nodes_sorted.begin(), nodes_sorted.end(), stable_comparator);
      for (Node* in : nodes_sorted) {
        add_work(in);
      }
    } else {
      for (Node* in : nodes) {
        add_work(in);
      }
    }
  }
}

void GetPostOrder(const Graph& g, std::vector<Node*>* order,
                  const NodeComparator& stable_comparator) {
  order->clear();
  DFS(g, nullptr, [order](Node* n) { order->push_back(n); }, stable_comparator);
}

void GetReversePostOrder(const Graph& g, std::vector<Node*>* order,
                         const NodeComparator& stable_comparator) {
  GetPostOrder(g, order, stable_comparator);
  std::reverse(order->begin(), order->end());
}

bool PruneForReverseReachability(Graph* g,
                                 std::unordered_set<const Node*> visited) {
  // Compute set of nodes that we need to traverse in order to reach
  // the nodes in "nodes" by performing a breadth-first search from those
  // nodes, and accumulating the visited nodes.
  std::deque<const Node*> queue;
  for (const Node* n : visited) {
    VLOG(2) << "Reverse reach init: " << n->name();
    queue.push_back(n);
  }
  while (!queue.empty()) {
    const Node* n = queue.front();
    queue.pop_front();
    for (const Node* in : n->in_nodes()) {
      if (visited.insert(in).second) {
        queue.push_back(in);
        VLOG(2) << "Reverse reach : " << n->name() << " from " << in->name();
      }
    }
  }

  // Make a pass over the graph to remove nodes not in "visited"
  std::vector<Node*> all_nodes;
  all_nodes.reserve(g->num_nodes());
  for (Node* n : g->nodes()) {
    all_nodes.push_back(n);
  }

  bool any_removed = false;
  for (Node* n : all_nodes) {
    if (visited.count(n) == 0 && !n->IsSource() && !n->IsSink()) {
      g->RemoveNode(n);
      any_removed = true;
    }
  }

  return any_removed;
}

bool FixupSourceAndSinkEdges(Graph* g) {
  // Connect all nodes with no incoming edges to source.
  // Connect all nodes with no outgoing edges to sink.
  bool changed = false;
  for (Node* n : g->nodes()) {
    if (!n->IsSource() && n->in_edges().empty()) {
      g->AddControlEdge(g->source_node(), n);
      changed = true;
    }
    if (!n->IsSink() && n->out_edges().empty()) {
      g->AddControlEdge(n, g->sink_node());
      changed = true;
    }
  }
  return changed;
}

}  // namespace tensorflow
