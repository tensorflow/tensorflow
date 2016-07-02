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

void DFS(const Graph& g, std::function<void(Node*)> enter,
         std::function<void(Node*)> leave) {
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

    // Arrange to work on descendants.
    for (Node* out : n->out_nodes()) {
      if (!visited[out->id()]) {
        // Note; we must not mark as visited until we actually process it.
        stack.push_back(Work{out, false});
      }
    }
  }
}

void ReverseDFS(const Graph& g, std::function<void(Node*)> enter,
                std::function<void(Node*)> leave) {
  // Stack of work to do.
  struct Work {
    Node* node;
    bool leave;  // Are we entering or leaving n?
  };
  std::vector<Work> stack;
  stack.push_back(Work{g.sink_node(), false});

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

    // Arrange to work on parents.
    for (Node* in : n->in_nodes()) {
      if (!visited[in->id()]) {
        // Note; we must not mark as visited until we actually process it.
        stack.push_back(Work{in, false});
      }
    }
  }
}

void GetPostOrder(const Graph& g, std::vector<Node*>* order) {
  order->clear();
  DFS(g, nullptr, [order](Node* n) { order->push_back(n); });
}

void GetReversePostOrder(const Graph& g, std::vector<Node*>* order) {
  GetPostOrder(g, order);
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
