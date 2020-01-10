#include "tensorflow/core/graph/algorithm.h"

#include <algorithm>
#include <deque>
#include <vector>

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

void GetPostOrder(const Graph& g, std::vector<Node*>* order) {
  order->clear();
  DFS(g, nullptr, [order](Node* n) { order->push_back(n); });
}

void GetReversePostOrder(const Graph& g, std::vector<Node*>* order) {
  GetPostOrder(g, order);
  std::reverse(order->begin(), order->end());
}

void PruneForReverseReachability(Graph* g,
                                 const std::unordered_set<const Node*>& nodes) {
  std::unordered_set<const Node*> visited;

  // Compute set of nodes that we need to traverse in order to reach
  // the nodes in "nodes" by performing a breadth-first search from those
  // nodes, and accumulating the visited nodes.
  std::deque<const Node*> queue;
  for (const Node* n : nodes) {
    queue.push_back(n);
  }
  while (!queue.empty()) {
    const Node* n = queue.front();
    queue.pop_front();
    if (visited.insert(n).second) {
      for (const Node* in : n->in_nodes()) {
        queue.push_back(in);
      }
    }
  }

  // Make a pass over the graph to remove nodes not in "visited"
  std::vector<Node*> all_nodes;
  for (Node* n : g->nodes()) {
    all_nodes.push_back(n);
  }

  for (Node* n : all_nodes) {
    if (visited.count(n) == 0 && !n->IsSource() && !n->IsSink()) {
      g->RemoveNode(n);
    }
  }

  // Reconnect nodes with no outgoing edges to the sink node
  FixupSourceAndSinkEdges(g);
}

void FixupSourceAndSinkEdges(Graph* g) {
  // Connect all nodes with no incoming edges to source.
  // Connect all nodes with no outgoing edges to sink.
  for (Node* n : g->nodes()) {
    if (!n->IsSource() && n->in_edges().empty()) {
      g->AddControlEdge(g->source_node(), n);
    }
    if (!n->IsSink() && n->out_edges().empty()) {
      g->AddControlEdge(n, g->sink_node());
    }
  }
}

}  // namespace tensorflow
