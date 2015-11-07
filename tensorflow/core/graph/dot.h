#ifndef TENSORFLOW_GRAPH_DOT_H_
#define TENSORFLOW_GRAPH_DOT_H_

#include <functional>
#include <string>
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

class Edge;
class Graph;
class Node;

struct DotOptions {
  bool (*include_node_function)(const Node*) = nullptr;

  // By default, all nodes with the same name prefix are collapsed into
  // a single node in the dot graph.  This regexp can be changed so that
  // only prefixes that match the regexp are collapsed in this fashion.
  // 'all' collapses all ops with prefixes, 'none' disables all collapsing.
  string prefix_collapse_regexp = "all";

  // A function that returns a label to embed into the per-node display.
  std::function<string(const Node*)> node_label;

  // A function that returns a label to attach to an edge.
  std::function<string(const Edge*)> edge_label;

  // A function that returns the "cost" of the node.  The dot display
  // makes a node size proportional to its cost.
  std::function<double(const Node*)> node_cost;

  // A function that returns the "cost" of the edge.  The dot display
  // makes a edge thickness proportional to its cost.
  std::function<double(const Edge*)> edge_cost;
};

// Return a string that contains a graphviz specification of the graph.
string DotGraph(const Graph& g, const DotOptions& opts);

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_DOT_H_
