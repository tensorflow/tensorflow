/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_GRAPH_DOT_H_
#define TENSORFLOW_GRAPH_DOT_H_

#include <functional>
#include <string>
#include "tensorflow/core/platform/types.h"

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
