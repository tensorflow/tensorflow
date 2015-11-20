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

#include "tensorflow/core/graph/dot.h"

#include <map>
#include <unordered_map>
#include <unordered_set>

#include "tensorflow/core/graph/colors.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

static string GraphNodeName(const DotOptions& opts, const Node* n) {
  return strings::StrCat("N", n->id());
}

bool ShoulDisplayOpType(const Node* n) {
  if (n->type_string() == "NoOp") {
    return false;
  }
  const string& op_name = n->def().name();
  if (op_name.find(n->type_string() + "_") == 0) {
    return false;
  }
  return true;
}

string DotGraph(const Graph& g, const DotOptions& opts) {
  RegexpStringPiece flag(opts.prefix_collapse_regexp);
  if (flag == "all") {
    flag = ".";
  } else if (flag == "none") {
    flag = "^$";
  }
  RE2 cluster_name_pattern(flag);
  string result;
  strings::StrAppend(&result, "digraph G {\n");
  strings::StrAppend(&result, "rankdir=\"BT\"\n");

  std::map<string, int> device_index;       // Map from device name to index.
  std::unordered_set<Node*> visible_nodes;  // Nodes to display.
  // Cluster name => set of nodes.
  std::unordered_map<string, std::unordered_set<Node*> > clusters;
  // Node* => Cluster
  std::unordered_map<Node*, string> node_cluster;
  for (Node* src : g.nodes()) {
    if (opts.include_node_function != nullptr &&
        !opts.include_node_function(src)) {
      continue;
    }
    // Do not display source and sink nodes
    if (src->IsSource() || src->IsSink()) {
      continue;
    }
    visible_nodes.insert(src);
    const string name_prefix = NodeNamePrefix(src->def().name()).ToString();
    if (!name_prefix.empty()) {
      clusters[name_prefix].insert(src);
      node_cluster[src] = name_prefix;
    }
    // Record device if present.
    if (src->IsOp()) {
      const string& d = src->assigned_device_name();
      if (!d.empty()) {
        device_index[d] = -1;  // Assigned later
      }
    }
  }

  // Add nodes whose name is exactly a cluster name to the cluster itself.
  for (Node* src : g.nodes()) {
    if (node_cluster.count(src) == 0) {
      const string name = src->def().name();
      auto it = clusters.find(name);
      if (it != clusters.end()) {
        it->second.insert(src);
        node_cluster[src] = name;
      }
    }
  }

  auto node_in_collapsed_cluster = [&node_cluster,
                                    &cluster_name_pattern](Node* n) {
    return node_cluster.count(n) > 0 &&
           RE2::PartialMatch(node_cluster[n], cluster_name_pattern);
  };

  // Assign device indices in sorted order.
  int num = 0;
  for (auto& e : device_index) {
    e.second = num++;
  }

  double total_node_cost = 0;
  double avg_node_cost = 1;
  if (opts.node_cost) {
    int node_count = 0;
    for (const Node* n : g.nodes()) {
      total_node_cost += opts.node_cost(n);
      ++node_count;
    }
    if (total_node_cost > 0) avg_node_cost = total_node_cost / node_count;
  }

  for (Node* src : g.nodes()) {
    if (visible_nodes.count(src) == 0 || node_in_collapsed_cluster(src)) {
      continue;
    }
    string label = src->name();
    if (ShoulDisplayOpType(src)) {
      // Append the op type if it is not directly deducible from the op name.
      strings::StrAppend(&label, "\\n(", src->type_string(), ")");
    }
    const char* shape = "box";
    const char* color = nullptr;
    if (src->IsSource()) {
      shape = "oval";
    } else if (src->IsSink()) {
      shape = "oval";
    } else {
      const string& d = src->assigned_device_name();
      const int dindex = (!d.empty()) ? device_index[d] : -1;
      if (dindex >= 0) {
        color = ColorFor(dindex);
      }

      shape = "box";
    }

    if (opts.node_label) {
      string extra = opts.node_label(src);
      if (!extra.empty()) {
        strings::StrAppend(&label, "\\n", extra);
      }
    }

    strings::StrAppend(&result, GraphNodeName(opts, src), "[shape=", shape,
                       ", label=\"", label, "\"");
    if (opts.node_cost && total_node_cost > 0) {
      // Pick fontsize in range [8..40] so that area is proportional to cost.
      const double cost = opts.node_cost(src);
      const double relcost = fabs(cost / avg_node_cost);
      // Average cost node has font size of 12.
      const int fs = 8 + static_cast<int>(4.0 * std::min(sqrt(relcost), 8.0));
      strings::StrAppend(&result, ", width=0, height=0, fontsize=", fs);
      VLOG(2) << "Node: " << cost << " => " << relcost << " => " << fs;
    }
    if (color != nullptr) {
      strings::StrAppend(&result, ", fillcolor=\"", color,
                         "\", fontcolor=\"white\", style=\"filled\"");
    }
    strings::StrAppend(&result, "]\n");
  }

  for (auto c : clusters) {
    const string& cluster_name = c.first;
    const std::unordered_set<Node*> nodes = c.second;
    std::unordered_map<string, int> node_colors;
    for (auto n : nodes) {
      const string& d = n->assigned_device_name();
      const int dindex = (!d.empty()) ? device_index[d] : -1;
      if (dindex >= 0) {
        ++node_colors[ColorFor(dindex)];
      }
    }

    string majority_color;
    if (node_colors.empty()) {
      majority_color = ColorFor(0);
    } else {
      majority_color = std::max_element(node_colors.begin(), node_colors.end(),
                                        [](const std::pair<string, int>& x,
                                           const std::pair<string, int>& y) {
                                          return x.second < y.second;
                                        })
                           ->first;
    }

    if (!RE2::PartialMatch(cluster_name, cluster_name_pattern)) {
      strings::StrAppend(&result, "subgraph cluster_", cluster_name, "{\n");
      for (auto n : nodes) {
        strings::StrAppend(&result, GraphNodeName(opts, n), ";\n");
      }
      strings::StrAppend(&result, "}\n");
    } else {
      strings::StrAppend(&result, cluster_name, " [shape=oval, fillcolor=\"",
                         majority_color, "\", label=\"", cluster_name,
                         "\", style=\"filled\", fontcolor=\"white\"]\n");
    }
  }

  std::unordered_set<string> edge_drawn;

  double max_edge_cost = 0;
  double total_edge_cost = 0;
  double avg_edge_cost = 1;
  if (opts.edge_cost && g.edges().size()) {
    for (const Edge* e : g.edges()) {
      auto cost = opts.edge_cost(e);
      total_edge_cost += cost;
      max_edge_cost = std::max(max_edge_cost, cost);
    }
    avg_edge_cost = total_edge_cost / g.edges().size();
  }
  VLOG(2) << "Edge cost tot/max/avg: " << total_edge_cost << "/"
          << max_edge_cost << "/" << avg_edge_cost;

  for (const Edge* e : g.edges()) {
    Node* src = e->src();
    Node* dst = e->dst();
    // If either endpoint isn't drawn in the graph, don't draw the edge
    if (visible_nodes.count(src) == 0 || visible_nodes.count(dst) == 0) {
      continue;
    }

    const string src_name = node_in_collapsed_cluster(src)
                                ? node_cluster[src]
                                : GraphNodeName(opts, src);
    const string dst_name = node_in_collapsed_cluster(dst)
                                ? node_cluster[dst]
                                : GraphNodeName(opts, dst);
    // Don't draw self edges
    if (src_name == dst_name) {
      continue;
    }
    // And previously drawn edges.
    const string& edge_name = strings::StrCat(src_name, ":", dst_name);
    if (edge_drawn.count(edge_name) > 0) {
      continue;
    }
    edge_drawn.insert(edge_name);

    strings::StrAppend(&result, src_name, " -> ", dst_name, "[");
    string label;
    if (e->IsControlEdge()) {
      strings::StrAppend(&result, " style=dotted");
    }
    if (opts.edge_label) {
      string label = opts.edge_label(e);
      if (!label.empty()) {
        strings::StrAppend(&result, " label=<", label, ">");
      }
    }
    // Make edge widths proportional to amount of data transferred.
    if (opts.edge_cost && max_edge_cost > 0) {
      const double cost = opts.edge_cost(e);
      const double relcost = fabs(cost / avg_edge_cost);
      // Pick penwidth in range [1..6] so that width is proportional to cost.
      const int pw = 1 + std::min(5, static_cast<int>(2.0 * relcost));
      strings::StrAppend(&result, " penwidth=", pw);
      // Use weight attributes [1..100] to keep heavier edges more vertical.
      const int weight = 1 + std::min(99, static_cast<int>(100.0 * relcost));
      strings::StrAppend(&result, " weight=", weight);
      VLOG(2) << "Edge: " << cost << " => " << relcost << " => " << pw << "/"
              << weight;
    }

    strings::StrAppend(&result, "]\n");
  }
  // Compute some statistics
  int op_nodes = 0;
  for (Node* n : g.nodes()) {
    if (n->IsOp()) {
      op_nodes++;
    }
  }

  // Emit legend
  strings::StrAppend(&result,
                     "{ rank = source; Legend [shape=box, margin=0, label=<",
                     "<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" ",
                     "CELLPADDING=\"4\">", "<TR><TD COLSPAN=\"2\">op_nodes: ",
                     op_nodes, "</TD></TR>\n");
  for (const auto& e : device_index) {
    const int dindex = e.second;
    strings::StrAppend(&result, "<TR><TD BGCOLOR=\"", ColorFor(dindex),
                       "\"><FONT COLOR=\"white\">", dindex, "</FONT></TD><TD>",
                       e.first, "</TD></TR>\n");
  }
  strings::StrAppend(&result, "</TABLE>>]}\n");

  strings::StrAppend(&result, "}\n");  // End digraph
  return result;
}

}  // namespace tensorflow
