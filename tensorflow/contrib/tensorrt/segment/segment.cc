/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/tensorrt/segment/segment.h"

#include <set>
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/tensorrt/segment/union_find.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tensorrt {
namespace segment {
using ::tensorflow::strings::StrAppend;
namespace {

bool check_cycles(const Graph* g, const Node* src,
                  const std::vector<Node*>& start) {
  struct Work {
    Node* node;
    bool leave;  // Are we entering or leaving n?
  };

  std::vector<Work> stack(start.size());
  for (int i = 0; i < start.size(); ++i) {
    stack[i] = Work{start[i], false};
  }

  std::vector<bool> visited(g->num_node_ids(), false);
  while (!stack.empty()) {
    Work w = stack.back();
    stack.pop_back();

    auto n = w.node;
    if (w.leave) {
      if (n == src) {
        return true;
      }
      continue;
    }

    if (visited[n->id()]) continue;
    visited[n->id()] = true;
    // Arrange to call leave(n) when all done with descendants.
    stack.push_back(Work{n, true});

    auto nodes = n->in_nodes();
    for (const auto node : nodes) {
      if (!visited[node->id()]) {
        stack.push_back(Work{node, false});
      }
    }
  }
  return false;
}

bool CanContractEdge(const Edge* edge, const Graph* graph) {
  const auto src = edge->src();
  const auto dst = edge->dst();

  // Can't contract edge if doing so would cause a cycle in the
  // graph. So, if there is a directed path from 'src' to 'dst', other
  // than 'edge' (or any other direct edge from 'src' to 'dst'), then
  // combining 'src' and 'dst' will cause a cycle along that path.
  //
  // In practice, to avoid modifying the graph and to take advantage
  // of existing graph functions, we perform an equivalent.
  //   1. Get all nodes incoming to 'dst', excluding 'src'
  //   2. Reverse DFS from those nodes
  //   3. If reverse DFS reaches 'src' then we have a cycle
  std::vector<Node*> dfs_start_nodes;
  for (Node* node : dst->in_nodes()) {
    if (node != src) {
      dfs_start_nodes.push_back(node);
    }
  }

  bool is_cycle = check_cycles(graph, src, dfs_start_nodes);
  return !is_cycle;
}
}  // namespace
Node::Node(const tensorflow::Node* node, const int id) : node_(node), id_(id) {
  if (node_) {
    in_edges_.reserve(node_->in_edges().size());
    out_edges_.reserve(node_->out_edges().size());
  }
}

Graph::Graph(const tensorflow::Graph* g) : g_(g) {
  int n_nodes = g_->num_node_ids();
  nodes_.resize(n_nodes, nullptr);
  nodes_[g->kSourceId] = new Node(g->source_node(), g->kSourceId);
  nodes_[g->kSinkId] = new Node(g->sink_node(), g->kSinkId);
  int n_edges = g->num_edge_ids();
  edges_.resize(n_edges, nullptr);
  for (int i = 2; i < n_nodes; i++) {
    const auto n = g->FindNodeId(i);
    if (n) {
      nodes_[i] = new Node(n, i);
    } else {
      node_ids_.insert(i);
    }
  }
  for (int i = 0; i < n_edges; i++) {
    const auto e = g->FindEdgeId(i);
    if (e) {
      const auto tfsrc = e->src();
      const auto tfdst = e->dst();
      bool is_control = e->IsControlEdge();
      auto src = nodes_[tfsrc->id()];
      auto dst = nodes_[tfdst->id()];
      auto edge =
          new Edge(i, src, e->src_output(), dst, e->dst_input(), is_control);
      edges_[i] = edge;
      src->out_edges_.push_back(edge);
      dst->in_edges_.push_back(edge);
    } else {
      edge_ids_.insert(i);
    }
  }
}

void Graph::AddEdge(Node* src, int out_port, Node* dst, int in_port) {
  int i = edges_.size();
  if (edge_ids_.size()) {
    auto it = edge_ids_.begin();
    i = *it;
    edge_ids_.erase(it);
  } else {
    edges_.push_back(0);
  }
  bool is_control = (out_port == tensorflow::Graph::kControlSlot);
  is_control |= (in_port == tensorflow::Graph::kControlSlot);
  auto edge = new Edge(i, src, out_port, dst, in_port, is_control);
  edges_[i] = edge;
  src->out_edges_.push_back(edge);
  dst->in_edges_.push_back(edge);
}

void Graph::AddControlEdge(Node* src, Node* dst) {
  AddEdge(src, tensorflow::Graph::kControlSlot, dst,
          tensorflow::Graph::kControlSlot);
}

void Graph::RemoveEdge(const Edge* edge) {
  auto src = edge->src();
  auto dst = edge->dst();
  for (auto it = src->out_edges_.begin(); it != src->out_edges_.end(); ++it) {
    if (*it == edge) {
      src->out_edges_.erase(it);
      break;
    }
  }
  for (auto it = dst->in_edges_.begin(); it != dst->in_edges_.end(); ++it) {
    if (*it == edge) {
      dst->in_edges_.erase(it);
      break;
    }
  }
}

Graph::~Graph() {
  for (auto x : nodes_) delete x;
  for (auto x : edges_) delete x;
}

void ContractEdge(Edge* edge, Graph* graph,
                  std::vector<const Edge*>* remove_edges) {
  // Transfer all inputs and outputs of 'dst' to 'src' except edges
  // connecting the two.
  auto src = edge->src();
  auto dst = edge->dst();

  // We can use '0' for input/output index because we don't need them
  // to be accurate for the way we are using the graph.
  std::vector<const Edge*> in_edges(dst->in_edges().begin(),
                                    dst->in_edges().end());
  for (const Edge* in_edge : in_edges) {
    if (in_edge->IsControlEdge()) {
      if (in_edge->src() != src) {
        Edge* e = const_cast<Edge*>(in_edge);
        graph->AddControlEdge(e->src(), src);
      }
    } else {
      if (in_edge->src() != src) {
        Edge* e = const_cast<Edge*>(in_edge);
        if (e->src() == graph->source_node()) {
          graph->AddEdge(e->src(), e->src_output(), src,
                         tensorflow::Graph::kControlSlot);
        } else {
          graph->AddEdge(e->src(), e->src_output(), src, 0 /* input index */);
        }
      }
    }
  }

  std::vector<const Edge*> out_edges(dst->out_edges().begin(),
                                     dst->out_edges().end());
  for (const Edge* out_edge : out_edges) {
    if (out_edge->IsControlEdge()) {
      Edge* e = const_cast<Edge*>(out_edge);
      graph->AddControlEdge(src, e->dst());
    } else {
      Edge* e = const_cast<Edge*>(out_edge);
      if (e->dst() == graph->sink_node()) {
        VLOG(1) << " edge to sink node " << src->name() << " -> "
                << e->dst()->name();
        graph->AddEdge(src, tensorflow::Graph::kControlSlot, e->dst(),
                       e->dst_input());
      } else {
        graph->AddEdge(src, 0 /* output index */, e->dst(), e->dst_input());
      }
    }
  }

  // Return the edges that must be removed to disconnect 'dst' from
  // the graph. We don't actually remove 'dst' since the caller holds
  // references to all the nodes.
  for (const auto& in_edge : dst->in_edges()) {
    remove_edges->push_back(in_edge);
  }
  for (const auto& out_edge : dst->out_edges()) {
    remove_edges->push_back(out_edge);
  }
}

tensorflow::Status SegmentGraph(
    const tensorflow::GraphDef& gdef,
    const std::function<bool(const tensorflow::Node*)>& candidate_fn,
    const SegmentOptions& options, SegmentNodesVector* segments) {
  // Create a Graph representation of the GraphDef.
  tensorflow::FunctionLibraryDefinition flib(tensorflow::OpRegistry::Global(),
                                             gdef.library());
  tensorflow::Graph graph(flib);
  TF_RETURN_IF_ERROR(tensorflow::ConvertGraphDefToGraph(
      tensorflow::GraphConstructorOptions(), gdef, &graph));
  return SegmentGraph(&graph, candidate_fn, options, segments);
}

tensorflow::Status SegmentGraph(
    tensorflow::Graph* tf_graph,
    const std::function<bool(const tensorflow::Node*)>& candidate_fn,
    const SegmentOptions& options, SegmentNodesVector* segments) {
  // tensorflow::DumpGraph("Pre-Segment", &graph);
  Graph* graph = new Graph(tf_graph);
  // Use a union-find to collect the nodes that belong to the same
  // segment. A node value of nullptr indicates that the node is not a candidate
  // for TRT.
  std::vector<UnionFind<Node*>> node_segments;
  for (int i = 0; i < graph->num_node_ids(); ++i) {
    Node* node = graph->FindNodeId(i);
    if (options.exclude_node_list.count(node->name()) != 0 ||
        !candidate_fn(node->tf_node())) {
      node = nullptr;
    }
    node_segments.emplace_back(node);
  }

  // The segmentation algorithm below visits nodes in reverse
  // topological order and attempts to merge nodes along output
  // edges. That means that subgraphs grow from the output-side of the
  // network towards the inputs. In general this is not guaranteed to
  // produce a globally optimal segmentation. In the future if we have
  // a measure of how beneficial it is to include a given node in a
  // TRT subgraph then we can revisit this algorithm to take advantage
  // of that information.
  std::vector<tensorflow::Node*> tforder;
  tensorflow::GetPostOrder(*tf_graph, &tforder);
  // use postorder implementation from tensorflow and construct mirror in
  // internal format
  std::vector<Node*> order;
  order.reserve(tforder.size());
  for (const auto tfnode : tforder) {
    order.push_back(graph->FindNodeId(tfnode->id()));
  }
  for (const Node* node : order) {
    // All output nodes of 'node' have been visited...
    VLOG(2) << "Trying node " << node->name() << " id=" << node->id();

    // 'node' must be a TRT candidate...
    if (node_segments[node->id()].Value() == nullptr) {
      VLOG(2) << "... not a TRT candidate";
      continue;
    }

    // Contract output edges to combine 'node' with output
    // nodes. Iterate since combining two nodes may unblock other
    // combining.
    while (true) {
      std::set<const Edge*> contract_edges;
      for (const Edge* out_edge : node->out_edges()) {
        VLOG(2) << "... out node " << out_edge->dst()->name() << " ( "
                << out_edge->dst()->id() << " <- " << node->id() << " )";
        if (out_edge->IsControlEdge()) {
          VLOG(2) << "... ... Control Edge, Skipping";
          continue;
        }
        // Out node must be TRT candidate...
        if (node_segments[out_edge->dst()->id()].Value() == nullptr) {
          VLOG(2) << "... ... not a TRT candidate";
          continue;
        }

        if (CanContractEdge(out_edge, graph)) {
          VLOG(2) << "... ... can contract";
          contract_edges.insert(out_edge);
        } else {
          VLOG(2) << "... ... cannot contract, would form cycle";
        }
      }

      if (contract_edges.empty()) {
        break;
      }

      // Contract edges and collect the adjacent nodes into the same
      // segment/subgraph.
      while (!contract_edges.empty()) {
        const Edge* contract_edge = *contract_edges.begin();
        const Node* src = contract_edge->src();
        const Node* dst = contract_edge->dst();

        VLOG(2) << "Merge " << src->name() << " <- " << dst->name() << " ("
                << src->id() << " <- " << dst->id();
        node_segments[src->id()].Merge(&node_segments[dst->id()]);

        // Contracting the edge leaves disconnected graph edges.
        // Remove these from the graph and from 'contract_edges' so we
        // don't visit them again.
        Edge* e = const_cast<Edge*>(contract_edge);
        std::vector<const Edge*> remove_edges;
        ContractEdge(e, graph, &remove_edges);

        for (const Edge* r : remove_edges) {
          contract_edges.erase(r);
          graph->RemoveEdge(r);
        }
      }
    }
  }

  // Collect the segments/subgraphs. Each subgraph is represented by a
  // set of the names of the nodes in that subgraph.
  std::unordered_map<string, std::set<string>> sg_map;
  std::unordered_map<string, std::set<string>> device_maps;
  for (auto& u : node_segments) {
    if ((u.Value() != nullptr) && (u.ParentValue() != nullptr)) {
      sg_map[u.ParentValue()->name()].insert(u.Value()->name());
      auto tf_node = u.Value()->tf_node();
      if (tf_node->has_assigned_device_name()) {
        device_maps[u.ParentValue()->name()].insert(
            tf_node->assigned_device_name());
      } else if (tf_node->requested_device().size() > 0) {
        device_maps[u.ParentValue()->name()].insert(
            tf_node->requested_device());
      } else {
        VLOG(1) << "Node " << tf_node->name()
                << " has no device assigned requested device is: "
                << tf_node->requested_device();
      }
    }
  }

  // Convert the segments into the expected return format
  for (const auto& itr : sg_map) {
    const auto& segment_node_names = itr.second;
    if (VLOG_IS_ON(1)) {
      string s;
      for (const auto& name : segment_node_names) {
        s += " " + name;
      }
      VLOG(1) << "Segment " << segments->size() << ":" << s;
    }

    // Don't use small segments.
    if (static_cast<int>(segment_node_names.size()) <
        options.minimum_segment_size) {
      VLOG(1) << "Segment " << segments->size() << " has only "
              << segment_node_names.size() << " nodes, dropping";
      continue;
    }
    const auto& dev_itr = device_maps.find(itr.first);
    if (dev_itr == device_maps.end() || dev_itr->second.size() == 0) {
      VLOG(1) << "No device assigned to segment " << segments->size();
      segments->emplace_back(std::make_pair(segment_node_names, string()));
    } else if (dev_itr->second.size() > 1) {
      string s("Segment ");
      StrAppend(&s, segments->size(), " has multiple devices attached: ");
      for (const auto& dev : dev_itr->second) {
        StrAppend(&s, dev, ", ");
      }
      LOG(WARNING) << s << " choosing " << *(dev_itr->second.begin());
      segments->emplace_back(
          std::make_pair(segment_node_names, *(dev_itr->second.begin())));
    } else {
      segments->emplace_back(
          std::make_pair(segment_node_names, *(dev_itr->second.begin())));
    }
  }
  for (const auto& d : device_maps) {
    string s("Segment ");
    StrAppend(&s, ": '", d.first, "' ");
    for (const auto& dd : d.second) {
      StrAppend(&s, dd, ", ");
    }
    VLOG(1) << "Devices " << s;
  }
  delete graph;
  return tensorflow::Status::OK();
}

}  // namespace segment
}  // namespace tensorrt
}  // namespace tensorflow
