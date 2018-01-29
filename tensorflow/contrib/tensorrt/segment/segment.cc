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

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorflow/contrib/tensorrt/segment/segment.h"

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/tensorrt/segment/union_find.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

//------------------------------------------------------------------------------
namespace tensorflow {
namespace tensorrt {
namespace segment {

//------------------------------------------------------------------------------
namespace {

//------------------------------------------------------------------------------
bool CanContractEdge(const tensorflow::Edge* edge,
                     const tensorflow::Graph& graph) {
  const tensorflow::Node* src = edge->src();
  const tensorflow::Node* dst = edge->dst();

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
  std::vector<tensorflow::Node*> dfs_start_nodes;
  for (tensorflow::Node* node : dst->in_nodes()) {
    if (node != src) {
      dfs_start_nodes.push_back(node);
    }
  }

  bool is_cycle = false;
  if (!dfs_start_nodes.empty()) {
    tensorflow::ReverseDFSFrom(graph, dfs_start_nodes, {},
                               [&is_cycle, src](tensorflow::Node* node) {
                                 if (node == src) {
                                   is_cycle = true;
                                 }
                               });
  }

  return !is_cycle;
}

//------------------------------------------------------------------------------
void ContractEdge(tensorflow::Edge* edge, tensorflow::Graph* graph,
                  std::vector<const tensorflow::Edge*>* remove_edges) {
  // Transfer all inputs and outputs of 'dst' to 'src' except edges
  // connecting the two.
  tensorflow::Node* src = edge->src();
  tensorflow::Node* dst = edge->dst();

  // We can use '0' for input/output index because we don't need them
  // to be accurate for the way we are using the graph.
  std::vector<const tensorflow::Edge*> in_edges(dst->in_edges().begin(),
                                                dst->in_edges().end());
  for (const tensorflow::Edge* in_edge : in_edges) {
    if (in_edge->src() != src) {
      tensorflow::Edge* e = const_cast<tensorflow::Edge*>(in_edge);
      if (e->src() == graph->source_node()) {
        graph->AddEdge(e->src(), e->src_output(), src,
                       tensorflow::Graph::kControlSlot);
      } else {
        graph->AddEdge(e->src(), e->src_output(), src, 0 /* input index */);
      }
    }
  }

  std::vector<const tensorflow::Edge*> out_edges(dst->out_edges().begin(),
                                                 dst->out_edges().end());
  for (const tensorflow::Edge* out_edge : out_edges) {
    tensorflow::Edge* e = const_cast<tensorflow::Edge*>(out_edge);
    if (e->dst() == graph->sink_node()) {
      graph->AddEdge(src, tensorflow::Graph::kControlSlot, e->dst(),
                     e->dst_input());
    } else {
      graph->AddEdge(src, 0 /* output index */, e->dst(), e->dst_input());
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

}  // namespace

//------------------------------------------------------------------------------
tensorflow::Status SegmentGraph(
    const tensorflow::GraphDef& gdef,
    const std::function<bool(const tensorflow::NodeDef&)>& candidate_fn,
    const SegmentOptions& options, SegmentNodesVector* segments) {
  // Create a Graph representation of the GraphDef.
  tensorflow::FunctionLibraryDefinition flib(tensorflow::OpRegistry::Global(),
                                             gdef.library());
  tensorflow::Graph graph(flib);
  TF_RETURN_IF_ERROR(tensorflow::ConvertGraphDefToGraph(
      tensorflow::GraphConstructorOptions(), gdef, &graph));

  // tensorflow::DumpGraph("Pre-Segment", &graph);

  // Use a union-find to collect the nodes that belong to the same
  // segment. A node value of nullptr indicates that the node is not a
  // candidate for TRT.
  std::vector<UnionFind<tensorflow::Node*>> node_segments;
  for (int i = 0; i < graph.num_node_ids(); ++i) {
    tensorflow::Node* node = graph.FindNodeId(i);
    if (options.exclude_node_list.count(node->name())!=0
        || !candidate_fn(node->def())) {
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
  std::vector<tensorflow::Node*> order;
  tensorflow::GetPostOrder(graph, &order);

  for (const tensorflow::Node* node : order) {
    // All output nodes of 'node' have been visited...
    VLOG(2) << "Trying node " << node->name();

    // 'node' must be a TRT candidate...
    if (node_segments[node->id()].Value() == nullptr) {
      VLOG(2) << "... not a TRT candidate";
      continue;
    }

    // Contract output edges to combine 'node' with output
    // nodes. Iterate since combining two nodes may unblock other
    // combining.
    while (true) {
      std::set<const tensorflow::Edge*> contract_edges;
      for (const tensorflow::Edge* out_edge : node->out_edges()) {
        VLOG(2) << "... out node " << out_edge->dst()->name();

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
        const tensorflow::Edge* contract_edge = *contract_edges.begin();
        const tensorflow::Node* src = contract_edge->src();
        const tensorflow::Node* dst = contract_edge->dst();

        VLOG(2) << "Merge " << src->name() << " <- " << dst->name();
        node_segments[src->id()].Merge(&node_segments[dst->id()]);

        // Contracting the edge leaves disconnected graph edges.
        // Remove these from the graph and from 'contract_edges' so we
        // don't visit them again.
        tensorflow::Edge* e = const_cast<tensorflow::Edge*>(contract_edge);
        std::vector<const tensorflow::Edge*> remove_edges;
        ContractEdge(e, &graph, &remove_edges);

        for (const tensorflow::Edge* r : remove_edges) {
          contract_edges.erase(r);
          graph.RemoveEdge(r);
        }
      }
    }
  }

  // Collect the segments/subgraphs. Each subgraph is represented by a
  // set of the names of the nodes in that subgraph.
  std::unordered_map<std::string, std::set<std::string>> sg_map;
  for (auto& u : node_segments) {
    if ((u.Value() != nullptr) && (u.ParentValue() != nullptr)) {
      sg_map[u.ParentValue()->name()].insert(u.Value()->name());
    }
  }

  // Cleanup the graph to remove disconnected nodes before outputting
  if (VLOG_IS_ON(2)) {
    for (tensorflow::Node* node : graph.nodes()) {
      if ((node->in_edges().size() == 0) && (node->out_edges().size() == 0)) {
        graph.RemoveNode(node);
      }
    }
  }

  // Convert the segments into the expected return format
  for (const auto& itr : sg_map) {
    const auto& segment_node_names = itr.second;
    if (VLOG_IS_ON(1)) {
      std::string s;
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

    segments->emplace_back(segment_node_names);
  }

  return tensorflow::Status::OK();
}

}  // namespace segment
}  // namespace tensorrt
}  // namespace tensorflow

#endif // GOOGLE_TENSORRT
#endif // GOOGLE_CUDA
