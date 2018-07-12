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

#include <queue>
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

// A simple graph representation to mirror tensorflow::Graph. This structure
// helps saving memory since segmenter modifies the graph in place, preventing
// the need to create a copy of the graph. It is composed of edges and nodes.
// Nodes keep pointers to original TF nodes.
class SimpleNode;
class SimpleGraph;
class SimpleEdge {
 public:
  SimpleEdge(int id, SimpleNode* src, int src_port, SimpleNode* dst,
             int dst_port, bool is_control = false)
      : id_(id),
        src_(src),
        src_port_(src_port),
        dst_(dst),
        dst_port_(dst_port),
        control_(is_control) {}
  ~SimpleEdge() {}

  SimpleNode* src() const { return src_; }
  SimpleNode* dst() const { return dst_; }
  int src_output() const { return src_port_; }
  int dst_input() const { return dst_port_; }
  int id() const { return id_; }
  bool IsControlEdge() const { return control_; }

 private:
  int id_;
  SimpleNode* src_;
  int src_port_;
  SimpleNode* dst_;
  int dst_port_;
  bool control_;
};

class SimpleNode {
 public:
  SimpleNode(const tensorflow::Node* node, const int id);

  const std::vector<SimpleEdge*>& in_edges() const { return in_edges_; }
  const std::vector<SimpleEdge*>& out_edges() const { return out_edges_; }
  std::vector<SimpleNode*> in_nodes() const {
    std::vector<SimpleNode*> res;
    res.reserve(in_edges_.size());
    for (const auto e : in_edges_) {
      if (e) res.push_back(e->src());
    }
    return res;
  }
  const string& name() const { return node_->name(); }
  const tensorflow::Node* tf_node() const { return node_; }
  int id() const { return id_; }

 private:
  const tensorflow::Node* node_;
  std::vector<SimpleEdge*> in_edges_;
  std::vector<SimpleEdge*> out_edges_;
  int id_;

  friend class SimpleGraph;
};

class SimpleGraph {
 public:
  explicit SimpleGraph(const tensorflow::Graph* g);
  ~SimpleGraph();

  void AddControlEdge(SimpleNode* src, SimpleNode* dst);
  void AddEdge(SimpleNode* src, int out_port, SimpleNode* dst, int in_port);
  void RemoveEdge(const SimpleEdge*);
  SimpleNode* FindNodeId(int node_id) {
    if (node_id < 0 || node_id > static_cast<int>(nodes_.size())) {
      return nullptr;
    }
    return nodes_[node_id];
  }
  int num_node_ids() const { return nodes_.size(); }
  const SimpleNode* source_node() const {
    return nodes_[tensorflow::Graph::kSourceId];
  }
  const SimpleNode* sink_node() const {
    return nodes_[tensorflow::Graph::kSinkId];
  }

 private:
  const tensorflow::Graph* g_;
  std::vector<SimpleNode*> nodes_;
  std::vector<SimpleEdge*> edges_;
  // free_edge_ids_ and free_node_ids_ contain freed indices.
  std::set<int> free_edge_ids_;
  std::set<int> free_node_ids_;
};

SimpleNode::SimpleNode(const tensorflow::Node* node, const int id)
    : node_(node), id_(id) {
  if (node_) {
    in_edges_.reserve(node_->in_edges().size());
    out_edges_.reserve(node_->out_edges().size());
  }
}

SimpleGraph::SimpleGraph(const tensorflow::Graph* g) : g_(g) {
  int n_nodes = g_->num_node_ids();
  nodes_.resize(n_nodes, nullptr);
  nodes_[g->kSourceId] = new SimpleNode(g->source_node(), g->kSourceId);
  nodes_[g->kSinkId] = new SimpleNode(g->sink_node(), g->kSinkId);
  int n_edges = g->num_edge_ids();
  edges_.resize(n_edges, nullptr);
  for (int i = 2; i < n_nodes; i++) {
    const auto n = g->FindNodeId(i);
    if (n) {
      nodes_[i] = new SimpleNode(n, i);
    } else {
      free_node_ids_.insert(i);
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
      auto edge = new SimpleEdge(i, src, e->src_output(), dst, e->dst_input(),
                                 is_control);
      edges_[i] = edge;
      src->out_edges_.push_back(edge);
      dst->in_edges_.push_back(edge);
    } else {
      free_edge_ids_.insert(i);
    }
  }
}

void SimpleGraph::AddEdge(SimpleNode* src, int out_port, SimpleNode* dst,
                          int in_port) {
  int i = edges_.size();
  if (!free_edge_ids_.empty()) {
    auto it = free_edge_ids_.begin();
    i = *it;
    free_edge_ids_.erase(it);
  } else {
    edges_.push_back(nullptr);
  }
  bool is_control = (out_port == tensorflow::Graph::kControlSlot);
  is_control |= (in_port == tensorflow::Graph::kControlSlot);
  auto edge = new SimpleEdge(i, src, out_port, dst, in_port, is_control);
  edges_[i] = edge;
  src->out_edges_.push_back(edge);
  dst->in_edges_.push_back(edge);
}

void SimpleGraph::AddControlEdge(SimpleNode* src, SimpleNode* dst) {
  AddEdge(src, tensorflow::Graph::kControlSlot, dst,
          tensorflow::Graph::kControlSlot);
}

void SimpleGraph::RemoveEdge(const SimpleEdge* edge) {
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

SimpleGraph::~SimpleGraph() {
  for (auto x : nodes_) delete x;
  for (auto x : edges_) delete x;
}

namespace {

bool CheckCycles(const std::unique_ptr<SimpleGraph>& g, const SimpleNode* src,
                 const std::vector<SimpleNode*>& start) {
  // Copied from TF ReverseDFS, which only works for tensorflow::Graph.
  struct Work {
    SimpleNode* node;
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

bool CanContractEdge(const SimpleEdge* edge,
                     const std::unique_ptr<SimpleGraph>& graph) {
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
  //
  // TODO(aaroey): there are several problems with the current approach:
  // 1. src->dst->src, this is not detected but it should be;
  // 2. src->dst->...(any node sequence that doesn't contain src)...->dst, this
  //    is detected but it should not be.
  //
  // Note that it's fine that dst connects back to src indirectly (i.e. through
  // a path with length > 1 that consists of intermedia nodes other than src).
  // While loops is one example.
  //
  // The goal is to make sure that the trt subgraph:
  // 1. has no loops (i.e. is a DAG), and
  // 2. if there is a path in the subgraph from X to Y (X and Y are both nodes
  //    in the subgraph), then all paths from X to Y are in the subgraph.
  //
  // To achieve this goal, the correct way seems to be:
  // 1. remove any direct edge from src->dst;
  // 2. detect if src can reach dst, if so they cannot be merged.
  std::vector<SimpleNode*> dfs_start_nodes;
  for (SimpleNode* node : dst->in_nodes()) {
    if (node != src) {
      dfs_start_nodes.push_back(node);
    }
  }

  const bool has_cycle = CheckCycles(graph, src, dfs_start_nodes);
  return !has_cycle;
}
}  // namespace

void ContractEdge(SimpleEdge* edge, SimpleGraph* graph,
                  std::vector<const SimpleEdge*>* remove_edges) {
  // Transfer all inputs and outputs of 'dst' to 'src' except edges
  // connecting the two.
  auto src = edge->src();
  auto dst = edge->dst();

  // We can use '0' for input/output index because we don't need them
  // to be accurate for the way we are using the graph.
  std::vector<const SimpleEdge*> in_edges(dst->in_edges().begin(),
                                          dst->in_edges().end());
  for (const SimpleEdge* in_edge : in_edges) {
    if (in_edge->IsControlEdge()) {
      if (in_edge->src() != src) {
        SimpleEdge* e = const_cast<SimpleEdge*>(in_edge);
        graph->AddControlEdge(e->src(), src);
      }
    } else {
      if (in_edge->src() != src) {
        SimpleEdge* e = const_cast<SimpleEdge*>(in_edge);
        if (e->src() == graph->source_node()) {
          graph->AddEdge(e->src(), e->src_output(), src,
                         tensorflow::Graph::kControlSlot);
        } else {
          graph->AddEdge(e->src(), e->src_output(), src, 0 /* input index */);
        }
      }
    }
  }

  std::vector<const SimpleEdge*> out_edges(dst->out_edges().begin(),
                                           dst->out_edges().end());
  for (const SimpleEdge* out_edge : out_edges) {
    if (out_edge->IsControlEdge()) {
      SimpleEdge* e = const_cast<SimpleEdge*>(out_edge);
      graph->AddControlEdge(src, e->dst());
    } else {
      SimpleEdge* e = const_cast<SimpleEdge*>(out_edge);
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
    const tensorflow::Graph* tf_graph,
    const std::function<bool(const tensorflow::Node*)>& candidate_fn,
    const std::function<bool(const tensorflow::Node*)>& input_candidate_fn,
    const std::function<bool(const tensorflow::Node*)>& output_candidate_fn,
    const SegmentOptions& options, SegmentNodesVector* segments) {
  // Steps:
  // 1. run the segmentation algorithm to find all the segments, which uses
  //    candidate_fn to determine the candidates segment nodes;
  // 2. for each segments, remove the nodes that are inputs/outputs of the
  //    segment but are not eligible, using input/output_candidate_fn to
  //    determine the eligibilities;
  // 3. convert the segment into expected return format and return the result.

  // --------------------------------- Step 1 ---------------------------------
  auto graph = std::unique_ptr<SimpleGraph>(new SimpleGraph(tf_graph));
  // Use a union-find to collect the nodes that belong to the same
  // segment. A node value of nullptr indicates that the node is not a candidate
  // for TRT.
  std::vector<UnionFind<SimpleNode*>> node_segments;
  for (int i = 0; i < graph->num_node_ids(); ++i) {
    SimpleNode* node = graph->FindNodeId(i);
    if (options.exclude_node_list.count(node->name()) != 0 ||
        !candidate_fn(node->tf_node())) {
      node = nullptr;
    }
    node_segments.emplace_back(node);
  }

  // The segmentation algorithm below visits nodes in reverse topological order
  // and attempts to merge nodes along output edges. That means that subgraphs
  // grow from the output-side of the network towards the inputs.
  //
  // In general this is not guaranteed to produce a globally optimal
  // segmentation. For exaample, consider graph with node {A, B, C, D} and edges
  // {A->B, A->C, B->D, C->D), where A, B, D are trt compatible but C is not, so
  // in theory we can choose to contract either A, B or B, D but not both, but
  // here it always choose to contract B, D.
  //
  // In the future if we have a measure of how beneficial it is to include a
  // given node in a TRT subgraph then we can revisit this algorithm to take
  // advantage of that information.
  std::vector<tensorflow::Node*> tforder;
  tensorflow::GetPostOrder(*tf_graph, &tforder);
  // use postorder implementation from tensorflow and construct mirror in
  // internal format
  std::vector<SimpleNode*> order;
  order.reserve(tforder.size());
  for (const auto tfnode : tforder) {
    order.push_back(graph->FindNodeId(tfnode->id()));
  }
  for (const SimpleNode* node : order) {
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
      std::set<const SimpleEdge*> contract_edges;
      for (const SimpleEdge* out_edge : node->out_edges()) {
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
        const SimpleEdge* contract_edge = *contract_edges.begin();
        const SimpleNode* src = contract_edge->src();
        const SimpleNode* dst = contract_edge->dst();

        VLOG(2) << "Merge " << src->name() << " <- " << dst->name() << " ("
                << src->id() << " <- " << dst->id();
        node_segments[src->id()].Merge(&node_segments[dst->id()]);

        // Contracting the edge leaves disconnected graph edges.
        // Remove these from the graph and from 'contract_edges' so we
        // don't visit them again.
        SimpleEdge* e = const_cast<SimpleEdge*>(contract_edge);
        std::vector<const SimpleEdge*> remove_edges;
        ContractEdge(e, graph.get(), &remove_edges);

        for (const SimpleEdge* r : remove_edges) {
          contract_edges.erase(r);
          graph->RemoveEdge(r);
        }
      }
    }
  }

  // Collect the segments/subgraphs. Each subgraph is represented by a
  // set of the names of the nodes in that subgraph.

  // A map from the segment identifier (currently the name of the root node of
  // the segment tree) to the segment nodes set.
  std::unordered_map<string, std::set<const tensorflow::Node*>> sg_map;

  // A map from the segment identifier (currently the name of the root node of
  // the segment tree) to the device names that the nodes in the segment are
  // assigned to.
  //
  // TODO(aaroey): nodes assigned to different devices should not be merged,
  // fix this.
  std::unordered_map<string, std::set<string>> device_maps;

  for (auto& u : node_segments) {
    if ((u.Value() != nullptr) && (u.ParentValue() != nullptr)) {
      sg_map[u.ParentValue()->name()].insert(u.Value()->tf_node());
      auto tf_node = u.Value()->tf_node();
      // has_assigned_device_name() is expected to return true
      // when called from optimization pass. However, since graph
      // is converted back and forth between graph and graphdef,
      // assigned devices demoted to requested devices. If the graph
      // is passed directly to this module, assigned devices will be set.
      if (tf_node->has_assigned_device_name()) {
        device_maps[u.ParentValue()->name()].insert(
            tf_node->assigned_device_name());
      } else if (!tf_node->requested_device().empty()) {
        device_maps[u.ParentValue()->name()].insert(
            tf_node->requested_device());
      } else {
        VLOG(1) << "Node " << tf_node->name()
                << " has no device assigned requested device is: "
                << tf_node->requested_device();
      }
    }
  }

  // --------------------------------- Step 2 ---------------------------------
  // Remove ineligible input/output nodes.
  for (auto& itr : sg_map) {
    std::set<const tensorflow::Node*>& segment_nodes = itr.second;
    VLOG(1) << "Segment original size: " << segment_nodes.size();
    while (true) {
      std::deque<const tensorflow::Node*> in_nodes_que, out_nodes_que;
      // Find an input node that is not eligible and add it to the queue.
      // Nodes that has no incoming edges should not be treated as "input",
      // as there are really no inputs to them. Similar for output nodes.
      for (auto node : segment_nodes) {
        bool added = false;
        for (const tensorflow::Edge* edge : node->in_edges()) {
          if (!edge->IsControlEdge() && !edge->src()->IsSource() &&
              !segment_nodes.count(edge->src())) {  // 'node' is an input node.
            if (!input_candidate_fn(node)) {
              in_nodes_que.push_back(node);
              added = true;
              break;
            }
          }
        }
        if (added) continue;  // Only adding the node once to either queue.
        for (const tensorflow::Edge* edge : node->out_edges()) {
          if (!edge->dst()->IsSink() && !edge->IsControlEdge() &&
              !segment_nodes.count(edge->dst())) {  // 'node' is an output node.
            if (!output_candidate_fn(node)) {
              out_nodes_que.push_back(node);
              break;
            }
          }
        }
      }
      if (in_nodes_que.empty() && out_nodes_que.empty()) {
        // No more ineligible input/output nodes.
        break;
      }
      // Now for each ineligible node, remove all of its inputs or outputs from
      // the subgraph.
      //
      // It can be proven that, if the original subgraph:
      // 1. is a DAG, and
      // 2. all paths between two nodes in the subgraph are all inside the
      //    subgraph
      // then after doing this operation the resulting subgraph will keep the
      // same properties 1 and 2.
      //
      // For simplicity we use heuristics: for input nodes remove all its
      // input, for output nodes remove all its output. In this way, for common
      // cases the number of removed nodes should be minimum.
      auto remove_nodes = [&segment_nodes](
          bool is_input_nodes,
          std::deque<const tensorflow::Node*>* que) {
        // Run a BFS on the queue to find all the input/output nodes.
        std::set<const tensorflow::Node*> visited;
        while (!que->empty()) {
          auto node = que->front();
          que->pop_front();
          if (!visited.insert(node).second) continue;
          segment_nodes.erase(node);
          for (auto in : is_input_nodes ? node->in_nodes() : node->out_nodes()) {
            if (segment_nodes.count(in)) {
              que->push_back(in);
              VLOG(2) << "Need to remove node " << in->name()
                         << " because one of its "
                         << (is_input_nodes ? "output" : "input")
                         << " nodes in the graph was removed: " << node->name();
            }
          }
        }
      };
      remove_nodes(true, &in_nodes_que);
      remove_nodes(false, &out_nodes_que);
    }
    VLOG(1) << "Segment new size: " << segment_nodes.size();
  }

  // --------------------------------- Step 3 ---------------------------------
  // Convert the segments into the expected return format
  for (const auto& itr : sg_map) {
    const std::set<const tensorflow::Node*>& segment_nodes = itr.second;
    if (VLOG_IS_ON(1)) {
      string s;
      for (auto node : segment_nodes) s += " " + node->name();
      VLOG(1) << "Segment " << segments->size() << ": " << s;
    }

    // Don't use small segments.
    if (static_cast<int>(segment_nodes.size()) <
        options.minimum_segment_size) {
      VLOG(1) << "Segment " << segments->size() << " has only "
              << segment_nodes.size() << " nodes, dropping";
      continue;
    }

    // TODO(sami): Make segmenter placement aware once trtscopes are in place
    std::set<string> segment_node_names;
    for (auto node : itr.second) segment_node_names.insert(node->name());
    const auto& dev_itr = device_maps.find(itr.first);
    if (dev_itr == device_maps.end() || dev_itr->second.empty()) {
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
  if (VLOG_IS_ON(1)) {
    for (const auto& d : device_maps) {
      string s("Segment ");
      StrAppend(&s, ": '", d.first, "' ");
      for (const auto& dd : d.second) {
        StrAppend(&s, dd, ", ");
      }
      VLOG(1) << "Devices " << s;
    }
  }
  return tensorflow::Status::OK();
}

}  // namespace segment
}  // namespace tensorrt
}  // namespace tensorflow
