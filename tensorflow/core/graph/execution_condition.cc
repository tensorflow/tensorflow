/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/execution_condition.h"

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace {

struct NodeHash {
  size_t operator()(Node* const n) const {
    return n->id();
  }
};

class TempEdgeManager {
 public:
  TempEdgeManager(Graph* g) : g_(g) {}

  void Remove(const Edge* edge) {
    removed_edges_.push_back(
      {edge->src(), edge->dst(), edge->src_output(), edge->dst_input()});
    g_->RemoveEdge(edge);
  }

  void Add(Node* src, int src_port, Node* dst, int dst_port) {
    added_edges_.push_back(g_->AddEdge(src, src_port, dst, dst_port));
  }

  void AddControl(Node* src, Node* dst) {
    added_edges_.push_back(g_->AddControlEdge(src, dst));
  }

  void Restore() {
    for (auto& e : removed_edges_) {
      g_->AddEdge(e.src, e.src_port, e.dst, e.dst_port);
    }
    removed_edges_.clear();

    for (auto e : added_edges_) {
      g_->RemoveEdge(e);
    }
    added_edges_.clear();
  }

 private:
  struct DetachedEdge {
    Node* src;
    Node* dst;
    int src_port, dst_port;
  };

  Graph* g_;
  std::vector<DetachedEdge> removed_edges_;
  std::vector<const Edge*> added_edges_;
};

class EdgeSetWrapper {
 private:
  friend class Iterator;
  class Iterator {
   public:
    typedef std::vector<const Edge*>::const_iterator iterator;

    Iterator(EdgeSetWrapper& owner, const EdgeSet& edge_set)
        : owner_(owner) {
      CHECK(!owner_.active_flag_);
      owner_.active_flag_ = true;
      for (const Edge* e : edge_set) {
        owner_.storage_.push_back(e);
      }
    }
    iterator begin() {
      return owner_.storage_.cbegin();
    }
    iterator end() {
      return owner_.storage_.cend();
    }
    ~Iterator() {
      owner_.active_flag_ = false;
      owner_.storage_.clear();
    }

   private:
    EdgeSetWrapper& owner_;
  };

  std::vector<const Edge*> storage_;
  bool active_flag_ = false;

 public:
  Iterator operator()(const EdgeSet& edge_set) {
    return Iterator(*this, edge_set);
  }
};

struct FrameInfoItem {
  string frame_name;
  std::vector<Node*> exits;
  std::vector<Node*> enters;
  Node* loop_cond = nullptr;

  Status GetIterationTrigger(Graph* g, Node*& ret) {
    if (iteration_trigger) {
      ret = iteration_trigger;
      return Status::OK();
    }

    string name_prefix = strings::StrCat(frame_name, "/iteration_trigger");
    const string& device = loop_cond->def().device();
    const string& assigned_device = loop_cond->assigned_device_name();

    Node* switch_node;
    TF_RETURN_IF_ERROR(
      NodeBuilder(g->NewName(strings::StrCat(name_prefix, "/Switch")),
                  "Switch")
        .Input(loop_cond)
        .Input(loop_cond)
        .Device(device)
        .Finalize(g, &switch_node));
    switch_node->set_assigned_device_name(assigned_device);

    Node* next_iteration_node;
    TF_RETURN_IF_ERROR(
      NodeBuilder(g->NewName(strings::StrCat(name_prefix, "/NextIteration")),
                  "NextIteration")
        .Input(switch_node, 1)
        .Device(device)
        .Finalize(g, &next_iteration_node));
    next_iteration_node->set_assigned_device_name(assigned_device);

    Node* const_node;
    TF_RETURN_IF_ERROR(
      NodeBuilder(g->NewName(strings::StrCat(name_prefix, "/Const")), "Const")
        .Attr("value", Tensor(DT_INT32))
        .Attr("dtype", DT_INT32)
        .ControlInput(next_iteration_node)
        .Device(device)
        .Finalize(g, &const_node));
    const_node->set_assigned_device_name(assigned_device);

    TF_RETURN_IF_ERROR(
      NodeBuilder(g->NewName(strings::StrCat(name_prefix, "/Switch_1")),
                  "Switch")
        .Input(const_node)
        .Input(next_iteration_node)
        .Device(device)
        .Finalize(g, &iteration_trigger));
    iteration_trigger->set_assigned_device_name(assigned_device);

    ret = iteration_trigger;
    return Status::OK();
  }

 private:
  Node* iteration_trigger = nullptr;
};

// Merge may only be used to combine outputs of an Enter node and a
// NextIteration node.
Status VerifyMerge(Node* n) {
  int flag = 0;
  if (n->in_edges().size() == 2) {
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) break;
      if (e->src()->IsEnter()) {
        flag += 1;
      }
      if (e->src()->IsNextIteration()) {
        flag += 2;
      }
    }
  }

  if (flag != 3) {
    return errors::InvalidArgument(
      "Merge nodes must take an Enter node and a NextIteration Node ",
      "as its only input: ", n->DebugString());
  }
  return Status::OK();
}

// Switch may only be used to conditionally exit a loop.
Status VerifySwitch(Node* n) {
  const Node* input_node;
  TF_RETURN_IF_ERROR(n->input_node(1, &input_node));
  if (!input_node->IsLoopCond()) {
    return errors::InvalidArgument(
      "The second input to ", n->DebugString(), " should be a LoopCond.");
  }

  for (const Edge* e : n->out_edges()) {
    if (e->src_output() == 0 && !e->dst()->IsExit()) {
      return errors::InvalidArgument(
        "The second output of ", n->DebugString(), " should be an Exit.");
    }
  }

  return Status::OK();
}

class FrameInfo {
 public:
  FrameInfo(Graph* g) : g_(g) {}

  Status Init() {
    TF_RETURN_IF_ERROR(BuildControlFlowInfo(g_, &control_flow_info_));

    for (Node* n : g_->nodes()) {
      if (n->IsMerge()) {
        TF_RETURN_IF_ERROR(VerifyMerge(n));
      }
      if (n->IsSwitch()) {
        TF_RETURN_IF_ERROR(VerifySwitch(n));
      }

      const string& frame_name = control_flow_info_[n->id()].frame_name;
      if (n->IsExit()) {
        frames_[frame_name].exits.push_back(n);
      }
      if (n->IsEnter()) {
        frames_[frame_name].enters.push_back(n);
      }
      if (n->IsLoopCond()) {
        FrameInfoItem& f = frames_[frame_name];
        if (f.loop_cond != nullptr) {
          return errors::InvalidArgument(
            "Frame cannot contain more than one LoopCond: %s.", frame_name);
        }
        f.loop_cond = n;
        f.frame_name = frame_name;
      }
    }

    for (auto name_and_frame : frames_) {
      if (name_and_frame.second.loop_cond == nullptr) {
        return errors::InvalidArgument(
          "Frame must contain one LoopCond: %s.", name_and_frame.first);
      }
    }

    return Status::OK();
  }

  typedef std::unordered_map<string, FrameInfoItem>::iterator base_iterator;

  class iterator : public base_iterator {
   public:
    iterator(const base_iterator& other) : base_iterator(other) {}

    FrameInfoItem& operator*() {
      return base_iterator::operator*().second;
    }

    const FrameInfoItem& operator*() const {
      return base_iterator::operator*().second;
    }
  };

  iterator begin() {
    return frames_.begin();
  }

  iterator end() {
    return frames_.end();
  }

  FrameInfoItem& operator[](Node* n) {
    string& frame_name = control_flow_info_[n->id()].frame_name;
    DCHECK(n->id() < control_flow_info_.size());
    DCHECK(frames_.find(frame_name) != frames_.end());
    return frames_[frame_name];
  }

  size_t size() {
    return frames_.size();
  }

 private:
  Graph* g_;
  std::vector<ControlFlowInfo> control_flow_info_;
  std::unordered_map<string, FrameInfoItem> frames_;
};

bool IsConnected(Node* src, Node* dst) {
  const EdgeSet& src_out = src->out_edges();
  const EdgeSet& dst_in = dst->in_edges();
  if (src_out.size() <= dst_in.size()) {
    for (const Edge* e : src_out) {
      if (e->dst() == dst) {
        return true;
      }
    }
  } else {
    for (const Edge* e : dst_in) {
      if (e->src() == src) {
        return true;
      }
    }
  }
  return false;
}

void EnsureConnected(Graph* g, Node* src, Node* dst) {
  if (!IsConnected(src, dst)) {
    g->AddControlEdge(src, dst);
  }
}

Status EnsureNotStall(Graph* g, Node* n, FrameInfo& frame_info,
                      EdgeSetWrapper& wrapper) {
  DCHECK(n->IsEnter());
  DCHECK(n->output_type(0) == DT_INT32);
  DCHECK(n->def().attr().at("is_constant").b());

  Node* iteration_trigger;
  TF_RETURN_IF_ERROR(frame_info[n].GetIterationTrigger(g, iteration_trigger));
  DCHECK(iteration_trigger->IsSwitch());

  Node* merge_node;
  TF_RETURN_IF_ERROR(
    NodeBuilder(g->NewName(strings::StrCat(n->name(), "/Merge")),
                "Merge")
      .Input(n)
      .Input(iteration_trigger, 0)
      .Device(n->def().device())
      .Finalize(g, &merge_node));
  merge_node->set_assigned_device_name(n->assigned_device_name());

  for (const Edge* e : wrapper(n->out_edges())) {
    if (e->IsControlEdge()) continue;

    Node* dst = e->dst();
    int dst_port = e->dst_input();
    g->RemoveEdge(e);
    g->AddEdge(merge_node, 0, dst, dst_port);
  }

  return Status::OK();
}

Status TransformMuxInputs(Graph* g, Node* n, FrameInfo& frame_info,
                          EdgeSetWrapper& wrapper) {
  DCHECK(n->IsMux());

  const string& device = n->def().device();
  const string& assigned_device = n->assigned_device_name();

  for (const Edge* e : wrapper(n->in_edges())) {
    if (e->IsControlEdge()) {
      continue;
    }

    Node* src = e->src();

    if (src->out_edges().size() == 1 && !src->IsEnter() &&
        !src->IsNextIteration() && !src->IsExit()) {
      continue;
    }

    if (e->dst_input() == 0 && src->IsEnter()) {
      TF_RETURN_IF_ERROR(EnsureNotStall(g, src, frame_info, wrapper));
      continue;
    }

    const char* op_type;
    if (IsRefType(n->input_type(e->dst_input()))) {
      op_type = "RefIdentity";
    } else {
      op_type = "Identity";
    }

    Node* identity_node;
    TF_RETURN_IF_ERROR(
      NodeBuilder(
          g->NewName(strings::StrCat(src->name(), "/", op_type)), op_type)
        .Input(src, e->src_output())
        .Device(device)
        .Finalize(g, &identity_node));
    identity_node->set_assigned_device_name(assigned_device);

    int port = e->dst_input();
    g->RemoveEdge(e);
    g->AddEdge(identity_node, 0, n, port);
  }

  return Status::OK();
}

Status TransformLoops(Graph* g, FrameInfo& frame_info,
                      TempEdgeManager& temp_edge_manager,
                      EdgeSetWrapper& wrapper,
                      std::vector<Node*>& loop_placeholders) {
  loop_placeholders.clear();
  loop_placeholders.reserve(frame_info.size());
  for (const FrameInfoItem& f : frame_info) {
    Node* placeholder;
    TF_RETURN_IF_ERROR(
      NodeBuilder(g->NewName(f.frame_name), "NoOp")
        .Finalize(g, &placeholder));
    loop_placeholders.push_back(placeholder);

    for (Node* n : f.enters) {
      for (const Edge* e : wrapper(n->out_edges())) {
        temp_edge_manager.Remove(e);
      }
      g->AddControlEdge(n, placeholder);
    }
    for (Node* n : f.exits) {
      for (const Edge* e : wrapper(n->out_edges())) {
        EnsureConnected(g, placeholder, e->dst());
        temp_edge_manager.Remove(e);
      }
      g->AddControlEdge(n, g->sink_node());
    }
  }

  for (Node* n : g->nodes()) {
    if (!n->IsNextIteration()) continue;

    for (const Edge* e : wrapper(n->out_edges())) {
      temp_edge_manager.Remove(e);
    }
    g->AddControlEdge(n, g->sink_node());
  }

  return Status::OK();
}

// Create indicators for each branch of a mux node.
Status CreateIndicators(Graph* g, Node* mux_node,
                        std::vector<Node*>& indicators,
                        TempEdgeManager& temp_edge_manager) {
  DCHECK(mux_node->IsMux());

  const Edge* index_edge;
  TF_RETURN_IF_ERROR(mux_node->input_edge(0, &index_edge));
  Node* index_node = index_edge->src();
  int index_port = index_edge ->src_output();

  DCHECK(!index_node->IsEnter());

  Node* const_node;
  TF_RETURN_IF_ERROR(
    NodeBuilder(g->NewName(strings::StrCat(index_node->name(), "/Const")),
                "Const")
      .Attr("value", Tensor())
      .Attr("dtype", DT_FLOAT)
      .ControlInput(index_node) // Ensure const_node is in the same frame.
      .Device(index_node->def().device())
      .Finalize(g, &const_node));
  const_node->set_assigned_device_name(index_node->assigned_device_name());
  temp_edge_manager.Remove(*const_node->in_edges().begin());

  Node* demux_node;
  TF_RETURN_IF_ERROR(
    NodeBuilder(g->NewName(strings::StrCat(index_node->name(), "/Demux")),
                "Demux")
      .Input(index_node, index_port)
      .Input(const_node, 0)
      .Attr("N", mux_node->num_inputs() - 1)
      .Device(index_node->def().device())
      .Finalize(g, &demux_node));
  demux_node->set_assigned_device_name(index_node->assigned_device_name());
  for (const Edge* e : demux_node->in_edges()) {
    if (e->dst_input() == 0) {
      temp_edge_manager.Remove(e);
      break;
    }
  }

  for (const Edge* e : mux_node->in_edges()) {
    if (e->IsControlEdge()) {
      continue;
    }

    DCHECK(e->src()->out_edges().size() == 1);
    DCHECK(!e->src()->IsEnter());
    DCHECK(!e->src()->IsExit());
    DCHECK(!e->src()->IsNextIteration());

    if (e->dst_input() == 0) {
      continue;
    }

    Node* src = e->src();
    Node*& indicator_node = indicators[src->id()];
    TF_RETURN_IF_ERROR(
      NodeBuilder(g->NewName(strings::StrCat(src->name(), "/Identity")),
                  "Identity")
        .Input(demux_node, e->dst_input() - 1)
        .Device(src->def().device())
        .Finalize(g, &indicator_node));
    indicator_node->set_assigned_device_name(src->assigned_device_name());
  }

  return Status::OK();
}

// Replace mux node with merge node.
Status ReplaceWithMerge(Graph* g, Node* n) {
  DCHECK(n->IsMux());

  // Backup input edges.
  int N = n->num_inputs() - 1;
  std::vector<NodeBuilder::NodeOut> data_inputs;
  data_inputs.reserve(N);
  std::vector<Node*> control_inputs;
  control_inputs.reserve(n->in_edges().size() - N);

  for (const Edge* e : n->in_edges()) {
    if (e->dst_input() == 0) continue;

    if (e->IsControlEdge()) {
      control_inputs.push_back(e->src());
    } else {
      data_inputs.emplace_back(e->src(), e->src_output());
    }
  }

  struct NodeAndPort {
    Node* node;
    int port;
  };

  // Backup output edges.
  std::vector<NodeAndPort> data_outputs;
  std::vector<Node*> control_outputs;
  for (const Edge* e : n->out_edges()) {
    if (e->IsControlEdge()) {
      control_outputs.push_back(e->dst());
    } else {
      data_outputs.push_back({e->dst(), e->dst_input()});
    }
  }

  const char* op_type;
  if (IsRefType(n->output_type(0))) {
    op_type = "RefMerge";
  } else {
    op_type = "Merge";
  }

  // Create a merge node with the same inputs as the mux node.
  Node* merge_node;
  TF_RETURN_IF_ERROR(
    NodeBuilder(g->NewName(strings::StrCat(n->name(), "/", op_type)), op_type)
      .Input(data_inputs)
      .ControlInputs(control_inputs)
      .Device(n->def().device())
      .Finalize(g, &merge_node));
  merge_node->set_assigned_device_name(n->assigned_device_name());

  g->RemoveNode(n);

  // Restore output edges.
  for (NodeAndPort node_and_port : data_outputs) {
    g->AddEdge(merge_node, 0, node_and_port.node, node_and_port.port);
  }
  for (Node* n : control_outputs) {
    g->AddControlEdge(merge_node, n);
  }

  return Status::OK();
}

Status TransformGraph(Graph* g, const std::vector<Node*>& target_nodes,
                      TempEdgeManager& temp_edge_manager,
                      EdgeSetWrapper& wrapper,
                      std::vector<Node*>& indicators,
                      std::vector<Node*>& loop_placeholders) {
  FrameInfo frame_info(g);
  TF_RETURN_IF_ERROR(frame_info.Init());

  for (const Edge* e : wrapper(g->sink_node()->in_edges())) {
    g->RemoveEdge(e);
  }
  for (const Edge* e : wrapper(g->source_node()->out_edges())) {
    g->RemoveEdge(e);
  }
  for (Node* n : target_nodes) {
    g->AddControlEdge(n, g->sink_node());
  }

  int num_reserved_node_ids = 0;
  for (Node* n : g->nodes()) {
    if (!n->IsMux()) continue;

    // Some nodes will be added by CreateIndicators() and ReplaceWithMerge().
    // For a mux node with N data inputs, CreateIndicators() adds N+2 nodes,
    // while ReplaceWithMerge() adds 1.
    num_reserved_node_ids += n->num_inputs() + 2;

    TF_RETURN_IF_ERROR(
      TransformMuxInputs(g, n, frame_info, wrapper));
  }

  TF_RETURN_IF_ERROR(TransformLoops(
    g, frame_info, temp_edge_manager, wrapper, loop_placeholders));

  indicators.clear();
  indicators.resize(g->num_node_ids() + num_reserved_node_ids, nullptr);

  for (Node* n : g->nodes()) {
    if (!n->IsMux()) continue;

    TF_RETURN_IF_ERROR(
      CreateIndicators(g, n, indicators, temp_edge_manager));

    CHECK(n->id() < indicators.size());

    // for (const Edge* e : wrapper(n->in_edges())) {
    //   if (e->IsControlEdge()) continue;

    //   DCHECK(e->src()->out_edges().size() == 1);
    //   g->AddControlEdge(e->src(), g->sink_node());
    //   temp_edge_manager.Remove(e);
    // }
  }

  return Status::OK();
}

std::vector<Node*> GetReverseExecutionOrder(Graph* g) {
  std::vector<int> pending_counts;
  pending_counts.reserve(g->num_node_ids());

  for (Node* n : g->nodes()) {
    pending_counts[n->id()] = n->out_edges().size();
  }

  std::vector<Node*> order;
  order.reserve(g->num_nodes());

  order.push_back(g->sink_node());

  for (int i = 0; i < order.size(); i++) {
    Node* n = order[i];
    for (const Edge* e : n->in_edges()) {
      Node* src = e->src();
      if (--pending_counts[src->id()] == 0) {
        order.push_back(src);
      }
    }
  }

  return order;
}

Status PropagateIndicators(Graph* g, std::vector<Node*>& indicators,
                           const std::vector<Node*>& reverse_exec_order) {
  // Collect indicators of output nodes.
  std::unordered_set<Node*, NodeHash> downstream_indicators;
  std::vector<NodeBuilder::NodeOut> node_out_vec;

  for (Node* n : reverse_exec_order) {
    if (n->IsSink()) continue;

    DCHECK(!n->out_edges().empty());

    Node*& ind = indicators[n->id()];

    if (ind != nullptr) continue;

    downstream_indicators.clear();

    for (Node* o : n->out_nodes()) {
      Node* out_ind = indicators[o->id()];
      if (out_ind == nullptr) {
        // One of the output nodes is executed unconditionally,
        // so this node is also executed unconditionally.
        ind = nullptr;
        goto _continue;
      }
      downstream_indicators.insert(out_ind);
    }

    if (downstream_indicators.size() == 1) {
      // All of the output nodes has the same indicator.
      ind = *downstream_indicators.begin();
      continue;
    }

    // The output nodes has differenct indicators. We have to create a merge
    // node as the indicator of this node.
    node_out_vec.clear();
    for (Node* di : downstream_indicators) {
      node_out_vec.emplace_back(di, 0);
    }
    TF_RETURN_IF_ERROR(
      NodeBuilder(g->NewName(strings::StrCat(n->name(), "/Merge")), "Merge")
        .Input(node_out_vec)
        .Device(n->def().device())
        .Finalize(g, &ind));
    ind->set_assigned_device_name(n->assigned_device_name());

    _continue:;
  }

  return Status::OK();
}

void AttachIndicators(Graph* g, const std::vector<Node*>& indicators,
                      const std::vector<Node*>& backbone_nodes) {
  for (Node* n : backbone_nodes) {
    Node* ind = indicators[n->id()];

    if (ind == nullptr) continue;

    for (const Edge* e : n->in_edges()) {
      if (indicators[e->src()->id()] == ind) goto _continue;
    }

    g->AddControlEdge(ind, n);

    _continue:;
  }
}

#ifndef NDEBUG

string DumpGraph(Graph* g) {
  GraphDef gd;
  g->ToGraphDef(&gd);
  string ret;
  protobuf::TextFormat::PrintToString(gd, &ret);
  return ret;
}

void AddInputToNodeDef(NodeDef& dst, StringPiece src_name, int src_slot) {
  if (src_slot == Graph::kControlSlot) {
    dst.add_input(strings::StrCat("^", src_name));
  } else if (src_slot == 0) {
    dst.add_input(src_name.data(), src_name.size());
  } else {
    dst.add_input(strings::StrCat(src_name, ":", src_slot));
  }
}

template<typename T> NodeDef NodeToDef(T node) {
  NodeDef node_def = node->def();

  // Use the node's assigned device, if any, instead of the device requested
  // in the NodeDef.
  if (!node->assigned_device_name().empty()) {
    node_def.set_device(node->assigned_device_name());
  }

  // Get the inputs for this Node.  We make sure control inputs are
  // after data inputs, as required by GraphDef.
  std::vector<const Edge*> inputs(node->num_inputs(), nullptr);
  for (const Edge* edge : node->in_edges()) {
    if (edge->IsControlEdge()) {
      inputs.push_back(edge);
    } else {
      CHECK(inputs[edge->dst_input()] == nullptr)
          << "Edge " << edge->src()->DebugString() << ":"
          << edge->dst()->DebugString() << " with dst_input "
          << edge->dst_input() << " and had pre-existing input edge "
          << inputs[edge->dst_input()]->src()->DebugString() << ":"
          << inputs[edge->dst_input()]->dst()->DebugString();

      inputs[edge->dst_input()] = edge;
    }
  }
  node_def.clear_input();
  for (size_t i = 0; i < inputs.size(); ++i) {
    const Edge* edge = inputs[i];
    if (edge == nullptr) {
      node_def.add_input(node->def().input(i));
    } else {
      const Node* src = edge->src();
      if (!src->IsOp()) continue;
      AddInputToNodeDef(node_def, src->name(), edge->src_output());
    }
  }

  return node_def;
}

template<typename T> string DumpNode(T n) {
  string ret;
  protobuf::TextFormat::PrintToString(NodeToDef(n), &ret);
  return ret;
}

template<typename T> string DumpNodes(T nodes) {
  GraphDef graph_def;
  for (auto n : nodes) {
    NodeDef* node_def = graph_def.add_node();
    *node_def = NodeToDef(n);
  }

  string ret;
  protobuf::TextFormat::PrintToString(graph_def, &ret);
  return ret;
}

struct EdgeSpec {
  std::string src;
  int src_port;
  std::string dst;
  int dst_port;
};

std::vector<EdgeSpec> GetEdges(const EdgeSet& es) {
  std::vector<EdgeSpec> ret;
  for (const Edge* e : es) {
    ret.push_back({e->src()->name(), e->src_output(),
                   e->dst()->name(), e->dst_input()});
  }
  return ret;
}

#endif // #ifndef NDEBUG

Status AddExecutionConditions(
    Graph* g, const std::vector<Node*>& target_nodes) {
  TempEdgeManager temp_edge_manager(g);
  EdgeSetWrapper wrapper;
  std::vector<Node*> indicators;
  std::vector<Node*> loop_placeholders;
  TF_RETURN_IF_ERROR(
    TransformGraph(g, target_nodes, temp_edge_manager, wrapper,
                   indicators, loop_placeholders));

  std::vector<Node*> reverse_exec_order = GetReverseExecutionOrder(g);
  TF_RETURN_IF_ERROR(PropagateIndicators(g, indicators, reverse_exec_order));
  AttachIndicators(g, indicators, reverse_exec_order);

  for (Node* n : loop_placeholders) {
    g->RemoveNode(n);
  }
  temp_edge_manager.Restore();

  for (Node* n : g->nodes()) {
    if (n->IsMux()) {
      TF_RETURN_IF_ERROR(ReplaceWithMerge(g, n));
    }
  }

  return Status::OK();
}

} // namespace

Status AddExecutionConditions(
    Graph* g, const std::unordered_set<const Node*>& target_nodes) {
  // Skip if there is no mux node at all.
  for (Node* n : g->nodes()) {
    if (n->IsMux()) goto _label_has_mux;
  }
  return Status::OK();
  _label_has_mux:;

  std::vector<Node*> mutable_target_nodes;
  mutable_target_nodes.reserve(target_nodes.size());

  for (const Node* n : target_nodes) {
    mutable_target_nodes.push_back(g->FindNodeId(n->id()));
  }

  return AddExecutionConditions(g, mutable_target_nodes);
}

}  // namespace tensorflow
