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

#include "tensorflow/core/graph/graph.h"

#include <vector>
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

// Node

string Node::DebugString() const {
  if (this == nullptr) {
    return "{nullptr}";
  }
  string ret = strings::StrCat("{name:'", name(), "' id:", id_);
  if (IsSource()) {
    strings::StrAppend(&ret, " source}");
  } else if (IsSink()) {
    strings::StrAppend(&ret, " sink}");
  } else {
    strings::StrAppend(&ret, " op device:");
    strings::StrAppend(&ret, "{", assigned_device_name_, "}");
    strings::StrAppend(&ret, " def:{", SummarizeNodeDef(def()), "}}");
  }
  return ret;
}

Node::Node()
    : id_(-1),
      cost_id_(-1),
      class_(NC_UNINITIALIZED),
      props_(nullptr),
      assigned_device_name_() {}

Node::~Node() {
  if (props_) {
    props_->Unref();
  }
}

void Node::Initialize(int id, int cost_id, Properties* props) {
  DCHECK_EQ(id_, -1);
  DCHECK(in_edges_.empty());
  DCHECK(out_edges_.empty());
  id_ = id;
  cost_id_ = cost_id;

  // Unref the old, assign the new properties.
  if (props_) {
    props_->Unref();
  }
  props_ = props;
  // Initialize the class_ based on the type string
  const string& ts = this->type_string();
  class_ = NC_UNINITIALIZED;

#define SET_CLASS(enum_val, ts, str1, str2)        \
  do {                                             \
    if ((((ts) == (str1)) || ((ts) == (str2)))) {  \
      /* Cannot be member of more than one class*/ \
      CHECK(class_ == NC_UNINITIALIZED);           \
      class_ = (enum_val);                         \
    }                                              \
  } while (0)

  SET_CLASS(NC_SWITCH, ts, "Switch", "RefSwitch");
  SET_CLASS(NC_MERGE, ts, "Merge", "RefMerge");
  SET_CLASS(NC_ENTER, ts, "Enter", "RefEnter");
  SET_CLASS(NC_EXIT, ts, "Exit", "RefExit");
  SET_CLASS(NC_NEXT_ITERATION, ts, "NextIteration", "RefNextIteration");
  SET_CLASS(NC_LOOP_COND, ts, "LoopCond", "");
  SET_CLASS(NC_CONTROL_TRIGGER, ts, "ControlTrigger", "");
  SET_CLASS(NC_SEND, ts, "_Send", "_HostSend");
  SET_CLASS(NC_RECV, ts, "_Recv", "_HostRecv");
  SET_CLASS(NC_CONSTANT, ts, "Const", "HostConst");
  SET_CLASS(NC_VARIABLE, ts, "Variable", "");
  SET_CLASS(NC_IDENTITY, ts, "Identity", "RefIdentity");
  SET_CLASS(NC_GET_SESSION_HANDLE, ts, "GetSessionHandle", "");
  SET_CLASS(NC_GET_SESSION_TENSOR, ts, "GetSessionTensor", "");
  SET_CLASS(NC_DELETE_SESSION_TENSOR, ts, "DeleteSessionTensor", "");
  if (class_ == NC_UNINITIALIZED) {
    class_ = NC_OTHER;  // Catch all
  }
#undef SET_CLASS
}

void Node::Clear() {
  in_edges_.clear();
  out_edges_.clear();
  id_ = -1;
  cost_id_ = -1;
  class_ = NC_UNINITIALIZED;

  if (props_) {
    props_->Unref();
    props_ = nullptr;
  }

  assigned_device_name_.clear();
}

gtl::iterator_range<NeighborIter> Node::out_nodes() const {
  return gtl::make_range(NeighborIter(out_edges_.begin(), false),
                         NeighborIter(out_edges_.end(), false));
}

gtl::iterator_range<NeighborIter> Node::in_nodes() const {
  return gtl::make_range(NeighborIter(in_edges_.begin(), true),
                         NeighborIter(in_edges_.end(), true));
}

void Node::MaybeCopyOnWrite() {
  // Properties may be shared between Nodes. Make a copy if so.
  if (!props_->RefCountIsOne()) {
    Properties* new_props =
        new Properties(props_->op_def_, props_->node_def_, props_->input_types_,
                       props_->output_types_);
    props_->Unref();
    props_ = new_props;
  }
}

void Node::ClearAttr(const string& name) {
  MaybeCopyOnWrite();
  (*props_->node_def_.mutable_attr()).erase(name);
}

// Node::Properties

Node::Properties::Properties(const OpDef* op_def, const NodeDef& node_def,
                             const DataTypeSlice inputs,
                             const DataTypeSlice outputs)
    : op_def_(op_def),
      node_def_(node_def),
      input_types_(inputs.begin(), inputs.end()),
      output_types_(outputs.begin(), outputs.end()) {}

Node::Properties::~Properties() {}

// Graph

Graph::Graph(const OpRegistryInterface* ops)
    : ops_(ops), arena_(8 << 10 /* 8kB */) {
  versions_.set_producer(TF_GRAPH_DEF_VERSION);
  versions_.set_min_consumer(TF_GRAPH_DEF_VERSION_MIN_CONSUMER);

  // Source and sink have no endpoints, just control edges.
  NodeDef def;
  def.set_name("_SOURCE");
  def.set_op("NoOp");
  Status status;
  Node* source = AddNode(def, &status);
  TF_CHECK_OK(status);
  CHECK_EQ(source->id(), kSourceId);

  def.set_name("_SINK");
  Node* sink = AddNode(def, &status);
  TF_CHECK_OK(status);
  CHECK_EQ(sink->id(), kSinkId);

  AddControlEdge(source, sink);
}

Graph::~Graph() {
  // Manually call the destructors for all the Nodes we constructed using
  // placement new.
  for (Node* node : nodes_) {
    if (node != nullptr) {
      node->~Node();
    }
  }
  for (Node* node : free_nodes_) {
    node->~Node();
  }
  // Edges have no destructor, and we arena-allocated them, so no need to
  // destroy them.
}

Node* Graph::AddNode(const NodeDef& node_def, Status* status) {
  const OpDef* op_def = ops_->LookUp(node_def.op(), status);
  if (op_def == nullptr) return nullptr;

  DataTypeVector inputs;
  DataTypeVector outputs;
  status->Update(InOutTypesForNode(node_def, *op_def, &inputs, &outputs));
  if (!status->ok()) {
    *status = AttachDef(*status, node_def);
    return nullptr;
  }

  Node* node = AllocateNode(
      new Node::Properties(op_def, node_def, inputs, outputs), nullptr);
  return node;
}

Node* Graph::CopyNode(Node* node) {
  DCHECK(!node->IsSource());
  DCHECK(!node->IsSink());
  Node::Properties* props = node->properties();
  props->Ref();
  Node* copy = AllocateNode(props, node);
  copy->set_assigned_device_name(node->assigned_device_name());
  return copy;
}

void Graph::RemoveNode(Node* node) {
  DCHECK(IsValidNode(node)) << node->DebugString();
  DCHECK(!node->IsSource());
  DCHECK(!node->IsSink());

  // Remove any edges involving this node.
  while (!node->in_edges_.empty()) {
    RemoveEdge(*node->in_edges_.begin());
  }
  while (!node->out_edges_.empty()) {
    RemoveEdge(*node->out_edges_.begin());
  }
  ReleaseNode(node);
}

const Edge* Graph::AddEdge(Node* source, int x, Node* dest, int y) {
  DCHECK(IsValidNode(source)) << source->DebugString();
  DCHECK(IsValidNode(dest)) << dest->DebugString();

  // source/sink must only be linked via control slots, and
  // control slots must only be linked to control slots.
  if (source == source_node() || dest == sink_node() || x == kControlSlot ||
      y == kControlSlot) {
    DCHECK_EQ(x, kControlSlot) << source->DebugString();
    DCHECK_EQ(y, kControlSlot) << dest->DebugString();
  }

  Edge* e = nullptr;
  if (free_edges_.empty()) {
    e = new (arena_.Alloc(sizeof(Edge))) Edge;  // placement new
  } else {
    e = free_edges_.back();
    free_edges_.pop_back();
  }
  e->id_ = edges_.size();
  e->src_ = source;
  e->dst_ = dest;
  e->src_output_ = x;
  e->dst_input_ = y;
  CHECK(source->out_edges_.insert(e).second);
  CHECK(dest->in_edges_.insert(e).second);
  edges_.push_back(e);
  edge_set_.insert(e);
  return e;
}

void Graph::RemoveEdge(const Edge* e) {
  DCHECK(IsValidNode(e->src_)) << e->src_->DebugString();
  DCHECK(IsValidNode(e->dst_)) << e->dst_->DebugString();
  CHECK_EQ(e->src_->out_edges_.erase(e), size_t{1});
  CHECK_EQ(e->dst_->in_edges_.erase(e), size_t{1});
  CHECK_EQ(e, edges_[e->id_]);

  CHECK_EQ(edge_set_.erase(e), size_t{1});
  edges_[e->id_] = nullptr;

  Edge* del = const_cast<Edge*>(e);
  del->src_ = nullptr;
  del->dst_ = nullptr;
  del->id_ = -1;
  del->src_output_ = kControlSlot - 1;
  del->dst_input_ = kControlSlot - 1;
  free_edges_.push_back(del);
}

namespace {

void AddInput(NodeDef* dst, StringPiece src_name, int src_slot) {
  if (src_slot == Graph::kControlSlot) {
    dst->add_input(strings::StrCat("^", src_name));
  } else if (src_slot == 0) {
    dst->add_input(src_name.data(), src_name.size());
  } else {
    dst->add_input(strings::StrCat(src_name, ":", src_slot));
  }
}

}  // namespace

void Graph::ToGraphDef(GraphDef* graph_def) const {
  graph_def->Clear();
  graph_def->mutable_versions()->CopyFrom(versions());
  std::vector<const Edge*>
      inputs;  // Construct this outside the loop for speed.
  for (const Node* node : nodes()) {
    if (!node->IsOp()) continue;
    NodeDef* node_def = graph_def->add_node();
    *node_def = node->def();

    // Use the node's assigned device, if any, instead of the device requested
    // in the NodeDef.
    if (!node->assigned_device_name().empty()) {
      node_def->set_device(node->assigned_device_name());
    }

    // Get the inputs for this Node.  We make sure control inputs are
    // after data inputs, as required by GraphDef.
    inputs.clear();
    inputs.resize(node->num_inputs(), nullptr);
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
    node_def->clear_input();
    for (size_t i = 0; i < inputs.size(); ++i) {
      const Edge* edge = inputs[i];
      if (edge == nullptr) {
        node_def->add_input(node->def().input(i));
      } else {
        const Node* src = edge->src();
        if (!src->IsOp()) continue;
        AddInput(node_def, src->name(), edge->src_output());
      }
    }
  }
}

string Graph::NewName(StringPiece prefix) {
  return strings::StrCat(prefix, "/_", name_counter_++);
}

gtl::iterator_range<NodeIter> Graph::nodes() const {
  // Note that NodeId 0 is always valid since we don't let the source
  // node be removed from the graph.
  return gtl::make_range(NodeIter(this, 0), NodeIter(this, num_node_ids()));
}

bool Graph::IsValidNode(Node* node) const {
  if (node == nullptr) return false;
  const int id = node->id();
  if (id < 0 || static_cast<size_t>(id) >= nodes_.size()) return false;
  return nodes_[id] == node;
}

Node* Graph::AllocateNode(Node::Properties* props, const Node* cost_node) {
  Node* node = nullptr;
  if (free_nodes_.empty()) {
    node = new (arena_.Alloc(sizeof(Node))) Node;  // placement new
  } else {
    node = free_nodes_.back();
    free_nodes_.pop_back();
  }
  const int id = nodes_.size();
  int cost_id = cost_node ? cost_node->cost_id() : id;
  node->Initialize(id, cost_id, props);
  nodes_.push_back(node);
  ++num_nodes_;
  return node;
}

void Graph::ReleaseNode(Node* node) {
  DCHECK(IsValidNode(node)) << node->DebugString();
  nodes_[node->id()] = nullptr;
  free_nodes_.push_back(node);
  --num_nodes_;
  node->Clear();
}

}  // namespace tensorflow
