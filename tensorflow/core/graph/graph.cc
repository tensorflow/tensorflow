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

const int Graph::kControlSlot = -1;

// Node

#define REF_CLASS(key, value) \
  {key, value}, { "Ref" key, value }

const std::unordered_map<string, Node::NodeClass>& Node::kNodeClassTable =
    *new std::unordered_map<string, Node::NodeClass>({
        // Keep in same order as NodeClass values
        REF_CLASS("Switch", NC_SWITCH),
        REF_CLASS("Merge", NC_MERGE),
        REF_CLASS("Enter", NC_ENTER),
        REF_CLASS("Exit", NC_EXIT),
        REF_CLASS("NextIteration", NC_NEXT_ITERATION),
        {"LoopCond", NC_LOOP_COND},
        {"ControlTrigger", NC_CONTROL_TRIGGER},
        {"_Send", NC_SEND},
        {"_HostSend", NC_HOST_SEND},
        {"_Recv", NC_RECV},
        {"_HostRecv", NC_HOST_RECV},
        {"Const", NC_CONSTANT},
        {"HostConst", NC_CONSTANT},
        {"Variable", NC_VARIABLE},
        {"VariableV2", NC_VARIABLE},
        REF_CLASS("Identity", NC_IDENTITY),
        {"GetSessionHandle", NC_GET_SESSION_HANDLE},
        {"GetSessionHandleV2", NC_GET_SESSION_HANDLE},
        {"GetSessionTensor", NC_GET_SESSION_TENSOR},
        {"DeleteSessionTensor", NC_DELETE_SESSION_TENSOR},
        {"Size", NC_METADATA},
        {"Shape", NC_METADATA},
        {"Rank", NC_METADATA},
    });

#undef REF_CLASS

Node::NodeClass Node::GetNodeClassForOp(const string& ts) {
  auto it = kNodeClassTable.find(ts);
  if (it != kNodeClassTable.end()) {
    return it->second;
  } else {
    return NC_OTHER;
  }
}

string Node::DebugString() const {
  string ret = strings::StrCat("{name:'", name(), "' id:", id_);
  if (IsSource()) {
    strings::StrAppend(&ret, " source}");
  } else if (IsSink()) {
    strings::StrAppend(&ret, " sink}");
  } else {
    strings::StrAppend(&ret, " op device:");
    strings::StrAppend(&ret, "{", assigned_device_name(), "}");
    strings::StrAppend(&ret, " def:{", SummarizeNode(*this), "}}");
  }
  return ret;
}

Node::Node()
    : id_(-1),
      cost_id_(-1),
      class_(NC_UNINITIALIZED),
      props_(nullptr),
      assigned_device_name_index_(0) {}

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
  class_ = GetNodeClassForOp(props->node_def_.op());
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

  assigned_device_name_index_ = 0;
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

Status Node::input_edge(int idx, const Edge** e) const {
  if (idx < 0 || idx >= num_inputs()) {
    return errors::InvalidArgument("Invalid input_edge index: ", idx, ", Node ",
                                   name(), " only has ", num_inputs(),
                                   " inputs.");
  }

  // This does a linear search over the edges.  In the common case,
  // the number of elements is small enough that this search isn't
  // expensive.  Should it become a bottleneck, one can make an
  // optimization where, if the number of edges is small, we use
  // linear iteration, and if the number of edges is large, we perform
  // an indexing step during construction that keeps an array of Edges
  // indexed by pointer.  This would keep the size of each Node small
  // in the common case but make this function faster when the number
  // of edges is large.
  for (const Edge* edge : in_edges()) {
    if (edge->dst_input() == idx) {
      *e = edge;
      return Status::OK();
    }
  }

  return errors::NotFound("Could not find input edge ", idx, " for ", name());
}

// Returns a vector of the non-control input edges to a node, indexed by ID.
Status Node::input_edges(std::vector<const Edge*>* input_edges) const {
  input_edges->clear();
  input_edges->resize(num_inputs(), nullptr);

  for (const Edge* edge : in_edges()) {
    if (edge->IsControlEdge()) continue;
    if (edge->dst_input() < 0 || edge->dst_input() >= num_inputs()) {
      return errors::Internal("Invalid edge input number ", edge->dst_input());
    }
    if ((*input_edges)[edge->dst_input()] != nullptr) {
      return errors::Internal("Duplicate edge input number: ",
                              edge->dst_input());
    }
    (*input_edges)[edge->dst_input()] = edge;
  }

  for (int i = 0; i < num_inputs(); ++i) {
    if ((*input_edges)[i] == nullptr) {
      return errors::InvalidArgument("Missing edge input number: ", i);
    }
  }
  return Status::OK();
}

Status Node::input_node(int idx, Node** n) const {
  const Edge* e;
  TF_RETURN_IF_ERROR(input_edge(idx, &e));
  if (e == nullptr) {
    *n = nullptr;
  } else {
    *n = e->src();
  }
  return Status::OK();
}

Status Node::input_node(int idx, const Node** const_n) const {
  Node* n;
  TF_RETURN_IF_ERROR(input_node(idx, &n));
  *const_n = n;
  return Status::OK();
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
    : ops_(ops, FunctionDefLibrary()), arena_(8 << 10 /* 8kB */) {
  versions_.set_producer(TF_GRAPH_DEF_VERSION);
  versions_.set_min_consumer(TF_GRAPH_DEF_VERSION_MIN_CONSUMER);

  // Initialize the name interning table for assigned_device_name.
  device_names_.push_back("");
  DCHECK_EQ(0, InternDeviceName(""));

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

Graph::Graph(const FunctionLibraryDefinition& flib_def)
    : Graph(flib_def.default_registry()) {
  Status s = ops_.AddLibrary(flib_def);
  CHECK(s.ok()) << s.error_message();
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
  const OpDef* op_def;
  status->Update(ops_.LookUpOpDef(node_def.op(), &op_def));
  if (!status->ok()) return nullptr;

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

  // Since the OpDef of a function may be owned by the Graph that owns 'node',
  // relookup the OpDef in the target graph. If it differs, then clone the
  // node properties with the updated OpDef.
  const OpDef* op_def;
  TF_CHECK_OK(ops_.LookUpOpDef(node->type_string(), &op_def));
  if (op_def != props->op_def_) {
    copy->MaybeCopyOnWrite();
    copy->props_->op_def_ = op_def;
  }

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
  ++num_edges_;
  return e;
}

void Graph::RemoveEdge(const Edge* e) {
  DCHECK(IsValidNode(e->src_)) << e->src_->DebugString();
  DCHECK(IsValidNode(e->dst_)) << e->dst_->DebugString();
  CHECK_EQ(e->src_->out_edges_.erase(e), size_t{1});
  CHECK_EQ(e->dst_->in_edges_.erase(e), size_t{1});
  CHECK_EQ(e, edges_[e->id_]);
  CHECK_GT(num_edges_, 0);

  edges_[e->id_] = nullptr;

  Edge* del = const_cast<Edge*>(e);
  del->src_ = nullptr;
  del->dst_ = nullptr;
  del->id_ = -1;
  del->src_output_ = kControlSlot - 1;
  del->dst_input_ = kControlSlot - 1;
  free_edges_.push_back(del);
  --num_edges_;
}

Status Graph::AddFunctionLibrary(const FunctionDefLibrary& fdef_lib) {
  for (const FunctionDef& fdef : fdef_lib.function()) {
    const FunctionDef* preexisting_fdef = ops_.Find(fdef.signature().name());
    if (preexisting_fdef != nullptr) {
      if (!FunctionDefsEqual(*preexisting_fdef, fdef)) {
        return errors::InvalidArgument(
            "Cannot add function '", fdef.signature().name(),
            "' because a different function with the same name already "
            "exists.");
      }
      // Ignore duplicate FunctionDefs
      continue;
    }
    TF_RETURN_IF_ERROR(ops_.AddFunctionDef(fdef));
  }
  for (const GradientDef& grad : fdef_lib.gradient()) {
    string preexisting_grad_func = ops_.FindGradient(grad.function_name());
    if (!preexisting_grad_func.empty()) {
      if (preexisting_grad_func != grad.gradient_func()) {
        return errors::InvalidArgument(
            "Cannot assign gradient function '", grad.gradient_func(), "' to '",
            grad.function_name(), "' because it already has gradient function ",
            "'", preexisting_grad_func, "'");
      }
      // Ignore duplicate GradientDefs
      continue;
    }
    TF_RETURN_IF_ERROR(ops_.AddGradientDef(grad));
  }
  return Status::OK();
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
  ToGraphDefSubRange(graph_def, 0);
}

void Graph::ToGraphDefSubRange(GraphDef* graph_def, int from_node_id) const {
  graph_def->Clear();
  *graph_def->mutable_versions() = versions();
  *graph_def->mutable_library() = ops_.ToProto();
  std::vector<const Edge*>
      inputs;  // Construct this outside the loop for speed.
  for (auto id = from_node_id; id < num_node_ids(); ++id) {
    const Node* node = FindNodeId(id);
    if (node == nullptr || !node->IsOp()) continue;
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
        node_def->add_input(node->requested_inputs()[i]);
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
  node->graph_ = this;
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

// Ensures that 'device_name' is present in the device name table, and returns
// the index of that device name. The index is stable, and can be used in
// calls to Node::set_assigned_device_name_index().
int Graph::InternDeviceName(const string& device_name) {
  // Special case, very common.  Also, this allows us to use a single map
  // lookup below, instead of two.  The 'if (index_cell > 0)' test below
  // relies on this check.
  if (device_name.empty()) {
    return 0;
  }

  int& index_cell = device_names_map_[device_name];
  if (index_cell > 0) {
    return index_cell;
  }

  const int index = device_names_map_.size();
  index_cell = index;
  device_names_.push_back(device_name);
  return index;
}

}  // namespace tensorflow
