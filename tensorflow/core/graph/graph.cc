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

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_properties.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/graph/while_context.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

const int Graph::kControlSlot = -1;

// Node
Node::NodeClass Node::GetNodeClassForOp(const std::string& ts) {
  static const absl::flat_hash_map<std::string, Node::NodeClass>*
      kNodeClassTable =
#define REF_CLASS(key, value) \
  {key, value}, { "Ref" key, value }
          new absl::flat_hash_map<std::string, Node::NodeClass>({
              // Keep in same order as NodeClass values
              REF_CLASS("Switch", NC_SWITCH),
              REF_CLASS("_SwitchN", NC_SWITCH),
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
              {"_ScopedAllocator", NC_SCOPED_ALLOCATOR},
              {"CollectiveReduce", NC_COLLECTIVE},
              {"CollectiveBcastSend", NC_COLLECTIVE},
              {"CollectiveBcastRecv", NC_COLLECTIVE},
              {"CollectiveGather", NC_COLLECTIVE},
              {"FakeParam", NC_FAKE_PARAM},
              {"PartitionedCall", NC_PARTITIONED_CALL},
              {"StatefulPartitionedCall", NC_PARTITIONED_CALL},
              {"SymbolicGradient", NC_SYMBOLIC_GRADIENT},
              {"If", NC_IF},
              {"StatelessIf", NC_IF},
              {"While", NC_WHILE},
              {"StatelessWhile", NC_WHILE},
              {"Case", NC_CASE},
              {"StatelessCase", NC_CASE},
              // Not using the constants defined in FunctionLibraryDefinition
              // for the
              // 4 ops below because android inference library does not link
              // tf.function related files.
              {"_Arg", NC_ARG},
              {"_DeviceArg", NC_ARG},
              {"_Retval", NC_RETVAL},
              {"_DeviceRetval", NC_RETVAL},
              {"_XlaMerge", NC_MERGE},
          });
#undef REF_CLASS

  auto it = kNodeClassTable->find(ts);
  if (it != kNodeClassTable->end()) {
    return it->second;
  } else {
    return NC_OTHER;
  }
}

std::string Node::DebugString() const {
  std::string ret = strings::StrCat("{name:'", name(), "' id:", id_);
  if (IsSource()) {
    strings::StrAppend(&ret, " source}");
  } else if (IsSink()) {
    strings::StrAppend(&ret, " sink}");
  } else {
    strings::StrAppend(&ret, " op device:", "{requested: '", requested_device(),
                       "', assigned: '", assigned_device_name(), "'}", " def:{",
                       SummarizeNode(*this), "}}");
  }
  return ret;
}

Node::Node()
    : id_(-1),
      cost_id_(-1),
      class_(NC_UNINITIALIZED),
      props_(nullptr),
      assigned_device_name_index_(0),
      while_ctx_(nullptr) {}

void Node::Initialize(int id, int cost_id,
                      std::shared_ptr<NodeProperties> props,
                      Node::NodeClass node_class) {
  DCHECK_EQ(id_, -1);
  DCHECK(in_edges_.empty());
  DCHECK(out_edges_.empty());
  id_ = id;
  cost_id_ = cost_id;

  props_ = std::move(props);
  class_ = node_class;
}

void Node::Clear() {
  in_edges_.clear();
  out_edges_.clear();
  id_ = -1;
  cost_id_ = -1;
  class_ = NC_UNINITIALIZED;
  props_.reset();
  assigned_device_name_index_ = 0;
}

void Node::UpdateProperties() {
  DataTypeVector inputs;
  DataTypeVector outputs;
  Status status =
      InOutTypesForNode(props_->node_def, *(props_->op_def), &inputs, &outputs);
  if (!status.ok()) {
    LOG(ERROR) << "Failed at updating node: " << status;
    return;
  }
  if (props_->input_types != inputs || props_->output_types != outputs) {
    if (TF_PREDICT_TRUE(props_.use_count() == 1)) {
      props_->input_types = inputs;
      props_->input_types_slice = props_->input_types;
      props_->output_types = outputs;
      props_->output_types_slice = props_->output_types;
    } else {
      props_ = std::make_shared<NodeProperties>(
          props_->op_def, std::move(props_->node_def), inputs, outputs);
    }
  }
}

const std::string& Node::name() const { return props_->node_def.name(); }
const std::string& Node::type_string() const { return props_->node_def.op(); }
const NodeDef& Node::def() const { return props_->node_def; }
const OpDef& Node::op_def() const { return *props_->op_def; }

NodeDef* Node::mutable_def() { return &props_->node_def; }

int32 Node::num_inputs() const { return props_->input_types.size(); }
DataType Node::input_type(int32 i) const { return props_->input_types[i]; }
const DataTypeVector& Node::input_types() const { return props_->input_types; }

int32 Node::num_outputs() const { return props_->output_types.size(); }
DataType Node::output_type(int32 o) const { return props_->output_types[o]; }
const DataTypeVector& Node::output_types() const {
  return props_->output_types;
}

AttrSlice Node::attrs() const { return AttrSlice(def()); }

const protobuf::RepeatedPtrField<std::string>& Node::requested_inputs() const {
  return def().input();
}

const std::string& Node::requested_device() const { return def().device(); }

gtl::iterator_range<NeighborIter> Node::out_nodes() const {
  return gtl::make_range(NeighborIter(out_edges_.begin(), false),
                         NeighborIter(out_edges_.end(), false));
}

gtl::iterator_range<NeighborIter> Node::in_nodes() const {
  return gtl::make_range(NeighborIter(in_edges_.begin(), true),
                         NeighborIter(in_edges_.end(), true));
}

void Node::MaybeCopyOnWrite() {
  // NodeProperties may be shared between Nodes. Make a copy if so.
  if (!props_.unique()) {
    props_ = std::make_shared<NodeProperties>(*props_);
  }
}

AttrValue* Node::AddAttrHelper(const std::string& name) {
  MaybeCopyOnWrite();
  return &((*props_->node_def.mutable_attr())[name]);
}

void Node::ClearAttr(const std::string& name) {
  MaybeCopyOnWrite();
  (*props_->node_def.mutable_attr()).erase(name);
}

void Node::set_name(std::string name) {
  MaybeCopyOnWrite();
  props_->node_def.set_name(std::move(name));
}

void Node::set_requested_device(const std::string& device) {
  MaybeCopyOnWrite();
  props_->node_def.set_device(device);
}

void Node::set_original_node_names(const std::vector<std::string>& names) {
  MaybeCopyOnWrite();
  props_->node_def.mutable_experimental_debug_info()
      ->clear_original_node_names();
  if (!names.empty()) {
    *props_->node_def.mutable_experimental_debug_info()
         ->mutable_original_node_names() = {names.begin(), names.end()};
  }
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

Status Node::input_tensor(int idx, OutputTensor* t) const {
  const Edge* e;
  TF_RETURN_IF_ERROR(input_edge(idx, &e));
  DCHECK(e != nullptr);
  *t = OutputTensor(e->src(), e->src_output());
  return Status::OK();
}

// NodeDebugInfo

NodeDebugInfo::NodeDebugInfo(const Node& n) : NodeDebugInfo(n.def()) {}
NodeDebugInfo::NodeDebugInfo(const NodeDef& ndef)
    : NodeDebugInfo(ndef.name(), ndef.has_experimental_debug_info(),
                    ndef.experimental_debug_info()) {}
NodeDebugInfo::NodeDebugInfo(
    StringPiece node_name, bool has_experimental_debug_info,
    const NodeDef_ExperimentalDebugInfo& experimental_debug_info)
    : name(node_name) {
  if (has_experimental_debug_info) {
    const auto& names = experimental_debug_info.original_node_names();
    original_node_names.assign(names.begin(), names.end());
  }
}

// InputTensor

bool InputTensor::operator==(const InputTensor& other) const {
  return node == other.node && index == other.index;
}

uint64 InputTensor::Hash::operator()(InputTensor const& s) const {
  return Hash64Combine(std::hash<const Node*>()(s.node),
                       std::hash<int>()(s.index));
}

// OutputTensor

bool OutputTensor::operator==(const OutputTensor& other) const {
  return node == other.node && index == other.index;
}

uint64 OutputTensor::Hash::operator()(OutputTensor const& s) const {
  return Hash64Combine(std::hash<const Node*>()(s.node),
                       std::hash<int>()(s.index));
}

// Graph

Graph::Graph(const OpRegistryInterface* ops)
    : ops_(ops, FunctionDefLibrary()),
      versions_(new VersionDef),
      arena_(8 << 10 /* 8kB */) {
  versions_->set_producer(TF_GRAPH_DEF_VERSION);
  versions_->set_min_consumer(TF_GRAPH_DEF_VERSION_MIN_CONSUMER);

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
  // Need a new-enough consumer to support the functions we add to the graph.
  if (flib_def.num_functions() > 0 && versions_->min_consumer() < 12) {
    versions_->set_min_consumer(12);
  }
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

std::unique_ptr<Graph> Graph::Clone() {
  std::unique_ptr<Graph> new_graph(new Graph(flib_def()));
  new_graph->Copy(*this);
  return new_graph;
}

const VersionDef& Graph::versions() const { return *versions_; }
void Graph::set_versions(const VersionDef& versions) { *versions_ = versions; }

void Graph::Copy(const Graph& src) {
  SetConstructionContext(src.GetConstructionContextInternal());
  for (Node* n : nodes()) {
    CHECK(n->IsSource() || n->IsSink()) << "*dest must be empty";
  }

  // Copy GraphDef versions
  set_versions(src.versions());

  // Copy the nodes.
  // "Node in src" -> "Node in *dest"
  gtl::FlatMap<const Node*, Node*> node_map;
  node_map.reserve(src.num_nodes());
  node_map[src.source_node()] = source_node();
  node_map[src.sink_node()] = sink_node();
  for (Node* n : src.op_nodes()) {
    auto copy = CopyNode(n);
    copy->in_edges_.reserve(n->in_edges().size());
    copy->out_edges_.reserve(n->out_edges().size());
    node_map[n] = copy;
  }

  // Copy the edges
  edges_.reserve(src.num_edges());
  for (const Edge* e : src.edges()) {
    Node* src_copy = node_map[e->src()];
    Node* dst_copy = node_map[e->dst()];
    AddEdge(src_copy, e->src_output(), dst_copy, e->dst_input());
  }

  types_ = src.types_;
  node_name_to_out_type_ = src.node_name_to_out_type_;
}

Node* Graph::AddNode(NodeDef node_def, Status* status) {
  const OpRegistrationData* op_reg_data;
  status->Update(ops_.LookUp(node_def.op(), &op_reg_data));
  if (!status->ok()) return nullptr;

  DataTypeVector inputs;
  DataTypeVector outputs;
  status->Update(
      InOutTypesForNode(node_def, op_reg_data->op_def, &inputs, &outputs));
  if (!status->ok()) {
    *status = AttachDef(*status, node_def);
    return nullptr;
  }

  Node::NodeClass node_class = op_reg_data->is_function_op
                                   ? Node::NC_FUNCTION_OP
                                   : Node::GetNodeClassForOp(node_def.op());

  Node* node = AllocateNode(
      std::make_shared<NodeProperties>(&op_reg_data->op_def,
                                       std::move(node_def), inputs, outputs),
      nullptr, node_class);
  return node;
}

Node* Graph::CopyNode(const Node* node) {
  DCHECK(!node->IsSource());
  DCHECK(!node->IsSink());
  Node* copy = AllocateNode(node->props_, node, node->class_);
  copy->set_assigned_device_name(node->assigned_device_name());

  // Since the OpDef of a function may be owned by the Graph that owns 'node',
  // relookup the OpDef in the target graph. If it differs, then clone the
  // node properties with the updated OpDef.
  const OpDef* op_def;
  TF_CHECK_OK(ops_.LookUpOpDef(node->type_string(), &op_def));
  if (op_def != node->props_->op_def) {
    copy->MaybeCopyOnWrite();
    copy->props_->op_def = op_def;
  }
  copy->SetStackTrace(node->GetStackTrace());

  return copy;
}

void Graph::RemoveNode(Node* node) {
  TF_DCHECK_OK(IsValidNode(node)) << node->DebugString();
  DCHECK(!node->IsSource());
  DCHECK(!node->IsSink());

  // Remove any edges involving this node.
  for (const Edge* e : node->in_edges_) {
    CHECK_EQ(e->src_->out_edges_.erase(e), size_t{1});
    edges_[e->id_] = nullptr;
    RecycleEdge(e);
    --num_edges_;
  }
  node->in_edges_.clear();
  for (const Edge* e : node->out_edges_) {
    CHECK_EQ(e->dst_->in_edges_.erase(e), size_t{1});
    edges_[e->id_] = nullptr;
    RecycleEdge(e);
    --num_edges_;
  }
  node->out_edges_.clear();
  ReleaseNode(node);
}

const Edge* Graph::AddEdge(Node* source, int x, Node* dest, int y) {
  TF_DCHECK_OK(IsValidNode(source)) << source->DebugString();
  TF_DCHECK_OK(IsValidNode(dest)) << dest->DebugString();

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
  TF_DCHECK_OK(IsValidNode(e->src_)) << e->src_->DebugString();
  TF_DCHECK_OK(IsValidNode(e->dst_)) << e->dst_->DebugString();
  CHECK_EQ(e->src_->out_edges_.erase(e), size_t{1});
  CHECK_EQ(e->dst_->in_edges_.erase(e), size_t{1});
  CHECK_EQ(e, edges_[e->id_]);
  CHECK_GT(num_edges_, 0);

  edges_[e->id_] = nullptr;
  RecycleEdge(e);
  --num_edges_;
}

void Graph::RecycleEdge(const Edge* e) {
  free_edges_.push_back(const_cast<Edge*>(e));
}

const Edge* Graph::AddControlEdge(Node* source, Node* dest,
                                  bool allow_duplicates) {
  if (!allow_duplicates) {
    for (const Edge* edge : dest->in_edges()) {
      if (edge->IsControlEdge() && edge->src() == source) {
        // The requested edge already exists.
        return nullptr;
      }
    }
  }
  // Modify dest's NodeDef if necessary.
  if (!source->IsSource() && !dest->IsSink() && !allow_duplicates) {
    // Check if this input is already in dest's NodeDef.
    const std::string new_input = strings::StrCat("^", source->name());
    bool input_exists = false;
    for (const std::string& input : dest->props_->node_def.input()) {
      if (input == new_input) {
        input_exists = true;
        break;
      }
    }
    if (!input_exists) {
      dest->MaybeCopyOnWrite();
      dest->props_->node_def.add_input(new_input);
    }
  }
  return AddEdge(source, kControlSlot, dest, kControlSlot);
}

void Graph::RemoveControlEdge(const Edge* e) {
  if (!e->src_->IsSource() && !e->dst_->IsSink()) {
    e->dst_->MaybeCopyOnWrite();
    std::string e_src_name = strings::StrCat("^", e->src_->name());
    auto* inputs = e->dst_->props_->node_def.mutable_input();
    for (auto it = inputs->begin(); it != inputs->end(); ++it) {
      if (*it == e_src_name) {
        inputs->erase(it);
        break;
      }
    }
  }
  RemoveEdge(e);
}

namespace {
const Edge* FindEdge(const Node* dst, int index) {
  for (const Edge* e : dst->in_edges()) {
    if (e->dst_input() == index) return e;
  }
  return nullptr;
}
}  // namespace

Status Graph::UpdateEdge(Node* new_src, int new_src_index, Node* dst,
                         int dst_index) {
  TF_RETURN_IF_ERROR(IsValidOutputTensor(new_src, new_src_index));
  TF_RETURN_IF_ERROR(IsValidInputTensor(dst, dst_index));
  const Edge* e = FindEdge(dst, dst_index);
  if (e == nullptr) {
    return errors::InvalidArgument("Couldn't find edge to ",
                                   FormatNodeForError(*dst));
  }
  RemoveEdge(e);
  AddEdge(new_src, new_src_index, dst, dst_index);
  dst->MaybeCopyOnWrite();
  (*dst->props_->node_def.mutable_input())[dst_index] =
      strings::StrCat(new_src->name(), ":", new_src_index);
  return Status::OK();
}

Status Graph::AddWhileInputHack(Node* new_src, int new_src_index, Node* dst) {
  if (!dst->IsWhileNode()) {
    return errors::Internal(
        "dst argument to AddWhileEdgeHack should be a While op, got: ",
        dst->DebugString());
  }
  TF_RETURN_IF_ERROR(IsValidOutputTensor(new_src, new_src_index));
  // Find the current number of data inputs. We'll add the new edge to the next
  // missing data input.
  int dst_index = 0;
  for (const Edge* edge : dst->in_edges()) {
    if (edge->IsControlEdge()) continue;
    ++dst_index;
  }
  TF_RETURN_IF_ERROR(IsValidInputTensor(dst, dst_index));
  AddEdge(new_src, new_src_index, dst, dst_index);
  dst->MaybeCopyOnWrite();
  dst->props_->node_def.add_input(
      strings::StrCat(new_src->name(), ":", new_src_index));
  return Status::OK();
}

Status Graph::AddFunctionLibrary(const FunctionDefLibrary& fdef_lib) {
  // Need a new-enough consumer to support the functions we add to the graph.
  if (fdef_lib.function_size() > 0 && versions_->min_consumer() < 12) {
    versions_->set_min_consumer(12);
  }
  return ops_.AddLibrary(fdef_lib);
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

GraphDef Graph::ToGraphDefDebug() const {
  GraphDef ret;
  ToGraphDef(&ret);
  return ret;
}

void Graph::ToGraphDefSubRange(GraphDef* graph_def, int from_node_id) const {
  graph_def->Clear();
  *graph_def->mutable_versions() = versions();
  *graph_def->mutable_library() = ops_.ToProto();

  graph_def->mutable_node()->Reserve(std::max(1, num_nodes() - from_node_id));

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
        DCHECK(edge->dst_input() < inputs.size())
            << "Edge " << edge->DebugString()
            << " is overflowing the expected number of inputs ("
            << node->num_inputs() << ") for node " << node->DebugString();
        CHECK(inputs[edge->dst_input()] == nullptr)
            << "Edge " << edge->src()->name() << "->" << edge->dst()->name()
            << " conflicts with pre-existing input edge "
            << inputs[edge->dst_input()]->src()->name() << "->"
            << inputs[edge->dst_input()]->dst()->name();

        inputs[edge->dst_input()] = edge;
      }
    }
    // Sort the control inputs for more predictable serialization.
    std::sort(inputs.begin() + node->num_inputs(), inputs.end(),
              [](const Edge* a, const Edge* b) -> bool {
                return a->src()->name() < b->src()->name();
              });
    node_def->clear_input();
    node_def->mutable_input()->Reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
      const Edge* edge = inputs[i];
      if (edge == nullptr) {
        if (i < node->requested_inputs().size()) {
          node_def->add_input(node->requested_inputs()[i]);
        } else {
          node_def->add_input("");
        }
      } else {
        const Node* src = edge->src();
        if (!src->IsOp()) continue;
        AddInput(node_def, src->name(), edge->src_output());
      }
    }
  }
}

std::string Graph::NewName(StringPiece prefix) {
  return strings::StrCat(prefix, "/_", name_counter_++);
}

Status Graph::IsValidNode(const Node* node) const {
  if (node == nullptr) {
    return errors::InvalidArgument("Node is null");
  }
  const int id = node->id();
  if (id < 0) {
    return errors::InvalidArgument("node id ", id, " is less than zero");
  }
  if (static_cast<size_t>(id) >= nodes_.size()) {
    return errors::InvalidArgument(
        "node id ", id, " is >= than number of nodes in graph ", nodes_.size());
  }
  if (nodes_[id] != node) {
    return errors::InvalidArgument("Node with id ", id,
                                   " is different from the passed in node. "
                                   "Does it belong to a different graph?");
  }
  return Status::OK();
}

Status Graph::IsValidOutputTensor(const Node* node, int idx) const {
  TF_RETURN_IF_ERROR(IsValidNode(node));
  if (idx >= node->num_outputs() || idx < 0) {
    return errors::OutOfRange("Node '", node->name(), "' (type: '",
                              node->op_def().name(),
                              "', num of outputs: ", node->num_outputs(),
                              ") does not have ", "output ", idx);
  }
  return Status::OK();
}

Status Graph::IsValidInputTensor(const Node* node, int idx) const {
  TF_RETURN_IF_ERROR(IsValidNode(node));
  if (idx >= node->num_inputs() || idx < 0) {
    return errors::OutOfRange("Node '", node->name(), "' (type: '",
                              node->op_def().name(),
                              "', num of inputs: ", node->num_inputs(),
                              ") does not have ", "input ", idx);
  }
  return Status::OK();
}

Node* Graph::AllocateNode(std::shared_ptr<NodeProperties> props,
                          const Node* cost_node, Node::NodeClass node_class) {
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
  node->Initialize(id, cost_id, std::move(props), node_class);
  nodes_.push_back(node);
  ++num_nodes_;
  return node;
}

void Graph::ReleaseNode(Node* node) {
  TF_DCHECK_OK(IsValidNode(node)) << node->DebugString();
  nodes_[node->id()] = nullptr;
  free_nodes_.push_back(node);
  --num_nodes_;
  node->Clear();
}

// Ensures that 'device_name' is present in the device name table, and returns
// the index of that device name. The index is stable, and can be used in
// calls to Node::set_assigned_device_name_index().
int Graph::InternDeviceName(const std::string& device_name) {
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

Status Graph::AddWhileContext(StringPiece frame_name,
                              std::vector<Node*> enter_nodes,
                              std::vector<Node*> exit_nodes,
                              OutputTensor cond_output,
                              std::vector<OutputTensor> body_inputs,
                              std::vector<OutputTensor> body_outputs,
                              WhileContext** result) {
  auto pair = while_ctxs_.insert(std::pair<std::string, WhileContext>(
      std::string(frame_name),
      WhileContext(frame_name, std::move(enter_nodes), std::move(exit_nodes),
                   cond_output, std::move(body_inputs),
                   std::move(body_outputs))));
  if (!pair.second) {
    *result = nullptr;
    return errors::InvalidArgument("WhileContext with frame name '", frame_name,
                                   "' already exists");
  }
  *result = &pair.first->second;
  return Status::OK();
}

std::unordered_map<std::string, Node*> Graph::BuildNodeNameIndex() const {
  std::unordered_map<std::string, Node*> result;
  for (Node* n : nodes()) {
    result[n->name()] = n;
  }
  return result;
}

void Graph::SetNodeType(StringPiece name, const FullTypeDef& ft) {
  TypeRef t = {std::make_shared<FullTypeDef>(ft)};
  auto ret = types_.emplace(t);
  if (ret.second == false) {
    t = *ret.first;
  }

  node_name_to_out_type_.emplace(string(name), t);
}

void Graph::NodeType(StringPiece name, FullTypeDef** result) {
  *result = nullptr;
  auto it = node_name_to_out_type_.find(string(name));
  if (it == node_name_to_out_type_.end()) {
    *result = nullptr;
    return;
  }
  *result = it->second.full_type.get();
}

std::string Edge::DebugString() const {
  return strings::Printf("[id=%d %s:%d -> %s:%d]", id_, src_->name().c_str(),
                         src_output_, dst_->name().c_str(), dst_input_);
}

}  // namespace tensorflow
