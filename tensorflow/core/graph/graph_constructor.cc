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

#include "tensorflow/core/graph/graph_constructor.h"

#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/versions.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

namespace {
inline bool IsMerge(const NodeDef& node_def) {
  return node_def.op() == "Merge" || node_def.op() == "RefMerge";
}

bool IsValidNodeName(StringPiece s, bool allow_internal_ops) {
  using ::tensorflow::strings::Scanner;
  return Scanner(s)
      .One(allow_internal_ops ? Scanner::LETTER_DIGIT_DOT_UNDERSCORE
                              : Scanner::LETTER_DIGIT_DOT)
      .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE)
      .Eos()
      .GetResult();
}

class GraphConstructor {
 public:
  struct Options {
    Options(const GraphConstructorOptions& in)
        : allow_internal_ops(in.allow_internal_ops),
          expect_device_spec(in.expect_device_spec),
          importing(false) {}
    Options(const ImportGraphDefOptions& in)
        : allow_internal_ops(false),
          expect_device_spec(false),
          prefix(in.prefix.empty() || StringPiece(in.prefix).ends_with("/")
                     ? in.prefix
                     : in.prefix + "/"),
          input_map(in.input_map),
          control_dependencies(in.control_dependencies),
          return_tensors(in.return_tensors),
          importing(true) {}

    bool allow_internal_ops;
    bool expect_device_spec;

    string prefix;
    std::map<TensorId, TensorId> input_map;
    std::vector<string> control_dependencies;
    std::vector<TensorId> return_tensors;

    // TODO(ashankar): This bool exists to separate out functionality required
    // to make ImportGraphDef a close equivalent of Python's import_graph_def
    // without affecting the behavior of ConvertGraphDefToGraph at the time
    // ImportGraphDef was added.
    //
    // That said, the functionality here (shape and op validation) seems
    // applicable to ConvertGraphDefToGraph as well, so make an attempt to
    // remove this.
    bool importing;
  };

  static Status Construct(const Options& opts, const GraphDef* gdef, Graph* g,
                          ShapeRefiner* refiner,
                          std::vector<std::pair<Node*, int>>* return_tensors) {
    TF_RETURN_IF_ERROR(CheckVersions(gdef->versions(), TF_GRAPH_DEF_VERSION,
                                     TF_GRAPH_DEF_VERSION_MIN_PRODUCER,
                                     "GraphDef", "graph"));
    GraphConstructor c(opts, gdef, g, refiner, return_tensors);
    const Status s = c.TryImport();
    if (!s.ok()) c.Undo();
    return s;
  }

 private:
  GraphConstructor(const Options& opts, const GraphDef* gdef, Graph* g,
                   ShapeRefiner* refiner,
                   std::vector<std::pair<Node*, int>>* return_tensors)
      : opts_(opts),
        gdef_(gdef),
        g_(g),
        original_versions_(g->versions()),
        refiner_(refiner),
        return_tensors_(return_tensors) {}

  Status TryImport() {
    TF_RETURN_IF_ERROR(EnsureNoNameCollisions());
    TF_RETURN_IF_ERROR(ValidateInputMapAndControlDependencies());
    TF_RETURN_IF_ERROR(BuildNodeIndex());
    TF_RETURN_IF_ERROR(InitFromEdges());
    TF_RETURN_IF_ERROR(Convert());
    TF_RETURN_IF_ERROR(AddBackEdges());
    TF_RETURN_IF_ERROR(UpdateVersionDef());
    TF_RETURN_IF_ERROR(PopulateReturnTensors());
    FixupSourceAndSinkEdges(g_);
    return Status::OK();
  }

  Status EnsureNoNameCollisions();
  Status ValidateInputMapAndControlDependencies();
  Status BuildNodeIndex();
  Status InitFromEdges();
  Status Convert();
  Status AddBackEdges();
  Status UpdateVersionDef();
  Status PopulateReturnTensors();

  void Undo();

  Status ValidateColocationConstraints(const NodeDef& node_def);
  Status MakeNode(const NodeDef& node_def, Node** node);
  Status MakeEdge(Node* src, int output_index, Node* dst, int input_index);
  Status ValidateShape(Node* node);
  Status ModifyNodeDefForImport(NodeDef* node_def);
  // Modifies node_def's inputs according to opts_.input_map.
  // input_already_exists is a pre-initialized vector of length
  // node_def->input_size(). This function will mark inputs that are remapped to
  // true.
  void RemapNodeDefInputs(NodeDef* node_def,
                          std::vector<bool>* input_already_exists);
  // input_already_exists is a pre-initialized vector of length
  // node_def->input_size(). This function will add and mark control inputs as
  // true.
  void AddControlDependencies(NodeDef* node_def,
                              std::vector<bool>* input_already_exists);
  void AddPrefixToNodeDef(const std::vector<bool>& input_already_exists,
                          NodeDef* node_def);

  // From constructor
  const Options opts_;
  const GraphDef* gdef_;
  Graph* g_;
  const VersionDef original_versions_;

  ShapeRefiner* refiner_;

  // May be null. Not owned.
  std::vector<std::pair<Node*, int>>* return_tensors_;

  // Mapping from node name to the index within gdef_
  struct NodeInfo {
    explicit NodeInfo(int i) : gdef_index(i), node(nullptr) {}
    // std::unordered_map<> requires that we have a default constructor.
    NodeInfo() : NodeInfo(-1) {}
    int gdef_index;
    Node* node;  // nullptr until the NodeDef is converted to a Node.
  };
  // TODO(vrv): Profile this data structure to see if we should use an
  // alternative implementation of std::unordered_map.
  std::unordered_map<StringPiece, NodeInfo, StringPiece::Hasher> gdef_nodes_;

  // Mapping from node name to the existing node in g_
  std::unordered_map<StringPiece, Node*, StringPiece::Hasher> existing_nodes_;

  // Index of NodeDefs in gdef_ with all inputs already converted.
  std::vector<int> ready_;

  // Mapping between index within gdef_ and the number of inputs that
  // still need to be converted.
  std::vector<int> pending_count_;

  // Mapping between index within gdef_ and the index within gdef_ of
  // all nodes it outputs to.
  std::vector<gtl::InlinedVector<int, 4>> outputs_;

  // Used in the conversion from gdef_ to g_ to represent the ith input
  // of a node.
  struct InputInfo {
    explicit InputInfo(const string& node_name, Node* n, int i)
        : name(node_name), node(n), index(i) {}
    // Use string instead of StringPiece so we don't have to manage lifetime
    string name;
    Node* node;
    int index;
  };

  // Used in the conversion from gdef_ to g_ to represent an edge from
  // the node named 'name' to node 'n'.
  struct EdgeInfo {
    explicit EdgeInfo(const string& name, int i1, Node* n, int i2)
        : src_name(name), src_index(i1), dst_node(n), dst_index(i2) {}
    // Use string instead of StringPiece so we don't have to manage lifetime
    string src_name;
    int src_index;
    Node* dst_node;
    int dst_index;
  };
  std::vector<EdgeInfo> back_edges_;
};

// This could be expensive but we don't expect to call it often, if at all (only
// if there are multiple nodes in g_ with the same name)
bool NodeNameInValues(const std::map<TensorId, TensorId>& input_map,
                      const StringPiece& node_name) {
  for (auto iter = input_map.begin(); iter != input_map.end(); ++iter) {
    if (iter->second.first == node_name) return true;
  }
  return false;
}

bool NodeNameInValues(const std::vector<string>& control_dependencies,
                      const StringPiece& node_name) {
  return std::find(control_dependencies.begin(), control_dependencies.end(),
                   node_name) != control_dependencies.end();
}

Status GraphConstructor::EnsureNoNameCollisions() {
  existing_nodes_.reserve(g_->num_nodes());
  for (Node* n : g_->nodes()) {
    bool already_exists = !existing_nodes_.insert({n->name(), n}).second;
    if (already_exists) {
      if (NodeNameInValues(opts_.input_map, n->name())) {
        return errors::InvalidArgument(
            "cannot resolve input_map because multiple nodes exist with name '",
            n->name(), "'");
      }
      if (NodeNameInValues(opts_.control_dependencies, n->name())) {
        return errors::InvalidArgument(
            "cannot resolve control_dependencies because multiple nodes exist "
            "with name '",
            n->name(), "'");
      }
    }
  }
  if (opts_.prefix.empty() && opts_.importing) {
    for (int n = 0; n < gdef_->node_size(); ++n) {
      const string& name = gdef_->node(n).name();
      if (existing_nodes_.find(name) != existing_nodes_.end()) {
        return errors::InvalidArgument("Node '", name,
                                       "' already exists in the Graph");
      }
    }
  } else if (!opts_.prefix.empty()) {
    // Importing nodes with a prefix. No nodes should exist with the same
    // prefix.
    StringPiece prefix_no_slash(opts_.prefix);
    prefix_no_slash.remove_suffix(1);
    if (!IsValidNodeName(prefix_no_slash, false)) {
      return errors::InvalidArgument("Imported node name prefix '",
                                     opts_.prefix,
                                     "' would lead to invalid node names");
    }
    for (const Node* n : g_->nodes()) {
      if (StringPiece(n->name()).starts_with(opts_.prefix)) {
        return errors::InvalidArgument(
            "Import node name prefix conflicts with names of nodes already in "
            "the Graph, such as '",
            n->name(), "'");
      }
    }
  }
  return Status::OK();
}

Status GraphConstructor::ValidateInputMapAndControlDependencies() {
  for (const auto& mapping : opts_.input_map) {
    TensorId src = mapping.first;
    TensorId dst = mapping.second;
    if (existing_nodes_.count(dst.first) == 0) {
      return errors::InvalidArgument(
          "node '", dst.first, "' in input_map does not exist in graph ",
          "(input_map entry: ", src.ToString(), "->", dst.ToString(), ")");
    }
    if ((src.second == Graph::kControlSlot) !=
        (dst.second == Graph::kControlSlot)) {
      return errors::InvalidArgument("input_map entry ", src.ToString(), "->",
                                     dst.ToString(), " between ",
                                     "control edge and non-control edge");
    }
  }
  for (const string& node : opts_.control_dependencies) {
    if (existing_nodes_.count(node) == 0) {
      return errors::InvalidArgument(
          "node '", node,
          "' in control_dependencies does not exist in "
          "graph");
    }
  }
  return Status::OK();
}

Status GraphConstructor::BuildNodeIndex() {
  // Validate the node names and add them to gdef_nodes_.
  for (int n = 0; n < gdef_->node_size(); ++n) {
    const NodeDef& node_def(gdef_->node(n));
    if (!IsValidNodeName(node_def.name(), opts_.allow_internal_ops)) {
      return errors::InvalidArgument(
          "Node '", node_def.name(),
          "': Node name contains invalid characters");
    }
    if (!gdef_nodes_
             .insert(std::make_pair(StringPiece(node_def.name()), NodeInfo(n)))
             .second) {
      return errors::InvalidArgument("Node '", node_def.name(),
                                     "' is not unique");
    }
    // Validate the operation's type.
    if (node_def.op().empty()) {
      return errors::InvalidArgument("Node '", node_def.name(),
                                     "' does not specify an operation");
    }
    if (opts_.expect_device_spec && node_def.device().empty()) {
      return errors::InvalidArgument("Node '", node_def.name(),
                                     "' is missing a device specification");
    }
    // Validate control edges at end
    bool in_control_dependence = false;
    for (int i = 0; i < node_def.input_size(); ++i) {
      StringPiece input_name = node_def.input(i);
      if (!input_name.empty() && input_name.starts_with("^")) {
        in_control_dependence = true;
      } else if (in_control_dependence) {
        return errors::InvalidArgument(
            "Node '", node_def.name(),
            "': Control dependencies must come after regular dependencies");
      }
    }
  }
  return Status::OK();
}

Status GraphConstructor::InitFromEdges() {
  const int num_nodes = gdef_->node_size();
  pending_count_.reserve(num_nodes);
  outputs_.resize(num_nodes);

  // Parse the inputs for each node.
  for (int n = 0; n < num_nodes; ++n) {
    const NodeDef& node_def(gdef_->node(n));
    if (IsMerge(node_def)) {
      // for merge only wait for one non-control input.
      int32 num_control_edges = 0;
      for (int i = 0; i < node_def.input_size(); ++i) {
        StringPiece input_name(node_def.input(i));
        if (input_name.starts_with("^")) {
          num_control_edges++;
        }
      }
      pending_count_.push_back(num_control_edges + 1);
    } else {
      pending_count_.push_back(node_def.input_size());
    }
    if (node_def.input_size() == 0) {
      ready_.push_back(n);
      continue;
    }
    for (int i = 0; i < node_def.input_size(); ++i) {
      StringPiece input_name = node_def.input(i);
      TensorId id(ParseTensorName(input_name));
      auto iter = gdef_nodes_.find(id.first);
      if (iter == gdef_nodes_.end()) {
        return errors::InvalidArgument("Node '", node_def.name(),
                                       "': Unknown input node '",
                                       node_def.input(i), "'");
      }
      outputs_[iter->second.gdef_index].push_back(n);
    }
  }
  return Status::OK();
}

Status GraphConstructor::ValidateColocationConstraints(
    const NodeDef& node_def) {
  if (!opts_.importing) return Status::OK();
  const auto iter = node_def.attr().find(kColocationAttrName);
  if (iter == node_def.attr().end()) return Status::OK();
  for (const string& c : iter->second.list().s()) {
    StringPiece s(c);
    if (s.Consume(kColocationGroupPrefix) &&
        gdef_nodes_.find(s) == gdef_nodes_.end()) {
      return errors::InvalidArgument(
          "Node '", node_def.name(),
          "' expects to be colocated with unknown node '", s, "'");
    }
  }
  return Status::OK();
}

Status GraphConstructor::MakeNode(const NodeDef& node_def, Node** node) {
  // Add the node to the graph.
  Status status;
  *node = g_->AddNode(node_def, &status);
  if (!status.ok()) return status;
  if (opts_.expect_device_spec) {
    (*node)->set_assigned_device_name(node_def.device());
  }
  return Status::OK();
}

Status GraphConstructor::ValidateShape(Node* node) {
  if (!opts_.importing) return Status::OK();
  TF_RETURN_IF_ERROR(refiner_->AddNode(node));
  // For nodes with the _output_shapes atttribute, override the shape.
  std::vector<TensorShapeProto> shape_attrs;
  const char* kAttrName = "_output_shapes";
  if (!GetNodeAttr(node->def(), kAttrName, &shape_attrs).ok()) {
    // No _output_shapes attribute, the AddNode call above was sufficient.
    return Status::OK();
  }
  auto* ic = refiner_->GetContext(node);
  DCHECK(ic != nullptr)
      << "ShapeRefiner::AddNode() should have created the InferenceContext";
  if (shape_attrs.size() != node->num_outputs()) {
    return errors::InvalidArgument(
        "Node '", node->name(), "' has ", node->num_outputs(),
        " outputs but the ", kAttrName, " attribute specifies shapes for ",
        shape_attrs.size(), " outputs");
  }
  for (int i = 0; i < shape_attrs.size(); ++i) {
    const TensorShapeProto& p = shape_attrs[i];
    shape_inference::ShapeHandle h;
    Status s = ic->MakeShapeFromShapeProto(p, &h);
    if (!s.ok()) {
      return errors::InvalidArgument("Node '", node->name(), " has an invalid ",
                                     kAttrName, " attribute (shape #", i,
                                     " error:'", s.error_message(), "'");
    }
    s = refiner_->SetShape(node, i, h);
    if (!s.ok()) {
      // If the output shape is incompatible with what is inferred
      // by the graph for a very specific whitelist of ops, then we
      // ignore this output shape.  This can happen if there is a
      // bug in the shape function for some operation, and the
      // serialized graph def has the incorrect shape set when
      // running on a newer binary with the fixed shape function.
      // This is an escape hatch that allows us to correct shape
      // functions that are not critical to correct execution but
      // would cause graphs to fail if imported after correcting.
      //
      const string& op = node->def().op();
      const std::vector<string> whitelist = {
          // To be removed after 2017/03/08.
          "RandomShuffleQueue", "PaddingFIFOQueue", "FIFOQueue",
          "PriorityQueue", "QueueSize", "Stack", "Barrier", "BarrierReadySize",
          "BarrierIncompleteSize", "HashTable", "MutableHashTable",
          "MutableHashTableOfTensors", "Mutex", "CuckooTable", "IndexTable",
          "WholeFileReader", "TextLineReader", "FixedLengthRecordReader",
          "TFRecordReader", "IdentityReader", "RefSwitch", "RefEnter",
          "RefNextIteration", "RefMerge", "RefIdentity",
          // To be removed after 2017/04/24.
          "ConditionalAccumulator", "SparseConditionalAccumulator", "Table",
      };
      if (std::find(whitelist.begin(), whitelist.end(), op) ==
          whitelist.end()) {
        return errors::InvalidArgument(
            "Node '", node->name(), "' has an ", kAttrName,
            " attribute inconsistent with the GraphDef for output #", i, ": ",
            s.error_message());
      }
    }
  }
  node->ClearAttr(kAttrName);
  return Status::OK();
}

Status GraphConstructor::ModifyNodeDefForImport(NodeDef* node_def) {
  const OpDef* op_def;
  TF_RETURN_IF_ERROR(g_->op_registry()->LookUpOpDef(node_def->op(), &op_def));
  AddDefaultsToNodeDef(*op_def, node_def);
  TF_RETURN_IF_ERROR(ValidateNodeDef(*node_def, *op_def));
  TF_RETURN_IF_ERROR(CheckOpDeprecation(*op_def, gdef_->versions().producer()));
  return Status::OK();
}

void RemoveInputs(NodeDef* node_def, const std::vector<int>& inputs_to_remove) {
  // TODO(skyewm): is there a better way to do this?
  std::vector<string> inputs;
  for (int i = 0; i < node_def->input_size(); ++i) {
    inputs.push_back(node_def->input(i));
  }
  node_def->clear_input();
  for (int i = 0, j = 0; i < inputs.size(); ++i) {
    if (j < inputs_to_remove.size() && i == inputs_to_remove[j]) {
      ++j;
    } else {
      node_def->add_input(inputs[i]);
    }
  }
}

void GraphConstructor::RemapNodeDefInputs(
    NodeDef* node_def, std::vector<bool>* input_already_exists) {
  DCHECK_EQ(input_already_exists->size(), node_def->input_size());
  std::set<TensorId> control_inputs;
  std::vector<int> inputs_to_remove;

  for (int i = 0; i < node_def->input_size(); ++i) {
    auto iter = opts_.input_map.find(ParseTensorName(node_def->input(i)));
    if (iter == opts_.input_map.end()) continue;

    TensorId new_input = iter->second;
    if (new_input.second == Graph::kControlSlot) {
      // Check if we've already remapped a different input to new_input, and if
      // so remove this input.
      if (control_inputs.count(new_input) > 0) {
        inputs_to_remove.push_back(i);
        continue;
      }
      control_inputs.insert(new_input);
    }
    node_def->set_input(i, new_input.ToString());
    (*input_already_exists)[i] = true;
  }
  if (!inputs_to_remove.empty()) RemoveInputs(node_def, inputs_to_remove);
}

void GraphConstructor::AddControlDependencies(
    NodeDef* node_def, std::vector<bool>* input_already_exists) {
  // To avoid adding redundant control dependencies to every imported node, skip
  // nodes that will inherit the dependencies from another imported node.
  bool inherits_deps = false;
  for (int i = 0; i < node_def->input_size(); ++i) {
    // Assume we won't inherit dependencies from remapped inputs that already
    // exist in the graph. Even if we're wrong, we'll only add redundant
    // dependencies.
    if ((*input_already_exists)[i]) continue;

    // If this input is a backedge, assume we won't inherit the dependencies.
    // TODO(skyewm): we have many redundant ParseTensorName calls. It could be
    // worth optimizing these.
    TensorId id(ParseTensorName(node_def->input(i)));
    auto iter = gdef_nodes_.find(id.first);
    DCHECK(iter != gdef_nodes_.end()) << id.first;
    if (iter->second.node == nullptr) {
      // Input hasn't been created yet, indicating it's a backedge.
      continue;
    }
    inherits_deps = true;
  }
  if (inherits_deps) return;

  // node_def either has no inputs or all remapped inputs, add the control
  // dependencies
  for (const string& control_dep : opts_.control_dependencies) {
    string input = TensorId(control_dep, Graph::kControlSlot).ToString();
    const protobuf::RepeatedPtrField<string>& inputs = node_def->input();
    if (std::find(inputs.begin(), inputs.end(), input) != inputs.end()) {
      // Control dependency already exists
      continue;
    }
    node_def->add_input(input);
    input_already_exists->push_back(true);
  }
}

void GraphConstructor::AddPrefixToNodeDef(
    const std::vector<bool>& input_already_exists, NodeDef* node_def) {
  const string& prefix = opts_.prefix;
  if (prefix.empty()) return;
  node_def->set_name(strings::StrCat(prefix, node_def->name()));
  // Update names of input nodes
  for (int i = 0; i < node_def->input_size(); ++i) {
    StringPiece input(node_def->input(i));
    // Skip remapped inputs (which already exist in g_ and are not being
    // imported)
    if (input_already_exists[i]) continue;
    if (input.Consume("^")) {
      node_def->set_input(i, strings::StrCat("^", prefix, input));
    } else {
      node_def->set_input(i, strings::StrCat(prefix, input));
    }
  }
  // Update names of colocation groups
  if (node_def->attr().find(kColocationAttrName) != node_def->attr().end()) {
    auto* list =
        node_def->mutable_attr()->at(kColocationAttrName).mutable_list();
    for (int i = 0; i < list->s_size(); ++i) {
      StringPiece v(list->s(i));
      if (v.Consume(kColocationGroupPrefix)) {
        list->set_s(i, strings::StrCat(kColocationGroupPrefix, prefix, v));
      }
    }
  }
}

Status GraphConstructor::Convert() {
  std::vector<InputInfo> inputs;
  int processed = 0;
  // Process the NodeDefs in topological order.
  // (InitFromEdges() sets this up by filling in ready_ with nodes that have no
  // inputs, pending_counts_ with the number of inputs for each node and
  // outputs_ with the outputs of each node).
  while (!ready_.empty()) {
    int o = ready_.back();
    ready_.pop_back();
    ++processed;
    inputs.clear();
    bool has_data_back_edge = false;

    const NodeDef& original_node_def = gdef_->node(o);
    NodeDef imported_node_def;
    const NodeDef* node_def;

    // input_already_exists[i] is true iff the i-th input of the node we're
    // importing refers to a preexisting node in g_ (i.e. input[i] existed prior
    // to importing gdef_).  Conversely, input_already_exists[i] is false iff
    // the input refers to a node in gdef_.
    std::vector<bool> input_already_exists(original_node_def.input_size(),
                                           false);

    if (opts_.importing) {
      // TODO(ashankar): The line below means an additional copy of the NodeDef,
      // which can be expensive if the NodeDef contains large tensors in it.
      // Might make sense to change the API for ImportGraphDef to take a mutable
      // GraphDef* and avoid the copying.
      imported_node_def = original_node_def;
      if (!opts_.input_map.empty()) {
        RemapNodeDefInputs(&imported_node_def, &input_already_exists);
      }
      if (!opts_.control_dependencies.empty()) {
        // Note that input_already_exists can grow here
        AddControlDependencies(&imported_node_def, &input_already_exists);
      }
      node_def = &imported_node_def;
    } else {
      node_def = &original_node_def;
    }

    TF_RETURN_IF_ERROR(ValidateColocationConstraints(*node_def));
    for (int i = 0; i < node_def->input_size(); ++i) {
      TensorId id(ParseTensorName(node_def->input(i)));
      Node* src_node;
      int src_index;

      if (!input_already_exists[i]) {
        // Locate input in newly-imported nodes
        auto iter = gdef_nodes_.find(id.first);
        DCHECK(iter != gdef_nodes_.end()) << id.first;
        src_node = iter->second.node;
        src_index = id.second;
        if (src_node == nullptr) has_data_back_edge = true;
      } else {
        // Input refers to preexistng node in graph
        auto iter = existing_nodes_.find(id.first);
        DCHECK(iter != existing_nodes_.end()) << id.first;
        src_node = iter->second;
        src_index = id.second;
      }

      if (src_node != nullptr && src_index >= src_node->num_outputs()) {
        return errors::InvalidArgument(
            "Node '", node_def->name(), "': Connecting to invalid output ",
            id.second, " of source node ", id.first, " which has ",
            src_node->num_outputs(), " outputs");
      }

      inputs.push_back(InputInfo(id.first.ToString(), src_node, src_index));
    }

    if (has_data_back_edge && !IsMerge(*node_def)) {
      return errors::InvalidArgument(
          "Node '", node_def->name(),
          "' had a back edge, but only Merge nodes can have back edges.");
    }

    Node* node;
    if (opts_.importing) {
      AddPrefixToNodeDef(input_already_exists, &imported_node_def);
      TF_RETURN_IF_ERROR(ModifyNodeDefForImport(&imported_node_def));
    }
    TF_RETURN_IF_ERROR(MakeNode(*node_def, &node));
    // Use original_node_def so name StringPiece remains valid
    gdef_nodes_[original_node_def.name()].node = node;

    // Add edges from inputs to *node to the graph.
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].node == nullptr) {
        // Record this back edge, which will be added after all nodes
        // are created.
        back_edges_.push_back(
            EdgeInfo(inputs[i].name, inputs[i].index, node, i));
      } else if (inputs[i].index == Graph::kControlSlot) {
        g_->AddControlEdge(inputs[i].node, node);
      } else {
        TF_RETURN_IF_ERROR(MakeEdge(inputs[i].node, inputs[i].index, node, i));
      }
    }
    TF_RETURN_IF_ERROR(ValidateShape(node));

    // Update pending_count_ for outputs.
    for (size_t i = 0; i < outputs_[o].size(); ++i) {
      const int output = outputs_[o][i];
      pending_count_[output]--;
      if (pending_count_[output] == 0) {
        ready_.push_back(output);
      }
    }
  }

  if (processed < gdef_->node_size()) {
    return errors::InvalidArgument(gdef_->node_size() - processed,
                                   " nodes in a cycle");
  }
  return Status::OK();
}

Status GraphConstructor::AddBackEdges() {
  // Add the back edges after all nodes are created.
  for (auto e : back_edges_) {
    Node* src_node = gdef_nodes_[e.src_name].node;
    if (e.src_index == Graph::kControlSlot) {
      g_->AddControlEdge(src_node, e.dst_node);
    } else {
      TF_RETURN_IF_ERROR(
          MakeEdge(src_node, e.src_index, e.dst_node, e.dst_index));
    }

    VLOG(2) << "Add back edge: " << src_node->name() << " -> "
            << e.dst_node->name();
  }
  return Status::OK();
}

Status GraphConstructor::UpdateVersionDef() {
  if (!opts_.importing) {
    g_->set_versions(gdef_->versions());
    return Status::OK();
  }
  VersionDef versions = g_->versions();
  versions.set_producer(
      std::min(versions.producer(), gdef_->versions().producer()));
  versions.set_min_consumer(
      std::max(versions.min_consumer(), gdef_->versions().min_consumer()));
  if (gdef_->versions().bad_consumers_size() > 0) {
    std::set<int> bad(versions.bad_consumers().begin(),
                      versions.bad_consumers().end());
    bad.insert(gdef_->versions().bad_consumers().begin(),
               gdef_->versions().bad_consumers().end());
    versions.clear_bad_consumers();
    for (int v : bad) {
      versions.add_bad_consumers(v);
    }
  }
  g_->set_versions(versions);
  return Status::OK();
}

Status GraphConstructor::PopulateReturnTensors() {
  if (opts_.return_tensors.empty()) return Status::OK();
  for (const TensorId& id : opts_.return_tensors) {
    auto iter = opts_.input_map.find(id);
    if (iter == opts_.input_map.end()) {
      // Locate id in imported nodes
      auto iter = gdef_nodes_.find(id.first);
      if (iter == gdef_nodes_.end()) {
        return errors::InvalidArgument("Requested return node '", id.first,
                                       "' not found in graph def");
      }
      int num_outputs = iter->second.node->num_outputs();
      if ((id.second < 0 || id.second >= num_outputs) &&
          id.second != Graph::kControlSlot) {
        return errors::InvalidArgument("Invalid return output ", id.second,
                                       " of node '", id.first, "', which has ",
                                       num_outputs, " outputs");
      }
      return_tensors_->push_back({iter->second.node, id.second});
    } else {
      // id was remapped to existing node
      TensorId remapped_id = iter->second;
      DCHECK_GT(existing_nodes_.count(remapped_id.first), 0);
      Node* node = existing_nodes_[remapped_id.first];
      return_tensors_->push_back({node, remapped_id.second});
    }
  }
  return Status::OK();
}

void GraphConstructor::Undo() {
  for (const auto& iter : gdef_nodes_) {
    if (iter.second.node != nullptr) {
      g_->RemoveNode(iter.second.node);
    }
  }
  g_->set_versions(original_versions_);
}

Status GraphConstructor::MakeEdge(Node* src, int output_index, Node* dst,
                                  int input_index) {
  DataType src_out = src->output_type(output_index);
  DataType dst_in = dst->input_type(input_index);
  if (!TypesCompatible(dst_in, src_out)) {
    return errors::InvalidArgument(
        "Input ", input_index, " of node ", dst->name(), " was passed ",
        DataTypeString(src_out), " from ", src->name(), ":", output_index,
        " incompatible with expected ", DataTypeString(dst_in), ".");
  }
  g_->AddEdge(src, output_index, dst, input_index);
  return Status::OK();
}

}  // namespace

Status ConvertGraphDefToGraph(const GraphConstructorOptions& opts,
                              const GraphDef& gdef, Graph* g) {
  ShapeRefiner refiner(gdef.versions().producer(), g->op_registry());
  return GraphConstructor::Construct(opts, &gdef, g, &refiner, nullptr);
}

Status ImportGraphDef(const ImportGraphDefOptions& opts, const GraphDef& gdef,
                      Graph* g, ShapeRefiner* refiner,
                      std::vector<std::pair<Node*, int>>* return_tensors) {
  ShapeRefiner default_refiner(gdef.versions().producer(), g->op_registry());
  if (refiner == nullptr) {
    refiner = &default_refiner;
  }

  if (!opts.return_tensors.empty()) {
    if (return_tensors == nullptr) {
      return errors::InvalidArgument(
          "return_tensors argument to ImportNodeDef() must be non-null if "
          "opts.return_tensors is non-empty");
    }
    if (!return_tensors->empty()) {
      return errors::InvalidArgument(
          "return_tensors argument to ImportNodeDef() should be empty (has "
          "size ",
          return_tensors->size(), ")");
    }
  }
  if (gdef.library().function_size() != 0) {
    return errors::Unimplemented(
        "Importing GraphDefs containing functions not yet implemented");
  }
  return GraphConstructor::Construct(opts, &gdef, g, refiner, return_tensors);
}

void CopyGraph(const Graph& src, Graph* dest) {
  for (Node* n : dest->nodes()) {
    CHECK(n->IsSource() || n->IsSink()) << "*dest must be empty";
  }

  // Copy GraphDef versions
  dest->set_versions(src.versions());

  // Copy the nodes
  std::unordered_map<Node*, Node*>
      node_map;  // "Node in src" -> "Node in *dest"
  node_map[src.source_node()] = dest->source_node();
  node_map[src.sink_node()] = dest->sink_node();
  for (Node* n : src.nodes()) {
    if (n->IsSource() || n->IsSink()) continue;
    CHECK(n->IsOp());
    node_map[n] = dest->CopyNode(n);
  }

  // Copy the edges
  for (const Edge* e : src.edges()) {
    Node* src_copy = node_map[e->src()];
    Node* dst_copy = node_map[e->dst()];
    dest->AddEdge(src_copy, e->src_output(), dst_copy, e->dst_input());
  }
}

}  // namespace tensorflow
