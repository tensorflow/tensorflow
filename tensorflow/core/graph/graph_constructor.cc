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
          importing(true) {}

    bool allow_internal_ops;
    bool expect_device_spec;

    string prefix;
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
                          ShapeRefiner* refiner) {
    TF_RETURN_IF_ERROR(CheckVersions(gdef->versions(), TF_GRAPH_DEF_VERSION,
                                     TF_GRAPH_DEF_VERSION_MIN_PRODUCER,
                                     "GraphDef", "graph"));
    GraphConstructor c(opts, gdef, g, refiner);
    const Status s = c.TryImport();
    if (!s.ok()) c.Undo();
    return s;
  }

 private:
  GraphConstructor(const Options& opts, const GraphDef* gdef, Graph* g,
                   ShapeRefiner* refiner)
      : opts_(opts),
        gdef_(gdef),
        g_(g),
        original_versions_(g->versions()),
        refiner_(refiner) {}

  Status TryImport() {
    TF_RETURN_IF_ERROR(EnsureNoNameCollisions());
    TF_RETURN_IF_ERROR(BuildNodeIndex());
    TF_RETURN_IF_ERROR(InitFromEdges());
    TF_RETURN_IF_ERROR(Convert());
    TF_RETURN_IF_ERROR(AddBackEdges());
    TF_RETURN_IF_ERROR(UpdateVersionDef());
    FixupSourceAndSinkEdges(g_);
    return Status::OK();
  }

  Status EnsureNoNameCollisions();
  Status BuildNodeIndex();
  Status InitFromEdges();
  Status Convert();
  Status AddBackEdges();
  Status UpdateVersionDef();

  void Undo();

  Status ValidateColocationConstraints(const NodeDef& node_def);
  Status MakeNode(const NodeDef& node_def, Node** node);
  Status MakeEdge(Node* src, int output_index, Node* dst, int input_index);
  Status ValidateShape(Node* node);
  Status ModifyNodeDefForImport(NodeDef* node_def);
  void AddPrefixToNodeDef(NodeDef* node_def);

  // From constructor
  const Options opts_;
  const GraphDef* gdef_;
  Graph* g_;
  const VersionDef original_versions_;

  ShapeRefiner* refiner_;

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
  std::unordered_map<StringPiece, NodeInfo, StringPiece::Hasher> name_index_;

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
    explicit InputInfo(StringPiece node_name, Node* n, int i)
        : name(node_name), node(n), index(i) {}
    StringPiece name;
    Node* node;
    int index;
  };

  // Used in the conversion from gdef_ to g_ to represent an edge from
  // the node named 'name' to node 'n'.
  struct EdgeInfo {
    explicit EdgeInfo(StringPiece name, int i1, Node* n, int i2)
        : src_name(name), src_index(i1), dst_node(n), dst_index(i2) {}
    StringPiece src_name;
    int src_index;
    Node* dst_node;
    int dst_index;
  };
  std::vector<EdgeInfo> back_edges_;
};

Status GraphConstructor::EnsureNoNameCollisions() {
  if (opts_.prefix.empty() && opts_.importing) {
    std::unordered_set<string> existing(g_->num_nodes());
    for (const Node* n : g_->nodes()) {
      existing.insert(n->name());
    }
    for (int n = 0; n < gdef_->node_size(); ++n) {
      const string& name = gdef_->node(n).name();
      if (existing.find(name) != existing.end()) {
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

Status GraphConstructor::BuildNodeIndex() {
  // Validate the node names and add them to name_index_.
  for (int n = 0; n < gdef_->node_size(); ++n) {
    const NodeDef& node_def(gdef_->node(n));
    if (!IsValidNodeName(node_def.name(), opts_.allow_internal_ops)) {
      return errors::InvalidArgument(
          "Node '", node_def.name(),
          "': Node name contains invalid characters");
    }
    if (!name_index_
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
      input_name.Consume("^");  // In case this is a control dependence.
      TensorId id(ParseTensorName(input_name));
      auto iter = name_index_.find(id.first);
      if (iter == name_index_.end()) {
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
        name_index_.find(s) == name_index_.end()) {
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
      // This can be removed after 2017/03/08.
      const string& op = node->def().op();
      const std::vector<string> whitelist = {"RandomShuffleQueue",
                                             "PaddingFIFOQueue",
                                             "FIFOQueue",
                                             "PriorityQueue",
                                             "QueueSize",
                                             "Stack",
                                             "Barrier",
                                             "BarrierReadySize",
                                             "BarrierIncompleteSize",
                                             "HashTable",
                                             "MutableHashTable",
                                             "MutableHashTableOfTensors",
                                             "Mutex",
                                             "CuckooTable",
                                             "IndexTable",
                                             "WholeFileReader",
                                             "TextLineReader",
                                             "FixedLengthRecordReader",
                                             "TFRecordReader",
                                             "IdentityReader",
                                             "RefSwitch",
                                             "RefEnter",
                                             "RefNextIteration",
                                             "RefMerge",
                                             "RefIdentity"};
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
  TF_RETURN_IF_ERROR(CheckOpDeprecation(*op_def, TF_GRAPH_DEF_VERSION));
  return Status::OK();
}

void GraphConstructor::AddPrefixToNodeDef(NodeDef* node_def) {
  const string& prefix = opts_.prefix;
  if (prefix.empty()) return;
  node_def->set_name(strings::StrCat(prefix, node_def->name()));
  // Update names of input nodes
  for (int i = 0; i < node_def->input_size(); ++i) {
    StringPiece input(node_def->input(i));
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
    const NodeDef& node_def(gdef_->node(o));
    inputs.clear();
    bool in_control_dependence = false;
    bool has_data_back_edge = false;
    TF_RETURN_IF_ERROR(ValidateColocationConstraints(node_def));
    for (int i = 0; i < node_def.input_size(); ++i) {
      StringPiece input_name(node_def.input(i));
      if (input_name.Consume("^")) {
        in_control_dependence = true;
      } else if (in_control_dependence) {
        return errors::InvalidArgument(
            "Node '", node_def.name(),
            "': Control dependencies must come after regular dependencies");
      }
      TensorId id(ParseTensorName(input_name));
      auto iter = name_index_.find(id.first);
      DCHECK(iter != name_index_.end());
      Node* src_node = iter->second.node;
      if (in_control_dependence) {
        inputs.push_back(InputInfo(id.first, src_node, -1));
      } else {
        if (src_node == nullptr) {
          has_data_back_edge = true;
          inputs.push_back(InputInfo(id.first, src_node, id.second));
        } else {
          if (id.second >= src_node->num_outputs()) {
            return errors::InvalidArgument(
                "Node '", node_def.name(), "': Connecting to invalid output ",
                id.second, " of source node ", id.first, " which has ",
                src_node->num_outputs(), " outputs");
          }
          inputs.push_back(InputInfo(id.first, src_node, id.second));
        }
      }
    }
    if (has_data_back_edge && !IsMerge(node_def)) {
      return errors::InvalidArgument(
          "Node '", node_def.name(),
          "' had a back edge, but only Merge nodes can have back edges.");
    }

    Node* node;
    if (opts_.importing) {
      // TODO(ashankar): The line below means an additional copy of the NodeDef,
      // which can be expensive if the NodeDef contains large tensors in it.
      // Might make sense to change the API for ImportGraphDef to take a mutable
      // GraphDef* and avoid the copying.
      NodeDef imported_node_def = node_def;
      AddPrefixToNodeDef(&imported_node_def);
      TF_RETURN_IF_ERROR(ModifyNodeDefForImport(&imported_node_def));
      TF_RETURN_IF_ERROR(MakeNode(imported_node_def, &node));
    } else {
      TF_RETURN_IF_ERROR(MakeNode(node_def, &node));
    }
    name_index_[node_def.name()].node = node;

    // Add edges from inputs to *node to the graph.
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].node == nullptr) {
        // Record this back edge, which will be added after all nodes
        // are created.
        back_edges_.push_back(
            EdgeInfo(inputs[i].name, inputs[i].index, node, i));
      } else if (inputs[i].index == -1) {
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
    Node* src_node = name_index_[e.src_name].node;
    if (e.src_index == -1) {
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
  // This new graph is being "produced" by the binary invoking ImportGraphDef.
  versions.set_producer(TF_GRAPH_DEF_VERSION);
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

void GraphConstructor::Undo() {
  for (const auto& iter : name_index_) {
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
  ShapeRefiner refiner(g->op_registry());
  return GraphConstructor::Construct(opts, &gdef, g, &refiner);
}

Status ImportGraphDef(const ImportGraphDefOptions& opts, const GraphDef& gdef,
                      Graph* g, ShapeRefiner* refiner) {
  ShapeRefiner default_refiner(g->op_registry());
  if (refiner == nullptr) {
    refiner = &default_refiner;
  }
  return GraphConstructor::Construct(opts, &gdef, g, refiner);
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
