#include "tensorflow/core/graph/graph_constructor.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/optimizer_cse.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {

namespace {
inline bool IsMerge(const NodeDef& node_def) {
  return node_def.op() == "Merge";
}
}  // namespace

namespace {

class GraphConstructor {
 public:
  GraphConstructor(const GraphConstructorOptions& opts, const GraphDef* gdef,
                   Graph* g, Status* status)
      : opts_(opts), gdef_(gdef), g_(g), status_(status) {
    BuildNodeIndex();
    InitFromEdges();
    Convert();
  }

 private:
  void SetError(const string& error);
  void SetNodeError(const NodeDef& node_def, const StringPiece& message) {
    SetError(strings::StrCat("Node '", node_def.name(), "': ", message));
  }
  void BuildNodeIndex();
  void InitFromEdges();
  Node* MakeNode(const NodeDef& node_def);
  void Convert();
  // Calls SetError() and returns false if the type of the output of
  // the source of the edge can't be consumed by destination of the edge.
  // REQUIRES: edge must be a data edge, not a control edge.
  bool TypeValidateEdge(const Edge* edge);

  // From constructor
  const GraphConstructorOptions opts_;
  const GraphDef* gdef_;
  Graph* g_;
  Status* status_;

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
};

void GraphConstructor::SetError(const string& error) {
  status_->Update(errors::InvalidArgument(error));
}

void GraphConstructor::BuildNodeIndex() {
  // Initialized outside the loop for efficiency
  const char* pattern;
  if (opts_.allow_internal_ops) {
    pattern = "[A-Za-z0-9._][A-Za-z0-9_.\\-/]*";
  } else {
    pattern = "[A-Za-z0-9.][A-Za-z0-9_.\\-/]*";
  }
  RE2 node_name_re(pattern);

  // Validate the node names and add them to name_index_.
  for (int n = 0; n < gdef_->node_size(); ++n) {
    const NodeDef& node_def(gdef_->node(n));
    if (!RE2::FullMatch(node_def.name(), node_name_re)) {
      SetNodeError(node_def, "Node name contains invalid characters");
      return;
    }
    if (!name_index_.insert(std::make_pair(StringPiece(node_def.name()),
                                           NodeInfo(n)))
             .second) {
      SetNodeError(node_def, "Node name is not unique");
      return;
    }
    // Validate the operation's type.
    if (node_def.op().empty()) {
      SetNodeError(node_def, "Does not specify a type");
      return;
    }
    if (opts_.expect_device_spec && node_def.device().empty()) {
      SetNodeError(node_def, strings::StrCat("Missing device specification."));
      return;
    }
  }
}

void GraphConstructor::InitFromEdges() {
  const int num_nodes = gdef_->node_size();
  ready_.reserve(num_nodes);
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
        if (StringPiece(input_name).starts_with("^")) {
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
      if (input_name.starts_with("^")) {
        // Control dependence
        input_name.remove_prefix(1);
      }
      TensorId id(ParseTensorName(input_name));
      auto iter = name_index_.find(id.first);
      if (iter == name_index_.end()) {
        SetNodeError(node_def,
                     strings::StrCat("Unknown input node ", node_def.input(i)));
        return;
      }
      outputs_[iter->second.gdef_index].push_back(n);
    }
  }
}

Node* GraphConstructor::MakeNode(const NodeDef& node_def) {
  // Add the node to the graph.
  Node* node = g_->AddNode(node_def, status_);
  if (node == nullptr) return nullptr;
  if (opts_.expect_device_spec) {
    node->set_assigned_device_name(node_def.device());
  }
  name_index_[node_def.name()].node = node;
  return node;
}

// Return the number of nodes in "g"
static int CountNodes(Graph* g) {
  int nodes = 0;
  for (Node* node : g->nodes()) {
    VLOG(1) << node;  // Dummy use to avoid compiler warning
    nodes++;
  }
  return nodes;
}

void GraphConstructor::Convert() {
  std::vector<InputInfo> inputs;
  std::vector<EdgeInfo> back_edges;
  int processed = 0;
  // Process the NodeDefs in topological order.
  while (!ready_.empty()) {
    int o = ready_.back();
    ready_.pop_back();
    ++processed;
    const NodeDef& node_def(gdef_->node(o));
    inputs.clear();
    bool in_control_dependence = false;
    bool has_data_back_edge = false;
    for (int i = 0; i < node_def.input_size(); ++i) {
      StringPiece input_name(node_def.input(i));
      if (StringPiece(input_name).starts_with("^")) {
        // A control dependence
        in_control_dependence = true;
        input_name.remove_prefix(1);
      } else {
        if (in_control_dependence) {
          SetNodeError(node_def, strings::StrCat(
                                     "Control dependencies must come after ",
                                     "regular dependencies: input ", input_name,
                                     " of source node ", node_def.name()));
          return;
        }
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
            SetNodeError(
                node_def,
                strings::StrCat("Connecting to invalid output ", id.second,
                                " of source node ", id.first, " which has ",
                                src_node->num_outputs(), " outputs"));
            return;
          }
          inputs.push_back(InputInfo(id.first, src_node, id.second));
        }
      }
    }
    if (has_data_back_edge && !IsMerge(node_def)) {
      SetError(strings::StrCat(
          node_def.name(),
          " had a back edge. But only Merge can have back edges."));
      return;
    }

    Node* node = MakeNode(node_def);
    if (node == nullptr) return;

    // Add edges from inputs to *node to the graph.
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].node == nullptr) {
        // Record this back edge, which will be added after all nodes
        // are created.
        back_edges.push_back(
            EdgeInfo(inputs[i].name, inputs[i].index, node, i));
      } else if (inputs[i].index == -1) {
        g_->AddControlEdge(inputs[i].node, node);
      } else {
        const Edge* edge =
            g_->AddEdge(inputs[i].node, inputs[i].index, node, i);
        if (!TypeValidateEdge(edge)) return;
      }
    }

    // Update pending_count_ for outputs.
    for (size_t i = 0; i < outputs_[o].size(); ++i) {
      const int output = outputs_[o][i];
      pending_count_[output]--;
      if (pending_count_[output] == 0) {
        ready_.push_back(output);
      }
    }
  }

  // Add the back edges after all nodes are created.
  for (auto e : back_edges) {
    Node* src_node = name_index_[e.src_name].node;
    if (e.src_index == -1) {
      g_->AddControlEdge(src_node, e.dst_node);
    } else {
      const Edge* edge =
          g_->AddEdge(src_node, e.src_index, e.dst_node, e.dst_index);
      if (!TypeValidateEdge(edge)) return;
    }

    VLOG(2) << "Add back edge: " << src_node->name() << " -> "
            << e.dst_node->name();
  }

  if (processed < gdef_->node_size()) {
    SetError(
        strings::StrCat(gdef_->node_size() - processed, " nodes in a cycle"));
    return;
  }

  if (status_->ok()) {
    FixupSourceAndSinkEdges(g_);

    if (opts_.optimizer_do_cse) {
      if (!back_edges.empty()) {
        LOG(WARNING) << "Not doing CSE.  We need to figure out how to handle "
                     << "loops in the CSE phase.";
      } else {
        VLOG(1) << "Starting CSE: graph of " << CountNodes(g_) << " nodes";
        OptimizeCSE(g_, opts_.cse_consider_function);
        VLOG(1) << "Finished CSE: graph of " << CountNodes(g_) << " nodes";
      }
    }
  }
}

bool GraphConstructor::TypeValidateEdge(const Edge* edge) {
  DataType src_out = edge->src()->output_type(edge->src_output());
  DataType dst_in = edge->dst()->input_type(edge->dst_input());
  if (!TypesCompatible(dst_in, src_out)) {
    SetError(strings::StrCat(
        "Input ", edge->dst_input(), " of node ", edge->dst()->name(),
        " was passed ", DataTypeString(src_out), " from ", edge->src()->name(),
        ":", edge->src_output(), " incompatible with expected ",
        DataTypeString(dst_in), "."));
    return false;
  }
  return true;
}

}  // namespace

// ----------------------------------------------------------------------------
// ConvertGraphDefToGraph
// ----------------------------------------------------------------------------

Status ConvertGraphDefToGraph(const GraphConstructorOptions& opts,
                              const GraphDef& gdef, Graph* g) {
  Status status;
  GraphConstructor constructor(opts, &gdef, g, &status);
  return status;
}

// ----------------------------------------------------------------------------
// CopyGraph
// ----------------------------------------------------------------------------
void CopyGraph(const Graph& src, Graph* dest) {
  for (Node* n : dest->nodes()) {
    CHECK(n->IsSource() || n->IsSink()) << "*dest must be empty";
  }

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
