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

#include "tensorflow/core/common_runtime/graph_constructor.h"

#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/versions.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

namespace {

// We remove duplicate control inputs before adding edges to the Graph, so we
// can skip expensive duplicates check in 'AddControlEdge'.
static constexpr const bool kDoNotCheckDuplicates = true;

inline bool IsMerge(const NodeDef& node_def) {
  return node_def.op() == "Merge" || node_def.op() == "RefMerge" ||
         node_def.op() == "_XlaMerge";
}

inline bool IsNextIteration(const NodeDef& node_def) {
  return node_def.op() == "NextIteration" ||
         node_def.op() == "RefNextIteration";
}

bool IsValidNodeName(StringPiece s, bool allow_internal_ops) {
  using ::tensorflow::strings::Scanner;
  Scanner scanner(s);
  scanner
      .One(allow_internal_ops ? Scanner::LETTER_DIGIT_DOT_UNDERSCORE
                              : Scanner::LETTER_DIGIT_DOT)
      .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE);

  while (true) {
    if (!scanner.GetResult())  // Some error in previous iteration.
      return false;
    if (scanner.empty())  // No error, but nothing left, good.
      return true;

    // Absorb another piece, starting with a '>'
    scanner.One(Scanner::RANGLE)
        .One(Scanner::LETTER_DIGIT_DOT)
        .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE);
  }
}

class GraphConstructor {
 public:
  struct Options {
    Options(const GraphConstructorOptions& in)  // NOLINT(runtime/explicit)
        : allow_internal_ops(in.allow_internal_ops),
          expect_device_spec(in.expect_device_spec),
          importing(false),
          validate_nodes(in.validate_nodes),
          validate_colocation_constraints(false),
          add_default_attributes(in.add_default_attributes) {}
    Options(const ImportGraphDefOptions& in)  // NOLINT(runtime/explicit)
        : allow_internal_ops(false),
          expect_device_spec(false),
          prefix(in.prefix.empty() || str_util::EndsWith(in.prefix, "/")
                     ? in.prefix
                     : in.prefix + "/"),
          uniquify_names(in.uniquify_names),
          uniquify_prefix(in.uniquify_prefix),
          input_map(in.input_map.begin(), in.input_map.end()),
          skip_mapped_nodes(in.skip_mapped_nodes),
          control_dependencies(in.control_dependencies),
          return_tensors(in.return_tensors.begin(), in.return_tensors.end()),
          return_nodes(in.return_nodes),
          importing(true),
          validate_nodes(true),
          validate_colocation_constraints(in.validate_colocation_constraints),
          validate_shape(in.validate_shape),
          default_device(in.default_device) {}

    bool allow_internal_ops;
    bool expect_device_spec;

    string prefix;
    bool uniquify_names;
    bool uniquify_prefix;
    std::map<TensorId, TensorId> input_map;
    bool skip_mapped_nodes;
    std::vector<string> control_dependencies;
    std::vector<TensorId> return_tensors;
    std::vector<string> return_nodes;

    // TODO(ashankar): This bool exists to separate out functionality required
    // to make ImportGraphDef a close equivalent of Python's import_graph_def
    // without affecting the behavior of ConvertGraphDefToGraph at the time
    // ImportGraphDef was added.
    //
    // That said, the functionality here (shape and op validation) seems
    // applicable to ConvertGraphDefToGraph as well, so make an attempt to
    // remove this.
    bool importing;
    // If true, validates that nodes being converted have all expected attrs
    // set and no unknown attrs set by calling ValidateNodeDef().
    // `validate_nodes` is always true when `importing` is set.
    bool validate_nodes;
    bool validate_colocation_constraints;
    bool validate_shape = true;

    // If true, GraphConstructor will add attributes with their default
    // value to the Node when they are missing from the NodeDef.
    bool add_default_attributes = true;

    string default_device;
  };

  typedef gtl::ArraySlice<const NodeDef*> NodeDefSlice;

  // versions and library may be nullptr
  static Status Construct(
      const Options& opts, NodeDefSlice node_defs, const VersionDef* versions,
      const FunctionDefLibrary* library, Graph* g, ShapeRefiner* refiner,
      std::vector<std::pair<Node*, int>>* return_tensors,
      std::vector<Node*>* return_nodes,
      std::vector<SafeTensorId>* missing_unused_input_map_keys);

  static Status Construct(
      const Options& opts, GraphDef&& graph_def, Graph* g,
      ShapeRefiner* refiner, std::vector<std::pair<Node*, int>>* return_tensors,
      std::vector<Node*>* return_nodes,
      std::vector<SafeTensorId>* missing_unused_input_map_keys);

 protected:
  GraphConstructor(const Options& opts, Graph* g, ShapeRefiner* refiner,
                   std::vector<std::pair<Node*, int>>* return_tensors,
                   std::vector<Node*>* return_nodes,
                   std::vector<SafeTensorId>* missing_unused_input_map_keys)
      : opts_(opts),
        g_(g),
        original_versions_(g->versions()),
        prefix_(opts.prefix),
        refiner_(refiner),
        return_tensors_(return_tensors),
        return_nodes_(return_nodes),
        missing_unused_input_map_keys_(missing_unused_input_map_keys) {}

  virtual ~GraphConstructor() {}

  Status TryImport() {
    TF_RETURN_IF_ERROR(EnsureNoNameCollisions());
    TF_RETURN_IF_ERROR(ValidateInputMapAndControlDependencies());
    TF_RETURN_IF_ERROR(BuildNodeIndex());
    TF_RETURN_IF_ERROR(InitFromEdges());

    // NOTE: Convert() invokes `consume_node_def()` on each node in the input
    // graph, so `get_node_def()` is no longer usable once it is called.
    TF_RETURN_IF_ERROR(Convert());

    TF_RETURN_IF_ERROR(AddBackEdges());
    TF_RETURN_IF_ERROR(UpdateVersionDef());
    TF_RETURN_IF_ERROR(PopulateReturnTensors());
    TF_RETURN_IF_ERROR(PopulateReturnNodes());
    TF_RETURN_IF_ERROR(PopulateMissingUnusedInputMapKeys());
    UpdateUniquifiedColocationNames();
    FixupSourceAndSinkEdges(g_);
    return Status::OK();
  }

 private:
  Status EnsureNoNameCollisions();
  Status ValidateInputMapAndControlDependencies();
  Status BuildNodeIndex();
  Status InitFromEdges();
  Status Convert();
  Status AddBackEdges();
  Status UpdateVersionDef();
  Status PopulateReturnTensors();
  Status PopulateReturnNodes();
  Status PopulateMissingUnusedInputMapKeys();

  void Undo();

  // Prints cycles in the graph.
  void PrintCycles();
  // Performs DFS starting at `cur_node` and prints any cycles found.
  void DFS(int cur_node, std::vector<int>* cur_branch,
           std::vector<bool>* is_on_cur_branch,
           absl::flat_hash_set<int>* unvisited,
           const std::vector<absl::string_view>& node_names);
  Status IsNodeFullyMapped(const NodeDef& node_def, bool* is_node_mapped);
  Status ValidateColocationConstraints(const NodeDef& node_def);
  Status MakeNode(NodeDef&& node_def, Node** node);
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

  // Modifies `node_def` if its name isn't unique, or if any of its inputs'
  // names have been uniquified. This must be called in topological order on all
  // nodes.
  void UniquifyNames(const std::vector<bool>& input_already_exists,
                     NodeDef* node_def);

  // Updates any constructed nodes' colocation group names if the name has been
  // updated by UniquifyNames. This is called after all the nodes have been
  // constructed so all the names have been uniquified if necessary.
  void UpdateUniquifiedColocationNames();

  // Returns true if `name` already exists in `g_` (either as a node name or
  // prefix).
  bool NameExistsInGraph(StringPiece name);

  // Returns true if `name` already exists in the GraphDef being imported
  // (either as a node name or prefix).
  bool NameExistsInGraphDef(StringPiece name);

  // Returns a unique version of `original_name`, or `original_name` if it's
  // already unique in the graph.
  string FindUniqueName(StringPiece original_name);

  // Decrement pending count for users of `processed` and add the ones that now
  // have all of their pending inputs satisfied to `ready_`.
  void UpdatePendingCountAndReady(int processed, bool is_next_iteration);

  // Subclasses override the following virtual methods to provide efficient
  // access to the original protocol buffer-based graph.

  // Returns the number of nodes in the graph.
  virtual size_t node_def_count() const = 0;
  // Returns the i^th node in the graph. Must not be called after
  // consume_node_def(i).
  virtual const NodeDef& get_node_def(int i) const = 0;
  // Destructively reads the i^th node in the graph, avoiding a copy if
  // possible. After calling this method, the result of get_node_def(i) is
  // undefined.
  virtual NodeDef consume_node_def(int i) = 0;
  // Returns the version information for the graph, or nullptr if none is
  // available.
  virtual const VersionDef* versions() const = 0;
  // Returns the function information for the graph, or nullptr if none is
  // available.
  virtual const FunctionDefLibrary* library() const = 0;

  // From constructor
  const Options opts_;
  Graph* g_;
  const VersionDef original_versions_;

  // A copy of opts_.prefix, possibly uniquified.
  string prefix_;

  ShapeRefiner* refiner_;

  // May be null. Not owned.
  std::vector<std::pair<Node*, int>>* return_tensors_;

  // May be null. Not owned.
  std::vector<Node*>* return_nodes_;

  // May be null. Not owned.
  std::vector<SafeTensorId>* missing_unused_input_map_keys_;

  // Intermediate datastructure used to populate
  // `missing_unused_input_map_keys_`.
  std::set<TensorId> used_input_map_keys_;

  // Intermediate datastructure used to track the destinations of back edges.
  absl::flat_hash_set<int> merge_node_indices_;

  // Mapping from node name to the index within node_defs_.
  struct NodeInfo {
    explicit NodeInfo(int i) : gdef_index(i), node(nullptr) {}
    // Containers require that we have a default constructor.
    NodeInfo() : NodeInfo(-1) {}
    int gdef_index;
    Node* node;  // nullptr until the NodeDef is converted to a Node.
  };
  absl::flat_hash_map<std::string, NodeInfo> gdef_nodes_;

  // Prefixes already used in the GraphDef being imported.
  absl::flat_hash_set<StringPiece> gdef_prefixes_;

  // Mapping from node name to the existing node in g_.
  absl::flat_hash_map<StringPiece, Node*> existing_nodes_;

  // Prefixes already used in the graph.
  absl::flat_hash_set<StringPiece> existing_prefixes_;

  // Imported node names that have been uniquified. The key is the original
  // name, the value is the new unique name.
  gtl::FlatMap<string, string> uniquified_names_;

  // Index of NodeDefs in node_defs_ with all inputs already converted. We use a
  // (sorted) set so nodes are created in the order defined in the GraphDef.
  std::set<int> ready_;

  // Mapping between index within node_defs_ and the number of inputs that
  // still need to be converted.
  std::vector<int> pending_count_;

  // Mapping between index within node_defs_ and the index within node_defs_ of
  // all nodes it outputs to.
  std::vector<gtl::InlinedVector<int, 4>> outputs_;

  // Used in the conversion from node_defs_ to g_ to represent the ith input
  // of a node.
  struct InputInfo {
    explicit InputInfo(const string& node_name, Node* n, int i)
        : name(node_name), node(n), index(i) {}
    // Use string instead of StringPiece so we don't have to manage lifetime
    string name;
    Node* node;
    int index;

    static bool IsControlInput(const InputInfo& input) {
      return input.index == Graph::kControlSlot;
    }
    static int CompareName(const InputInfo& lhs, const InputInfo& rhs) {
      return lhs.name < rhs.name;
    }
    static bool IsSameName(const InputInfo& lhs, const InputInfo& rhs) {
      return lhs.name == rhs.name;
    }
  };

  // Used in the conversion from node_defs_ to g_ to represent an edge from
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

  TF_DISALLOW_COPY_AND_ASSIGN(GraphConstructor);
};

// Implementation of GraphConstructor that does not take ownership of the
// input NodeDef messages and thus copies the nodes into the constructed Graph*.
//
// NOTE(mrry): Whenever possible, use NodeDefMovingGraphConstructor, which
// avoids copying each NodeDef into the constructed Graph*.
class NodeDefCopyingGraphConstructor : public GraphConstructor {
 public:
  NodeDefCopyingGraphConstructor(
      const Options& opts, NodeDefSlice node_defs, const VersionDef* versions,
      const FunctionDefLibrary* library, Graph* g, ShapeRefiner* refiner,
      std::vector<std::pair<Node*, int>>* return_tensors,
      std::vector<Node*>* return_nodes,
      std::vector<SafeTensorId>* missing_unused_input_map_keys)
      : GraphConstructor(opts, g, refiner, return_tensors, return_nodes,
                         missing_unused_input_map_keys),
        node_defs_(node_defs),
        versions_(versions),
        library_(library) {}

 private:
  size_t node_def_count() const override { return node_defs_.size(); }
  const NodeDef& get_node_def(int i) const override { return *node_defs_[i]; }
  NodeDef consume_node_def(int i) override { return *node_defs_[i]; }
  const VersionDef* versions() const override { return versions_; }
  const FunctionDefLibrary* library() const override { return library_; }

  const NodeDefSlice node_defs_;
  const VersionDef* const versions_;
  const FunctionDefLibrary* const library_;
};

// Implementation of GraphConstructor that takes ownership of the input
// GraphDef, and can perform destructive reads.
class NodeDefMovingGraphConstructor : public GraphConstructor {
 public:
  NodeDefMovingGraphConstructor(
      const Options& opts, GraphDef&& graph_def, Graph* g,
      ShapeRefiner* refiner, std::vector<std::pair<Node*, int>>* return_tensors,
      std::vector<Node*>* return_nodes,
      std::vector<SafeTensorId>* missing_unused_input_map_keys)
      : GraphConstructor(opts, g, refiner, return_tensors, return_nodes,
                         missing_unused_input_map_keys),
        graph_def_(std::move(graph_def)),
        is_consumed_(graph_def_.node_size(), false) {}

 private:
  size_t node_def_count() const override { return graph_def_.node().size(); }
  const NodeDef& get_node_def(int i) const override {
    CHECK(!is_consumed_[i])
        << "NodeDef " << i << " accessed after it was consumed.";
    return graph_def_.node(i);
  }
  NodeDef consume_node_def(int i) override {
    CHECK(!is_consumed_[i]) << "NodeDef " << i << " consumed twice.";
    is_consumed_[i] = true;
    return std::move(*graph_def_.mutable_node(i));
  }
  const VersionDef* versions() const override { return &graph_def_.versions(); }
  const FunctionDefLibrary* library() const override {
    return &graph_def_.library();
  }

  GraphDef graph_def_;
  std::vector<bool> is_consumed_;
};

bool ForwardCompatibilityWindowPassed(const VersionDef& versions) {
  // TF_GRAPH_DEF_VERSION is incremented daily.
  // TF has a 3 week forward compatibility guarantee.
  return (versions.producer() - TF_GRAPH_DEF_VERSION) > 21;
}

Status MaybeAppendVersionWarning(const VersionDef* versions,
                                 const Status& import_status) {
  if (versions && ForwardCompatibilityWindowPassed(*versions)) {
    return Status(
        import_status.code(),
        absl::StrCat(
            "Converting GraphDef to Graph has failed with an error: '",
            import_status.error_message(),
            "' The binary trying to import the GraphDef was built when "
            "GraphDef version was ",
            TF_GRAPH_DEF_VERSION,
            ". The GraphDef was produced by a binary built when GraphDef "
            "version was ",
            versions->producer(),
            ". The difference between these versions is larger than "
            "TensorFlow's forward compatibility guarantee, and might be the "
            "root cause for failing to import the GraphDef."));
  }
  return import_status;
}

/* static */ Status GraphConstructor::Construct(
    const Options& opts, NodeDefSlice node_defs, const VersionDef* versions,
    const FunctionDefLibrary* library, Graph* g, ShapeRefiner* refiner,
    std::vector<std::pair<Node*, int>>* return_tensors,
    std::vector<Node*>* return_nodes,
    std::vector<SafeTensorId>* missing_unused_input_map_keys) {
  if (versions) {
    TF_RETURN_IF_ERROR(CheckVersions(*versions, TF_GRAPH_DEF_VERSION,
                                     TF_GRAPH_DEF_VERSION_MIN_PRODUCER,
                                     "GraphDef", "graph"));
  }
  NodeDefCopyingGraphConstructor c(opts, node_defs, versions, library, g,
                                   refiner, return_tensors, return_nodes,
                                   missing_unused_input_map_keys);
  Status s = c.TryImport();
  if (!s.ok()) {
    c.Undo();
    s = MaybeAppendVersionWarning(versions, s);
  }
  return s;
}

/* static */ Status GraphConstructor::Construct(
    const Options& opts, GraphDef&& graph_def, Graph* g, ShapeRefiner* refiner,
    std::vector<std::pair<Node*, int>>* return_tensors,
    std::vector<Node*>* return_nodes,
    std::vector<SafeTensorId>* missing_unused_input_map_keys) {
  TF_RETURN_IF_ERROR(CheckVersions(graph_def.versions(), TF_GRAPH_DEF_VERSION,
                                   TF_GRAPH_DEF_VERSION_MIN_PRODUCER,
                                   "GraphDef", "graph"));
  VersionDef version_def = graph_def.versions();
  NodeDefMovingGraphConstructor c(opts, std::move(graph_def), g, refiner,
                                  return_tensors, return_nodes,
                                  missing_unused_input_map_keys);
  Status s = c.TryImport();
  if (!s.ok()) {
    c.Undo();
    s = MaybeAppendVersionWarning(&version_def, s);
  }
  return s;
}

void GraphConstructor::UpdatePendingCountAndReady(int processed,
                                                  bool is_next_iteration) {
  for (size_t i = 0; i < outputs_[processed].size(); ++i) {
    const int output = outputs_[processed][i];
    // We didn't consider NextIteration->Merge edges when computing
    // pending_counts_ so we should not have to consider it here either.
    bool is_next_iteration_to_merge_edge =
        is_next_iteration && merge_node_indices_.count(output) == 1;
    if (!is_next_iteration_to_merge_edge) {
      int* current_pending_count = &pending_count_[output];
      CHECK_GT(*current_pending_count, 0);
      (*current_pending_count)--;
      if (*current_pending_count == 0) {
        ready_.insert(output);
      }
    }
  }
}

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

// Adds any prefixes of `node_name` (not including the full name itself) to
// `prefixes`.
void AddPrefixes(StringPiece node_name,
                 absl::flat_hash_set<StringPiece>* prefixes) {
  size_t idx = -1;
  while ((idx = node_name.find('/', idx + 1)) != StringPiece::npos) {
    prefixes->insert(node_name.substr(0, idx));
  }
}

Status GraphConstructor::EnsureNoNameCollisions() {
  existing_nodes_.reserve(g_->num_nodes());
  // Populate existing_nodes_ and existing_prefixes_.
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
    AddPrefixes(n->name(), &existing_prefixes_);
  }
  if (prefix_.empty() && opts_.importing && !opts_.uniquify_names) {
    for (size_t i = 0; i < node_def_count(); ++i) {
      const string& name = get_node_def(i).name();
      if (NameExistsInGraph(name)) {
        return errors::InvalidArgument("Node name '", name,
                                       "' already exists in the Graph");
      }
    }
  } else if (!prefix_.empty()) {
    StringPiece prefix_no_slash(prefix_);
    prefix_no_slash.remove_suffix(1);
    if (!IsValidNodeName(prefix_no_slash, false)) {
      return errors::InvalidArgument("Imported node name prefix '", prefix_,
                                     "' would lead to invalid node names");
    }
    if (NameExistsInGraph(prefix_no_slash) && opts_.uniquify_prefix) {
      prefix_ = strings::StrCat(FindUniqueName(prefix_no_slash), "/");
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
  // Validate the node names and add them to gdef_nodes_ and gdef_prefixes_.
  for (int n = 0; n < node_def_count(); ++n) {
    const NodeDef& node_def = get_node_def(n);
    if (!IsValidNodeName(node_def.name(), opts_.allow_internal_ops)) {
      return errors::InvalidArgument(
          "Node '", node_def.name(),
          "': Node name contains invalid characters");
    }
    if (!gdef_nodes_.insert(std::make_pair(node_def.name(), NodeInfo(n)))
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
    if (IsMerge(node_def)) {
      merge_node_indices_.insert(n);
    }
    // Validate control edges at end
    bool in_control_dependence = false;
    for (int i = 0; i < node_def.input_size(); ++i) {
      StringPiece input_name = node_def.input(i);
      if (!input_name.empty() && absl::StartsWith(input_name, "^")) {
        in_control_dependence = true;
      } else if (in_control_dependence) {
        return errors::InvalidArgument(
            "Node '", node_def.name(),
            "': Control dependencies must come after regular dependencies");
      }
    }
    // Update gdef_prefixes_.
    AddPrefixes(node_def.name(), &gdef_prefixes_);
  }
  return Status::OK();
}

Status GraphConstructor::InitFromEdges() {
  const int num_nodes = node_def_count();
  pending_count_.reserve(num_nodes);
  outputs_.resize(num_nodes);
  gtl::FlatSet<string> next_iteration_nodes;
  for (int n = 0; n < node_def_count(); ++n) {
    const NodeDef& node_def = get_node_def(n);
    if (IsNextIteration(node_def)) {
      next_iteration_nodes.insert(node_def.name());
    }
  }

  // Parse the inputs for each node.
  for (int n = 0; n < num_nodes; ++n) {
    const NodeDef& node_def = get_node_def(n);
    int pending_count = node_def.input_size();
    if (IsMerge(node_def)) {
      // Cycles in the graph are only allowed for while loops. A while loop is
      // identified by an edge from a NextIteration node to a Merge node. For
      // such Merge nodes, only wait for one non-control input before
      // considering the node ready to process in Convert().
      int32_t num_control_edges = 0;
      bool has_loop_back_edge = false;
      for (int i = 0; i < node_def.input_size(); ++i) {
        StringPiece input_name(node_def.input(i));
        if (absl::StartsWith(input_name, "^")) {
          num_control_edges++;
        } else {
          TensorId id(ParseTensorName(input_name));
          if (next_iteration_nodes.find(string(id.first)) !=
              next_iteration_nodes.end()) {
            has_loop_back_edge = true;
          }
        }
      }
      if (has_loop_back_edge) {
        pending_count = num_control_edges + 1;
      }
    }
    for (int i = 0; i < node_def.input_size(); ++i) {
      StringPiece input_name = node_def.input(i);
      TensorId id(ParseTensorName(input_name));
      if (opts_.input_map.count(id) == 0) {
        // If an input is not mapped, then the input should appear in the graph
        // being imported.
        auto iter = gdef_nodes_.find(id.first);
        if (iter == gdef_nodes_.end()) {
          return errors::InvalidArgument("Node '", node_def.name(),
                                         "': Unknown input node '",
                                         node_def.input(i), "'");
        }
        outputs_[iter->second.gdef_index].push_back(n);
      } else {
        // This input is mapped to an existing edge. Therefore this input is
        // as good as being already processed.
        --pending_count;
        DCHECK_GE(pending_count, 0);
      }
    }
    if (pending_count == 0) {
      ready_.insert(n);
    }
    pending_count_.push_back(pending_count);
  }
  return Status::OK();
}

Status GraphConstructor::ValidateColocationConstraints(
    const NodeDef& node_def) {
  if (!opts_.validate_colocation_constraints || !opts_.importing)
    return Status::OK();
  const auto iter = node_def.attr().find(kColocationAttrName);
  if (iter == node_def.attr().end()) return Status::OK();
  for (const string& c : iter->second.list().s()) {
    StringPiece s(c);
    if (absl::ConsumePrefix(&s, kColocationGroupPrefix) &&
        gdef_nodes_.find(s) == gdef_nodes_.end()) {
      return errors::InvalidArgument(
          "Node '", node_def.name(),
          "' expects to be colocated with unknown node '", s, "'");
    }
  }
  return Status::OK();
}

Status GraphConstructor::MakeNode(NodeDef&& node_def, Node** node) {
  // Add the node to the graph.
  Status status;
  *node = g_->AddNode(std::move(node_def), &status);
  if (!status.ok()) return status;
  if (opts_.expect_device_spec) {
    (*node)->set_assigned_device_name((*node)->def().device());
  }
  return Status::OK();
}

Status GraphConstructor::ValidateShape(Node* node) {
  if (!opts_.importing || !opts_.validate_shape) return Status::OK();
  TF_RETURN_IF_ERROR(refiner_->AddNode(node));
  // For nodes with the _output_shapes attribute, override the shape.
  std::vector<const TensorShapeProto*> shape_attrs;
  const char* kAttrName = "_output_shapes";
  if (!TryGetNodeAttr(node->attrs(), kAttrName, &shape_attrs)) {
    // No _output_shapes attribute, the AddNode call above was sufficient.
    return Status::OK();
  }
  auto* ic = refiner_->GetContext(node);
  DCHECK(ic != nullptr)
      << "ShapeRefiner::AddNode() should have created the InferenceContext";
  if (shape_attrs.size() < node->num_outputs()) {
    return errors::InvalidArgument(
        "Node '", node->name(), "' has ", node->num_outputs(),
        " outputs but the ", kAttrName, " attribute specifies shapes for ",
        shape_attrs.size(), " outputs");
  }
  // NOTE(skyewm): we don't raise an error here because some users depend on
  // this behavior, even though it's unsafe.
  // TODO(b/74619486): raise an error.
  if (shape_attrs.size() > node->num_outputs()) {
    LOG(WARNING) << "Node '" << node->name() << "' has " << node->num_outputs()
                 << " outputs but the " << kAttrName
                 << " attribute specifies shapes for " << shape_attrs.size()
                 << " outputs. Output shapes may be inaccurate.";
  }
  for (int i = 0; i < node->num_outputs(); ++i) {
    const TensorShapeProto& p = *shape_attrs[i];
    shape_inference::ShapeHandle h;
    Status s = ic->MakeShapeFromShapeProto(p, &h);
    if (!s.ok()) {
      return errors::InvalidArgument("Node '", node->name(), " has an invalid ",
                                     kAttrName, " attribute (shape #", i,
                                     " error:'", s.error_message(), "'");
    }
    s = refiner_->SetShape(node, i, h);
    if (!s.ok()) {
      return errors::InvalidArgument(
          "Node '", node->name(), "' has an ", kAttrName,
          " attribute inconsistent with the GraphDef for output #", i, ": ",
          s.error_message());
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
  if (versions()) {
    TF_RETURN_IF_ERROR(CheckOpDeprecation(*op_def, versions()->producer()));
  }
  return Status::OK();
}

void RemoveInputs(const std::vector<int>& inputs_to_remove, NodeDef* node_def,
                  std::vector<bool>* input_already_exists) {
  // Remove 'inputs_to_remove' from 'node_def'
  NodeDef copy;
  copy.mutable_input()->Reserve(node_def->input_size() -
                                inputs_to_remove.size());
  for (int i = 0, j = 0; i < node_def->input_size(); ++i) {
    if (j < inputs_to_remove.size() && i == inputs_to_remove[j]) {
      ++j;
    } else {
      copy.add_input()->swap(*node_def->mutable_input(i));
    }
  }
  node_def->mutable_input()->Swap(copy.mutable_input());
  // Remove 'inputs_to_remove' from 'input_already_exists'
  for (int idx : inputs_to_remove) {
    input_already_exists->erase(input_already_exists->begin() + idx);
  }
  DCHECK_EQ(input_already_exists->size(), node_def->input_size());
}

void GraphConstructor::RemapNodeDefInputs(
    NodeDef* node_def, std::vector<bool>* input_already_exists) {
  DCHECK_EQ(input_already_exists->size(), node_def->input_size());
  std::set<TensorId> control_inputs;
  std::vector<int> inputs_to_remove;

  for (int i = 0; i < node_def->input_size(); ++i) {
    auto iter = opts_.input_map.find(ParseTensorName(node_def->input(i)));
    if (iter == opts_.input_map.end()) continue;
    used_input_map_keys_.insert(iter->first);

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
  if (!inputs_to_remove.empty()) {
    RemoveInputs(inputs_to_remove, node_def, input_already_exists);
  }
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
    bool found = false;
    for (int i = node_def->input_size() - 1; i >= 0; --i) {
      const string& node_input = node_def->input(i);
      if (node_input[0] != '^') {
        // Control inputs are at the end. Break when we reach the non-control
        // inputs.
        break;
      }
      if (node_input == input) {
        // Control dependency already exists
        found = true;
        break;
      }
    }
    if (found) {
      continue;
    }
    node_def->add_input(input);
    input_already_exists->push_back(true);
  }
}

void GraphConstructor::AddPrefixToNodeDef(
    const std::vector<bool>& input_already_exists, NodeDef* node_def) {
  if (prefix_.empty()) return;
  node_def->set_name(strings::StrCat(prefix_, node_def->name()));
  // Update names of input nodes
  for (int i = 0; i < node_def->input_size(); ++i) {
    // Skip remapped inputs (which already exist in g_ and are not being
    // imported).
    if (input_already_exists[i]) continue;
    StringPiece input(node_def->input(i));
    if (absl::ConsumePrefix(&input, "^")) {
      node_def->set_input(i, strings::StrCat("^", prefix_, input));
    } else {
      node_def->set_input(i, strings::StrCat(prefix_, input));
    }
  }
  // Update names of colocation groups
  if (node_def->attr().find(kColocationAttrName) != node_def->attr().end()) {
    auto* list =
        node_def->mutable_attr()->at(kColocationAttrName).mutable_list();
    for (int i = 0; i < list->s_size(); ++i) {
      StringPiece v(list->s(i));
      if (absl::ConsumePrefix(&v, kColocationGroupPrefix)) {
        list->set_s(i, strings::StrCat(kColocationGroupPrefix, prefix_, v));
      }
    }
  }
}

void GraphConstructor::UniquifyNames(
    const std::vector<bool>& input_already_exists, NodeDef* node_def) {
  if (NameExistsInGraph(node_def->name())) {
    string old_name = node_def->name();
    node_def->set_name(FindUniqueName(node_def->name()));
    uniquified_names_[old_name] = node_def->name();
    // Note that we don't have to update gdef_nodes_ or gdef_prefixes_ with
    // `name` because we guarantee the original NodeDef names are unique,
    // meaning we won't generate this name again.
  }
  for (int i = 0; i < node_def->input_size(); ++i) {
    // Skip remapped inputs (which already exist in g_ and are not being
    // imported).
    if (input_already_exists[i]) continue;
    TensorId id = ParseTensorName(node_def->input(i));
    // We require that UniquifyNames() is called on all NodeDefs in topological
    // order. This guarantees that node_def's inputs will already be uniquified
    // if necessary.
    auto iter = uniquified_names_.find(string(id.first));
    if (iter == uniquified_names_.end()) continue;
    id.first = iter->second;
    node_def->set_input(i, id.ToString());
  }
}

void GraphConstructor::UpdateUniquifiedColocationNames() {
  for (const auto& pair : gdef_nodes_) {
    Node* node = pair.second.node;
    if (node == nullptr) continue;
    std::vector<string> coloc_values;
    if (!TryGetNodeAttr(node->attrs(), kColocationAttrName, &coloc_values))
      continue;
    bool updated = false;
    for (size_t i = 0; i < coloc_values.size(); ++i) {
      StringPiece val(coloc_values[i]);
      if (absl::ConsumePrefix(&val, kColocationGroupPrefix)) {
        auto name_pair = uniquified_names_.find(string(val));
        if (name_pair == uniquified_names_.end()) continue;
        updated = true;
        coloc_values[i] =
            strings::StrCat(kColocationGroupPrefix, name_pair->second);
      }
    }
    if (updated) {
      node->AddAttr(kColocationAttrName, std::move(coloc_values));
    }
  }
}

bool GraphConstructor::NameExistsInGraph(StringPiece name) {
  if (existing_nodes_.find(name) != existing_nodes_.end()) return true;
  if (existing_prefixes_.find(name) != existing_prefixes_.end()) return true;
  return false;
}

bool GraphConstructor::NameExistsInGraphDef(StringPiece name) {
  if (gdef_nodes_.find(name) != gdef_nodes_.end()) return true;
  if (gdef_prefixes_.find(name) != gdef_prefixes_.end()) return true;
  return false;
}

string GraphConstructor::FindUniqueName(StringPiece original_name) {
  string name(original_name);
  int count = 0;
  // Check that any generated names don't collide with imported NodeDefs (as
  // well as nodes in g_).
  while (NameExistsInGraph(name) || (count > 0 && NameExistsInGraphDef(name))) {
    name = strings::StrCat(original_name, "_", ++count);
  }
  return name;
}

Status GraphConstructor::IsNodeFullyMapped(const NodeDef& node_def,
                                           bool* is_node_mapped) {
  const OpDef* op_def;
  TF_RETURN_IF_ERROR(g_->op_registry()->LookUpOpDef(node_def.op(), &op_def));
  for (int i = 0; i < op_def->output_arg_size(); ++i) {
    if (opts_.input_map.find({node_def.name(), i}) == opts_.input_map.end()) {
      *is_node_mapped = false;
      return Status::OK();
    }
  }
  *is_node_mapped = true;
  return Status::OK();
}

void GraphConstructor::DFS(int cur_node, std::vector<int>* cur_branch,
                           std::vector<bool>* is_on_cur_branch,
                           absl::flat_hash_set<int>* unvisited,
                           const std::vector<absl::string_view>& node_names) {
  cur_branch->push_back(cur_node);
  is_on_cur_branch->at(cur_node) = true;
  for (auto next_node : outputs_[cur_node]) {
    if (unvisited->find(next_node) != unvisited->end()) {
      if (is_on_cur_branch->at(next_node)) {
        auto iter =
            std::find(cur_branch->begin(), cur_branch->end(), next_node);
        LOG(WARNING) << "Cycle detected:";
        while (iter != cur_branch->end()) {
          const absl::string_view name = node_names[*iter];
          DCHECK(!name.empty());
          LOG(WARNING) << "node id=" << *iter << ", name=" << name;
          ++iter;
        }
        LOG(WARNING) << "End of cycle";
      } else {
        DFS(next_node, cur_branch, is_on_cur_branch, unvisited, node_names);
      }
    }
  }
  cur_branch->pop_back();
  is_on_cur_branch->at(cur_node) = false;
  unvisited->erase(cur_node);
}

void GraphConstructor::PrintCycles() {
  int num_nodes = outputs_.size();

  std::vector<absl::string_view> node_names;
  node_names.resize(num_nodes);
  for (const auto& named_node : gdef_nodes_) {
    DCHECK_GE(named_node.second.gdef_index, 0);
    DCHECK_LT(named_node.second.gdef_index, num_nodes);
    node_names[named_node.second.gdef_index] = named_node.first;
  }

  absl::flat_hash_set<int> unvisited;
  for (int i = 0; i < num_nodes; i++) {
    unvisited.insert(i);
  }

  while (!unvisited.empty()) {
    int cur_node = *unvisited.begin();
    // Nodes on the current branch of DFS in traversal order. This is used for
    // printing the nodes in the cycle.
    std::vector<int> cur_branch;
    // This is just to make lookups O(1).
    // is_on_cur_branch[i] ==
    //   (std::find(cur_branch.start(),
    //              cur_branch.end(), i) != cur_branch.end())
    std::vector<bool> is_on_cur_branch(num_nodes, false);
    DFS(cur_node, &cur_branch, &is_on_cur_branch, &unvisited, node_names);
  }
}

Status GraphConstructor::Convert() {
  // Import functions before adding nodes, since imported nodes may refer to
  // functions
  if (library()) {
    // TODO(b/135705010): Add rvalue overloads into the function library, to
    // avoid unnecessarily copying `*library()` here.
    TF_RETURN_IF_ERROR(g_->AddFunctionLibrary(*library()));
  }

  std::vector<InputInfo> inputs;
  int processed = 0;

  std::vector<bool> input_already_exists;

  // Process the NodeDefs in topological order.
  // (InitFromEdges() sets this up by filling in ready_ with nodes that have no
  // inputs, pending_counts_ with the number of inputs for each node and
  // outputs_ with the outputs of each node).
  while (!ready_.empty()) {
    int o = *ready_.begin();
    ready_.erase(ready_.begin());
    ++processed;
    inputs.clear();
    bool has_data_back_edge = false;

    NodeDef node_def = consume_node_def(o);

    // input_already_exists[i] is true iff the i-th input of the node we're
    // importing refers to a preexisting node in g_ (i.e. input[i] existed prior
    // to importing node_defs_).  Conversely, input_already_exists[i] is false
    // iff the input refers to a node in node_defs_.
    input_already_exists.clear();
    input_already_exists.resize(node_def.input_size(), false);

    std::string node_name = node_def.name();

    if (opts_.importing) {
      if (opts_.skip_mapped_nodes) {
        bool is_node_mapped = false;
        TF_RETURN_IF_ERROR(IsNodeFullyMapped(node_def, &is_node_mapped));
        if (is_node_mapped) {
          // Skip this node after updating pending_count_ for outputs
          UpdatePendingCountAndReady(o, IsNextIteration(node_def));
          continue;
        }
      }

      if (!opts_.input_map.empty()) {
        // Note that input_already_exists can shrink here
        RemapNodeDefInputs(&node_def, &input_already_exists);
      }
      if (!opts_.control_dependencies.empty()) {
        // Note that input_already_exists can grow here
        AddControlDependencies(&node_def, &input_already_exists);
      }
      if (!opts_.default_device.empty() && node_def.device().empty()) {
        node_def.set_device(opts_.default_device);
      }
    }

    DCHECK_EQ(node_def.input_size(), input_already_exists.size());
    TF_RETURN_IF_ERROR(ValidateColocationConstraints(node_def));
    for (int i = 0; i < node_def.input_size(); ++i) {
      TensorId tensor_id = ParseTensorName(node_def.input(i));
      Node* src_node;
      int src_index;

      if (!input_already_exists[i]) {
        // Locate input in newly-imported nodes
        auto iter = gdef_nodes_.find(tensor_id.node());
        DCHECK(iter != gdef_nodes_.end()) << tensor_id.node();
        src_node = iter->second.node;
        src_index = tensor_id.index();
        if (src_node == nullptr) has_data_back_edge = true;
      } else {
        // Input refers to preexistng node in graph
        auto iter = existing_nodes_.find(tensor_id.node());
        DCHECK(iter != existing_nodes_.end()) << tensor_id.node();
        src_node = iter->second;
        src_index = tensor_id.index();
      }

      if (src_node != nullptr && src_index >= src_node->num_outputs()) {
        std::ostringstream out;
        out << "Node '" << node_def.name() << "': Connecting to invalid output "
            << tensor_id.index() << " of source node " << tensor_id.node()
            << " which has " << src_node->num_outputs() << " outputs.";

        if (src_node->type_string() == "If" ||
            src_node->type_string() == "StatelessIf" ||
            src_node->type_string() == "While" ||
            src_node->type_string() == "StatelessWhile") {
          out << " Try using "
              << "tf.compat.v1.experimental.output_all_intermediates(True).";
        }
        return errors::InvalidArgument(out.str());
      }

      inputs.emplace_back(string(tensor_id.node()), src_node, src_index);
    }

    if (has_data_back_edge && !IsMerge(node_def)) {
      return errors::InvalidArgument(
          "Node '", node_def.name(),
          "' had a back edge, but only Merge nodes can have back edges.");
    }

    Node* node;
    if (opts_.importing) {
      if (!prefix_.empty()) {
        AddPrefixToNodeDef(input_already_exists, &node_def);
      }
      // Note: no need to uniquify names if the prefix already guarantees
      // uniqueness
      if (opts_.uniquify_names && (prefix_.empty() || !opts_.uniquify_prefix)) {
        UniquifyNames(input_already_exists, &node_def);
      }
    }

    if (opts_.importing) {
      TF_RETURN_IF_ERROR(ModifyNodeDefForImport(&node_def));
    } else {
      const OpDef* op_def;
      TF_RETURN_IF_ERROR(
          g_->op_registry()->LookUpOpDef(node_def.op(), &op_def));
      if (opts_.add_default_attributes) {
        AddDefaultsToNodeDef(*op_def, &node_def);
      }
      if (opts_.validate_nodes) {
        TF_RETURN_IF_ERROR(ValidateNodeDef(node_def, *op_def));
      }
    }

    TF_RETURN_IF_ERROR(MakeNode(std::move(node_def), &node));

    gdef_nodes_[node_name].node = node;

    // Remove duplicate control inputs before adding edges to the graph. It
    // will allow us to skip expensive duplicates check in 'AddControlEdge'.
    auto first_control = absl::c_find_if(inputs, &InputInfo::IsControlInput);
    auto first_control_copy = first_control;
    std::sort(first_control, inputs.end(), &InputInfo::CompareName);
    inputs.erase(
        std::unique(first_control_copy, inputs.end(), &InputInfo::IsSameName),
        inputs.end());

    // Add edges from inputs to *node to the graph.
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].node == nullptr) {
        // Record this back edge, which will be added after all nodes
        // are created.
        back_edges_.emplace_back(inputs[i].name, inputs[i].index, node, i);
      } else if (inputs[i].index == Graph::kControlSlot) {
        g_->AddControlEdge(inputs[i].node, node, kDoNotCheckDuplicates);
      } else {
        TF_RETURN_IF_ERROR(MakeEdge(inputs[i].node, inputs[i].index, node, i));
      }
    }

    TF_RETURN_IF_ERROR(ValidateShape(node));

    // Update pending_count_ for outputs.
    UpdatePendingCountAndReady(o, node->IsNextIteration());
  }

  if (processed < node_def_count()) {
    LOG(WARNING) << "IN " << __func__ << " " << (node_def_count() - processed)
                 << " NODES IN A CYCLE";
    for (int64_t i = 0; i < node_def_count(); i++) {
      if (pending_count_[i] != 0) {
        LOG(WARNING) << "PENDING: " << SummarizeNodeDef(get_node_def(i))
                     << " WITH PENDING COUNT = " << pending_count_[i];
      }
    }
    PrintCycles();
    return errors::InvalidArgument(node_def_count() - processed,
                                   " nodes in a cycle");
  }

  return Status::OK();
}

Status GraphConstructor::AddBackEdges() {
  // Add the back edges after all nodes are created.
  for (const auto& e : back_edges_) {
    Node* src_node = gdef_nodes_[e.src_name].node;
    if (e.src_index == Graph::kControlSlot) {
      g_->AddControlEdge(src_node, e.dst_node, kDoNotCheckDuplicates);
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
  if (versions() == nullptr) return Status::OK();

  if (!opts_.importing) {
    g_->set_versions(*versions());
    return Status::OK();
  }
  VersionDef g_versions = g_->versions();
  g_versions.set_producer(
      std::min(g_versions.producer(), versions()->producer()));
  g_versions.set_min_consumer(
      std::max(g_versions.min_consumer(), versions()->min_consumer()));
  if (versions()->bad_consumers_size() > 0) {
    std::set<int> bad(g_versions.bad_consumers().begin(),
                      g_versions.bad_consumers().end());
    bad.insert(versions()->bad_consumers().begin(),
               versions()->bad_consumers().end());
    g_versions.clear_bad_consumers();
    for (int v : bad) {
      g_versions.add_bad_consumers(v);
    }
  }
  g_->set_versions(g_versions);
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
        return errors::InvalidArgument("Requested return tensor '",
                                       id.ToString(),
                                       "' not found in graph def");
      }
      int num_outputs = iter->second.node->num_outputs();
      if ((id.second < 0 || id.second >= num_outputs) &&
          id.second != Graph::kControlSlot) {
        return errors::InvalidArgument("Invalid return output ", id.second,
                                       " of node '", id.first, "', which has ",
                                       num_outputs, " output(s)");
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

Status GraphConstructor::PopulateReturnNodes() {
  if (opts_.return_nodes.empty()) return Status::OK();
  for (StringPiece name : opts_.return_nodes) {
    auto iter = gdef_nodes_.find(name);
    if (iter == gdef_nodes_.end()) {
      return errors::InvalidArgument("Requested return node '", name,
                                     "' not found in graph def");
    }
    return_nodes_->push_back(iter->second.node);
  }
  return Status::OK();
}

Status GraphConstructor::PopulateMissingUnusedInputMapKeys() {
  if (missing_unused_input_map_keys_ == nullptr) return Status::OK();
  for (const auto& input_map_pair : opts_.input_map) {
    TensorId key = input_map_pair.first;
    if (used_input_map_keys_.count(key) > 0) continue;

    auto pair = gdef_nodes_.find(key.first);
    if (pair == gdef_nodes_.end()) {
      // key's node doesn't exist in GraphDef
      missing_unused_input_map_keys_->push_back(key);
      continue;
    }

    // Check that key's index is in bounds. Get the number of outputs from the
    // NodeDef, rather than the imported Node, since the Node may not exist if
    // opts_.skip_mapped_nodes is true.
    const NodeDef& node_def = get_node_def(pair->second.gdef_index);
    const OpDef* op_def;
    TF_RETURN_IF_ERROR(g_->op_registry()->LookUpOpDef(node_def.op(), &op_def));
    int num_outputs;
    TF_RETURN_IF_ERROR(NumOutputsForNode(node_def, *op_def, &num_outputs));
    if (key.second >= num_outputs) {
      // key's index out of bounds
      missing_unused_input_map_keys_->push_back(key);
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
  if (output_index >= src->num_outputs()) {
    return errors::InvalidArgument(
        "Output ", output_index, " of node ", src->name(),
        " does not exist. Node only has ", src->num_outputs(), " outputs.");
  }
  if (input_index >= dst->num_inputs()) {
    return errors::InvalidArgument(
        "Input ", input_index, " of node ", dst->name(),
        " does not exist. Node only has ", dst->num_inputs(), " inputs.");
  }

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
  return GraphConstructor::Construct(
      opts, gdef.node(), &gdef.versions(), &gdef.library(), g, &refiner,
      /*return_tensors=*/nullptr, /*return_nodes=*/nullptr,
      /*missing_unused_input_map_keys=*/nullptr);
}

Status ConvertGraphDefToGraph(const GraphConstructorOptions& opts,
                              GraphDef&& gdef, Graph* g) {
  ShapeRefiner refiner(gdef.versions().producer(), g->op_registry());
  return GraphConstructor::Construct(opts, std::move(gdef), g, &refiner,
                                     /*return_tensors=*/nullptr,
                                     /*return_nodes=*/nullptr,
                                     /*missing_unused_input_map_keys=*/nullptr);
}

Status ConvertNodeDefsToGraph(const GraphConstructorOptions& opts,
                              gtl::ArraySlice<NodeDef> nodes, Graph* g) {
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, g->op_registry());
  // TODO(irving): Copy will go away once NodeInfo exists
  std::vector<const NodeDef*> node_defs;
  node_defs.reserve(nodes.size());
  for (const auto& n : nodes) {
    node_defs.push_back(&n);
  }
  return GraphConstructor::Construct(opts, node_defs, nullptr, nullptr, g,
                                     &refiner, /*return_tensors=*/nullptr,
                                     /*return_nodes=*/nullptr,
                                     /*missing_unused_input_map_keys=*/nullptr);
}

Status ImportGraphDef(const ImportGraphDefOptions& opts, const GraphDef& gdef,
                      Graph* g, ShapeRefiner* refiner,
                      ImportGraphDefResults* results) {
  if (!opts.return_tensors.empty()) {
    if (results == nullptr) {
      return errors::InvalidArgument(
          "results argument to ImportGraphDef() must be non-null if "
          "opts.return_tensors is non-empty");
    }
  }

  if (!opts.return_nodes.empty()) {
    if (opts.skip_mapped_nodes) {
      return errors::InvalidArgument(
          "Requesting return_nodes with skip_mapped_nodes set is not currently "
          "supported");
    }
    if (results == nullptr) {
      return errors::InvalidArgument(
          "results argument to ImportGraphDef() must be non-null if "
          "opts.return_nodes is non-empty");
    }
  }

  if (results != nullptr) {
    if (!results->return_tensors.empty() || !results->return_nodes.empty() ||
        !results->missing_unused_input_map_keys.empty()) {
      return errors::InvalidArgument(
          "All fields in results argument to ImportGraphDef() must be empty.");
    }
  }

  ShapeRefiner default_refiner(gdef.versions().producer(), g->op_registry());
  if (refiner == nullptr) {
    refiner = &default_refiner;
  } else {
    // Log a warning if we are importing a GraphDef at an older
    // producer version after already having added non-source/sink
    // nodes to the graph in the past.
    if (gdef.versions().producer() > 0 &&
        gdef.versions().producer() < refiner->graph_def_version() &&
        g->num_nodes() > 2) {
      LOG(WARNING) << "Importing a graph with a lower producer version "
                   << gdef.versions().producer()
                   << " into an existing graph with producer version "
                   << refiner->graph_def_version() << ". Shape inference will "
                   << "have run different parts of the graph with different "
                   << "producer versions.";
    }
  }

  // Set the graph def version of the refiner as the min of the
  // current value and the version from the graph we are about to
  // import.
  //
  // Note: to match Run() semantics, we should re-run shape inference
  // on the entire graph if the producer version has changed.  For now
  // we log the warning above.
  refiner->set_graph_def_version(
      std::min(refiner->graph_def_version(), gdef.versions().producer()));

  if (results == nullptr) {
    return GraphConstructor::Construct(opts, gdef.node(), &gdef.versions(),
                                       &gdef.library(), g, refiner, nullptr,
                                       nullptr, nullptr);
  } else {
    return GraphConstructor::Construct(
        opts, gdef.node(), &gdef.versions(), &gdef.library(), g, refiner,
        &results->return_tensors, &results->return_nodes,
        &results->missing_unused_input_map_keys);
  }
}

void CopyGraph(const Graph& src, Graph* dest) { dest->Copy(src); }

}  // namespace tensorflow
