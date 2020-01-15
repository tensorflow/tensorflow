/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/tf2xla_util.h"

#include <functional>
#include <queue>
#include <random>
#include <set>
#include <unordered_map>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/sharding_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

namespace {

Status ValidateTensorId(const tf2xla::TensorId& id) {
  if (id.node_name().empty()) {
    return errors::InvalidArgument("TensorId node_name must be non-empty");
  }
  if (id.output_index() < 0) {
    return errors::InvalidArgument("TensorId output_index must be positive");
  }
  return Status::OK();
}

Status CheckNameDuplicates(const string& kind, const string& name,
                           std::set<string>* names) {
  if (!name.empty()) {
    if (!names->insert(name).second) {
      return errors::InvalidArgument("duplicate ", kind, " name: ", name);
    }
  }
  return Status::OK();
}

Status CheckFeedFetchNameConflicts(const string& kind,
                                   const std::set<string>& names) {
  // We don't allow the feeds or fetches to contain both "foo" and "foo_data",
  // since that will cause a collision in codegen symbols.
  for (const string& name : names) {
    const string name_data(name + "_data");
    if (names.find(name_data) != names.end()) {
      return errors::InvalidArgument("conflicting ", kind, " name: ", name,
                                     " and ", name_data);
    }
  }
  return Status::OK();
}

// For graph `g`, copy all function call nodes' FunctionDef from `lookup_fld` to
// `fld`. This is to ensure that `fld` can instantiate FunctionDef of graph `g`.
Status CopyAssociatedFunctions(Graph* g,
                               const FunctionLibraryDefinition* lookup_fld,
                               FunctionLibraryDefinition* fld) {
  for (Node* n : g->op_nodes()) {
    for (const auto& associated_function :
         GetAssociatedFunctions(*n, lookup_fld)) {
      switch (associated_function.type()) {
        case AssociatedFunctionInfo::kFunctionCallNode: {
          const FunctionDef* fdef =
              lookup_fld->Find(associated_function.func_name());
          if (!fdef) {
            return errors::Internal(
                "Cannot find function ", associated_function.func_name(),
                " for function call node ", n->DebugString());
          }
          TF_RETURN_IF_ERROR(fld->AddFunctionDef(*fdef));
          break;
        }
        case AssociatedFunctionInfo::kSymbolicGradient:
        case AssociatedFunctionInfo::kFunctionAttr:
          break;
      }
    }
  }
  return Status::OK();
}

// For graph `g`, replaces _Arg nodes whose "index" attribute is in
// `const_input_index_to_node` with Const nodes.
Status ReplaceArgUsageWithConstNode(
    Graph* g,
    const std::unordered_map<int, const Node*>& const_input_index_to_node) {
  // Collect all _Arg nodes.
  std::unordered_map<int, Node*> arg_nodes;
  for (Node* n : g->op_nodes()) {
    if (n->IsArg()) {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      arg_nodes[index] = n;
    }
  }

  for (const auto& iter : const_input_index_to_node) {
    int arg_index = iter.first;
    NodeDef const_def = iter.second->def();
    const_def.set_name(g->NewName(const_def.name()));
    Status s;
    Node* const_node = g->AddNode(const_def, &s);
    TF_RETURN_IF_ERROR(s);

    Node* arg_node = arg_nodes[arg_index];

    // Collect all usages of the _Arg node.
    struct OutEdgeInfo {
      int dst_node_id, dst_input;
    };
    std::vector<OutEdgeInfo> usages;
    for (const Edge* e : arg_node->out_edges()) {
      if (e->IsControlEdge()) {
        continue;
      }
      usages.push_back({e->dst()->id(), e->dst_input()});
    }

    for (int i = 0; i < usages.size(); i++) {
      // Make a copy of `usage_node`, and change its input to const node.
      Node* usage_node = g->FindNodeId(usages[i].dst_node_id);
      NodeDef replace_def = usage_node->def();
      *replace_def.mutable_input(usages[i].dst_input) = const_node->name();
      TF_ASSIGN_OR_RETURN(Node * replace_node,
                          ReplaceNode(g, usage_node, replace_def));
      const Edge* usage_edge;
      TF_RETURN_IF_ERROR(
          replace_node->input_edge(usages[i].dst_input, &usage_edge));
      g->RemoveEdge(usage_edge);
      g->AddEdge(const_node, 0, replace_node, usages[i].dst_input);

      // Later entries in `usages` might have `usage_node` as dst node, but
      // `usage_node` is removed. Replace such entries with `replace_node`.
      for (int j = i + 1; j < usages.size(); j++) {
        if (usages[j].dst_node_id == usages[i].dst_node_id) {
          usages[j].dst_node_id = replace_node->id();
        }
      }
    }
  }
  return Status::OK();
}

// For a node's function attr (e.g. then/else branch for "If" nodes), rewrites
// the function to replace _Arg nodes in `const_input_index_to_node` with Const
// inputs.
Status PropagateConstIntoFuncAttr(
    Node* n, const string& attr_name,
    const std::unordered_map<int, const Node*>& const_input_index_to_node,
    const FunctionLibraryDefinition* lookup_fld,
    FunctionLibraryDefinition* fld) {
  // Instantiate the function.
  NameAttrList func_attr;
  TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), attr_name, &func_attr));
  const FunctionDef* fdef = lookup_fld->Find(func_attr.name());
  if (!fdef) {
    return errors::Internal("Cannot find function ", func_attr.name(),
                            " for node ", n->name());
  }
  std::unique_ptr<FunctionBody> fbody;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
      *fdef, AttrSlice(&func_attr.attr()), lookup_fld, &fbody));

  // Rewrite _Arg usages with Const node.
  Graph* func_graph = fbody->graph;
  TF_RETURN_IF_ERROR(
      ReplaceArgUsageWithConstNode(func_graph, const_input_index_to_node));

  // Save rewritten function.
  FunctionDef replace_fdef;
  string new_func_name =
      fld->UniqueFunctionName(absl::StrCat(func_attr.name(), "_const_"));
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(*func_graph, new_func_name, &replace_fdef));
  TF_RETURN_IF_ERROR(fld->AddFunctionDef(replace_fdef));

  // Change the node to use rewritten function.
  func_attr.set_name(new_func_name);
  n->ClearAttr(attr_name);
  n->AddAttr(attr_name, func_attr);

  // Copy associated functions.
  TF_RETURN_IF_ERROR(CopyAssociatedFunctions(func_graph, lookup_fld, fld));

  return Status::OK();
}

// For an "If" node in graph `g`, if it has Const node inputs, rewrite its
// then/else branch function to replace _Arg nodes with those Const inputs.
Status PropagateConstIntoIfNode(Graph* g, Node* if_node,
                                const FunctionLibraryDefinition* lookup_fld,
                                FunctionLibraryDefinition* fld) {
  // Notice that first input for If node is predicate; other inputs are function
  // inputs.
  std::unordered_map<int, const Node*> const_input_index_to_node;
  for (int i = 1; i < if_node->num_inputs(); i++) {
    const Node* input_node;
    TF_RETURN_IF_ERROR(if_node->input_node(i, &input_node));
    if (input_node->type_string() == "Const") {
      const_input_index_to_node[i - 1] = input_node;
    }
  }
  if (const_input_index_to_node.empty()) {
    return Status::OK();
  }

  // Rewrite "then_branch" and "else_branch" function, replace usage of those
  // _Arg nodes with corresponding const node.
  for (const auto& attr_name :
       std::vector<string>{"then_branch", "else_branch"}) {
    TF_RETURN_IF_ERROR(PropagateConstIntoFuncAttr(
        if_node, attr_name, const_input_index_to_node, lookup_fld, fld));
  }

  return Status::OK();
}

// For a "While" node in graph `g`, if it has Const node inputs, rewrite its
// cond/body function to replace _Arg nodes with those Const inputs.
Status PropagateConstIntoWhileNode(Graph* g, Node* while_node,
                                   const FunctionLibraryDefinition* lookup_fld,
                                   FunctionLibraryDefinition* fld) {
  // For "While" node, we should only replace _Arg nodes which are loop
  // invariants. For such _Arg nodes, the return value's input will come
  // directly from the corresponding arg.
  std::unordered_map<int, const Node*> const_input_index_to_node;
  NameAttrList body_attr;
  TF_RETURN_IF_ERROR(GetNodeAttr(while_node->def(), "body", &body_attr));
  const FunctionDef* body_func = lookup_fld->Find(body_attr.name());
  if (!body_func) {
    return errors::Internal("Cannot find body function ", body_attr.name(),
                            " for While node ", while_node->name());
  }
  for (int i = 0; i < while_node->num_inputs(); i++) {
    const Node* input_node;
    TF_RETURN_IF_ERROR(while_node->input_node(i, &input_node));
    if (input_node->type_string() != "Const") {
      continue;
    }

    // Check if i-th retval's input comes from i-th arg directly.
    // For resource variable input of While nodes, TF2XLA convention is to place
    // them at the end of all inputs (after all data inputs), and *not* return
    // them. So number of While node inputs might be larger than number of its
    // outputs.
    if (i >= body_func->signature().output_arg_size()) {
      continue;
    }
    const OpDef_ArgDef& output_arg = body_func->signature().output_arg(i);
    auto output_arg_input = body_func->ret().find(output_arg.name());
    if (output_arg_input == body_func->ret().end()) {
      return errors::Internal("Cannot find input for output arg ",
                              output_arg.name(), " in function ",
                              body_attr.name());
    }
    const OpDef_ArgDef& input_arg = body_func->signature().input_arg(i);
    if (output_arg_input->second != input_arg.name()) {
      continue;
    }

    const_input_index_to_node[i] = input_node;
  }
  if (const_input_index_to_node.empty()) {
    return Status::OK();
  }

  // Rewrite "cond" and "body" function, replace usage of those _Arg nodes with
  // corresponding const node.
  for (const auto& attr_name : std::vector<string>{"cond", "body"}) {
    TF_RETURN_IF_ERROR(PropagateConstIntoFuncAttr(
        while_node, attr_name, const_input_index_to_node, lookup_fld, fld));
  }
  return Status::OK();
}

}  // namespace

const char kXlaOutsideCompilationAttrName[] = "_xla_outside_compilation";

Status ValidateConfig(const tf2xla::Config& config) {
  std::set<string> names;
  for (const tf2xla::Feed& feed : config.feed()) {
    TF_RETURN_IF_ERROR(ValidateTensorId(feed.id()));
    TF_RETURN_IF_ERROR(TensorShape::IsValidShape(feed.shape()));
    TF_RETURN_IF_ERROR(CheckNameDuplicates("feed", feed.name(), &names));
  }
  TF_RETURN_IF_ERROR(CheckFeedFetchNameConflicts("feed", names));
  names.clear();
  for (const tf2xla::Fetch& fetch : config.fetch()) {
    TF_RETURN_IF_ERROR(ValidateTensorId(fetch.id()));
    TF_RETURN_IF_ERROR(CheckNameDuplicates("fetch", fetch.name(), &names));
  }
  TF_RETURN_IF_ERROR(CheckFeedFetchNameConflicts("fetch", names));
  if (config.fetch().empty()) {
    return errors::InvalidArgument("fetches must be specified");
  }
  return Status::OK();
}

Status AddPlaceholdersForFeeds(
    const tf2xla::Config& config, const OpRegistryInterface* op_registry,
    std::unordered_map<string, string>* feed_remapping, GraphDef* graph_def) {
  struct PlaceholderInfo {
    const tf2xla::Feed* feed = nullptr;  // point to Feed in <config>.
    string placeholder_name;
    DataType data_type = DT_INVALID;
  };

  // Put each fed tensor into a map by name:port. A map is used for determinism
  // when creating placeholders (genrules want deterministic output).
  std::map<string, PlaceholderInfo> placeholder_info;
  for (int i = 0; i < config.feed_size(); ++i) {
    const tf2xla::Feed* feed = &config.feed(i);
    const string name_port = TensorIdToString(feed->id());
    PlaceholderInfo& info = placeholder_info[name_port];
    info.feed = feed;
    info.placeholder_name = absl::StrCat("aot_feed_", feed->id().output_index(),
                                         "/", feed->id().node_name());
    (*feed_remapping)[name_port] = info.placeholder_name;
  }

  // Verify node exists and determine data type.
  std::unordered_map<string, const NodeDef*> name_to_node;
  for (int i = 0; i < graph_def->node_size(); ++i) {
    name_to_node[graph_def->node(i).name()] = &graph_def->node(i);
  }
  for (auto it = placeholder_info.begin(); it != placeholder_info.end(); ++it) {
    PlaceholderInfo& info = it->second;
    const tf2xla::TensorId& feed_id = info.feed->id();

    // Find the existing node and determine data type.
    auto node_it = name_to_node.find(feed_id.node_name());
    if (node_it == name_to_node.end()) {
      return errors::NotFound("Can't find feed node: ",
                              TensorIdToString(feed_id));
    }
    const NodeDef* existing = node_it->second;

    if (info.feed->type() != DT_INVALID) {
      info.data_type = info.feed->type();
    } else {
      // Build the node in order to infer its type.

      // Must first add default attrs as well, so do this in a copied GraphDef.
      GraphDef gd;
      *gd.mutable_versions() = graph_def->versions();
      *gd.add_node() = *existing;
      MergeDebugInfo(NodeDebugInfo(*existing), gd.mutable_node(0));
      TF_RETURN_IF_ERROR(
          AddDefaultAttrsToGraphDef(&gd, *op_registry, 0 /*node_offset*/));

      // Now build the node from the copied node def.
      Graph g(op_registry);
      g.set_versions(graph_def->versions());
      Status status;
      Node* feed_node = g.AddNode(gd.node(0), &status);
      TF_RETURN_IF_ERROR(status);

      if (info.feed->id().output_index() < feed_node->num_outputs()) {
        info.data_type =
            BaseType(feed_node->output_type(info.feed->id().output_index()));
      } else {
        return errors::InvalidArgument(
            "Invalid output_index ", info.feed->id().output_index(),
            " for feed node ", info.feed->id().node_name());
      }
    }
  }

  // Create placeholders. Note that we could avoid creating a placeholder for
  // feeds which are already placeholders, but we omit that to avoid more cases
  // in this code.
  for (auto it = placeholder_info.begin(); it != placeholder_info.end(); ++it) {
    const PlaceholderInfo& info = it->second;
    // TODO(shikharagarwal): Add original node information.
    NodeDef* d = graph_def->add_node();
    d->set_name(info.placeholder_name);
    d->set_op("PlaceholderV2");
    auto& attr_map = *d->mutable_attr();
    attr_map["dtype"].set_type(info.data_type);
    *attr_map["shape"].mutable_shape() = info.feed->shape();
  }

  // Rewrite references to the fed tensors to refer to the placeholder.
  for (int i = 0; i < graph_def->node_size(); ++i) {
    NodeDef* node_def = graph_def->mutable_node(i);
    for (int j = 0; j < node_def->input_size(); ++j) {
      auto id = ParseTensorName(node_def->input(j));
      auto it = placeholder_info.find(id.ToString());
      if (it != placeholder_info.end()) {
        node_def->set_input(j, it->second.placeholder_name);
      }
    }
  }

  return Status::OK();
}

Status PruneGraphDefInto(const tf2xla::Config& config, const GraphDef& in,
                         GraphDef* out) {
  *out = in;
  out->clear_node();

  // Tensors needed for feeding.
  std::set<std::pair<string, int>> feed_tensors;
  for (const tf2xla::Feed& feed : config.feed()) {
    feed_tensors.insert(
        std::make_pair(feed.id().node_name(), feed.id().output_index()));
  }

  // Maps node name to reachability.
  std::unordered_map<string, std::pair<bool, const NodeDef*>> node_by_name;
  for (const NodeDef& node : in.node()) {
    node_by_name[node.name()] = std::pair<bool, const NodeDef*>(false, &node);
  }

  // Traverse.
  std::queue<string> name_queue;
  for (int i = 0; i < config.fetch_size(); ++i) {
    name_queue.push(config.fetch(i).id().node_name());
  }
  while (!name_queue.empty()) {
    const string name = name_queue.front();
    name_queue.pop();

    auto find_it = node_by_name.find(name);
    if (find_it == node_by_name.end()) {
      return errors::InvalidArgument("While pruning graph, node ", name,
                                     " needed but not found in the graph.");
    }
    auto& map_entry = find_it->second;
    if (map_entry.first) {
      continue;
    }
    map_entry.first = true;

    // Push input nodes of the currently visited node to name_queue.
    for (const string& in_edge : map_entry.second->input()) {
      auto id = ParseTensorName(in_edge);
      const string node_name = string(id.first);
      if (feed_tensors.find(std::make_pair(node_name, id.second)) ==
          feed_tensors.end()) {
        name_queue.push(node_name);
      } else {
        // The input tensor is from an edge that is being fed. Therefore,
        // we skip recursing down that edge, to avoid requiring nodes that
        // may not be needed (note that the input node may still be added
        // to name_queue later if one of its output edges is not being fed).
      }
    }
  }

  // Copy over, preserving order of original and only nodes that are reachable
  // from the fetches.
  out->mutable_node()->Reserve(in.node_size());
  for (const NodeDef& node : in.node()) {
    if (node_by_name[node.name()].first) {
      *out->add_node() = node;
    }
  }
  return Status::OK();
}

string TensorIdToString(const tf2xla::TensorId& id) {
  return absl::StrCat(id.node_name(), ":", id.output_index());
}

Status SetNodeShardingFromNeighbors(Node* n, bool out_edges) {
  int core = -1;
  const Node* matching_node = nullptr;
  for (const Edge* edge : (out_edges ? n->out_edges() : n->in_edges())) {
    if (edge->IsControlEdge()) continue;
    const Node* possible_match = out_edges ? edge->dst() : edge->src();
    TF_ASSIGN_OR_RETURN(
        absl::optional<xla::OpSharding> sharding,
        ParseShardingFromDevice(
            *possible_match,
            /*num_cores_per_replica=*/std::numeric_limits<int32>::max()));
    if (sharding && sharding->type() == xla::OpSharding::MAXIMAL) {
      const int core_annotation = sharding.value().tile_assignment_devices(0);
      if (core == -1 || core > core_annotation) {
        core = core_annotation;
        matching_node = possible_match;
      }
    }
  }
  if (matching_node != nullptr) {
    n->set_assigned_device_name(matching_node->assigned_device_name());
    n->set_requested_device(matching_node->requested_device());
  }
  return Status::OK();
}

void AddDtypeToKernelDefConstraint(absl::string_view name, DataType dtype,
                                   KernelDef* kdef) {
  for (KernelDef::AttrConstraint& constraint : *kdef->mutable_constraint()) {
    if (constraint.name() == name) {
      constraint.mutable_allowed_values()->mutable_list()->add_type(dtype);
    }
  }
}

namespace {
uint32 InitialRandomSeed() {
  // Support plumbing the TF seed through to XLA is being worked on.
  // If a user wants deterministic behavior, their best option
  // is to start with a known checkpoint. This also handles issues when
  // multiple random calls can be invoked in any order by TF executor.
  // Another option is to use stateless random ops. They have much cleaner
  // semantics.
  // If a user really wants to set a deterministic seed for XLA-based
  // devices, this is the place to do it.
  std::random_device rd;
  // Make the starting value odd.
  return rd() | 1;
}
}  // namespace

uint32 GetXLARandomSeed() {
  // We initialize counter with an odd number and increment it by two
  // everytime. This ensures that it will never be zero, even
  // after an overflow. When seeded with zero, some XLA backends
  // can return all zeros instead of random numbers.
  static std::atomic<uint32> counter(InitialRandomSeed());
  uint32 seed = counter.fetch_add(2);
  std::srand(seed);
  return std::rand() | 1;
}

// TODO(b/77601805): add tests for associated function related stuff.
bool HasAssociatedFunction(const NodeDef& node_def,
                           const FunctionLibraryDefinition* fld) {
  if (fld->Contains(node_def.op())) {
    return true;
  }

  if (node_def.op() == FunctionLibraryDefinition::kGradientOp) {
    // Gradient op has "f" attr, which is set to the function we are getting
    // gradient for. We need to functionalize the gradient function.
    return true;
  }

  if (node_def.op() == "XlaHostCompute") {
    // XlaHostCompute has "shape_inference_graph" func attr, but that's not
    // related to graph execution.
    return false;
  }

  for (const auto& iter : node_def.attr()) {
    if (iter.second.has_func()) {
      return true;
    }
  }

  return false;
}

std::vector<AssociatedFunctionInfo> GetAssociatedFunctions(
    const Node& node, const FunctionLibraryDefinition* fld) {
  std::vector<AssociatedFunctionInfo> results;
  const string& op = node.type_string();
  if (fld->Contains(op)) {
    // This is a function call node.
    AttrValueMap attrs(node.attrs().begin(), node.attrs().end());
    results.emplace_back(AssociatedFunctionInfo::FunctionCall(op, attrs));
  } else if (node.type_string() == FunctionLibraryDefinition::kGradientOp) {
    // This is a SymbolicGradient op.
    AttrValueMap attrs(node.attrs().begin(), node.attrs().end());
    results.emplace_back(AssociatedFunctionInfo::SymbolicGradient(op, attrs));
  } else if (node.type_string() == "XlaHostCompute") {
    // XlaHostCompute has "shape_inference_graph" func attr, but that's not
    // related to graph execution.
  } else {
    // Collect all function attrs for the node.
    for (auto& iter : node.attrs()) {
      if (iter.second.has_func()) {
        VLOG(2) << "Found function attr for node " << node.name() << ": "
                << iter.first << " = " << iter.second.func().name();
        results.emplace_back(AssociatedFunctionInfo::FunctionAttr(
            iter.second.func().name(), iter.second.func().attr(), iter.first));
      }
    }
  }
  return results;
}

Status RewriteAssociatedFunction(
    Graph* graph, Node* node, FunctionLibraryDefinition* fld,
    const AssociatedFunctionInfo& associated_function,
    const string& rewritten_function_name) {
  switch (associated_function.type()) {
    case AssociatedFunctionInfo::kFunctionCallNode: {
      // Change this node to call the new function.
      NodeDebugInfo debug_info(*node);
      NodeDefBuilder builder(node->name(), rewritten_function_name, fld,
                             &debug_info);
      for (auto attr : node->attrs()) {
        builder.Attr(attr.first, attr.second);
      }
      for (int i = 0; i < node->num_inputs(); i++) {
        Node* input_node;
        TF_RETURN_IF_ERROR(node->input_node(i, &input_node));
        builder.Input(input_node->name(), i, node->input_type(i));
      }
      builder.Device(node->assigned_device_name().empty()
                         ? node->requested_device()
                         : node->assigned_device_name());
      NodeDef node_def;
      TF_RETURN_IF_ERROR(builder.Finalize(&node_def));
      Status s;
      Node* new_node = graph->AddNode(node_def, &s);
      TF_RETURN_IF_ERROR(s);
      for (auto edge : node->in_edges()) {
        graph->AddEdge(edge->src(), edge->src_output(), new_node,
                       edge->dst_input());
      }
      for (auto edge : node->out_edges()) {
        graph->AddEdge(new_node, edge->src_output(), edge->dst(),
                       edge->dst_input());
      }
      graph->RemoveNode(node);
      break;
    }
    case AssociatedFunctionInfo::kSymbolicGradient: {
      NameAttrList func;
      TF_RETURN_IF_ERROR(GetNodeAttr(
          node->attrs(), FunctionLibraryDefinition::kFuncAttr, &func));
      GradientDef gradient_def;
      gradient_def.set_function_name(func.name());
      gradient_def.set_gradient_func(rewritten_function_name);
      string original_grad_func = fld->FindGradient(func.name());
      if (original_grad_func.empty()) {
        TF_RETURN_IF_ERROR(fld->AddGradientDef(gradient_def));
      } else if (original_grad_func != rewritten_function_name) {
        TF_RETURN_IF_ERROR(fld->ReplaceGradient(gradient_def));
      }
      break;
    }
    case AssociatedFunctionInfo::kFunctionAttr: {
      // Change function attr to rewritten functions.
      NameAttrList func;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(node->attrs(), associated_function.attr_name(), &func));
      node->ClearAttr(associated_function.attr_name());
      func.set_name(rewritten_function_name);
      node->AddAttr(associated_function.attr_name(), func);
      break;
    }
  }

  return Status::OK();
}

Status CachedFunctionHandles::GetOrInstantiate(
    const string& func_name, AttrSlice attrs,
    FunctionLibraryRuntime::Handle* handle) {
  string canonicalized_name = Canonicalize(func_name, attrs);
  auto iter = handles_.find(canonicalized_name);
  if (iter != handles_.end()) {
    *handle = iter->second;
    return Status::OK();
  }

  TF_RETURN_IF_ERROR(flr_->Instantiate(func_name, attrs, handle));
  handles_[canonicalized_name] = *handle;
  return Status::OK();
}

Status CachedFunctionHandles::ReleaseAllHandles() {
  Status result;
  for (auto iter : handles_) {
    result.Update(flr_->ReleaseHandle(iter.second));
  }
  handles_.clear();
  return result;
}

xla::StatusOr<Node*> ReplaceNode(Graph* g, Node* n, const NodeDef& node_def) {
  // Create the replacement node.
  Status s;
  Node* new_node = g->AddNode(node_def, &s);
  if (!s.ok()) {
    return s;
  }

  // Record original node's output edges and remove them first. This is to avoid
  // multiple producers for dst nodes' input.
  std::vector<OutEdgeInfo> out_edge_info;
  std::vector<const Edge*> out_edges;
  for (const Edge* edge : n->out_edges()) {
    out_edges.push_back(edge);
    out_edge_info.push_back(
        {edge->dst(), edge->src_output(), edge->dst_input()});
  }
  for (const Edge* edge : out_edges) {
    g->RemoveEdge(edge);
  }

  // Add original node's input and output edges to the replacement node.
  for (const Edge* in_edge : n->in_edges()) {
    g->AddEdge(in_edge->src(), in_edge->src_output(), new_node,
               in_edge->dst_input());
  }
  for (const OutEdgeInfo& out_edge : out_edge_info) {
    g->AddEdge(new_node, out_edge.src_output, out_edge.dst, out_edge.dst_input);
  }

  // Remove the original node.
  g->RemoveNode(n);

  return new_node;
}

xla::StatusOr<Node*> BuildIdentityNode(
    Graph* graph, const string& node_name, DataType dtype, const Node* input,
    absl::optional<string> requested_device) {
  // Create identity node.
  NodeDef ndef;
  ndef.set_name(node_name);
  ndef.set_op("Identity");
  if (input) {
    ndef.add_input(input->name());
  }
  if (requested_device) {
    ndef.set_device(*requested_device);
  }
  AddNodeAttr("T", dtype, &ndef);
  Status s;
  Node* id_node = graph->AddNode(ndef, &s);
  TF_RETURN_IF_ERROR(s);
  return id_node;
}

Status PropagateConstIntoFunctionalNodes(
    Graph* g, const FunctionLibraryDefinition* lookup_fld,
    FunctionLibraryDefinition* fld) {
  for (Node* n : g->op_nodes()) {
    if (n->IsIfNode()) {
      TF_RETURN_IF_ERROR(PropagateConstIntoIfNode(g, n, lookup_fld, fld));
    } else if (n->IsWhileNode()) {
      TF_RETURN_IF_ERROR(PropagateConstIntoWhileNode(g, n, lookup_fld, fld));
    }
  }
  return Status::OK();
}

Status PruneUnreachableFunctionsFromGraph(const Graph& g,
                                          FunctionLibraryDefinition* fld) {
  GraphDef graph_def;
  g.ToGraphDef(&graph_def);
  FunctionLibraryDefinition reachable_functions =
      fld->ReachableDefinitions(graph_def);
  for (const string& func_name : fld->ListFunctionNames()) {
    if (!reachable_functions.Find(func_name)) {
      TF_RETURN_IF_ERROR(fld->RemoveFunction(func_name));
    }
  }
  return Status::OK();
}

Status RewriteTensorListWithConstElement(Graph* g,
                                         FunctionLibraryDefinition* fld) {
  for (Node* n : g->nodes()) {
    if (n->type_string() != "EmptyTensorList") {
      continue;
    }

    // Find the forward While op.
    std::vector<const Edge*> fwd_while_edges;
    for (const Edge* e : n->out_edges()) {
      if (!e->IsControlEdge() && e->dst()->IsWhileNode()) {
        fwd_while_edges.push_back(e);
      }
    }
    if (fwd_while_edges.size() != 1) {
      // No forward While op found, or multiple forward While ops.
      continue;
    }

    // Find the backward While op.
    Node* fwd_while = fwd_while_edges[0]->dst();
    int fwd_while_dst_input = fwd_while_edges[0]->dst_input();
    std::vector<const Edge*> bwd_while_edges;
    for (const Edge* e : fwd_while->out_edges()) {
      if (e->src_output() == fwd_while_dst_input && e->dst()->IsWhileNode()) {
        bwd_while_edges.push_back(e);
      }
    }
    if (bwd_while_edges.size() != 1) {
      // No backward While op found, or multiple backward While ops.
      continue;
    }

    Node* bwd_while = bwd_while_edges[0]->dst();
    int bwd_while_dst_input = bwd_while_edges[0]->dst_input();

    // Look into forward While body function and check if TensorListPushBack op
    // has a Const input.
    NameAttrList fwd_body_attr;
    TF_CHECK_OK(GetNodeAttr(fwd_while->def(), "body", &fwd_body_attr));
    const FunctionDef* fwd_body = fld->Find(fwd_body_attr.name());
    if (!fwd_body) {
      return errors::InvalidArgument("Cannot find function ",
                                     fwd_body_attr.name(), " for While node ",
                                     fwd_while->DebugString());
    }
    std::unique_ptr<FunctionBody> fwd_fbody;
    TF_CHECK_OK(FunctionDefToBodyHelper(
        *fwd_body, AttrSlice(&fwd_body_attr.attr()), fld, &fwd_fbody));

    // Find the TensorListPushBack node; it's one of fwd_arg's successors.
    Node* fwd_arg = fwd_fbody->arg_nodes[fwd_while_dst_input];
    std::vector<Node*> tl_push_nodes;
    for (const Edge* out_edge : fwd_arg->out_edges()) {
      if (out_edge->dst()->type_string() == "TensorListPushBack") {
        tl_push_nodes.push_back(out_edge->dst());
      }
    }
    if (tl_push_nodes.size() != 1) {
      // No TensorListPushBack found, or multiple TensorListPushBack.
      continue;
    }

    // Get input for the TensorListPushBack node.
    Node* input_node;
    TF_CHECK_OK(tl_push_nodes[0]->input_node(1, &input_node));
    if (input_node->type_string() != "Const") {
      // Input for the TensorList is not Const node.
      continue;
    }

    NodeDef const_input_nodedef = input_node->def();

    // Rewrite backward While body function, replace usages of
    // TensorListPopBack with a Const node.
    NameAttrList bwd_body_attr;
    TF_CHECK_OK(GetNodeAttr(bwd_while->def(), "body", &bwd_body_attr));
    const FunctionDef* bwd_body = fld->Find(bwd_body_attr.name());
    if (!bwd_body) {
      return errors::InvalidArgument("Cannot find function ",
                                     bwd_body_attr.name(), " for While node ",
                                     bwd_while->DebugString());
    }
    std::unique_ptr<FunctionBody> bwd_fbody;
    TF_CHECK_OK(FunctionDefToBodyHelper(
        *bwd_body, AttrSlice(&bwd_body_attr.attr()), fld, &bwd_fbody));

    // Find the TensorListPopBack node; it's one of bwd_arg's successors.
    Node* bwd_arg = bwd_fbody->arg_nodes[bwd_while_dst_input];
    std::vector<Node*> tl_pop_nodes;
    for (const Edge* out_edge : bwd_arg->out_edges()) {
      if (out_edge->dst()->type_string() == "TensorListPopBack") {
        tl_pop_nodes.push_back(out_edge->dst());
      }
    }
    if (tl_pop_nodes.size() != 1) {
      // No TensorListPopBack found, or multiple TensorListPopBack.
      continue;
    }

    // Replace TensorListPopBack usages with Const node.
    std::vector<const Edge*> edges_to_replace;
    for (const Edge* e : tl_pop_nodes[0]->out_edges()) {
      if (e->src_output() == 1) {
        edges_to_replace.push_back(e);
      }
    }
    if (edges_to_replace.empty()) {
      continue;
    }
    Status s;
    const_input_nodedef.set_name(
        bwd_fbody->graph->NewName(const_input_nodedef.name()));
    Node* const_node = bwd_fbody->graph->AddNode(const_input_nodedef, &s);
    TF_RETURN_IF_ERROR(s);
    for (const Edge* e : edges_to_replace) {
      Node* dst = e->dst();
      int dst_input = e->dst_input();
      bwd_fbody->graph->RemoveEdge(e);
      bwd_fbody->graph->AddEdge(const_node, 0, dst, dst_input);
    }

    // Add rewritten backward While body function.
    FunctionDef new_fdef;
    string new_name = fld->UniqueFunctionName(
        absl::StrCat(bwd_body_attr.name(), "_tl_rewrite_"));
    TF_RETURN_IF_ERROR(
        GraphToFunctionDef(*bwd_fbody->graph, new_name, &new_fdef));
    TF_RETURN_IF_ERROR(fld->AddFunctionDef(new_fdef));

    // Change backward While op to use the new body function.
    bwd_body_attr.set_name(new_name);
    bwd_while->ClearAttr("body");
    bwd_while->AddAttr("body", bwd_body_attr);
  }
  return Status::OK();
}

}  // namespace tensorflow
