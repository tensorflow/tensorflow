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

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/sharding_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

namespace {

Status ValidateTensorId(const tf2xla::TensorId& id) {
  if (id.node_name().empty()) {
    return errors::InvalidArgument("TensorId node_name must be non-empty");
  }
  if (id.output_index() < 0) {
    return errors::InvalidArgument("TensorId output_index must be positive");
  }
  return OkStatus();
}

Status CheckNameDuplicates(const string& kind, const string& name,
                           std::set<string>* names) {
  if (!name.empty()) {
    if (!names->insert(name).second) {
      return errors::InvalidArgument("duplicate ", kind, " name: ", name);
    }
  }
  return OkStatus();
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
  return OkStatus();
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
  return OkStatus();
}

// Replaces the single edge feeding into {dst,dst_input} with a new
// src/src_output specified by {with,with_output}.
StatusOr<Node*> ReplaceEdge(Graph* g, Node* dst, int dst_input, Node* with,
                            int with_output) {
  NodeDef replace_def = dst->def();
  *replace_def.mutable_input(dst_input) = with->name();
  TF_ASSIGN_OR_RETURN(Node * replace_node, ReplaceNode(g, dst, replace_def));
  const Edge* usage_edge;
  TF_RETURN_IF_ERROR(replace_node->input_edge(dst_input, &usage_edge));
  g->RemoveEdge(usage_edge);
  g->AddEdge(with, with_output, replace_node, dst_input);
  return replace_node;
}

// Replaces usages of the given `src_output` index of the given `src` node with
// the given `replacement` node (assumes the :0 output of `replacement`).
Status ReplaceSrcOutputUsageWithNode(Graph* g, Node* src, int src_output,
                                     Node* replacement) {
  VLOG(1) << "Replace usages of output " << src_output << " of node "
          << (VLOG_IS_ON(3) ? src->DebugString() : src->name()) << " with "
          << (VLOG_IS_ON(3) ? replacement->DebugString() : replacement->name());
  // Collect all usages of the specified src output (src->out_edges() iterator
  // will not be stable under modifications).
  struct OutEdgeInfo {
    int dst_node_id, dst_input;
  };
  std::vector<OutEdgeInfo> usages;
  for (const Edge* e : src->out_edges()) {
    if (e->IsControlEdge() || e->src_output() != src_output) {
      continue;
    }
    usages.push_back({e->dst()->id(), e->dst_input()});
  }

  // Now, replace each usage.
  for (int i = 0, end = usages.size(); i < end; i++) {
    // Make a copy of `usage_node`, and change its input to const node.
    Node* usage_node = g->FindNodeId(usages[i].dst_node_id);
    VLOG(2) << "  Replace usage by " << usage_node->DebugString();
    // Note: Replacement output index is presumed to be 0.
    TF_ASSIGN_OR_RETURN(
        Node * replace_node,
        ReplaceEdge(g, usage_node, usages[i].dst_input, replacement, 0));
    // Later entries in `usages` might have `usage_node` as dst node, but
    // `usage_node` is removed. Replace such entries with `replace_node`.
    for (int j = i + 1, end = usages.size(); j < end; j++) {
      if (usages[j].dst_node_id == usages[i].dst_node_id) {
        usages[j].dst_node_id = replace_node->id();
      }
    }
  }
  return OkStatus();
}

// For graph `g`, replaces _Arg nodes whose "index" attribute is in
// `const_input_index_to_node` with Const nodes.
Status ReplaceArgUsageWithConstNode(
    Graph* g,
    const absl::flat_hash_map<int, const Node*>& const_input_index_to_node) {
  // Collect all _Arg nodes.
  absl::flat_hash_map<int, Node*> arg_nodes;
  for (Node* n : g->op_nodes()) {
    if (n->IsArg()) {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      arg_nodes[index] = n;
    }
  }

  for (const auto& iter : const_input_index_to_node) {
    int arg_index = iter.first;
    VLOG(2) << "Replace usages of _Arg " << arg_index;
    NodeDef const_def = iter.second->def();
    const_def.set_name(g->NewName(const_def.name()));
    TF_ASSIGN_OR_RETURN(Node * const_node, g->AddNode(const_def));
    Node* arg_node = arg_nodes[arg_index];
    TF_RETURN_IF_ERROR(
        ReplaceSrcOutputUsageWithNode(g, arg_node, 0, const_node));
  }
  return OkStatus();
}

// Replaces the single input to _Retval nodes with an index in the keys of
// const_input_index_to_node with the single output of the corresponding _Arg
// node.
Status ReplaceRetvalInputWithArg(
    Graph* g,
    const absl::flat_hash_map<int, const Node*>& const_input_index_to_node) {
  absl::flat_hash_map<int, Node*> arg_nodes;
  absl::flat_hash_map<int, Node*> ret_nodes;
  for (Node* n : g->op_nodes()) {
    if (n->IsRetval() || n->IsArg()) {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      if (n->IsRetval()) {
        ret_nodes[index] = n;
      } else {
        arg_nodes[index] = n;
      }
    }
  }

  for (const auto& iter : const_input_index_to_node) {
    int arg_index = iter.first;
    VLOG(2) << "Bind _Retval " << arg_index << " to _Arg " << arg_index;
    TF_RETURN_IF_ERROR(
        ReplaceEdge(g, ret_nodes[arg_index], 0, arg_nodes[arg_index], 0)
            .status());
  }
  return OkStatus();
}

// For a node's function attr (e.g. then/else branch for "If" nodes), rewrites
// the function to replace _Arg nodes in `const_input_index_to_node` with Const
// inputs.
Status PropagateConstIntoFuncAttr(
    Node* n, const string& attr_name,
    const absl::flat_hash_map<int, const Node*>& const_input_index_to_node,
    const FunctionLibraryDefinition* lookup_fld, FunctionLibraryDefinition* fld,
    bool passthrough_arg_to_retval = false) {
  VLOG(1) << "Propagate const into " << attr_name << " of node " << n->name();
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
  if (passthrough_arg_to_retval) {
    TF_RETURN_IF_ERROR(
        ReplaceRetvalInputWithArg(func_graph, const_input_index_to_node));
  }

  // Save rewritten function.
  FunctionDef replace_fdef;
  string new_func_name =
      fld->UniqueFunctionName(absl::StrCat(func_attr.name(), "_const_"));
  const StackTracesMap* stack_traces =
      lookup_fld->GetStackTraces(func_attr.name());
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(*func_graph, new_func_name, &replace_fdef));
  if (stack_traces != nullptr) {
    TF_RETURN_IF_ERROR(fld->AddFunctionDef(replace_fdef, *stack_traces));
  } else {
    TF_RETURN_IF_ERROR(fld->AddFunctionDef(replace_fdef, {}));
  }

  VLOG(1) << "replace func " << func_attr.name() << " with " << new_func_name;
  // Change the node to use rewritten function.
  func_attr.set_name(new_func_name);
  n->ClearAttr(attr_name);
  n->AddAttr(attr_name, func_attr);

  // Copy associated functions.
  TF_RETURN_IF_ERROR(CopyAssociatedFunctions(func_graph, lookup_fld, fld));

  return OkStatus();
}

// For an "If" node in graph `g`, if it has Const node inputs, rewrite its
// then/else branch function to replace _Arg nodes with those Const inputs.
Status PropagateConstIntoIfNode(Graph* g, Node* if_node,
                                const FunctionLibraryDefinition* lookup_fld,
                                FunctionLibraryDefinition* fld) {
  // Notice that first input for If node is predicate; other inputs are function
  // inputs.
  absl::flat_hash_map<int, const Node*> const_input_index_to_node;
  for (int i = 1; i < if_node->num_inputs(); i++) {
    const Node* input_node;
    TF_RETURN_IF_ERROR(if_node->input_node(i, &input_node));
    if (input_node->type_string() == "Const") {
      const_input_index_to_node[i - 1] = input_node;
    }
  }
  if (const_input_index_to_node.empty()) {
    return OkStatus();
  }

  // Rewrite "then_branch" and "else_branch" function, replace usage of those
  // _Arg nodes with corresponding const node.
  for (const auto& attr_name :
       std::vector<string>{"then_branch", "else_branch"}) {
    TF_RETURN_IF_ERROR(PropagateConstIntoFuncAttr(
        if_node, attr_name, const_input_index_to_node, lookup_fld, fld));
  }

  return OkStatus();
}

using GraphCache = absl::flat_hash_map<string, std::unique_ptr<FunctionBody>>;

StatusOr<FunctionBody*> FindOrInsert(
    GraphCache* cache, const NameAttrList& body_attr,
    const FunctionLibraryDefinition* lookup_fld,
    const FunctionLibraryDefinition* fallback_fld) {
  const string name = body_attr.name();
  std::unique_ptr<FunctionBody>& value = (*cache)[name];
  if (!value) {
    const FunctionDef* body_func = lookup_fld->Find(name);
    if (!body_func && fallback_fld != nullptr) {
      body_func = fallback_fld->Find(name);
    }
    if (!body_func) {
      return errors::Internal("Traverse: Cannot find body function ", name);
    }
    std::unique_ptr<FunctionBody> fbody;
    Status s = FunctionDefToBodyHelper(*body_func, AttrSlice(&body_attr.attr()),
                                       lookup_fld, &fbody);
    if (!s.ok() && fallback_fld != nullptr) {
      TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
          *body_func, AttrSlice(&body_attr.attr()), fallback_fld, &fbody));
    }
    value = std::move(fbody);
  }
  return value.get();
}
// Determines whether a loop body is invariant for the given argument index.
StatusOr<bool> IsLoopInvariant(const FunctionBody* loop_body, int index,
                               const FunctionLibraryDefinition* lookup_fld,
                               const FunctionLibraryDefinition* fallback_fld,
                               GraphCache* cache);

// Traces backward through non-modifying ops such as Identity and loop-invariant
// While, to find a preceding source edge.
StatusOr<const Edge*> TraverseUnmodifiedPathBackward(
    const Edge* src, const FunctionLibraryDefinition* lookup_fld,
    const FunctionLibraryDefinition* fallback_fld, GraphCache* cache) {
  const Edge* e = src;
  VLOG(2) << "Traverse: Begin at " << e->DebugString();
  // TODO(b/184727356): Also traverse If/Case nodes.
  // Begin walking back from the output node.
  while (IsConstTraversableOpType(e->src())) {
    VLOG(3) << e->DebugString();

    if (e->src()->IsWhileNode()) {
      NameAttrList body_attr;
      TF_RETURN_IF_ERROR(GetNodeAttr(e->src()->def(), "body", &body_attr));
      TF_ASSIGN_OR_RETURN(
          FunctionBody * fbody,
          FindOrInsert(cache, body_attr, lookup_fld, fallback_fld));
      TF_ASSIGN_OR_RETURN(bool is_loop_invariant,
                          IsLoopInvariant(fbody, e->src_output(), lookup_fld,
                                          fallback_fld, cache));
      if (!is_loop_invariant) {
        VLOG(2) << "Non-loop-invariant: index " << e->src_output() << " of "
                << body_attr.name();
        break;
      }
    }  // if While|StatelessWhile
    // Proceed backward to the src's input corresponding with the output index.
    TF_RETURN_IF_ERROR(e->src()->input_edge(e->src_output(), &e));
  }
  VLOG(2) << "Traverse: Finish at " << e->DebugString();

  return e;
}

// Determines whether a loop body is invariant for the given argument index.
StatusOr<bool> IsLoopInvariant(const FunctionBody* loop_body, int index,
                               const FunctionLibraryDefinition* lookup_fld,
                               const FunctionLibraryDefinition* fallback_fld,
                               GraphCache* cache) {
  const Edge* e;
  TF_RETURN_IF_ERROR(loop_body->ret_nodes[index]->input_edge(0, &e));
  TF_ASSIGN_OR_RETURN(
      const Edge* reachable,
      TraverseUnmodifiedPathBackward(e, lookup_fld, fallback_fld, cache));
  if (reachable->src()->id() == loop_body->arg_nodes[index]->id()) {
    VLOG(2) << "Index " << index << " is loop invariant.";
    return true;
  }
  VLOG(2) << "Index " << index << " not loop invariant: "
          << "walk backward from " << e->src()->DebugString() << " to "
          << reachable->src()->DebugString() << " did not reach "
          << loop_body->arg_nodes[index]->DebugString();
  return false;
}

// For a "While" node in graph `g`, if it has Const node inputs, rewrite its
// cond/body function to replace _Arg nodes with those Const inputs. Then,
// propagate these Const to consumers of the relevant outputs of the while loop.
Status PropagateConstIntoAndAroundWhileNode(
    Graph* g, Node* while_node, const FunctionLibraryDefinition* lookup_fld,
    FunctionLibraryDefinition* fld) {
  VLOG(1) << "Propagate const into " << while_node->name();

  // For "While" node, we should only replace _Arg nodes which are loop
  // invariants. For such _Arg nodes, the return value's input will come
  // directly from the corresponding arg.
  absl::flat_hash_map<int, const Node*> const_input_index_to_node;
  absl::flat_hash_map<int, Node*> const_input_index_to_mutable_node;
  NameAttrList body_attr;
  TF_RETURN_IF_ERROR(GetNodeAttr(while_node->def(), "body", &body_attr));
  const string fn_name = body_attr.name();
  const FunctionDef* body_func = lookup_fld->Find(fn_name);
  if (!body_func) {
    return errors::Internal("Propagate: Cannot find body function ", fn_name,
                            " for While node ", while_node->name());
  }
  std::unique_ptr<FunctionBody> fbody;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
      *body_func, AttrSlice(&body_attr.attr()), lookup_fld, &fbody));
  GraphCache cache;
  for (int i = 0; i < while_node->num_inputs(); i++) {
    // Check if i-th retval's input comes from i-th arg directly.
    // For resource variable input of While nodes, TF2XLA convention is to place
    // them at the end of all inputs (after all data inputs), and *not* return
    // them. So number of While node inputs might be larger than number of its
    // outputs.
    if (i >= body_func->signature().output_arg_size()) {
      break;
    }

    const Edge* input_edge;
    TF_RETURN_IF_ERROR(while_node->input_edge(i, &input_edge));
    TF_ASSIGN_OR_RETURN(input_edge, TraverseUnmodifiedPathBackward(
                                        input_edge, lookup_fld, fld, &cache));
    if (!input_edge->src()->IsConstant()) {
      VLOG(2) << "Input " << i << " is not Const; is "
              << input_edge->src()->type_string();
      continue;
    }

    TF_ASSIGN_OR_RETURN(
        bool is_loop_invariant,
        IsLoopInvariant(fbody.get(), i, lookup_fld, fld, &cache));
    if (!is_loop_invariant) {
      VLOG(2) << "While state not loop-invariant; not propagating Const " << i;
      continue;
    }
    VLOG(2) << "While state is loop-invariant; propagating Const " << i;

    const_input_index_to_mutable_node[i] = input_edge->src();
    const_input_index_to_node[i] = input_edge->src();
  }
  if (const_input_index_to_node.empty()) {
    return OkStatus();
  }

  // Rewrite "cond" and "body" function, replace usage of those _Arg nodes with
  // corresponding const node.
  for (const auto& attr_name : std::vector<string>{"cond", "body"}) {
    TF_RETURN_IF_ERROR(PropagateConstIntoFuncAttr(
        while_node, attr_name, const_input_index_to_node, lookup_fld, fld,
        /*passthrough_arg_to_retval=*/attr_name == "body"));
  }

  // Rewrite usages of the output edges corresponding to loop-invariant const
  // inputs to refer instead to the Const node.
  for (const auto& it : const_input_index_to_mutable_node) {
    TF_RETURN_IF_ERROR(
        ReplaceSrcOutputUsageWithNode(g, while_node, it.first, it.second));
  }
  return OkStatus();
}

}  // namespace

StatusOr<bool> IsLoopInvariant(const FunctionBody* loop_body, int index,
                               const FunctionLibraryDefinition* lookup_fld) {
  GraphCache cache;
  return IsLoopInvariant(loop_body, index, lookup_fld,
                         /*fallback_fld=*/nullptr, &cache);
}

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
  return OkStatus();
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
      TF_ASSIGN_OR_RETURN(Node * feed_node, g.AddNode(gd.node(0)));

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
    d->set_op("Placeholder");
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

  return OkStatus();
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
  return OkStatus();
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
        std::optional<xla::OpSharding> sharding,
        ParseShardingFromDevice(
            *possible_match,
            /*num_cores_per_replica=*/std::numeric_limits<int32>::max(),
            /*add_metadata=*/false));
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
  return OkStatus();
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
      for (const auto& attr : node->attrs()) {
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
      TF_ASSIGN_OR_RETURN(Node * new_node, graph->AddNode(node_def));
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
      // Save the original function name in case TFRT fallbacks to use
      // TPUPartitionedCall op in the runtime.
      if (node->type_string() == "TPUPartitionedCall") {
        node->AddAttr("_orig_f", func.name());
      }
      node->ClearAttr(associated_function.attr_name());
      func.set_name(rewritten_function_name);
      node->AddAttr(associated_function.attr_name(), func);
      break;
    }
  }

  return OkStatus();
}

Status CachedFunctionHandles::GetOrInstantiate(
    const string& func_name, AttrSlice attrs,
    FunctionLibraryRuntime::Handle* handle) {
  string canonicalized_name = Canonicalize(func_name, attrs);
  auto iter = handles_.find(canonicalized_name);
  if (iter != handles_.end()) {
    *handle = iter->second;
    return OkStatus();
  }

  TF_RETURN_IF_ERROR(flr_->Instantiate(func_name, attrs, handle));
  handles_[canonicalized_name] = *handle;
  return OkStatus();
}

Status CachedFunctionHandles::ReleaseAllHandles() {
  Status result;
  for (const auto& iter : handles_) {
    result.Update(flr_->ReleaseHandle(iter.second));
  }
  handles_.clear();
  return result;
}

StatusOr<Node*> ReplaceNode(Graph* g, Node* n, const NodeDef& node_def) {
  // Create the replacement node.
  TF_ASSIGN_OR_RETURN(Node * new_node, g->AddNode(node_def));

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

StatusOr<Node*> BuildIdentityNode(Graph* graph, const string& node_name,
                                  DataType dtype, const Node* input,
                                  std::optional<string> requested_device) {
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
  TF_ASSIGN_OR_RETURN(Node * id_node, graph->AddNode(ndef));
  return id_node;
}

Status PropagateConstIntoFunctionalNodes(
    Graph* g, const FunctionLibraryDefinition* lookup_fld,
    FunctionLibraryDefinition* fld) {
  absl::flat_hash_set<int> done_node_ids;

  // Because we may propagate Const around a while node as well as into it,
  // we restart the op_nodes() iterator after each pass and keep track of which
  // nodes we've already dealt with.
  bool should_continue = true;
  while (should_continue) {
    should_continue = false;
    for (Node* n : g->op_nodes()) {
      if (!done_node_ids.contains(n->id())) {
        if (n->IsIfNode()) {
          VLOG(1) << "PropagateConstIntoIfNode: " << n->name();
          TF_RETURN_IF_ERROR(PropagateConstIntoIfNode(g, n, lookup_fld, fld));
          done_node_ids.emplace(n->id());
          VLOG(1) << "Done PropagateConstIntoIfNode: " << n->name();
        } else if (n->IsWhileNode()) {
          VLOG(1) << "PropagateConstIntoWhileNode: " << n->name();
          TF_RETURN_IF_ERROR(
              PropagateConstIntoAndAroundWhileNode(g, n, lookup_fld, fld));
          done_node_ids.emplace(n->id());
          should_continue = true;
          VLOG(1) << "Done PropagateConstIntoWhileNode: " << n->name();
          break;
        }
      }
    }
  }
  return OkStatus();
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
  return OkStatus();
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
    const_input_nodedef.set_name(
        bwd_fbody->graph->NewName(const_input_nodedef.name()));
    TF_ASSIGN_OR_RETURN(Node * const_node,
                        bwd_fbody->graph->AddNode(const_input_nodedef));
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
  return OkStatus();
}

}  // namespace tensorflow
