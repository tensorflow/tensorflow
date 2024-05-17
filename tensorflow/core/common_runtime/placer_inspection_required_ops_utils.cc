/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/placer_inspection_required_ops_utils.h"

#include <unordered_map>
#include <unordered_set>

#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/refcount.h"

namespace tensorflow {
namespace {

bool IsFunctionCall(const Node& node) {
  // TODO(iga): Handle non-PCO functions when we add multi-device support
  // to regular function calls. Also, the GetFunctionDefAndAttrs assumes that
  // the function name is stored in the `f` attribute of the node. That code
  // will need to change as well.
  const string& op_type = node.op_def().name();
  return op_type == "PartitionedCall" || op_type == "StatefulPartitionedCall";
}

// Utility to set node's value in `cache` and `is_deep` to `value`.
Status Set(const Node& node, bool value, bool* is_deep,
           std::vector<absl::optional<bool>>* cache) {
  *is_deep = value;
  (*cache)[node.id()] = value;
  return absl::OkStatus();
}

}  // namespace

PlacerInspectionRequiredOpChecker::PlacerInspectionRequiredOpChecker(
    const Graph* graph, const FunctionLibraryDefinition* flib_def)
    : graph_(*graph), flib_def_(*flib_def) {
  cache_.resize(graph_.num_node_ids());
}

Status PlacerInspectionRequiredOpChecker::IsPlacerInspectionRequired(
    const Node& node, bool* is_deep) {
  if (cache_[node.id()].has_value()) {
    *is_deep = cache_[node.id()].value();
    return absl::OkStatus();
  }

  if (!IsFunctionCall(node)) {
    return Set(node, false, is_deep, &cache_);
  }
  core::RefCountPtr<FunctionRecord> fdef;
  NameAttrList func;
  TF_RETURN_IF_ERROR(GetFunctionDefAndAttrs(flib_def_, node, &fdef, &func));
  DataTypeVector types;
  TF_RETURN_IF_ERROR(OutputTypesForNode(AttrSlice(&func.attr()),
                                        fdef->fdef().signature(), &types));
  for (DataType type : types) {
    if (type == DT_RESOURCE) {
      return Set(node, true, is_deep, &cache_);
    }
  }
  return Set(node, false, is_deep, &cache_);
}

Status GetFunctionDefAndAttrs(const FunctionLibraryDefinition& flib_def,
                              const Node& node,
                              core::RefCountPtr<FunctionRecord>* fdef,
                              NameAttrList* func) {
  TF_RETURN_IF_ERROR(GetNodeAttr(node.def(), "f", func));
  const string& function_name = func->name();
  *fdef = flib_def.FindRecord(function_name);
  if (*fdef == nullptr) {
    return errors::InvalidArgument(
        "Failed to find function \"", function_name,
        "\" in function library: ", flib_def.ToProto().DebugString());
  }
  return absl::OkStatus();
}

FunctionStack::FunctionStack(const string& function_name)
    : current_function_name_(function_name) {}

FunctionStack FunctionStack::Push(const Node* node_in_current_function,
                                  const string& new_current_function) const {
  FunctionStack new_stack(new_current_function);
  new_stack.frames_ = frames_;
  new_stack.frames_.emplace_back(current_function_name_,
                                 node_in_current_function);
  return new_stack;
}

bool FunctionStack::HasFunction(const string& function_name) const {
  if (current_function_name_ == function_name) {
    return true;
  }
  for (const Frame& frame : frames_) {
    if (frame.function_name == function_name) {
      return true;
    }
  }
  return false;
}

string FunctionStack::FormatForError() const {
  std::vector<string> msgs;
  for (int i = 0; i < frames_.size(); ++i) {
    if (frames_[i].function_name.empty()) {
      // Empty function body should only happen at the top level, i.e. i = 0.
      // All internal frames should have valid function names.
      msgs.push_back(absl::StrCat("Graph contains node ",
                                  FormatNodeForError(*frames_[i].node)));

    } else {
      msgs.push_back(absl::StrCat(
          "Function ", errors::FormatFunctionForError(frames_[i].function_name),
          " contains node ", FormatNodeForError(*frames_[i].node)));
    }
    const string& fname = (i + 1 < frames_.size())
                              ? frames_[i + 1].function_name
                              : current_function_name_;
    msgs.push_back(absl::StrCat("Node ", FormatNodeForError(*frames_[i].node),
                                " calls function ",
                                errors::FormatFunctionForError(fname)));
  }
  return absl::StrJoin(msgs, "\n  ");
}

namespace {

using OutputEdgeMap = std::vector<std::vector<const Edge*>>;

constexpr char kIdentityOp[] = "Identity";

string Uniquify(const string& candidate_name,
                std::unordered_set<string>* node_names) {
  if (node_names->find(candidate_name) == node_names->end()) {
    node_names->insert(candidate_name);
    return candidate_name;
  }

  for (int counter = 0;; ++counter) {
    string candidate = absl::StrCat(candidate_name, "_", counter);
    if (node_names->find(candidate) == node_names->end()) {
      node_names->insert(candidate);
      return candidate;
    }
  }
}

Status AddInputIdentity(Node* node, int input_idx, Graph* graph,
                        std::unordered_set<string>* node_names) {
  const Edge* edge;
  TF_RETURN_IF_ERROR(node->input_edge(input_idx, &edge));

  string identity_name = Uniquify(
      absl::StrCat(edge->src()->name(), "_", node->name()), node_names);

  NodeDefBuilder builder(identity_name, kIdentityOp);
  builder.Attr("T", node->input_type(input_idx));
  NodeDefBuilder::NodeOut input(edge->src()->name(), edge->src_output(),
                                node->input_type(input_idx));
  builder.Input(input);
  NodeDef identity_def;
  TF_RETURN_IF_ERROR(builder.Finalize(&identity_def));
  MergeDebugInfo(NodeDebugInfo(*node), &identity_def);

  VLOG(6) << "Adding identity into " << edge->src()->name() << ":"
          << edge->src_output() << " -> " << edge->dst()->name() << ":"
          << input_idx << " \n"
          << identity_def.DebugString();

  TF_ASSIGN_OR_RETURN(Node * identity_node, graph->AddNode(identity_def));
  graph->AddEdge(edge->src(), edge->src_output(), identity_node, 0);

  // Replace node's `input_idx` input with the new identity's 0'th output
  TF_RETURN_IF_ERROR(graph->UpdateEdge(identity_node, 0, node, input_idx));

  VLOG(6) << "Successfully inserted identity. Modified node: \n"
          << node->DebugString();
  return absl::OkStatus();
}

struct EdgePtrCompare {
  bool operator()(const Edge* lhs, const Edge* rhs) const {
    return lhs->id() < rhs->id();
  }
};

Status AddOutputIdentities(Node* node, Graph* graph,
                           std::unordered_set<string>* node_names) {
  auto add_identity = [&](int src_output, const string& identity_name,
                          Node** identity_node) {
    NodeDefBuilder builder(identity_name, kIdentityOp);
    builder.Attr("T", node->output_type(src_output));
    NodeDefBuilder::NodeOut input(node->name(), src_output,
                                  node->output_type(src_output));
    builder.Input(input);
    NodeDef identity_def;
    TF_RETURN_IF_ERROR(builder.Finalize(&identity_def));
    MergeDebugInfo(NodeDebugInfo(*node), &identity_def);

    TF_ASSIGN_OR_RETURN(*identity_node, graph->AddNode(identity_def));
    graph->AddEdge(node, src_output, *identity_node, 0);
    return absl::OkStatus();
  };

  // output_used[i] == true iff `node`'s i'th output is used
  // in this graph
  std::vector<bool> output_used(node->num_outputs(), false);
  // Copy the set of edges since EdgeSet does not allow modifications
  // to graph edges during iteration.
  const EdgeSet& out_edges = node->out_edges();
  std::vector<const Edge*> edge_vector(out_edges.begin(), out_edges.end());
  std::sort(edge_vector.begin(), edge_vector.end(), EdgePtrCompare());
  for (const Edge* edge : edge_vector) {
    if (edge->IsControlEdge()) {
      continue;
    }
    output_used[edge->src_output()] = true;

    Node* dst = edge->dst();
    int dst_input = edge->dst_input();
    int src_output = edge->src_output();
    string identity_name =
        Uniquify(absl::StrCat(node->name(), "_", dst->name()), node_names);
    Node* identity_node;
    TF_RETURN_IF_ERROR(add_identity(src_output, identity_name, &identity_node));
    VLOG(6) << "Adding identity into " << node->name() << ":" << src_output
            << " -> " << dst->name() << ":" << dst_input << " \n"
            << identity_node->DebugString();

    // Make original dst node consume the new identity's output instead of
    // `node`'s output.
    TF_RETURN_IF_ERROR(graph->UpdateEdge(identity_node, 0, dst, dst_input));
  }

  for (int output_idx = 0; output_idx < node->num_outputs(); ++output_idx) {
    if (output_used[output_idx]) {
      continue;
    }
    // The output is unused in the graph. Just add an identity
    // consuming it.
    string identity_name = Uniquify(node->name(), node_names);
    Node* identity_node;
    TF_RETURN_IF_ERROR(add_identity(output_idx, identity_name, &identity_node));
    VLOG(6) << "Added identity into " << node->name() << ":" << output_idx
            << " -> <no consumer>: \n"
            << identity_node->DebugString();
  }
  return absl::OkStatus();
}

Status IsolateNode(Node* node, Graph* graph) {
  // We use `node_names` to make sure we pick unique names.
  // We don't use graph->NewName() because it produces verbose names and
  // does not actually ensure that they are unique (it assumes all names
  // are generated using it, which is not true today).
  std::unordered_set<string> node_names(graph->num_nodes());
  for (Node* n : graph->nodes()) {
    node_names.insert(n->name());
  }

  for (int i = 0; i < node->num_inputs(); ++i) {
    TF_RETURN_IF_ERROR(AddInputIdentity(node, i, graph, &node_names));
  }
  TF_RETURN_IF_ERROR(AddOutputIdentities(node, graph, &node_names));
  return absl::OkStatus();
}

}  // namespace

Status IsolatePlacerInspectionRequiredOps(
    const FunctionLibraryDefinition& flib_def, Graph* graph) {
  PlacerInspectionRequiredOpChecker checker(graph, &flib_def);
  // It is OK to add nodes to the graph during iteration.
  // New nodes will get ids above current ids. The loop
  // will loop over current nodes only because the op_nodes()
  // iterator uses node ids to iterate.
  // Because the new nodes will be higher ids, the caching in
  // the checker will also work fine as new nodes are added.
  for (Node* node : graph->op_nodes()) {
    bool should_be_isolated = false;
    TF_RETURN_IF_ERROR(
        checker.IsPlacerInspectionRequired(*node, &should_be_isolated));
    if (!should_be_isolated) {
      continue;
    }
    TF_RETURN_IF_ERROR(IsolateNode(node, graph));
  }

  return absl::OkStatus();
}

}  // namespace tensorflow
