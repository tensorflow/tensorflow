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

#include "tensorflow/core/framework/graph_to_functiondef.h"

#include <unordered_map>
#include <unordered_set>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace {

// TODO(phawkins) add a canonical copy of these operator names and refactor
// everything to use it.
const char* const kArgOp = "_Arg";
const char* const kRetValOp = "_Retval";

// Class that maintains a one-to-one original node name -> new name mapping.
// We have to normalize the names used as input and output arguments to
// match regexp "[a-z][a-z0-9_]*". Once we rename them, we risk creating
// a name collision with the other node names, so if necessary we add
// a suffix to make names unique.  So if we have an input named "A" and a
// node in the function body named "a", they will be renamed to "a" and "a_0".
class NodeNameMapping {
 public:
  NodeNameMapping() = default;

  // Normalize the input/output name and then make it unique.
  string Normalize(const string& name);

  // Make the node name unique.
  string Uniquify(const string& name);

  // Look up how a node name was previously normalized/uniquified.
  // Returns empty if name was never seen.
  string Renormalize(const string& name) const;

 private:
  string NormalizeHelper(string name) const;
  string UniquifyHelper(string name);

  std::unordered_set<string> used_names_;
  std::unordered_map<string, string> name_mapping_;
};

string NodeNameMapping::NormalizeHelper(string name) const {
  // Convert letters to lowercase and non-alphanumeric characters to '_'.
  if (name.empty()) name = "unknown";
  const int n = name.size();
  for (int i = 0; i < n; i++) {
    char c = name[i];
    if (isalnum(c)) {
      if (isupper(c)) {
        name[i] = tolower(c);
      }
    } else {
      name[i] = '_';
    }
  }
  return name;
}

string NodeNameMapping::UniquifyHelper(string name) {
  // If the name hasn't been used yet, use it as-is.
  if (used_names_.insert(name).second) return name;
  // Add a suffix to name to make it unique.
  for (int i = 0;; ++i) {
    const string candidate = strings::StrCat(name, "_", i);
    if (used_names_.insert(candidate).second) return candidate;
  }
}

string NodeNameMapping::Normalize(const string& name) {
  const string normalized = UniquifyHelper(NormalizeHelper(name));
  name_mapping_[name] = normalized;
  return normalized;
}

string NodeNameMapping::Uniquify(const string& name) {
  const string uniqued = UniquifyHelper(name);
  name_mapping_[name] = uniqued;
  return uniqued;
}

string NodeNameMapping::Renormalize(const string& name) const {
  const auto iter = name_mapping_.find(name);
  if (iter == name_mapping_.end()) return string();
  return iter->second;
}

}  // anonymous namespace

// Graph to FunctionDef conversion. This code is closely modeled on the Python
// code in tensorflow/python/framework/function.py.

Status GraphToFunctionDef(const Graph& graph, const string& name,
                          FunctionDef* fdef) {
  fdef->mutable_signature()->set_name(name);

  std::unordered_map<string, string> tensor_renaming;
  std::unordered_map<string, string> return_values;
  NodeNameMapping node_names;

  for (Node const* node : graph.op_nodes()) {
    if (node->type_string() == kArgOp) {
      int index;
      DataType type;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &type));
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &index));
      while (fdef->signature().input_arg_size() <= index) {
        fdef->mutable_signature()->add_input_arg();
      }
      OpDef::ArgDef* argdef =
          fdef->mutable_signature()->mutable_input_arg(index);
      argdef->set_type(type);
      const string normalized = node_names.Normalize(node->name());
      argdef->set_name(normalized);
      tensor_renaming[strings::StrCat(node->name(), ":0")] = normalized;
      continue;
    }

    if (node->type_string() == kRetValOp) {
      int index;
      DataType type;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &type));
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &index));
      while (fdef->signature().output_arg_size() <= index) {
        fdef->mutable_signature()->add_output_arg();
      }
      OpDef::ArgDef* argdef =
          fdef->mutable_signature()->mutable_output_arg(index);
      argdef->set_type(type);
      const string normalized = node_names.Normalize(node->name());
      argdef->set_name(normalized);
      Edge const* edge;
      TF_RETURN_IF_ERROR(node->input_edge(0, &edge));
      return_values[normalized] =
          strings::StrCat(edge->src()->name(), ":", edge->src_output());
      continue;
    }

    NodeDef* node_def = fdef->add_node_def();
    *node_def = node->def();
    if (!node->assigned_device_name().empty()) {
      node_def->set_device(node->assigned_device_name());
    }
    node_def->set_name(node_names.Uniquify(node->name()));
    MergeDebugInfo(NodeDebugInfo(node->def()), node_def);

    // Reset input names based on graph rather than the NodeDef.
    node_def->clear_input();

    // Edges, indexed by dst_input.
    std::vector<const Edge*> in_edges;
    std::vector<const Edge*> control_edges;
    for (Edge const* edge : node->in_edges()) {
      if (edge->src()->IsSource()) continue;

      if (edge->IsControlEdge()) {
        control_edges.push_back(edge);
      } else {
        if (in_edges.size() <= edge->dst_input()) {
          in_edges.resize(edge->dst_input() + 1);
        }
        in_edges[edge->dst_input()] = edge;
      }
    }

    // Add regular inputs
    for (std::vector<const Edge*>::size_type i = 0; i < in_edges.size(); ++i) {
      const Edge* edge = in_edges[i];
      if (edge == nullptr) {
        return errors::InvalidArgument(
            "Nonconsecutive input edges; missing "
            "input edge ",
            i, " for node ", node->name());
      }
      node_def->add_input(
          strings::StrCat(edge->src()->name(), ":", edge->src_output()));
    }

    // Add control inputs
    for (const Edge* edge : control_edges) {
      node_def->add_input(strings::StrCat("^", edge->src()->name()));
    }

    // Populate tensor_renaming.
    NameRangeMap output_ranges;
    TF_RETURN_IF_ERROR(
        NameRangesForNode(*node, node->op_def(), nullptr, &output_ranges));
    for (const auto& output : output_ranges) {
      for (int i = output.second.first; i < output.second.second; ++i) {
        const string tensor_name = strings::StrCat(
            node_def->name(), ":", output.first, ":", i - output.second.first);
        tensor_renaming[strings::StrCat(node->name(), ":", i)] = tensor_name;
      }
    }
  }

  // Detect missing function inputs.
  for (int i = 0; i < fdef->signature().input_arg_size(); ++i) {
    const string& input_name = fdef->signature().input_arg(i).name();
    if (input_name.empty()) {
      return errors::InvalidArgument("Missing input ", i, " to function ",
                                     name);
    }
  }

  // Remap input names.  We do this as a second pass to allow the nodes to be in
  // any order.
  for (int n_index = 0; n_index < fdef->node_def_size(); ++n_index) {
    NodeDef* node_def = fdef->mutable_node_def(n_index);
    for (int i = 0; i < node_def->input_size(); ++i) {
      if (str_util::StartsWith(node_def->input(i), "^")) {
        // Control input
        const string normalized =
            node_names.Renormalize(node_def->input(i).substr(1));
        if (normalized.empty()) {
          return errors::InvalidArgument(
              "Could not remap control input ", i, ", '", node_def->input(i),
              "', of node '", node_def->name(), "' in function ", name);
        }
        *node_def->mutable_input(i) = strings::StrCat("^", normalized);
      } else {
        const auto iter = tensor_renaming.find(node_def->input(i));
        if (iter == tensor_renaming.end()) {
          return errors::InvalidArgument(
              "Could not remap input ", i, ", '", node_def->input(i),
              "', of node '", node_def->name(), "' in function ", name);
        }
        *node_def->mutable_input(i) = iter->second;
      }
    }
  }

  // Remap return values.
  for (int r = 0; r < fdef->signature().output_arg_size(); ++r) {
    const string& ret_name = fdef->signature().output_arg(r).name();
    if (ret_name.empty()) {
      return errors::InvalidArgument("Missing output ", r, " to function ",
                                     name);
    }
    const string& return_value = return_values[ret_name];
    const auto iter = tensor_renaming.find(return_value);
    if (iter == tensor_renaming.end()) {
      return errors::InvalidArgument("Could not remap return value ", r, ", '",
                                     ret_name, "', of '", return_value,
                                     "' in function ", name);
    }
    (*fdef->mutable_ret())[ret_name] = iter->second;
  }

  return Status::OK();
}

}  // namespace tensorflow
