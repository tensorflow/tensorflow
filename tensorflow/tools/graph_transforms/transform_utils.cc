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

#include "tensorflow/tools/graph_transforms/transform_utils.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace graph_transforms {

namespace {
inline bool IsMerge(const NodeDef& node_def) {
  return node_def.op() == "Merge" || node_def.op() == "RefMerge";
}

void RecordMatchedNodes(const NodeMatch& match,
                        std::set<string>* matched_nodes) {
  matched_nodes->insert(match.node.name());
  for (const NodeMatch& input_match : match.inputs) {
    RecordMatchedNodes(input_match, matched_nodes);
  }
}

inline uint64 Hash64String(const string& input) {
  return Hash64(input.data(), input.size());
}
}  // namespace

void MatchedNodesAsArray(const NodeMatch& match, std::vector<NodeDef>* result) {
  std::set<string> found_nodes;
  std::vector<NodeMatch> current_matches = {match};
  while (!current_matches.empty()) {
    std::vector<NodeMatch> next_matches;
    for (const NodeMatch& current_match : current_matches) {
      if (found_nodes.count(current_match.node.name())) {
        continue;
      }
      found_nodes.insert(current_match.node.name());
      result->push_back(current_match.node);
      for (const NodeMatch& input_match : current_match.inputs) {
        next_matches.push_back(input_match);
      }
    }
    current_matches = next_matches;
  }
}

void MapNamesToNodes(const GraphDef& graph_def,
                     std::map<string, const NodeDef*>* result) {
  for (const NodeDef& node : graph_def.node()) {
    (*result)[node.name()] = &node;
  }
}

void MapNodesToOutputs(const GraphDef& graph_def,
                       std::map<string, std::vector<const NodeDef*>>* result) {
  std::map<string, const NodeDef*> node_map;
  MapNamesToNodes(graph_def, &node_map);
  for (const NodeDef& node : graph_def.node()) {
    for (const string& input : node.input()) {
      string input_node_name = NodeNameFromInput(input);
      (*result)[input_node_name].push_back(&node);
    }
  }
}

void NodeNamePartsFromInput(const string& input_name, string* prefix,
                            string* node_name, string* suffix) {
  std::vector<string> input_parts = str_util::Split(input_name, ':');
  if (input_parts.size() < 2) {
    *suffix = "";
  } else {
    *suffix = ":" + input_parts[1];
  }
  StringPiece node_name_piece(input_parts[0]);
  if (str_util::ConsumePrefix(&node_name_piece, "^")) {
    *prefix = "^";
  } else {
    *prefix = "";
  }
  *node_name = std::string(node_name_piece);
}

string NodeNameFromInput(const string& input_name) {
  string prefix;
  string node_name;
  string suffix;
  NodeNamePartsFromInput(input_name, &prefix, &node_name, &suffix);
  return node_name;
}

string CanonicalInputName(const string& input_name) {
  string prefix;
  string node_name;
  string suffix;
  NodeNamePartsFromInput(input_name, &prefix, &node_name, &suffix);
  if (suffix.empty()) {
    suffix = ":0";
  }
  return prefix + node_name + suffix;
}

uint64 HashNodeDef(const NodeDef& node) {
  uint64 hash = Hash64String(node.op());
  hash = Hash64Combine(hash, Hash64String(node.name()));
  for (const string& input : node.input()) {
    hash = Hash64Combine(hash, Hash64String(CanonicalInputName(input)));
  }
  hash = Hash64Combine(hash, Hash64String(node.device()));
  std::vector<string> attr_names;
  attr_names.reserve(node.attr().size());
  for (const auto& attr : node.attr()) {
    attr_names.push_back(attr.first);
  }
  std::sort(attr_names.begin(), attr_names.end());
  string attr_serialized;
  for (const string& attr_name : attr_names) {
    auto attr = node.attr().at(attr_name);
    attr.SerializeToString(&attr_serialized);
    hash = Hash64Combine(hash, Hash64String(attr_serialized));
  }
  return hash;
}

void AddNodeInput(const string& input_name, NodeDef* node) {
  *(node->mutable_input()->Add()) = input_name;
}

void CopyNodeAttr(const NodeDef& source, const string& source_key,
                  const string& dest_key, NodeDef* dest) {
  CHECK_NE(0, source.attr().count(source_key))
      << "No key '" << source_key << "' found in " << source.DebugString();
  (*(dest->mutable_attr()))[dest_key] = source.attr().at(source_key);
}

Tensor GetNodeTensorAttr(const NodeDef& node, const string& key) {
  TensorProto tensor_proto = node.attr().at(key).tensor();
  Tensor tensor;
  CHECK(tensor.FromProto(tensor_proto));
  return tensor;
}

void FilterGraphDef(const GraphDef& input_graph_def,
                    std::function<bool(const NodeDef&)> selector,
                    GraphDef* output_graph_def) {
  output_graph_def->mutable_node()->Clear();
  for (const NodeDef& node : input_graph_def.node()) {
    if (selector(node)) {
      *output_graph_def->mutable_node()->Add() = node;
    }
  }
}

void RemoveAttributes(const GraphDef& input_graph_def,
                      const std::vector<string>& attributes,
                      GraphDef* output_graph_def) {
  output_graph_def->mutable_node()->Clear();
  for (const NodeDef& node : input_graph_def.node()) {
    NodeDef* new_node = output_graph_def->mutable_node()->Add();
    *new_node = node;
    for (const string& attribute : attributes) {
      new_node->mutable_attr()->erase(attribute);
    }
  }
}

Status SortByExecutionOrder(const GraphDef& input_graph_def,
                            GraphDef* output_graph_def) {
  const int num_nodes = input_graph_def.node_size();
  std::vector<int> ready;
  std::vector<int> pending_count;
  pending_count.reserve(num_nodes);
  std::vector<gtl::InlinedVector<int, 4>> outputs(num_nodes);

  std::map<string, int> name_index;
  for (int i = 0; i < input_graph_def.node_size(); ++i) {
    const NodeDef& node(input_graph_def.node(i));
    name_index[node.name()] = i;
  }

  // Parse the inputs for each node.
  for (int n = 0; n < num_nodes; ++n) {
    const NodeDef& node_def(input_graph_def.node(n));
    if (IsMerge(node_def)) {
      // for merge only wait for one non-control input.
      int32 num_control_edges = 0;
      for (int i = 0; i < node_def.input_size(); ++i) {
        if (str_util::StartsWith(node_def.input(i), "^")) {
          num_control_edges++;
        }
      }
      pending_count.push_back(num_control_edges + 1);
    } else {
      pending_count.push_back(node_def.input_size());
    }
    if (node_def.input_size() == 0) {
      ready.push_back(n);
      continue;
    }
    for (int i = 0; i < node_def.input_size(); ++i) {
      const string& input_name = node_def.input(i);
      const string& input_node_name = NodeNameFromInput(input_name);
      if (!name_index.count(input_node_name)) {
        return errors::InvalidArgument("Node '", node_def.name(),
                                       "': Unknown input node '",
                                       node_def.input(i), "'");
      }
      outputs[name_index[input_node_name]].push_back(n);
    }
  }

  int processed = 0;
  output_graph_def->Clear();
  // Process the NodeDefs in topological order.
  // Code above sets this up by filling in ready_ with nodes that have no
  // inputs, pending_counts_ with the number of inputs for each node and
  // outputs_ with the outputs of each node.
  while (!ready.empty()) {
    int o = ready.back();
    ready.pop_back();
    ++processed;
    const NodeDef& node_def(input_graph_def.node(o));
    *output_graph_def->mutable_node()->Add() = node_def;

    // Update pending_count for outputs.
    for (size_t i = 0; i < outputs[o].size(); ++i) {
      const int output = outputs[o][i];
      pending_count[output]--;
      if (pending_count[output] == 0) {
        ready.push_back(output);
      }
    }
  }

  if (processed < num_nodes) {
    LOG(WARNING) << "IN " << __func__ << (num_nodes - processed)
                 << " NODES IN A CYCLE";
    for (int64 i = 0; i < num_nodes; i++) {
      if (pending_count[i] != 0) {
        LOG(WARNING) << "PENDING: " << SummarizeNodeDef(input_graph_def.node(i))
                     << "WITH PENDING COUNT = " << pending_count[i];
      }
    }
    return errors::InvalidArgument(num_nodes - processed, " nodes in a cycle");
  }
  return Status::OK();
}

string OpTypePattern::DebugString() const {
  string result = "{" + op + ", {";
  for (const OpTypePattern& input : inputs) {
    result += input.DebugString() + ",";
  }
  result += "}}";
  return result;
}

string NodeMatch::DebugString() const {
  string result = "{";
  result += node.DebugString();
  result += ", {";
  for (const NodeMatch& input : inputs) {
    result += input.DebugString() + ",";
  }
  result += "}}";
  return result;
}

GraphMatcher::GraphMatcher(const GraphDef& graph_def) {
  SortByExecutionOrder(graph_def, &graph_def_).IgnoreError();
  MapNamesToNodes(graph_def_, &node_map_);
}

Status GraphMatcher::GetOpTypeMatches(const OpTypePattern& pattern,
                                      std::vector<NodeMatch>* matches) {
  std::set<string> matched_nodes;
  for (const NodeDef& node : graph_def_.node()) {
    // Skip any nodes that are already part of a match.
    if (matched_nodes.count(node.name())) {
      continue;
    }
    NodeMatch match;
    if (DoesOpTypeMatch(node, pattern, matched_nodes, &match)) {
      RecordMatchedNodes(match, &matched_nodes);
      matches->push_back(match);
    }
  }
  return Status::OK();
}

bool GraphMatcher::DoesOpTypeMatch(
    const NodeDef& node, const OpTypePattern& pattern,
    const std::set<string>& previously_matched_nodes, NodeMatch* match) {
  VLOG(1) << "Looking at node " << node.DebugString();
  VLOG(1) << "pattern=" << pattern.DebugString();
  VLOG(1) << "match=" << match->DebugString();
  if (previously_matched_nodes.count(node.name())) {
    VLOG(1) << "node " << node.name() << " has been previously matched";
    return false;
  }
  bool pattern_matched = false;
  if (pattern.op == "*") {
    pattern_matched = true;
  } else {
    std::vector<string> pattern_ops = str_util::Split(pattern.op, '|');
    for (const string& pattern_op : pattern_ops) {
      if (node.op() == pattern_op) {
        pattern_matched = true;
      }
    }
  }
  if (!pattern_matched) {
    VLOG(1) << "node.op() != pattern.op()";
    return false;
  }
  match->node = node;
  // Ignore any control inputs for pattern-matching purposes
  std::vector<string> non_control_inputs;
  for (const string& input : node.input()) {
    if (!input.empty() && (input[0] != '^')) {
      non_control_inputs.push_back(input);
    }
  }
  if (pattern.inputs.empty()) {
    // If there are no inputs, assume that's the end of the pattern.
    return true;
  }
  if (non_control_inputs.size() != pattern.inputs.size()) {
    VLOG(1) << "non_control_inputs.size() != pattern.inputs.size()";
    return false;
  }
  for (int i = 0; i < pattern.inputs.size(); ++i) {
    const string& input_node_name = NodeNameFromInput(non_control_inputs[i]);
    const NodeDef& input_node = *(node_map_[input_node_name]);
    const OpTypePattern& input_pattern = pattern.inputs[i];
    match->inputs.push_back(NodeMatch());
    NodeMatch* input_match = &(match->inputs.back());
    if (!DoesOpTypeMatch(input_node, input_pattern, previously_matched_nodes,
                         input_match)) {
      return false;
    }
  }
  return true;
}

Status ReplaceMatchingOpTypes(
    const GraphDef& input_graph_def, const OpTypePattern& pattern,
    const std::function<Status(const NodeMatch&, const std::set<string>&,
                               const std::set<string>&, std::vector<NodeDef>*)>&
        node_generator,
    const ReplaceMatchingOpTypesOptions& options, GraphDef* output_graph_def) {
  // Start off by retrieving all the matching subgraphs.
  GraphMatcher matcher(input_graph_def);
  std::vector<NodeMatch> matches;
  TF_RETURN_IF_ERROR(matcher.GetOpTypeMatches(pattern, &matches));

  // Do some housekeeping so we can easily look up the resulting matches given
  // a node name.
  std::set<string> matched_nodes;
  std::map<string, const NodeMatch*> matches_by_head_name;
  for (const NodeMatch& match : matches) {
    matches_by_head_name[match.node.name()] = &match;
    RecordMatchedNodes(match, &matched_nodes);
  }
  std::map<string, std::vector<const NodeDef*>> outputs_map;
  MapNodesToOutputs(input_graph_def, &outputs_map);

  // Go through all the nodes in the input graph, see if they are part of a
  // match or if they can be left untouched.
  output_graph_def->Clear();
  for (const NodeDef& input_node : input_graph_def.node()) {
    if (matches_by_head_name.count(input_node.name())) {
      // This node is the beginning of a match, so call the replacement function
      // after setting up some information it will need.
      const NodeMatch* match = matches_by_head_name[input_node.name()];
      std::vector<NodeDef> matched_nodes_array;
      MatchedNodesAsArray(*match, &matched_nodes_array);
      // This tells us whether a node is part of the current match.
      std::set<string> matched_nodes_lookup;
      for (const NodeDef& matched_node : matched_nodes_array) {
        matched_nodes_lookup.insert(matched_node.name());
      }
      // These are helper arrays that the replacement function can use to tell
      // whether it can safely remove an internal node (because nothing outside
      // of the match uses it) or whether external nodes depend on it.
      std::set<string> input_nodes;
      std::set<string> output_nodes;
      for (const NodeDef& matched_node : matched_nodes_array) {
        // Look through all of this node's inputs, and if any of them come from
        // outside the match, then this should be noted as one of the external
        // inputs of the subgraph.
        for (const string& input_name : matched_node.input()) {
          string input_node_name = NodeNameFromInput(input_name);
          if (!matched_nodes_lookup.count(input_node_name)) {
            input_nodes.insert(matched_node.name());
          }
        }
        // Do a reverse input lookup, to see which other nodes use the current
        // one as an input. If any of those nodes are outside the match
        // subgraph, then the current node is marked as an output node that
        // shouldn't be removed.
        if (outputs_map.count(matched_node.name())) {
          for (const NodeDef* dependent_node :
               outputs_map[matched_node.name()]) {
            if (!matched_nodes_lookup.count(dependent_node->name())) {
              output_nodes.insert(matched_node.name());
            }
          }
        }
      }
      // Call the generator function and add all the returned nodes to the
      // graph.
      std::vector<NodeDef> new_nodes;
      TF_RETURN_IF_ERROR(
          node_generator(*match, input_nodes, output_nodes, &new_nodes));
      std::set<string> new_node_names;
      for (const NodeDef& new_node : new_nodes) {
        new_node_names.insert(new_node.name());
      }
      // Check to make sure the generator function preserved all of the nodes
      // that are used elsewhere in the graph, and add them back in if not.
      bool abort_replacement = false;
      if (!options.allow_inconsistencies) {
        for (const string& expected_output : output_nodes) {
          if (!new_node_names.count(expected_output)) {
            LOG(WARNING) << "Expected " << expected_output
                         << " to be preserved.";
            abort_replacement = true;
          }
        }
      }
      if (abort_replacement) {
        LOG(WARNING) << "Generator function didn't preserve needed nodes, "
                     << "copying old replacements back in instead.";
        std::vector<NodeDef> old_nodes;
        MatchedNodesAsArray(*match, &old_nodes);
        for (const NodeDef& old_node : old_nodes) {
          NodeDef* added_node = output_graph_def->mutable_node()->Add();
          *added_node = old_node;
        }
      } else {
        for (const NodeDef& new_node : new_nodes) {
          NodeDef* added_node = output_graph_def->mutable_node()->Add();
          *added_node = new_node;
        }
      }
    } else if (!matched_nodes.count(input_node.name())) {
      // This node isn't part of any match, so just copy it over.
      NodeDef* added_node = output_graph_def->mutable_node()->Add();
      *added_node = input_node;
    } else {
      // Do nothing, because this is an internal part of a matching subgraph,
      // and so will have been replaced by a new replacement subgraph.
    }
  }

  return Status::OK();
}

Status RenameNodeInputs(const GraphDef& input_graph_def,
                        const std::map<string, string>& inputs_to_rename,
                        const std::unordered_set<string>& nodes_to_ignore,
                        GraphDef* output_graph_def) {
  std::map<string, std::vector<std::pair<string, string>>>
      canonical_inputs_to_rename;
  for (const auto& input_to_rename : inputs_to_rename) {
    canonical_inputs_to_rename[NodeNameFromInput(input_to_rename.first)]
        .push_back({input_to_rename.first, input_to_rename.second});
  }

  output_graph_def->Clear();
  for (const NodeDef& node : input_graph_def.node()) {
    NodeDef* new_node = output_graph_def->mutable_node()->Add();
    *new_node = node;
    new_node->mutable_input()->Clear();
    for (const string& input_name : node.input()) {
      std::set<string> already_visited;
      string new_input_name = input_name;
      while (
          canonical_inputs_to_rename.count(NodeNameFromInput(new_input_name))) {
        string input_node_name = NodeNameFromInput(new_input_name);
        if (already_visited.count(input_node_name)) {
          return errors::InvalidArgument(
              "RenameNodeInputs argument contains a cycle for ",
              input_node_name);
        }
        already_visited.insert(input_node_name);
        if (nodes_to_ignore.count(node.name())) {
          break;
        }
        bool any_match_found = false;
        for (const std::pair<string, string>& input_to_rename :
             canonical_inputs_to_rename.at(input_node_name)) {
          const string& source_name = input_to_rename.first;
          const string& dest_name = input_to_rename.second;
          bool is_match;
          string match_name;
          if (str_util::EndsWith(source_name, ":*")) {
            is_match = true;
            string prefix;
            string unused_node_name;
            string suffix;
            NodeNamePartsFromInput(new_input_name, &prefix, &unused_node_name,
                                   &suffix);
            match_name = prefix + dest_name + suffix;
          } else {
            is_match = (CanonicalInputName(source_name) ==
                        CanonicalInputName(new_input_name));
            match_name = dest_name;
          }
          if (is_match) {
            new_input_name = match_name;
            any_match_found = true;
          }
        }
        if (!any_match_found) {
          break;
        }
      }
      *(new_node->mutable_input()->Add()) = new_input_name;
    }
  }
  return Status::OK();
}

void CopyOriginalMatch(const NodeMatch& match,
                       std::vector<NodeDef>* new_nodes) {
  std::vector<NodeDef> old_nodes;
  MatchedNodesAsArray(match, &old_nodes);
  for (const NodeDef& old_node : old_nodes) {
    new_nodes->push_back(old_node);
  }
}

TransformRegistry* GetTransformRegistry() {
  static TransformRegistry transform_registry;
  return &transform_registry;
}

void FindInvalidInputs(const GraphDef& graph_def,
                       std::vector<std::pair<string, string>>* invalid_inputs) {
  std::map<string, const NodeDef*> node_map;
  MapNamesToNodes(graph_def, &node_map);

  for (const NodeDef& node : graph_def.node()) {
    for (const string& input : node.input()) {
      string input_node = NodeNameFromInput(input);
      if (!node_map.count(input_node)) {
        invalid_inputs->push_back({node.name(), input_node});
      }
    }
  }
}

Status IsGraphValid(const GraphDef& graph_def) {
  std::vector<std::pair<string, string>> invalid_inputs;
  FindInvalidInputs(graph_def, &invalid_inputs);
  if (!invalid_inputs.empty()) {
    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(graph_def, &node_map);
    for (const std::pair<string, string>& invalid_input : invalid_inputs) {
      LOG(ERROR) << "Invalid input " << invalid_input.second << " for node "
                 << invalid_input.first << " - "
                 << node_map[invalid_input.first]->DebugString();
    }
    return errors::Internal(
        "Invalid graph with inputs referring to nonexistent nodes");
  }
  return Status::OK();
}

Status GetInOutTypes(const NodeDef& node_def, DataTypeVector* inputs,
                     DataTypeVector* outputs) {
  const OpDef* op_def;
  TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef(node_def.op(), &op_def));
  TF_RETURN_IF_ERROR(InOutTypesForNode(node_def, *op_def, inputs, outputs));
  return Status::OK();
}

Status TensorShapeFromString(const string& shape_string, TensorShape* result) {
  if (shape_string.empty()) {
    return errors::InvalidArgument("Specificed shape is empty.");
  }
  std::vector<int64> dims;
  if (!str_util::SplitAndParseAsInts(shape_string, ',', &dims)) {
    return errors::InvalidArgument("Could parse as shape: '", shape_string,
                                   "'");
  }
  *result = TensorShape(dims);
  return Status::OK();
}

int TransformFuncContext::CountParameters(const string& name) const {
  if (params.count(name)) {
    return params.at(name).size();
  } else {
    return 0;
  }
}

Status TransformFuncContext::GetOneStringParameter(const string& name,
                                                   const string& default_value,
                                                   string* result) const {
  const int params_count = CountParameters(name);
  if (params_count == 0) {
    *result = default_value;
    return Status::OK();
  } else if (params_count == 1) {
    *result = params.at(name).at(0);
    return Status::OK();
  } else {
    return errors::InvalidArgument("Expected a single '", name,
                                   "' parameter, but found ", params_count,
                                   " occurrences");
  }
}

Status TransformFuncContext::GetOneInt32Parameter(const string& name,
                                                  int32 default_value,
                                                  int32* result) const {
  const int params_count = CountParameters(name);
  if (params_count == 0) {
    *result = default_value;
    return Status::OK();
  }
  string string_value;
  TF_RETURN_IF_ERROR(GetOneStringParameter(name, "", &string_value));
  if (!strings::safe_strto32(StringPiece(string_value), result)) {
    return errors::InvalidArgument("Couldn't interpret the ", name,
                                   " argument as a number:", string_value);
  }
  return Status::OK();
}

Status TransformFuncContext::GetOneInt64Parameter(const string& name,
                                                  int64 default_value,
                                                  int64* result) const {
  const int params_count = CountParameters(name);
  if (params_count == 0) {
    *result = default_value;
    return Status::OK();
  }
  string string_value;
  TF_RETURN_IF_ERROR(GetOneStringParameter(name, "", &string_value));
  if (!strings::safe_strto64(StringPiece(string_value), result)) {
    return errors::InvalidArgument("Couldn't interpret the ", name,
                                   " argument as a number:", string_value);
  }
  return Status::OK();
}

Status TransformFuncContext::GetOneFloatParameter(const string& name,
                                                  float default_value,
                                                  float* result) const {
  const int params_count = CountParameters(name);
  if (params_count == 0) {
    *result = default_value;
    return Status::OK();
  }
  string string_value;
  TF_RETURN_IF_ERROR(GetOneStringParameter(name, "", &string_value));
  if (!strings::safe_strtof(string_value.c_str(), result)) {
    return errors::InvalidArgument(
        "Couldn't interpret the ", name,
        " argument as a float number:", string_value);
  }
  return Status::OK();
}

Status TransformFuncContext::GetOneBoolParameter(const string& name,
                                                 bool default_value,
                                                 bool* result) const {
  const int params_count = CountParameters(name);
  if (params_count == 0) {
    *result = default_value;
    return Status::OK();
  }
  string string_value;
  TF_RETURN_IF_ERROR(GetOneStringParameter(name, "", &string_value));
  if (string_value == "true" || string_value == "1") {
    *result = true;
  } else if (string_value == "false" || string_value == "0") {
    *result = false;
  } else {
    return errors::InvalidArgument("Couldn't interpret the ", name,
                                   " argument as a boolean:", string_value,
                                   " (expected true, false, 0 or 1)");
  }
  return Status::OK();
}

}  // namespace graph_transforms
}  // namespace tensorflow
