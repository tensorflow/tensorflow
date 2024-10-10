/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/split_utils.h"

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/ascii.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {
namespace split_utils {

namespace {

bool ArgDefIsList(const OpDef::ArgDef& arg_def) {
  return !arg_def.number_attr().empty() || !arg_def.type_list_attr().empty();
}

// Returns map from node name to NodeDef in a function.
absl::flat_hash_map<absl::string_view, const NodeDef*> NameToNode(
    const FunctionDef& function) {
  absl::flat_hash_map<absl::string_view, const NodeDef*> name_to_node;
  for (const NodeDef& node : function.node_def()) {
    name_to_node.insert({node.name(), &node});
  }
  return name_to_node;
}

// Returns true if the input string in a FunctionDef node refers to a function
// argument, as opposed to a node output.
bool IsFunctionArgument(absl::string_view input_str) {
  // Arguments are in the form "fun_in" or "fun_in:number", where "fun_in" is
  // the input arg name and "number" is the output index.
  size_t pos = input_str.find(':');
  return pos == absl::string_view::npos ||
         absl::ascii_isdigit(input_str[pos + 1]);
}

size_t FindArgDefIndex(
    const protobuf::RepeatedPtrField<OpDef::ArgDef>& arg_defs,
    absl::string_view name) {
  for (int i = 0; i < arg_defs.size(); i++) {
    if (arg_defs[i].name() == name) {
      return i;
    }
  }
  return -1;
}

// Helper class to SplitFunction(). When adding nodes to `second`, some node
// inputs may refer to nodes in `first`. This class handles this case by adding
// an output argument to `first` and a corresponding input argument to `second`.
// The input of the node in `second` is rewritten to refer to the newly created
// input argument.
class InputRewriter {
 public:
  // Note `original_function` must not have any list arguments.
  InputRewriter(
      const FunctionDef& original_function,
      const absl::flat_hash_set<absl::string_view>& nodes_in_first_func,
      int64_t num_captured_inputs, const FunctionLibraryDefinition& library,
      FunctionDef* first_function, FunctionDef* second_function,
      std::vector<DataType>* first_function_output_types)
      : original_function_(original_function),
        nodes_in_first_func_(nodes_in_first_func),
        num_captured_inputs_(num_captured_inputs),
        library_(library),
        name_to_node_(NameToNode(original_function)),
        first_function_(first_function),
        second_function_(second_function),
        first_function_output_types_(first_function_output_types) {
    for (const NodeDef& node_def : original_function_.node_def()) {
      used_names_.insert(node_def.name());
    }

    for (const OpDef::ArgDef& input_arg :
         original_function_.signature().input_arg()) {
      used_names_.insert(input_arg.name());
    }
  }

  // Rewrite an input of a node that is being moved to the second function.
  // If the input is in the first function, an output argument will be added to
  // the first function and a corresponding input argument will be added to the
  // second function. In this case, the input argument's name will be returned.
  // If the input is in the second function, the input will not be rewritten.
  //
  // *new_input_str will be set to the empty string if the input should be
  // removed, which occurs if it is a control dependency for a node in the first
  // function.
  absl::Status RewriteInput(absl::string_view input_str, string* new_input_str);

 private:
  bool IsInFirstFunction(absl::string_view node_name) {
    return nodes_in_first_func_.contains(node_name);
  }

  // Rewrite a control input. input_str is in the form "^node_name"
  absl::Status RewriteControlInput(absl::string_view input_str,
                                   string* new_input_str);

  // Rewrite an input that is an argument to original_function_. input_str is in
  // the form "fun_in" or "fun_in:number".
  absl::Status RewriteArgumentInput(absl::string_view input_str,
                                    string* new_input_str);

  // Rewrite an input that is the output of a node. input_str is in the form
  // "node:out" or "node:out:number"
  absl::Status RewriteNodeInput(absl::string_view input_str,
                                string* new_input_str);

  // Rewrites an input, `input_str`, where the node producing `input_str` is in
  // first_function_ and the node consuming `input_str` is in second_function_.
  // This function adds an output argument to first_function_ and an input
  // argument to second_function_. "input_arg_def" is the ArgDef corresponding
  // to input_str, and must have the type() field set.
  absl::Status RewriteCrossFunctionInput(absl::string_view input_str,
                                         const OpDef::ArgDef& input_arg_def,
                                         string* new_input_str);

  string unique_name(const std::string& name) {
    if (used_names_.count(name) == 0) {
      used_names_.insert(name);
      return name;
    }

    for (int64_t suffix = 0; true; suffix++) {
      string new_name = absl::StrCat(name, "_", suffix);
      auto iter = used_names_.insert(new_name);
      if (iter.second) {
        return new_name;
      }
    }
  }

  const FunctionDef& original_function_;
  const absl::flat_hash_set<absl::string_view>& nodes_in_first_func_;
  const int64_t num_captured_inputs_;
  const FunctionLibraryDefinition& library_;

  // Map from node name to NodeDef in original_function_.node_def()
  const absl::flat_hash_map<absl::string_view, const NodeDef*> name_to_node_;

  FunctionDef* const first_function_;
  FunctionDef* const second_function_;
  std::vector<DataType>* const first_function_output_types_;

  // Caches results of RewriteInput(), so that if the same input string is
  // passed, it is rewritten to the same string.
  absl::flat_hash_map<absl::string_view, string> input_map_;

  // Node and argument names that are used in either function. Used to uniquify
  // argument names.
  std::unordered_set<string> used_names_;
};

absl::Status InputRewriter::RewriteInput(absl::string_view input_str,
                                         string* new_input_str) {
  auto iter = input_map_.find(input_str);
  if (iter != input_map_.end()) {
    *new_input_str = iter->second;
    return absl::OkStatus();
  }

  if (IsControlInput(input_str)) {
    TF_RETURN_IF_ERROR(RewriteControlInput(input_str, new_input_str));
  } else if (IsFunctionArgument(input_str)) {
    TF_RETURN_IF_ERROR(RewriteArgumentInput(input_str, new_input_str));
  } else {
    TF_RETURN_IF_ERROR(RewriteNodeInput(input_str, new_input_str));
  }
  input_map_.insert({input_str, *new_input_str});
  return absl::OkStatus();
}

absl::Status InputRewriter::RewriteControlInput(absl::string_view input_str,
                                                string* new_input_str) {
  DCHECK_EQ(input_str.at(0), '^');
  absl::string_view node_name = input_str.substr(1);
  if (IsInFirstFunction(node_name)) {
    *new_input_str = "";
  } else {
    *new_input_str = string{input_str};
  }
  return absl::OkStatus();
}

absl::Status InputRewriter::RewriteArgumentInput(absl::string_view input_str,
                                                 string* new_input_str) {
  std::vector<string> components = absl::StrSplit(input_str, ':');
  if (components.size() != 1 && components.size() != 2) {
    return errors::Internal("Found node with invalid argument input: ",
                            input_str);
  }
  string argument_name = components[0];
  if (components.size() == 2 && components[1] != "0") {
    // It is required that `original_function` must not have any list arguments.
    return errors::Internal(
        "Input string \"", input_str,
        "\" has a last component which is not 0, but it is expected to be 0 "
        "because corresponding argument is not a list");
  }

  int i = FindArgDefIndex(original_function_.signature().input_arg(),
                          argument_name);
  if (i == -1) {
    return errors::Internal(
        "Input string \"", input_str,
        "\" refers to an argument which does not exist. Argument \"",
        argument_name, "\" does not appear in following FunctionDef: ",
        original_function_.DebugString());
  }
  if (i >=
      original_function_.signature().input_arg_size() - num_captured_inputs_) {
    // Argument is a captured input. No need to modify argument string.
    *new_input_str = string{input_str};
    return absl::OkStatus();
  }
  const OpDef::ArgDef* found_arg_def =
      &original_function_.signature().input_arg(i);

  if (ArgDefIsList(*found_arg_def)) {
    return errors::Unimplemented(
        "Splitting a function where an edge is a list of tensors is "
        "unsupported. ArgDef representing edge: ",
        found_arg_def->DebugString());
  }
  if (!found_arg_def->type_attr().empty()) {
    return errors::Unimplemented(
        "Splitting a function where an edge's ArgDef has a type attribute is "
        "unsupported. ArgDef representing argument: ",
        found_arg_def->DebugString());
  }

  return RewriteCrossFunctionInput(input_str, *found_arg_def, new_input_str);
}

absl::Status InputRewriter::RewriteNodeInput(absl::string_view input_str,
                                             string* new_input_str) {
  std::vector<string> components = absl::StrSplit(input_str, ':');
  if (components.size() != 2 && components.size() != 3) {
    return errors::Internal("Found node with invalid node input: ", input_str);
  }
  const string& node_name = components[0];
  const string& node_output_arg = components[1];
  const string& list_output_index =
      components.size() == 3 ? components[2] : "0";
  if (!IsInFirstFunction(node_name)) {
    *new_input_str = string{input_str};
    return absl::OkStatus();
  }

  auto index_iter = name_to_node_.find(node_name);
  if (index_iter == name_to_node_.end()) {
    return errors::Internal("Found input referring to nonexistent node: ",
                            node_name);
  }
  const NodeDef& node = *index_iter->second;

  const OpRegistrationData* op_reg_data = nullptr;
  TF_RETURN_IF_ERROR(library_.LookUp(node.op(), &op_reg_data));
  int i = FindArgDefIndex(op_reg_data->op_def.output_arg(), node_output_arg);
  if (i == -1) {
    return errors::Internal("Could not found input \"", node_output_arg,
                            "\" for OpDef ", op_reg_data->op_def.name());
  }
  OpDef::ArgDef found_arg_def = op_reg_data->op_def.output_arg(i);

  if (ArgDefIsList(found_arg_def)) {
    return errors::Unimplemented(
        "Splitting a function where an edge is a list of tensors is "
        "unsupported. ArgDef representing edge: ",
        found_arg_def.DebugString());
  }
  if (list_output_index != "0") {
    return errors::Internal(
        "Input string \"", input_str,
        "\" has a last component which is not 0, but it is expected to be 0 "
        "because corresponding output is not a list");
  }

  if (!found_arg_def.type_attr().empty()) {
    const string& attr = found_arg_def.type_attr();
    auto attr_iter = node.attr().find(attr);
    if (attr_iter == node.attr().end()) {
      return errors::Internal("Failed to find attr ", attr, " on node ",
                              node.name());
    }
    if (!attr_iter->second.placeholder().empty()) {
      return errors::Unimplemented(
          "Splitting a function where an edge between functions has an "
          "AttrValue placeholder dtype is unsupported.");
    }
    DataType dtype = attr_iter->second.type();
    if (dtype == DT_INVALID) {
      return errors::Internal("Attr ", attr, " is not a dtype attr");
    }
    found_arg_def.mutable_type_attr()->clear();
    found_arg_def.set_type(dtype);
  }

  return RewriteCrossFunctionInput(input_str, found_arg_def, new_input_str);
}

absl::Status InputRewriter::RewriteCrossFunctionInput(
    absl::string_view input_str, const OpDef::ArgDef& input_arg_def,
    string* new_input_str) {
  DCHECK(input_arg_def.type() != DT_INVALID);
  if (input_arg_def.is_ref() || IsRefType(input_arg_def.type())) {
    // This case is untested and is not important to support, so an
    // Unimplemented error is raised.
    return errors::Unimplemented(
        "Splitting a function where an edge between functions is a ref is "
        "unsupported. Input ",
        input_str, " is a ref type.");
  }
  OpDef::ArgDef* added_output_arg =
      first_function_->mutable_signature()->add_output_arg();
  *added_output_arg = input_arg_def;
  size_t output_index = first_function_->signature().output_arg_size() - 1;
  added_output_arg->set_name(absl::StrCat("output_", output_index));
  added_output_arg->set_description(absl::StrCat(
      "Output ", output_index, ", corresponding to input ", input_str));
  first_function_->mutable_ret()->insert(
      {added_output_arg->name(), string{input_str}});
  first_function_output_types_->push_back(input_arg_def.type());

  OpDef::ArgDef* added_input_arg =
      second_function_->mutable_signature()->add_input_arg();
  *added_input_arg = input_arg_def;
  size_t input_index = second_function_->signature().input_arg_size() - 1;
  added_input_arg->set_name(unique_name(absl::StrCat("input_", input_index)));
  added_input_arg->set_description(absl::StrCat("Input ", input_index));

  *new_input_str = added_input_arg->name();
  return absl::OkStatus();
}

void InitializeSignatures(
    const FunctionDef& original_function_, FunctionDef* first_function_,
    FunctionDef* second_function_,
    const absl::flat_hash_set<absl::string_view>& nodes_in_first_function,
    const FunctionDefLibrary& func_def_lib_) {
  // Initialize first_function_->signature().
  *first_function_->mutable_signature() = original_function_.signature();
  graph_utils::SetUniqueGraphFunctionName(
      original_function_.signature().name() + "_first_split", &func_def_lib_,
      first_function_);
  first_function_->mutable_signature()->clear_output_arg();
  first_function_->mutable_signature()->clear_control_output();
  first_function_->mutable_signature()->set_description(absl::StrCat(
      "The function \"", original_function_.signature().name(),
      "\" was split into two pieces in the make_deterministic Grappler pass. "
      "This function is the first piece."));
  first_function_->mutable_signature()->set_is_commutative(false);
  first_function_->mutable_signature()->set_is_aggregate(false);

  // Initialize second_function_->signature().
  *second_function_->mutable_signature() = original_function_.signature();
  graph_utils::SetUniqueGraphFunctionName(
      original_function_.signature().name() + "_second_split", &func_def_lib_,
      second_function_);
  second_function_->mutable_signature()->clear_input_arg();
  second_function_->mutable_signature()->clear_control_output();
  second_function_->mutable_signature()->set_description(absl::StrCat(
      "The function \"", original_function_.signature().name(),
      "\" was split into two pieces in the make_deterministic Grappler pass. "
      "This function is the second piece."));
  second_function_->mutable_signature()->set_is_commutative(false);
  second_function_->mutable_signature()->set_is_aggregate(false);

  // Initialize the control_ret fields of the two signatures.
  for (const auto& it : original_function_.control_ret()) {
    if (nodes_in_first_function.contains(it.second)) {
      first_function_->mutable_control_ret()->insert(it);
    } else {
      second_function_->mutable_control_ret()->insert(it);
    }
  }
}

}  // namespace

absl::StatusOr<SplitResults> SplitFunction(
    const FunctionDef& function,
    const absl::flat_hash_set<absl::string_view>& nodes_in_first_function,
    int64_t num_captured_inputs, const FunctionLibraryDefinition& library) {
  for (const auto& attr : function.attr()) {
    if (attr.first != data::kTFDataFunction &&
        attr.first != "_construction_context") {
      return errors::Unimplemented(
          "Cannot split function with unknown attribute key: ", attr.first);
    }
  }

  for (int i = 0; i < function.signature().input_arg_size(); i++) {
    // Processing list arguments is more complicated and not yet implemented.
    if (ArgDefIsList(function.signature().input_arg(i))) {
      return errors::Unimplemented(
          "Cannot split function when an input argument is a list of tensors "
          "instead of a single tensor.");
    }
  }

  for (const NodeDef& node_def : function.node_def()) {
    if (IsControlFlow(node_def)) {
      return errors::Unimplemented(
          "Cannot split function with control flow ops");
    }
  }

  SplitResults results;
  InitializeSignatures(function, &results.first_function,
                       &results.second_function, nodes_in_first_function,
                       library.ToProto());

  // Insert _construction_context attribute into functions, if it exists on
  // original_function_.
  auto contruction_ctx_iter = function.attr().find("_construction_context");
  if (contruction_ctx_iter != function.attr().end()) {
    results.first_function.mutable_attr()->insert(
        {contruction_ctx_iter->first, contruction_ctx_iter->second});
    results.second_function.mutable_attr()->insert(
        {contruction_ctx_iter->first, contruction_ctx_iter->second});
  }

  InputRewriter rewriter{function,
                         nodes_in_first_function,
                         num_captured_inputs,
                         library,
                         &results.first_function,
                         &results.second_function,
                         &results.first_function_output_types};

  for (const NodeDef& orig_node_def : function.node_def()) {
    if (!nodes_in_first_function.contains(orig_node_def.name())) {
      // Add node to second function and rewrite its inputs.
      NodeDef& new_node_def = *results.second_function.add_node_def();
      new_node_def = orig_node_def;
      new_node_def.clear_input();

      for (const string& input_str : orig_node_def.input()) {
        string* new_input_str = new_node_def.add_input();
        TF_RETURN_IF_ERROR(rewriter.RewriteInput(input_str, new_input_str));
        if (new_input_str->empty()) {
          new_node_def.mutable_input()->RemoveLast();
          VLOG(3) << "Removed input " << input_str << " from node "
                  << orig_node_def.name();
        } else if (*new_input_str != input_str) {
          VLOG(3) << "Rewrote input " << input_str << " to " << new_input_str
                  << " of node " << orig_node_def.name();
        }
      }
    } else {
      // Add node to first function, and check that all its inputs are also in
      // the first function.
      *results.first_function.add_node_def() = orig_node_def;
      for (const string& input_str : orig_node_def.input()) {
        std::vector<string> components = absl::StrSplit(input_str, ':');
        if (!IsControlInput(input_str) && !IsFunctionArgument(input_str) &&
            !nodes_in_first_function.contains(components[0])) {
          return errors::Internal("Node ", orig_node_def.name(),
                                  " is in first function but has input ",
                                  input_str,
                                  " which is not in first function.");
        }
      }
    }
  }

  // Add return values to second_fuction.ret()
  for (const OpDef::ArgDef& arg_def : function.signature().output_arg()) {
    auto it = function.ret().find(arg_def.name());
    if (it == function.ret().end()) {
      return errors::Internal(
          "Failed to find output_arg '", arg_def.name(),
          "' in 'ret' section. FunctionDef: ", function.DebugString());
    }
    string& new_ret = (*results.second_function.mutable_ret())[arg_def.name()];
    TF_RETURN_IF_ERROR(rewriter.RewriteInput(it->second, &new_ret));
    DCHECK(!new_ret.empty());
  }

  // Add captured inputs to second_function.input_arg()
  for (int i = function.signature().input_arg_size() - num_captured_inputs;
       i < function.signature().input_arg_size(); i++) {
    *results.second_function.mutable_signature()->add_input_arg() =
        function.signature().input_arg(i);
  }

  return results;
}

}  // namespace split_utils
}  // namespace grappler
}  // namespace tensorflow
