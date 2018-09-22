/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/fusion_utils.h"

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace fusion_utils {

namespace {
string ParseNodeConnection(const string& name) {
  // If input/output node name has semicolon, take the prefix.  Otherwise take
  // the whole string.
  return name.substr(0, name.find(':'));
}

string ParseOutputNode(const string& name) {
  if (name.find(':') == string::npos) return {};
  return name.substr(name.find(':'), string::npos);
}

string GetOutputNode(const FunctionDef& function, int output_idx) {
  const auto& ret_output_name =
      function.signature().output_arg(output_idx).name();
  return function.ret().at(ret_output_name);
}

string& GetMutableOutputNode(FunctionDef* function, int output_idx) {
  const auto& ret_output_name =
      function->signature().output_arg(output_idx).name();
  return function->mutable_ret()->at(ret_output_name);
}

template <typename Iterable>
StringCollection GetNames(const Iterable& iterable, int allocate_size) {
  StringCollection names;
  names.reserve(allocate_size);
  for (auto& arg : iterable) names.push_back(arg.name());
  return names;
}

template <typename Iterable>
gtl::FlatSet<string> GetNodeNamesSet(const Iterable& nodes) {
  // NOTE(prazek): Cases where the set is not modified after construction
  // could use sorted vector with binary_search instead, to make it faster.
  gtl::FlatSet<string> names;
  for (const auto& node : nodes) {
    CHECK(gtl::InsertIfNotPresent(&names, node.name()))
        << "Functions should have unique node names. Node with name "
        << node.name() << " already exists";
  }
  return names;
}

template <typename Iterable>
gtl::FlatMap<string, string> GetUniqueNames(const Iterable& first_iterable,
                                            const Iterable& second_iterable) {
  gtl::FlatMap<string, string> changed_node_names;
  const auto first_names = GetNodeNamesSet(first_iterable);
  auto second_names = GetNodeNamesSet(first_iterable);
  int id = second_iterable.size();

  for (const auto& node : second_iterable) {
    string name_before = node.name();
    string name = name_before;
    bool changed_name = false;

    while (first_names.count(name) ||
           (changed_name && second_names.count(name))) {
      name = strings::StrCat(name_before, "/_", id);
      changed_name = true;
      ++id;
    }
    if (changed_name) {
      changed_node_names[name_before] = name;
      // We don't want to pick a new name that would collide with another new
      // name.
      second_names.insert(std::move(name));
    }
  }
  return changed_node_names;
}

// We need to rename them and the connections of the inputs that refer to them.
// Nodes that will be added to the function can have the same name as the nodes
// from parent function.
void RenameFunctionNodes(const FunctionDef& first_function,
                         protobuf::RepeatedPtrField<NodeDef>* nodes_to_fuse,
                         protobuf::Map<string, string>* rets_to_fuse) {
  const gtl::FlatMap<string, string> changed_node_names =
      GetUniqueNames(first_function.node_def(), *nodes_to_fuse);

  auto update_name = [&changed_node_names](string* input) {
    string input_node = ParseNodeConnection(*input);
    auto iter = changed_node_names.find(input_node);
    if (iter != changed_node_names.end()) {
      *input = iter->second + ParseOutputNode(*input);
    }
  };

  for (NodeDef& function_node : *nodes_to_fuse) {
    if (const string* new_name =
            gtl::FindOrNull(changed_node_names, function_node.name())) {
      function_node.set_name(*new_name);
    }

    for (string& input : *function_node.mutable_input()) {
      update_name(&input);
    }
  }

  for (auto& ret : *rets_to_fuse) update_name(&ret.second);
}

StringCollection GetFunctionInputs(const FunctionDef& function) {
  return GetNames(function.signature().input_arg(),
                  function.signature().input_arg_size());
}

// This function produces signature having names that do not conflict with
// `first_signature`.  The input of returns and nodes that will be fused are
// updated to use new names.
OpDef GetUniqueSignature(const OpDef& first_signature,
                         const OpDef& second_signature,
                         protobuf::Map<string, string>* rets_to_fuse,
                         protobuf::RepeatedPtrField<NodeDef>* nodes_to_fuse) {
  const gtl::FlatMap<string, string> changed_input_names =
      GetUniqueNames(first_signature.input_arg(), second_signature.input_arg());
  OpDef signature;
  signature.set_name(second_signature.name());

  for (const auto& input_arg : second_signature.input_arg()) {
    auto& input = *signature.add_input_arg();
    input = input_arg;
    if (const string* new_name =
            gtl::FindOrNull(changed_input_names, input.name())) {
      input.set_name(*new_name);
    }
  }
  const gtl::FlatMap<string, string> changed_output_names = GetUniqueNames(
      first_signature.output_arg(), second_signature.output_arg());

  for (const auto& output_arg : second_signature.output_arg()) {
    auto& output = *signature.add_output_arg();
    output = output_arg;
    if (const string* new_name =
            gtl::FindOrNull(changed_output_names, output.name())) {
      output.set_name(*new_name);
    }
  }

  protobuf::Map<string, string> new_rets;
  for (const auto& ret : *rets_to_fuse) {
    const auto& key = changed_output_names.count(ret.first)
                          ? changed_output_names.at(ret.first)
                          : ret.first;
    const auto& input = ParseNodeConnection(ret.second);
    const auto& value =
        changed_input_names.count(input)
            ? changed_input_names.at(input) + ParseOutputNode(ret.second)
            : ret.second;
    new_rets[key] = value;
  }
  *rets_to_fuse = std::move(new_rets);

  for (NodeDef& function_node : *nodes_to_fuse) {
    for (auto& node_input : *function_node.mutable_input()) {
      const auto& input = ParseNodeConnection(node_input);
      if (const string* new_name =
              gtl::FindOrNull(changed_input_names, input)) {
        node_input = *new_name + ParseOutputNode(node_input);
      }
    }
  }

  return signature;
}

// This function adds new nodes and changes their input to the output nodes
// of parent function.  It assumes that the name of nodes to fuse are not
// conflicting.
void FuseFunctionNodes(const StringCollection& first_inputs,
                       const StringCollection& second_inputs,
                       const StringCollection& first_outputs,
                       const SetInputFn& set_input,
                       protobuf::RepeatedPtrField<NodeDef>* nodes_to_fuse) {
  for (NodeDef& function_node : *nodes_to_fuse) {
    for (auto& node_input : *function_node.mutable_input()) {
      auto parsed_name = ParseNodeConnection(node_input);

      auto input_it =
          std::find(second_inputs.begin(), second_inputs.end(), parsed_name);
      if (input_it == second_inputs.end()) continue;

      auto arg_num = std::distance(second_inputs.begin(), input_it);
      node_input =
          set_input(first_inputs, second_inputs, first_outputs, arg_num);
    }
  }
}

// This function looks for direct edges from input to return and rewrites
// them to the corresponding input of the return of `first_function`.
void FuseReturns(const StringCollection& first_inputs,
                 const StringCollection& second_inputs,
                 const StringCollection& first_outputs,
                 const SetInputFn& set_input,
                 protobuf::Map<string, string>* fused_ret) {
  for (auto& ret : *fused_ret) {
    auto return_input = ParseNodeConnection(ret.second);
    auto input_it =
        std::find(second_inputs.begin(), second_inputs.end(), return_input);
    if (input_it == second_inputs.end()) continue;

    auto input_idx = std::distance(second_inputs.begin(), input_it);
    ret.second =
        set_input(first_inputs, second_inputs, first_outputs, input_idx);
  }
}

// Returns collection of node names that are used as a return from function.
StringCollection GetFunctionOutputs(const FunctionDef& function) {
  const auto number_of_outputs = function.signature().output_arg_size();
  StringCollection outputs;
  outputs.reserve(number_of_outputs);

  for (int output_idx = 0; output_idx < number_of_outputs; output_idx++)
    outputs.push_back(GetOutputNode(function, output_idx));
  return outputs;
}

FunctionDef* CreateFalsePredicate(
    const protobuf::RepeatedPtrField<OpDef_ArgDef>& fake_args,
    FunctionDefLibrary* library) {
  GraphDef graph;
  MutableGraphView graph_view(&graph);
  auto* node = graph_utils::AddScalarConstNode(false, &graph_view);
  auto* false_predicate = library->add_function();
  graph_utils::SetUniqueGraphFunctionName("false_predicate", library,
                                          false_predicate);

  int num = 0;
  for (const auto& fake_arg : fake_args) {
    auto* arg = false_predicate->mutable_signature()->add_input_arg();
    arg->set_type(fake_arg.type());
    arg->set_name(strings::StrCat("fake_arg", num));
    num++;
  }

  auto* output = false_predicate->mutable_signature()->add_output_arg();
  output->set_name("false_out");
  output->set_type(DT_BOOL);

  (*false_predicate->mutable_ret())["false_out"] = node->name() + ":output:0";
  *false_predicate->mutable_node_def() = std::move(*graph.mutable_node());
  return false_predicate;
}

void CheckIfCanCompose(const OpDef& first_signature,
                       const OpDef& second_signature) {
  CHECK(CanCompose(first_signature, second_signature))
      << "The number of input arguments of function " << second_signature.name()
      << " should be the same as the number of output arguments of function "
      << first_signature.name() << ".";
}

}  // namespace

void MergeNodes(const FunctionDef& first_function,
                const FunctionDef& second_function, FunctionDef* fused_function,
                FunctionDefLibrary* library) {
  // Copy all nodes from first_function.
  fused_function->mutable_node_def()->CopyFrom(first_function.node_def());
  // Copy transformed nodes from the second function.
  fused_function->mutable_node_def()->MergeFrom(second_function.node_def());
}

bool CanCompose(const OpDef& first_signature, const OpDef& second_signature) {
  // TODO(prazek): Functions can have additional inputs being placeholders
  // for a values used in function.  We should be able to also fuse these
  // functions.
  return first_signature.output_arg_size() == second_signature.input_arg_size();
}

string ComposeInput(const StringCollection& first_inputs,
                    const StringCollection& second_inputs,
                    const StringCollection& first_outputs, int arg_num) {
  // Take corresponding parent output.
  return first_outputs.at(arg_num);
}

void ComposeSignature(const OpDef& first_signature,
                      const OpDef& second_signature, OpDef* fused_signature) {
  CheckIfCanCompose(first_signature, second_signature);

  // Copy input signature from parent function.
  *fused_signature->mutable_input_arg() = first_signature.input_arg();
  // Copy output signature from second function.
  *fused_signature->mutable_output_arg() = second_signature.output_arg();
}

void ComposeOutput(const protobuf::Map<string, string>& first_ret,
                   const protobuf::Map<string, string>& second_ret,
                   protobuf::Map<string, string>* fused_ret) {
  *fused_ret = second_ret;
}

void CombineSignature(const OpDef& first_signature,
                      const OpDef& second_signature, OpDef* fused_signature) {
  CheckIfCanCompose(first_signature, second_signature);
  // Copy input and output signature from parent function.
  *fused_signature = first_signature;

  // Add new output parameter.
  fused_signature->mutable_output_arg()->MergeFrom(
      second_signature.output_arg());
}

void CombineOutput(const protobuf::Map<string, string>& first_ret,
                   const protobuf::Map<string, string>& second_ret,
                   protobuf::Map<string, string>* fused_ret) {
  *fused_ret = first_ret;
  fused_ret->insert(second_ret.begin(), second_ret.end());
}

string SameInput(const StringCollection& first_inputs,
                 const StringCollection& second_inputs,
                 const StringCollection& first_outputs, int arg_num) {
  return first_inputs.at(arg_num);
}

bool HasSameSignature(const OpDef& first_signature,
                      const OpDef& second_signature) {
  return first_signature.input_arg_size() ==
             second_signature.input_arg_size() &&
         first_signature.output_arg_size() ==
             second_signature.output_arg_size();
}

void SameSignature(const OpDef& first_signature, const OpDef& second_signature,
                   OpDef* fused_signature) {
  CHECK(HasSameSignature(first_signature, second_signature))
      << "Functions do not have the same signature";
  // Copy signature from first function.
  *fused_signature = first_signature;
}

void LazyConjunctionNodes(const FunctionDef& first_function,
                          const FunctionDef& second_function,
                          FunctionDef* fused_function,
                          FunctionDefLibrary* library) {
  fused_function->mutable_node_def()->CopyFrom(first_function.node_def());

  NodeDefBuilder if_builder("", "If");
  if_builder.Input(GetOutputNode(first_function, 0), 0, DT_BOOL);
  DataTypeVector in_arg_types;
  std::vector<NodeDefBuilder::NodeOut> inputs;
  for (const auto& input_arg : first_function.signature().input_arg()) {
    inputs.push_back({input_arg.name(), 0, input_arg.type()});
    in_arg_types.push_back(input_arg.type());
  }
  if_builder.Attr("Tin", in_arg_types);

  if_builder.Attr("Tcond", DT_BOOL);
  if_builder.Attr("Tout", DataTypeVector{DT_BOOL});
  if_builder.Attr("_lower_using_switch_merge", true);

  NameAttrList then_branch;
  then_branch.set_name(second_function.signature().name());
  if_builder.Attr("then_branch", then_branch);

  auto* false_predicate =
      CreateFalsePredicate(first_function.signature().input_arg(), library);

  NameAttrList else_branch;
  else_branch.set_name(false_predicate->signature().name());
  if_builder.Attr("else_branch", else_branch);
  if_builder.Input(inputs);

  auto* if_node = fused_function->add_node_def();
  // This is guaranteed to succeed.
  TF_CHECK_OK(if_builder.Finalize(if_node));
  function_utils::SetUniqueFunctionNodeName("cond", fused_function, if_node);

  GetMutableOutputNode(fused_function, 0) = if_node->name() + ":output:0";
}

void LazyConjunctionOutput(const protobuf::Map<string, string>& first_ret,
                           const protobuf::Map<string, string>& second_ret,
                           protobuf::Map<string, string>* fused_ret) {
  CHECK_EQ(first_ret.size(), 1);
  CHECK_EQ(second_ret.size(), 1);
  // Temporarily copy returns from first_ret.  We are going to change the
  // output node after creating it.
  *fused_ret = first_ret;
}

FunctionDef* FuseFunctions(
    const FunctionDef& first_function, const FunctionDef& second_function,
    StringPiece fused_name_prefix, const SetFunctionSignatureFn& set_signature,
    const SetInputFn& set_input, const SetOutputFn& set_output,
    const SetNodesFn& set_nodes, FunctionDefLibrary* library) {
  if (first_function.attr_size() != 0 || second_function.attr_size() != 0)
    return nullptr;  // Functions with attributes are currently not supported

  // This function will be used as a clone of second function, having unique
  // names.
  FunctionDef setup_function = second_function;
  *setup_function.mutable_signature() = GetUniqueSignature(
      first_function.signature(), setup_function.signature(),
      setup_function.mutable_ret(), setup_function.mutable_node_def());

  FunctionDef* fused_function = library->add_function();

  set_signature(first_function.signature(), setup_function.signature(),
                fused_function->mutable_signature());

  graph_utils::SetUniqueGraphFunctionName(fused_name_prefix, library,
                                          fused_function);

  RenameFunctionNodes(first_function, setup_function.mutable_node_def(),
                      setup_function.mutable_ret());
  set_output(first_function.ret(), setup_function.ret(),
             fused_function->mutable_ret());

  CHECK(fused_function->signature().output_arg_size() ==
        fused_function->ret_size())
      << "Fused function must have the same number of returns as output "
         "args.  Output size: "
      << fused_function->signature().output_arg_size()
      << ", ret size: " << fused_function->ret_size();

  const auto first_inputs = GetFunctionInputs(first_function);
  const auto second_inputs = GetFunctionInputs(setup_function);
  const auto first_outputs = GetFunctionOutputs(first_function);
  FuseFunctionNodes(first_inputs, second_inputs, first_outputs, set_input,
                    setup_function.mutable_node_def());
  FuseReturns(first_inputs, second_inputs, first_outputs, set_input,
              fused_function->mutable_ret());

  set_nodes(first_function, setup_function, fused_function, library);

  return fused_function;
}

}  // end namespace fusion_utils
}  // end namespace grappler
}  // end namespace tensorflow
