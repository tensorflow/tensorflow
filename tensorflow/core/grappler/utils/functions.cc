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
#include "tensorflow/core/grappler/utils/functions.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/strings/scanner.h"

namespace tensorflow {
namespace grappler {

GrapplerFunctionItem::GrapplerFunctionItem(
    string func_name, string description, AttrSlice func_attr,
    std::vector<const FunctionDef::ArgAttrs*> arg_attr,
    std::vector<InputArgInstantiation> input_args,
    std::vector<OutputArgInstantiation> output_args,
    std::vector<ControlOutput> control_outputs, const int graph_def_version,
    const bool is_stateful, GraphDef&& function_body)
    : description_(std::move(description)),
      func_attr_(func_attr),
      arg_attr_(std::move(arg_attr)),
      input_args_(std::move(input_args)),
      output_args_(std::move(output_args)),
      control_outputs_(std::move(control_outputs)),
      is_stateful_(is_stateful) {
  id = std::move(func_name);
  graph = std::move(function_body);
  graph.mutable_versions()->set_producer(graph_def_version);

  // Fill the feed nodes with function input arguments.
  for (const InputArgInstantiation& input_arg : input_args_) {
    feed.push_back({input_arg.node_name, Tensor()});
  }
  // Fill the fetch nodes with outputs.
  for (const OutputArgInstantiation& output_arg : output_args_) {
    fetch.push_back(output_arg.node_name);
  }
  // We must keep all control output nodes.
  for (const ControlOutput& control_output : control_outputs_) {
    keep_ops.push_back(control_output.node_name);
  }

  // Tensorflow functions execution semantics is different from the main graph,
  // and we need to preserve it when we do graph optimizations.
  optimization_options().allow_pruning_stateful_and_dataset_ops = false;
}

const string& GrapplerFunctionItem::description() const { return description_; }

const std::vector<InputArgInstantiation>& GrapplerFunctionItem::inputs() const {
  return input_args_;
}

const InputArgInstantiation& GrapplerFunctionItem::input(int i) const {
  return input_args_[i];
}

const std::size_t GrapplerFunctionItem::input_size() const {
  return input_args_.size();
}

const std::vector<OutputArgInstantiation>& GrapplerFunctionItem::outputs()
    const {
  return output_args_;
}

const OutputArgInstantiation& GrapplerFunctionItem::output(int i) const {
  return output_args_[i];
}

const std::size_t GrapplerFunctionItem::output_size() const {
  return output_args_.size();
}

const std::vector<ControlOutput>& GrapplerFunctionItem::control_outputs()
    const {
  return control_outputs_;
}

const std::size_t GrapplerFunctionItem::control_output_size() const {
  return control_outputs_.size();
}

const AttrSlice& GrapplerFunctionItem::func_attr() const { return func_attr_; }

const std::vector<const FunctionDef::ArgAttrs*>&
GrapplerFunctionItem::arg_attr() const {
  return arg_attr_;
}

const GraphDef& GrapplerFunctionItem::function_body() const { return graph; }

GraphDef& GrapplerFunctionItem::mutable_function_body() { return graph; }

bool GrapplerFunctionItem::is_stateful() const { return is_stateful_; }

GrapplerFunctionItem& GrapplerFunctionItem::SwapFunctionBody(GraphDef&& other) {
  graph = std::move(other);
  return *this;
}

bool HasParametrizedType(const FunctionDef& func) {
  const auto is_type_parametrized = [](const OpDef::ArgDef& arg) {
    return !arg.type_attr().empty() || !arg.number_attr().empty() ||
           !arg.type_list_attr().empty();
  };

  const auto& input = func.signature().input_arg();
  const auto& output = func.signature().output_arg();
  return std::any_of(input.begin(), input.end(), is_type_parametrized) ||
         std::any_of(output.begin(), output.end(), is_type_parametrized);
}

bool HasParametrizedBody(const FunctionDef& func) {
  const auto is_parametrized = [&](const NodeDef& node) {
    for (const auto& attr : node.attr()) {
      if (!attr.second.placeholder().empty()) return true;
    }
    return false;
  };
  return std::any_of(func.node_def().begin(), func.node_def().end(),
                     is_parametrized);
}

bool IsParametrized(const FunctionDef& func) {
  return HasParametrizedType(func) || HasParametrizedBody(func);
}

Status InstantiationTypeParameters(
    const FunctionDef& func, const AttrSlice& func_instantiation_attr,
    absl::flat_hash_map<string, DataType>* type_parameters) {
  if (!type_parameters->empty()) {
    return errors::InvalidArgument("Type parameters output map must be empty");
  }

  const auto resolve_type_attr = [&](const OpDef::ArgDef& arg) -> Status {
    if (!arg.type_attr().empty()) {
      DataType dtype;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(func_instantiation_attr, arg.type_attr(), &dtype));
      type_parameters->emplace(arg.type_attr(), dtype);

    } else if (!arg.type_list_attr().empty()) {
      std::vector<DataType> dtypes;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(func_instantiation_attr, arg.type_list_attr(), &dtypes));
      int index = 0;
      for (const DataType& dtype : dtypes) {
        type_parameters->emplace(absl::StrCat(arg.type_list_attr(), ":", index),
                                 dtype);
        ++index;
      }
    }
    return Status::OK();
  };

  for (const auto& input : func.signature().input_arg())
    TF_RETURN_IF_ERROR(resolve_type_attr(input));
  for (const auto& output : func.signature().output_arg())
    TF_RETURN_IF_ERROR(resolve_type_attr(output));

  return Status::OK();
}

Status InstantiationBodyParameters(
    const FunctionDef& func, const AttrSlice& func_instantiation_attr,
    absl::flat_hash_map<string, AttrValue>* body_parameters) {
  if (!body_parameters->empty()) {
    return errors::InvalidArgument("Body parameters output map must be empty");
  }

  for (const NodeDef& func_body_node : func.node_def()) {
    for (auto& attr : func_body_node.attr()) {
      const string& placeholder = attr.second.placeholder();

      if (placeholder.empty() || body_parameters->contains(placeholder)) {
        continue;
      }

      const AttrValue* placeholder_value =
          func_instantiation_attr.Find(placeholder);
      if (placeholder_value) {
        body_parameters->insert({placeholder, *placeholder_value});
      } else {
        return errors::InvalidArgument("Can't resolve placeholder: ",
                                       placeholder);
      }
    }
  }

  return Status::OK();
}

Status MakeGrapplerFunctionItem(const FunctionDef& func,
                                const AttrSlice& func_instantiation_attr,
                                const FunctionLibraryDefinition& flib,
                                const int graph_def_version,
                                GrapplerFunctionItem* item) {
  const OpDef& signature = func.signature();

  if (signature.name().empty()) {
    return errors::InvalidArgument("Function name must be specified");
  }

  // Function types will be resolved from function instantiation attributes. All
  // other attributes will be lost during conversion to FunctionDef.
  for (const OpDef::AttrDef& attr : signature.attr()) {
    if (attr.type() != "type") {
      return errors::InvalidArgument(
          "Function signature must have only type attributes");
    }
  }

  // Instantiate function into a statically defined FunctionBody Graph.
  std::unique_ptr<FunctionBody> fbody;
  TF_RETURN_IF_ERROR(
      FunctionDefToBodyHelper(func, func_instantiation_attr, &flib, &fbody));

  GraphDef function_body;
  fbody->graph->ToGraphDef(&function_body);

  // Function body shares the library with the graph that instantiated it. We do
  // not need a full copy of the function library, just the reachable subset.
  *function_body.mutable_library() = flib.ReachableDefinitions(func).ToProto();

  VLOG(3) << absl::Substitute(
      "Deleted $0 unreachable functions from the Grappler function item "
      "instantiation of $1 (library size = $2)",
      flib.num_functions() - function_body.library().function_size(),
      signature.name(), function_body.library().function_size());

  const int num_instantiated_inputs = fbody->arg_types.size();
  const int num_instantiated_outputs = fbody->ret_types.size();

  std::vector<InputArgInstantiation> inputs;
  inputs.reserve(num_instantiated_inputs);

  for (int in_id = 0; in_id < num_instantiated_inputs; ++in_id) {
    const Node* node = fbody->arg_nodes[in_id];
    const DataType& dtype = fbody->arg_types[in_id];
    inputs.emplace_back(node->name(), dtype);
  }

  std::vector<OutputArgInstantiation> outputs;
  outputs.reserve(num_instantiated_outputs);

  for (int out_id = 0; out_id < num_instantiated_outputs; ++out_id) {
    const Node* node = fbody->ret_nodes[out_id];
    const DataType& dtype = fbody->ret_types[out_id];
    outputs.emplace_back(node->name(), dtype);
  }

  // Control outputs ensure that all side-effectful nodes in the function body
  // will execute, even if they are not required to compute regular output args.
  std::vector<ControlOutput> control_outputs;
  control_outputs.reserve(func.control_ret_size());
  for (const auto& control_ret : func.control_ret()) {
    control_outputs.push_back({control_ret.first, control_ret.second});
  }
  // Sort control outputs to keep FunctionDef output stable. The sort order of
  // map entries in func.control_ret() are not stable.
  // See b/174715578 for context on why stability is desired.
  std::sort(control_outputs.begin(), control_outputs.end());

  std::vector<const FunctionDef::ArgAttrs*> arg_attr(inputs.size(), nullptr);
  for (const auto& attr : func.arg_attr()) {
    arg_attr.at(attr.first) = &attr.second;
  }

  *item = GrapplerFunctionItem(
      /*func_name=*/signature.name(),
      /*description=*/signature.description(),
      /*func_attr=*/AttrSlice(&func.attr()), std::move(arg_attr),
      std::move(inputs), std::move(outputs), std::move(control_outputs),
      graph_def_version, signature.is_stateful(), std::move(function_body));
  return Status::OK();
}

Status MakeGrapplerFunctionItem(const FunctionDef& func,
                                const FunctionLibraryDefinition& flib,
                                const int graph_def_version,
                                GrapplerFunctionItem* item) {
  return MakeGrapplerFunctionItem(func, AttrSlice(), flib, graph_def_version,
                                  item);
}

Status ReplaceInputWithConst(const NodeDef& input_const, int input_index,
                             GrapplerFunctionItem* item) {
  if (!IsConstant(input_const)) {
    return errors::InvalidArgument("Input node is not a constant: ",
                                   SummarizeNodeDef(input_const));
  }
  const int item_input_size = item->input_size();
  if (input_index < 0 || input_index >= item_input_size) {
    return errors::InvalidArgument(
        "Function input index is out of bound: index=", input_index,
        " input_size=", item->input_size());
  }

  const InputArgInstantiation& input_arg = item->input(input_index);

  for (NodeDef& node : *item->graph.mutable_node()) {
    // Replace '_Arg' node in the function body with a 'Const' node.
    if (node.name() == input_arg.node_name) {
      node = input_const;
      node.set_name(input_arg.node_name);
      node.clear_input();
      node.clear_device();  // device placement is defined by instantiating node
    }

    // Update index in all inputs after the removed const input.
    if (IsArg(node)) {
      auto attrs = AttrSlice(node);
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "index", &index));
      if (index >= input_index) {
        (*node.mutable_attr())["index"].set_i(index - 1);
      }
    }
  }

  item->input_args_.erase(item->input_args_.begin() + input_index);
  item->arg_attr_.erase(item->arg_attr_.begin() + input_index);

  return Status::OK();
}

Status RemoveFunctionOutputs(const absl::flat_hash_set<int>& remove_outputs,
                             GrapplerFunctionItem* item,
                             std::vector<std::pair<int, int>>* output_mapping) {
  DCHECK(output_mapping->empty());

  // Do some sanity checking of the removed outputs positions.
  for (int remove_output : remove_outputs) {
    const int item_output_size = item->output_size();
    if (remove_output < 0 || remove_output >= item_output_size) {
      return errors::InvalidArgument(
          "Function output index is out of bound: index=", remove_output,
          " output_size=", item->output_size());
    }
  }

  absl::flat_hash_set<const OutputArgInstantiation*> remove_output_args;
  const auto is_remove_output_arg = [&](const OutputArgInstantiation& output) {
    return remove_output_args.find(&output) != remove_output_args.end();
  };

  for (int i = 0, end = item->output_size(); i < end; ++i) {
    const OutputArgInstantiation& output = item->output(i);
    if (remove_outputs.contains(i)) {
      VLOG(3) << "Remove functions output: name=" << output.node_name
              << "(index = " << i << ")";
      remove_output_args.insert(&output);
    } else if (!remove_output_args.empty()) {
      // Add output mapping only if output position changed.
      output_mapping->push_back({i, i - remove_output_args.size()});
    }
  }

  // Update 'index' attribute in all '_Retval' nodes that are in output mapping.
  for (NodeDef& node : *item->graph.mutable_node()) {
    if (IsRetval(node)) {
      auto attrs = AttrSlice(node);
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "index", &index));

      for (const auto& mapping : *output_mapping) {
        const int from = mapping.first;
        const int to = mapping.second;
        if (index == from) {
          (*node.mutable_attr())["index"].set_i(to);
        }
      }
    }
  }

  auto& o = item->output_args_;
  o.erase(std::remove_if(o.begin(), o.end(), is_remove_output_arg), o.end());

  return Status::OK();
}

namespace {

// FunctionDef uses different connectivity encoding for the function body nodes,
// than a GraphDef (see function.proto for details). This is a helper class that
// converts inputs in GraphDef format (node[:position]) to the FunctionDef
// format (node:output[:position]).
class MakeFunctionDefHelper {
 public:
  MakeFunctionDefHelper() = default;

  Status Initialize(const GrapplerFunctionItem& item,
                    const FunctionLibraryDefinition& flib);

  // Converts input name from GraphDef format (name[:position]) to the
  // FunctionDef input format (name[:output][:position]) using registered input
  // arg instantiations and function body outputs.
  Status AsFunctionDefInput(const string& graph_def_input,
                            string* func_def_input) const;

  // Updates Node inputs from GraphDef to FunctionDef format.
  Status AsFunctionDefNode(NodeDef* function_body_node) const;

  bool IsInputNode(const NodeDef& node) const {
    return input_nodes_.contains(node.name());
  }

  bool IsOutputNode(const NodeDef& node) const {
    return output_nodes_.contains(node.name());
  }

 private:
  absl::flat_hash_set<absl::string_view> input_nodes_;
  absl::flat_hash_set<absl::string_view> output_nodes_;
  // Mapping from function body node name to output names range map.
  absl::flat_hash_map<string, tensorflow::NameRangeMap> function_body_outputs_;
};

Status MakeFunctionDefHelper::Initialize(
    const GrapplerFunctionItem& item, const FunctionLibraryDefinition& flib) {
  for (const InputArgInstantiation& input_arg : item.inputs()) {
    input_nodes_.insert(input_arg.node_name);
  }
  for (const OutputArgInstantiation& output_arg : item.outputs()) {
    output_nodes_.insert(output_arg.node_name);
  }

  for (const NodeDef& node : item.function_body().node()) {
    const OpRegistrationData* registration;
    TF_RETURN_IF_ERROR(flib.LookUp(node.op(), &registration));

    tensorflow::NameRangeMap outputs_range_map;
    TF_RETURN_IF_ERROR(tensorflow::NameRangesForNode(
        node, registration->op_def, nullptr, &outputs_range_map));

    function_body_outputs_.emplace(node.name(), std::move(outputs_range_map));
  }

  return Status::OK();
}

Status MakeFunctionDefHelper::AsFunctionDefInput(const string& graph_def_input,
                                                 string* func_def_input) const {
  if (IsControlInput(graph_def_input)) {
    *func_def_input = graph_def_input;
    return Status::OK();
  }

  const SafeTensorId tensor = ParseTensorName(graph_def_input);
  DCHECK_GE(tensor.index(), 0);

  // Graph def input corresponds to one of the function inputs.
  const auto is_input = input_nodes_.find(tensor.node());
  if (is_input != input_nodes_.end()) {
    DCHECK_EQ(tensor.index(), 0);
    *func_def_input = tensor.node();
    return Status::OK();
  }

  // Or it must be output from one of the function body nodes
  const auto is_body_output = function_body_outputs_.find(tensor.node());
  if (is_body_output != function_body_outputs_.end()) {
    const tensorflow::NameRangeMap& outputs_range_map = is_body_output->second;

    for (const auto& el : outputs_range_map) {
      const auto& output_name = el.first;
      const auto& output_range = el.second;
      if (tensor.index() >= output_range.first &&
          tensor.index() < output_range.second) {
        *func_def_input = absl::StrCat(tensor.node(), ":", output_name, ":",
                                       tensor.index() - output_range.first);
        return Status::OK();
      }
    }
  }

  return errors::InvalidArgument("Unknown graph def input: ", graph_def_input);
}

Status MakeFunctionDefHelper::AsFunctionDefNode(
    NodeDef* function_body_node) const {
  string func_def_input;

  for (int i = 0; i < function_body_node->input_size(); ++i) {
    TF_RETURN_IF_ERROR(
        AsFunctionDefInput(function_body_node->input(i), &func_def_input));
    function_body_node->set_input(i, func_def_input);
  }

  return Status::OK();
}

}  // namespace

Status MakeFunctionDef(const GrapplerFunctionItem& item,
                       const FunctionLibraryDefinition& flib,
                       FunctionDef* func) {
  func->mutable_signature()->set_name(item.id);
  func->mutable_signature()->set_description(item.description());
  func->mutable_signature()->set_is_stateful(item.is_stateful());

  MakeFunctionDefHelper helper;
  TF_RETURN_IF_ERROR(helper.Initialize(item, flib));

  // Mapping from the '_Retval' node name to the output tensor.
  absl::flat_hash_map<absl::string_view, string> output_tensors;
  for (const NodeDef& func_body_node : item.function_body().node()) {
    if (!helper.IsOutputNode(func_body_node)) continue;
    if (func_body_node.input_size() != 1) {
      return errors::Internal("_Retval node must have single input: ",
                              SummarizeNodeDef(func_body_node));
    }
    output_tensors.emplace(func_body_node.name(), func_body_node.input(0));
  }

  for (const InputArgInstantiation& input_arg : item.inputs()) {
    OpDef::ArgDef arg_def;
    arg_def.set_name(input_arg.node_name);
    arg_def.set_type(input_arg.data_type);
    arg_def.set_is_ref(IsRefType(input_arg.data_type));
    *func->mutable_signature()->add_input_arg() = arg_def;
  }

  // Add function output arguments.
  for (const OutputArgInstantiation& output_arg : item.outputs()) {
    const string output_name =
        absl::StrReplaceAll(output_arg.node_name, {{"_RetVal", ""}});

    OpDef::ArgDef arg_def;
    arg_def.set_name(output_name);
    arg_def.set_type(output_arg.data_type);
    arg_def.set_is_ref(IsRefType(output_arg.data_type));
    *func->mutable_signature()->add_output_arg() = arg_def;

    auto it = output_tensors.find(output_arg.node_name);
    if (it == output_tensors.end()) {
      return errors::Internal(
          "Can't find an output tensor for the output node: ",
          output_arg.node_name);
    }

    TF_RETURN_IF_ERROR(helper.AsFunctionDefInput(
        it->second, &(*func->mutable_ret())[output_name]));
  }

  // Add function control outputs.
  for (const ControlOutput& control_out : item.control_outputs()) {
    func->mutable_control_ret()->insert(
        {control_out.output_name, control_out.node_name});
    *func->mutable_signature()->add_control_output() = control_out.output_name;
  }

  // Copy function definition specific attributes.
  for (const auto& attr : item.func_attr()) {
    const auto& attr_name = attr.first;
    const auto& attr_value = attr.second;
    (*func->mutable_attr())[attr_name] = attr_value;
  }

  // Copy function arg attributes.
  for (int i = 0, end = item.arg_attr().size(); i < end; ++i) {
    const auto* attr = item.arg_attr().at(i);
    if (attr != nullptr) {
      (*func->mutable_arg_attr())[i] = *attr;
    }
  }

  // Copy function body nodes to the FunctionDef and update input format
  for (const NodeDef& func_node : item.function_body().node()) {
    // Skip original `_Arg` and `_Retval` nodes. If node was converted to some
    // other type (e.g. inputs converted to placeholders), we need to check that
    // it's not registered as function input or output node.
    if (IsArg(func_node) || IsRetval(func_node) ||
        helper.IsInputNode(func_node) || helper.IsOutputNode(func_node))
      continue;

    NodeDef* func_def_node = func->add_node_def();
    *func_def_node = func_node;
    TF_RETURN_IF_ERROR(helper.AsFunctionDefNode(func_def_node));
  }

  return Status::OK();
}

}  // end namespace grappler
}  // end namespace tensorflow
