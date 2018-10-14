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

#include <unordered_map>

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
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/scanner.h"

namespace tensorflow {
namespace grappler {

namespace {

Status RegisterFunctionBodyOutputs(const OpRegistrationData& registration,
                                   const NodeDef& node,
                                   GrapplerFunctionConnectivity* connectivity) {
  tensorflow::NameRangeMap outputs_range_map;
  TF_RETURN_IF_ERROR(tensorflow::NameRangesForNode(
      node, registration.op_def, nullptr, &outputs_range_map));
  connectivity->RegisterFunctionBodyOutputs(node.name(),
                                            std::move(outputs_range_map));
  return Status::OK();
}

Status RegisterFunctionBodyOutputs(const FunctionLibraryDefinition& flib,
                                   const NodeDef& node,
                                   GrapplerFunctionConnectivity* connectivity) {
  const OpRegistrationData* registration;
  TF_RETURN_IF_ERROR(flib.LookUp(node.op(), &registration));
  return RegisterFunctionBodyOutputs(*registration, node, connectivity);
}

// Replace the placeholder attribute values with the values specified in
// instantiation attributes.
Status ResolveFunctionBodyNodeAttrPlaceholders(
    const AttrValueMap& func_instantiation_attr, NodeDef* node) {
  for (auto& attr : *node->mutable_attr()) {
    const string& placeholder = attr.second.placeholder();
    if (placeholder.empty()) continue;

    auto it = func_instantiation_attr.find(placeholder);
    if (it != func_instantiation_attr.end()) {
      attr.second = it->second;
    } else {
      return errors::InvalidArgument("Can't resolve placeholder: ",
                                     placeholder);
    }
  }
  return Status::OK();
}

}  // namespace

void GrapplerFunctionConnectivity::RegisterInputArgExpansion(
    InputArgExpansion input_arg_expansion) {
  string input_name = input_arg_expansion.input_name;
  const auto& placeholders = input_arg_expansion.placeholders;

  for (int i = 0; i < placeholders.size(); ++i) {
    const string& placeholder = input_arg_expansion.placeholders[i];
    input_arg_placeholders_.insert(
        {placeholder, InputArgPlaceholder{input_name, /*position=*/i}});
  }
  input_arg_expansions_.insert(
      {std::move(input_name), std::move(input_arg_expansion)});
}

void GrapplerFunctionConnectivity::RegisterFunctionBodyOutputs(
    const string& node_name, tensorflow::NameRangeMap&& outputs) {
  function_body_outputs_[node_name] = std::move(outputs);
}

Status GrapplerFunctionConnectivity::ExpandFunctionDefInput(
    const string& func_def_input, std::vector<string>* graph_def_inputs) const {
  using ::tensorflow::strings::Scanner;

  if (IsControlInput(func_def_input)) {
    graph_def_inputs->push_back(func_def_input);
    return Status::OK();
  }

  // Parse input format: "node_name[:node_output][:position]"
  string node_name;
  string node_output;
  int position = -1;

  StringPiece capture;
  StringPiece remaining;

  // Parse "node_name"
  if (Scanner(func_def_input)
          .One(strings::Scanner::LETTER_DIGIT_DOT_UNDERSCORE)
          .Any(strings::Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE)
          .GetResult(&remaining, &capture)) {
    node_name = string(capture.data(), capture.size());
  }

  // Parse "node_output" if it exists
  if (Scanner(remaining)
          .OneLiteral(":")
          .RestartCapture()
          .One(strings::Scanner::LETTER)
          .Any(strings::Scanner::LETTER_DIGIT_UNDERSCORE)
          .GetResult(&remaining, &capture)) {
    node_output = string(capture.data(), capture.size());
  }

  // Parse "position" if it exists
  if (Scanner(remaining)
          .OneLiteral(":")
          .RestartCapture()
          .Many(strings::Scanner::DIGIT)
          .GetResult(nullptr, &capture)) {
    CHECK(strings::safe_strto32(capture, &position));
  }

  // If "node_output" is not empty, it must be an output of a function body node
  bool is_function_body_output = !node_output.empty();

  // Function input argument: "node_name[:position]"
  if (!is_function_body_output) {
    auto input_arg = input_arg_expansions_.find(node_name);
    if (input_arg != input_arg_expansions_.end()) {
      const InputArgExpansion& input_arg_expansion = input_arg->second;
      const auto& placeholders = input_arg_expansion.placeholders;

      if (position == -1) {
        // If position is not defined use all placeholders
        graph_def_inputs->reserve(placeholders.size());
        for (const string& placeholder : placeholders) {
          graph_def_inputs->push_back(placeholder);
        }
      } else {
        if (position > input_arg_expansion.placeholders.size() - 1) {
          return errors::InvalidArgument("Invalid input ", node_name,
                                         "position: ", position,
                                         " (out of range)");
        }
        graph_def_inputs->push_back(input_arg_expansion.placeholders[position]);
      }

      return Status::OK();
    }
  }

  // Function body output: "node_name:node_output[:position]"
  if (is_function_body_output) {
    auto function_body_outputs = function_body_outputs_.find(node_name);
    if (function_body_outputs != function_body_outputs_.end()) {
      const tensorflow::NameRangeMap& outputs = function_body_outputs->second;
      auto output = outputs.find(node_output);
      if (output != outputs.end()) {
        const auto& output_range = output->second;

        if (position == -1) {
          graph_def_inputs->reserve(graph_def_inputs->size() +
                                    output_range.second - output_range.first);
          // If position is not defined expand node output range
          for (int i = output_range.first; i < output_range.second; ++i) {
            graph_def_inputs->push_back(
                i == 0 ? node_name : strings::StrCat(node_name, ":", i));
          }
        } else {
          if (position > (output_range.second - output_range.first)) {
            return errors::InvalidArgument(
                "Invalid node ", node_name, " output ", node_output,
                " position: ", position, " (out of range)");
          }
          int pos = output_range.first + position;
          graph_def_inputs->push_back(
              pos == 0 ? node_name : strings::StrCat(node_name, ":", pos));
        }

        return Status::OK();
      }
    }
  }

  return errors::InvalidArgument("Failed to expand a function def input: ",
                                 func_def_input);
}

Status GrapplerFunctionConnectivity::ExpandNodeInputs(
    NodeDef* function_body_node) const {
  std::vector<string> expanded_inputs;

  for (const string& function_def_input : function_body_node->input()) {
    TF_RETURN_IF_ERROR(
        ExpandFunctionDefInput(function_def_input, &expanded_inputs));
  }

  function_body_node->clear_input();
  for (string& expanded_input : expanded_inputs)
    function_body_node->add_input(std::move(expanded_input));
  return Status::OK();
}

Status GrapplerFunctionConnectivity::AsFunctionDefInput(
    const string& graph_def_input, string* func_def_input) const {
  using gtl::FindOrNull;

  if (IsControlInput(graph_def_input)) {
    *func_def_input = graph_def_input;
    return Status::OK();
  }

  int position;
  string node_name = ParseNodeName(graph_def_input, &position);
  CHECK_GE(position, 0);

  // Check if it's an input arg placeholder
  if (position == 0) {
    const InputArgPlaceholder* placeholder =
        FindOrNull(input_arg_placeholders_, node_name);
    if (placeholder != nullptr) {
      *func_def_input =
          strings::StrCat(placeholder->input_name, ":", placeholder->position);
      return Status::OK();
    }
  }

  // It must be output from one of the function body nodes
  const tensorflow::NameRangeMap* outputs_range_map =
      FindOrNull(function_body_outputs_, node_name);
  if (outputs_range_map != nullptr) {
    for (const auto& el : *outputs_range_map) {
      const auto& output_name = el.first;
      const auto& output_range = el.second;
      if (position >= output_range.first && position < output_range.second) {
        int pos = position - output_range.first;
        *func_def_input =
            strings::StrCat(node_name, ":", output_name, ":", pos);
        return Status::OK();
      }
    }
  }

  return errors::InvalidArgument("Unknown graph def input: ", graph_def_input);
}

Status GrapplerFunctionConnectivity::AsFunctionDefNode(
    NodeDef* function_body_node) const {
  string func_def_input;

  for (int i = 0; i < function_body_node->input_size(); ++i) {
    TF_RETURN_IF_ERROR(
        AsFunctionDefInput(function_body_node->input(i), &func_def_input));
    function_body_node->set_input(i, func_def_input);
  }

  return Status::OK();
}

Status GrapplerFunctionItemInstantiation::GetTypeAttr(
    const string& type_attr_name, DataType* data_type) const {
  auto it = func_instantiation_attr_->find(type_attr_name);
  if (it == func_instantiation_attr_->end()) {
    return errors::InvalidArgument("Type attribute ", type_attr_name,
                                   " is not defined");
  } else if (it->second.type() == DT_INVALID) {
    return errors::InvalidArgument("Type attribute ", type_attr_name,
                                   " is not defined with a valid type");
  } else {
    *data_type = it->second.type();
  }
  return Status::OK();
}

Status GrapplerFunctionItemInstantiation::GetArgType(
    const OpDef::ArgDef& arg, DataType* data_type) const {
  if (arg.type() != DT_INVALID) {
    *data_type = arg.type();
  } else {
    if (!arg.type_list_attr().empty() || !arg.number_attr().empty()) {
      return errors::InvalidArgument(
          "Arguments with sequence of tensors are not supported. Unsupported "
          "argument name: ",
          arg.name());
    }
    TF_RETURN_IF_ERROR(GetTypeAttr(arg.type_attr(), data_type));
  }
  return Status::OK();
}

GrapplerFunctionItem::GrapplerFunctionItem(
    string func_name, string description, AttrValueMap func_attr,
    std::vector<InputArgExpansion> input_arg_expansions,
    std::vector<OutputArgExpansion> output_arg_expansions,
    std::vector<string> keep_nodes, const int graph_def_version,
    const bool is_stateful, GraphDef&& function_body)
    : description_(std::move(description)),
      func_attr_(std::move(func_attr)),
      input_arg_expansions_(std::move(input_arg_expansions)),
      output_arg_expansions_(std::move(output_arg_expansions)),
      is_stateful_(is_stateful) {
  // Move assign GrapplerItem members.
  keep_ops = std::move(keep_nodes);
  id = std::move(func_name);
  graph = std::move(function_body);

  graph.mutable_versions()->set_producer(graph_def_version);
  // Fill the feed nodes with input placeholders.
  for (const InputArgExpansion& input_arg : input_arg_expansions_) {
    for (const string& placeholder : input_arg.placeholders) {
      feed.push_back({placeholder, Tensor()});
      input_arg_placeholders_.insert(placeholder);
    }
  }
  // Fill the fetch nodes with outputs.
  for (const OutputArgExpansion& output_arg : output_arg_expansions_) {
    for (const string& output_tensor : output_arg.output_tensors) {
      fetch.push_back(output_tensor);
    }
  }
  // Stateful and Send (it's not stateful) nodes must be preserved in the graph.
  for (const NodeDef& node : graph.node()) {
    if (IsSend(node)) {
      keep_ops.push_back(node.name());
    }
  }
}

const string& GrapplerFunctionItem::description() const { return description_; }

const std::vector<InputArgExpansion>& GrapplerFunctionItem::inputs() const {
  return input_arg_expansions_;
}

const InputArgExpansion& GrapplerFunctionItem::input(int i) const {
  return input_arg_expansions_[i];
}

const std::size_t GrapplerFunctionItem::input_size() const {
  return input_arg_expansions_.size();
}

bool GrapplerFunctionItem::IsInputPlaceholder(const string& node_name) const {
  return input_arg_placeholders_.find(node_name) !=
         input_arg_placeholders_.end();
}

const std::vector<OutputArgExpansion>& GrapplerFunctionItem::outputs() const {
  return output_arg_expansions_;
}

const OutputArgExpansion& GrapplerFunctionItem::output(int i) const {
  return output_arg_expansions_[i];
}

const std::size_t GrapplerFunctionItem::output_size() const {
  return output_arg_expansions_.size();
}

const AttrValueMap& GrapplerFunctionItem::func_attr() const {
  return func_attr_;
}

const GraphDef& GrapplerFunctionItem::function_body() const { return graph; }

GraphDef& GrapplerFunctionItem::mutable_function_body() { return graph; }

bool GrapplerFunctionItem::is_stateful() const { return is_stateful_; }

GrapplerFunctionItem& GrapplerFunctionItem::SwapFunctionBody(GraphDef&& other) {
  graph.Swap(&other);
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
    const FunctionDef& func, const AttrValueMap& func_instantiation_attr,
    std::unordered_map<string, DataType>* type_parameters) {
  if (!type_parameters->empty()) {
    return errors::InvalidArgument("Type parameters output map must be empty");
  }

  GrapplerFunctionItemInstantiation instantiation(&func_instantiation_attr);

  const auto resolve_type_attr = [&](const OpDef::ArgDef& arg) {
    // Check if it's unknown and unresolved type.
    if (arg.type() == DT_INVALID &&
        type_parameters->find(arg.type_attr()) == type_parameters->end()) {
      DataType data_type;
      TF_RETURN_IF_ERROR(instantiation.GetArgType(arg, &data_type));
      type_parameters->insert({arg.type_attr(), data_type});
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
    const FunctionDef& func, const AttrValueMap& func_instantiation_attr,
    std::unordered_map<string, AttrValue>* body_parameters) {
  if (!body_parameters->empty()) {
    return errors::InvalidArgument("Body parameters output map must be empty");
  }

  for (const NodeDef& func_body_node : func.node_def()) {
    for (auto& attr : func_body_node.attr()) {
      const string& placeholder = attr.second.placeholder();

      if (placeholder.empty() ||
          body_parameters->find(placeholder) != body_parameters->end()) {
        continue;
      }

      auto it = func_instantiation_attr.find(placeholder);
      if (it != func_instantiation_attr.end()) {
        body_parameters->insert({placeholder, it->second});
      } else {
        return errors::InvalidArgument("Can't resolve placeholder: ",
                                       placeholder);
      }
    }
  }

  return Status::OK();
}

Status MakeGrapplerFunctionItem(const FunctionDef& func,
                                const AttrValueMap& func_instantiation_attr,
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

  // Helper methods to lookup function instantiation attributes
  GrapplerFunctionItemInstantiation instantiation(&func_instantiation_attr);

  // Mapping from FunctionDef input format (name[:output][:position]) to
  // GraphDef input format (name[:position])
  GrapplerFunctionConnectivity connectivity;

  // Function body shares the library with the graph that instantiated it.
  GraphDef function_body;
  *function_body.mutable_library() = flib.ToProto();

  // TODO(ezhulenev): support functions with tensor sequence inputs/outputs

  // Make sure that there is no tensor sequences in outputs
  for (const OpDef::ArgDef& output : signature.output_arg()) {
    if (!output.type_list_attr().empty() || !output.number_attr().empty()) {
      return errors::InvalidArgument(
          "Outputs with sequence of tensors are not supported. Unsupported "
          "output: ",
          output.name());
    }
  }

  std::vector<InputArgExpansion> inputs;
  inputs.reserve(signature.input_arg_size());

  // For each input argument create a placeholder in function body.
  for (const OpDef::ArgDef& input : signature.input_arg()) {
    if (!input.type_list_attr().empty() || !input.number_attr().empty()) {
      return errors::InvalidArgument(
          "Inputs with sequence of tensors are not supported. Unsupported "
          "input: ",
          input.name());
    }

    DataType input_data_type;
    TF_RETURN_IF_ERROR(instantiation.GetArgType(input, &input_data_type));

    NodeDef* placeholder = function_body.add_node();
    placeholder->set_name(input.name());
    placeholder->set_op("Placeholder");
    (*placeholder->mutable_attr())["dtype"].set_type(input_data_type);
    (*placeholder->mutable_attr())["shape"].mutable_shape()->set_unknown_rank(
        true);

    InputArgExpansion input_expansion{/*input_name=*/input.name(),
                                      /*data_type=*/input_data_type,
                                      /*is_ref*/ input.is_ref(),
                                      /*placeholders=*/{input.name()}};
    connectivity.RegisterInputArgExpansion(input_expansion);
    inputs.push_back(std::move(input_expansion));
  }

  std::vector<string> keep_nodes;
  // Add all function nodes to the function body
  for (const NodeDef& func_def_node : func.node_def()) {
    NodeDef* new_node = function_body.add_node();
    *new_node = func_def_node;

    const OpRegistrationData* registration;
    TF_RETURN_IF_ERROR(flib.LookUp(func_def_node.op(), &registration));

    // Resolve all placeholder values using function instantiation attributes.
    TF_RETURN_IF_ERROR(ResolveFunctionBodyNodeAttrPlaceholders(
        func_instantiation_attr, new_node));

    // Register node output range in a function connectivity.
    TF_RETURN_IF_ERROR(RegisterFunctionBodyOutputs(*registration, func_def_node,
                                                   &connectivity));

    // Stateful and Send nodes must be preserved in a function body
    if (registration->op_def.is_stateful() || IsSend(func_def_node)) {
      keep_nodes.push_back(func_def_node.name());
    }
  }

  // Rewrite inputs to use GraphDef format
  for (NodeDef& node : *function_body.mutable_node()) {
    TF_RETURN_IF_ERROR(connectivity.ExpandNodeInputs(&node));
  }

  std::vector<OutputArgExpansion> outputs;
  outputs.reserve(signature.output_arg_size());
  // Add function outputs
  for (const OpDef::ArgDef& out : signature.output_arg()) {
    std::vector<string> output_tensors;
    auto ret = func.ret().find(out.name());
    TF_RETURN_IF_ERROR(
        ret != func.ret().end()
            // Expand outputs using provided output mapping
            ? connectivity.ExpandFunctionDefInput(ret->second, &output_tensors)
            // Otherwise output must be one of the function inputs
            : connectivity.ExpandFunctionDefInput(out.name(), &output_tensors));

    DataType output_data_type;
    TF_RETURN_IF_ERROR(instantiation.GetArgType(out, &output_data_type));

    OutputArgExpansion output{/*output_name=*/out.name(),
                              /*data_type=*/output_data_type,
                              /*is_ref=*/out.is_ref(),
                              /*output_tensors=*/std::move(output_tensors)};
    outputs.push_back(std::move(output));
  }

  bool is_stateful = signature.is_stateful();

  *item = GrapplerFunctionItem(
      /*func_name=*/signature.name(), /*description=*/signature.description(),
      /*func_attr=*/AttrValueMap(func.attr().begin(), func.attr().end()),
      std::move(inputs), std::move(outputs), std::move(keep_nodes),
      graph_def_version, is_stateful, std::move(function_body));
  return Status::OK();
}

Status MakeGrapplerFunctionItem(const FunctionDef& func,
                                const FunctionLibraryDefinition& flib,
                                const int graph_def_version,
                                GrapplerFunctionItem* item) {
  return MakeGrapplerFunctionItem(func, AttrValueMap(), flib, graph_def_version,
                                  item);
}

// Register GrapplerFunctionItem input arg expansion and function body outputs
// in the GrapplerFunctionConnectivity.
Status RegisterGrapplerFunctionConnectivity(
    const GrapplerFunctionItem& item, const FunctionLibraryDefinition& flib,
    GrapplerFunctionConnectivity* connectivity) {
  for (const InputArgExpansion& input : item.inputs()) {
    connectivity->RegisterInputArgExpansion(input);
  }
  for (const NodeDef& func_body_node : item.function_body().node()) {
    TF_RETURN_IF_ERROR(
        RegisterFunctionBodyOutputs(flib, func_body_node, connectivity));
  }
  return Status::OK();
}

Status ReplaceInputWithConst(const NodeDef& input_const, int input_position,
                             GrapplerFunctionItem* item) {
  if (!IsConstant(input_const)) {
    return errors::InvalidArgument("Input node ", input_const.name(),
                                   " is not a constant");
  }

  auto& inputs = item->input_arg_expansions_;

  // Find input arg expansion and input placeholder position in it for the
  // given function input position.
  InputArgExpansion* input_arg_expansion = nullptr;
  int placeholder_idx = input_position;

  for (InputArgExpansion& input : inputs) {
    if (placeholder_idx < input.placeholders.size()) {
      input_arg_expansion = &input;
      break;
    }
    placeholder_idx -= input.placeholders.size();
  }

  if (input_arg_expansion == nullptr) {
    return errors::InvalidArgument(
        "Input placeholder not found: input_position=", input_position,
        " function=", item->id);
  }

  // Delete placeholder from input expansion.
  string placeholder_name = input_arg_expansion->placeholders[placeholder_idx];
  item->input_arg_placeholders_.erase(placeholder_name);
  input_arg_expansion->placeholders.erase(
      input_arg_expansion->placeholders.begin() + placeholder_idx);

  // Delete empty input expansions.
  inputs.erase(std::remove_if(inputs.begin(), inputs.end(),
                              [](const InputArgExpansion& input) {
                                return input.placeholders.empty();
                              }),
               inputs.end());

  // Replace placeholder node in the function body with a const node.
  for (NodeDef& node : *item->graph.mutable_node()) {
    if (node.name() == placeholder_name) {
      node = input_const;
      node.set_name(placeholder_name);
      node.clear_input();   // remove potential control inputs
      node.clear_device();  // device placement is defined by instantiating node
    }
  }

  return Status::OK();
}

Status MakeFunctionDef(const GrapplerFunctionItem& item,
                       const FunctionLibraryDefinition& flib,
                       FunctionDef* func) {
  func->mutable_signature()->set_name(item.id);
  func->mutable_signature()->set_description(item.description());
  func->mutable_signature()->set_is_stateful(item.is_stateful());

  // Build a GrapplerFunctionConnectivity from inputs and new function body.
  GrapplerFunctionConnectivity connectivity;
  TF_RETURN_IF_ERROR(
      RegisterGrapplerFunctionConnectivity(item, flib, &connectivity));

  // Add function input arguments.
  for (const InputArgExpansion& input_arg : item.inputs()) {
    CHECK(input_arg.placeholders.size() == 1)  // do some sanity checking
        << "Inputs of tensor sequences are not supported";

    OpDef::ArgDef arg_def;
    arg_def.set_name(input_arg.input_name);
    arg_def.set_type(input_arg.data_type);
    arg_def.set_is_ref(input_arg.is_ref);
    *func->mutable_signature()->add_input_arg() = arg_def;
  }

  // Add function output arguments.
  for (const OutputArgExpansion& output_arg : item.outputs()) {
    CHECK(output_arg.output_tensors.size() == 1)  // do some sanity checking
        << "Outputs of tensor sequences are not supported";

    OpDef::ArgDef arg_def;
    arg_def.set_name(output_arg.output_name);
    arg_def.set_type(output_arg.data_type);
    arg_def.set_is_ref(output_arg.is_ref);
    *func->mutable_signature()->add_output_arg() = arg_def;

    string ret;
    for (const string& output_tensor : output_arg.output_tensors) {
      TF_RETURN_IF_ERROR(connectivity.AsFunctionDefInput(output_tensor, &ret));
      (*func->mutable_ret())[output_arg.output_name] = ret;
    }
  }

  // Copy function definition specific attributes.
  for (const auto& attr : item.func_attr()) {
    const auto& attr_name = attr.first;
    const auto& attr_value = attr.second;
    (*func->mutable_attr())[attr_name] = attr_value;
  }

  // Copy function body nodes to the FunctionDef and update input format
  for (const NodeDef& func_body_node : item.function_body().node()) {
    // Do not copy input placeholders
    if (item.IsInputPlaceholder(func_body_node.name())) continue;

    NodeDef* func_def_node = func->add_node_def();
    *func_def_node = func_body_node;
    TF_RETURN_IF_ERROR(connectivity.AsFunctionDefNode(func_def_node));
  }

  return Status::OK();
}

}  // end namespace grappler
}  // end namespace tensorflow
