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
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/strings/scanner.h"

namespace tensorflow {
namespace grappler {

void GrapplerFunctionConnectivity::RegisterInputArgExpansion(
    const InputArgExpansion& input_arg_expansion) {
  input_arg_expansions_.insert(
      {input_arg_expansion.input_name, input_arg_expansion});
}

void GrapplerFunctionConnectivity::RegisterFunctionBodyOutputs(
    const string& node_name, const tensorflow::NameRangeMap& outputs) {
  function_body_outputs_.insert({node_name, outputs});
}

Status GrapplerFunctionConnectivity::ExpandFunctionDefInput(
    const string& func_def_input, std::vector<string>* graph_def_inputs) const {
  using ::tensorflow::strings::Scanner;

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
          .One(strings::Scanner::LOWERLETTER)
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
          // If position is not defined expand node output range
          for (int i = output_range.first; i < output_range.second; ++i) {
            i == 0 ? graph_def_inputs->push_back(node_name)
                   : graph_def_inputs->push_back(
                         strings::StrCat(node_name, ":", i));
          }
        } else {
          if (position > (output_range.second - output_range.first)) {
            return errors::InvalidArgument(
                "Invalid node ", node_name, " output ", node_output,
                " position: ", position, " (out of range)");
          }
          int pos = output_range.first + position;
          pos == 0 ? graph_def_inputs->push_back(node_name)
                   : graph_def_inputs->push_back(
                         strings::StrCat(node_name, ":", pos));
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
    if (!IsControlInput(function_def_input))
      TF_RETURN_IF_ERROR(
          ExpandFunctionDefInput(function_def_input, &expanded_inputs));
    else
      expanded_inputs.push_back(function_def_input);
  }

  function_body_node->clear_input();
  for (const string& expanded_input : expanded_inputs)
    function_body_node->add_input(expanded_input);
  return Status::OK();
}

Status GrapplerFunctionItemBuilder::GetTypeAttr(const string& type_attr_name,
                                                DataType* data_type) const {
  auto it = func_attr_->find(type_attr_name);
  if (it == func_attr_->end()) {
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

Status GrapplerFunctionItemBuilder::GetArgType(const OpDef::ArgDef& arg,
                                               DataType* data_type) const {
  if (arg.type() != DT_INVALID) {
    *data_type = arg.type();
  } else {
    TF_RETURN_IF_ERROR(GetTypeAttr(arg.type_attr(), data_type));
  }
  return Status::OK();
}

GrapplerFunctionItem::GrapplerFunctionItem(
    const string& function_name,
    const std::vector<InputArgExpansion>& input_arg_expansions,
    const std::vector<OutputArgExpansion>& output_arg_expansions,
    GraphDef&& function_body)
    : function_name_(function_name),
      input_arg_expansions_(input_arg_expansions),
      output_arg_expansions_(output_arg_expansions) {
  graph.Swap(&function_body);
}

const string& GrapplerFunctionItem::function_name() const {
  return function_name_;
}

const std::vector<InputArgExpansion>& GrapplerFunctionItem::inputs() const {
  return input_arg_expansions_;
}

const InputArgExpansion& GrapplerFunctionItem::input(int i) const {
  return input_arg_expansions_[i];
}

const std::size_t GrapplerFunctionItem::input_size() const {
  return input_arg_expansions_.size();
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

const GraphDef& GrapplerFunctionItem::function_body() const { return graph; }

GraphDef& GrapplerFunctionItem::mutable_function_body() { return graph; }

std::vector<string> OutputTensors(const GrapplerFunctionItem& item) {
  std::vector<string> output_tensors;
  for (const OutputArgExpansion& output : item.outputs()) {
    for (const string& tensor : output.output_tensors) {
      output_tensors.push_back(tensor);
    }
  }
  return output_tensors;
}

Status MakeGrapplerFunctionItem(
    const FunctionDef& func,
    const std::unordered_map<string, AttrValue>& func_attr,
    const FunctionLibraryDefinition& func_library, GrapplerFunctionItem* item) {
  const OpDef& signature = func.signature();

  if (signature.name().empty()) {
    return errors::InvalidArgument("Function name must be specified");
  }

  // Helper methods to lookup function attributes
  GrapplerFunctionItemBuilder builder(&func_attr);

  // Mapping from FunctionDef input format (name[:output][:position]) to
  // GraphDef input format (name[:position])
  GrapplerFunctionConnectivity connectivity;

  std::vector<InputArgExpansion> inputs;
  std::vector<OutputArgExpansion> outputs;
  GraphDef function_body;

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

  // For each input argument create a placeholder in function body.
  for (const OpDef::ArgDef& input : signature.input_arg()) {
    if (!input.type_list_attr().empty() || !input.number_attr().empty()) {
      return errors::InvalidArgument(
          "Inputs with sequence of tensors are not supported. Unsupported "
          "input: ",
          input.name());
    }

    DataType input_data_type;
    TF_RETURN_IF_ERROR(builder.GetArgType(input, &input_data_type));

    NodeDef* placeholder = function_body.add_node();
    placeholder->set_name(input.name());
    placeholder->set_op("Placeholder");
    (*placeholder->mutable_attr())["T"].set_type(input_data_type);

    InputArgExpansion input_expansion{/*input_name=*/input.name(),
                                      /*placeholders=*/{input.name()}};
    connectivity.RegisterInputArgExpansion(input_expansion);
    inputs.push_back(input_expansion);
  }

  // Add all function nodes to the function body
  for (const NodeDef& func_def_node : func.node_def()) {
    NodeDef* new_node = function_body.add_node();
    *new_node = func_def_node;

    // Replace the placeholder attribute values with the specified value
    for (auto& attr : *new_node->mutable_attr()) {
      const string& ph_name = attr.second.placeholder();
      auto it = func_attr.find(ph_name);
      if (it != func_attr.end()) {
        attr.second = it->second;
      }
    }

    // Functions use a custom format to encode connectivity. Map these custom
    // strings to regular ones.
    tensorflow::NameRangeMap outputs_range_map;
    const OpRegistrationData* registration;
    TF_RETURN_IF_ERROR(func_library.LookUp(func_def_node.op(), &registration));
    TF_RETURN_IF_ERROR(tensorflow::NameRangesForNode(
        func_def_node, registration->op_def, nullptr, &outputs_range_map));
    connectivity.RegisterFunctionBodyOutputs(func_def_node.name(),
                                             outputs_range_map);
  }

  // Rewrite inputs to use GraphDef format
  for (NodeDef& node : *function_body.mutable_node()) {
    TF_RETURN_IF_ERROR(connectivity.ExpandNodeInputs(&node));
  }

  // Add function outputs
  for (const OpDef::ArgDef& out : signature.output_arg()) {
    std::vector<string> output_tensors;
    auto ret = func.ret().find(out.name());
    if (ret != func.ret().end()) {
      // Expand outputs using provided output mapping
      TF_RETURN_IF_ERROR(
          connectivity.ExpandFunctionDefInput(ret->second, &output_tensors));
    } else {
      // Otherwise output must be one of the function inputs
      TF_RETURN_IF_ERROR(
          connectivity.ExpandFunctionDefInput(out.name(), &output_tensors));
    }
    outputs.push_back({out.name(), output_tensors});
  }

  *item = GrapplerFunctionItem(signature.name(), inputs, outputs,
                               std::move(function_body));
  return Status::OK();
}

}  // end namespace grappler
}  // end namespace tensorflow
