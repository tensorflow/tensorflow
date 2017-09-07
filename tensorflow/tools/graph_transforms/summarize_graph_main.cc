/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// This program prints out a summary of a GraphDef file's contents, listing
// things that are useful for debugging and reusing the model it contains. For
// example it looks at the graph structure and op types to figure out likely
// input and output nodes, and shows which ops are used by the graph. To use it,
// run something like this:
//
// bazel build tensorflow/tools/graph_transforms:summarize_graph
// bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
// --in_graph=my_graph.pb

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/file_utils.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {
namespace {

void PrintNodeInfo(const NodeDef* node) {
  string shape_description = "None";
  if (node->attr().count("shape")) {
    TensorShapeProto shape_proto = node->attr().at("shape").shape();
    Status shape_status = PartialTensorShape::IsValidShape(shape_proto);
    if (shape_status.ok()) {
      shape_description = PartialTensorShape(shape_proto).DebugString();
    } else {
      shape_description = shape_status.error_message();
    }
  }
  DataType dtype = DT_INVALID;
  if (node->attr().count("dtype")) {
    dtype = node->attr().at("dtype").type();
  }
  std::cout << "(name=" << node->name();
  std::cout << ", type=" << DataTypeString(dtype) << "(" << dtype << ")";
  std::cout << ", shape=" << shape_description << ") ";
}

void PrintBenchmarkUsage(const std::vector<const NodeDef*>& placeholders,
                         const std::vector<const NodeDef*>& variables,
                         const std::vector<const NodeDef*> outputs,
                         const string& graph_path) {
  std::vector<const NodeDef*> all_inputs(placeholders);
  all_inputs.insert(all_inputs.end(), variables.begin(), variables.end());

  std::vector<string> input_layers;
  std::vector<string> input_layer_types;
  std::vector<string> input_layer_shapes;
  for (const NodeDef* node : all_inputs) {
    input_layers.push_back(node->name());
    DataType dtype = DT_INVALID;
    if (node->attr().count("dtype")) {
      dtype = node->attr().at("dtype").type();
    }
    input_layer_types.push_back(DataTypeString(dtype));
    std::vector<int64> sizes;
    PartialTensorShape shape;
    if (node->attr().count("shape")) {
      TensorShapeProto shape_proto = node->attr().at("shape").shape();
      if (PartialTensorShape::IsValid(shape_proto)) {
        shape = PartialTensorShape(shape_proto);
      }
    }
    string sizes_string;
    if (shape.dims() == -1) {
      // Unknown shapes can have -1 for dims, so leave these blank.
      sizes_string = "";
    } else {
      sizes.reserve(shape.dims());
      for (int i = 0; i < shape.dims(); ++i) {
        sizes.push_back(shape.dim_size(i));
      }
      sizes_string = str_util::Join(sizes, ",");
    }
    input_layer_shapes.push_back(sizes_string);
  }
  std::vector<string> output_layers;
  output_layers.reserve(outputs.size());
  for (const NodeDef* node : outputs) {
    output_layers.push_back(node->name());
  }
  string input_layer_value = str_util::Join(input_layers, ",");
  string input_layer_type_value = str_util::Join(input_layer_types, ",");
  string input_layer_shape_value = str_util::Join(input_layer_shapes, ":");
  string output_layer_value = str_util::Join(output_layers, ",");

  std::cout << "To use with tensorflow/tools/benchmark:benchmark_model try "
               "these arguments:"
            << std::endl;
  std::cout << "bazel run tensorflow/tools/benchmark:benchmark_model --";
  std::cout << " --graph=" << graph_path;
  std::cout << " --show_flops";
  std::cout << " --input_layer=" << input_layer_value;
  std::cout << " --input_layer_type=" << input_layer_type_value;
  std::cout << " --input_layer_shape=" << input_layer_shape_value;
  std::cout << " --output_layer=" << output_layer_value;
  std::cout << std::endl;
}

Status PrintStructure(const GraphDef& graph) {
  GraphDef sorted_graph;
  TF_RETURN_IF_ERROR(SortByExecutionOrder(graph, &sorted_graph));
  for (const NodeDef& node : sorted_graph.node()) {
    std::cout << node.name() << " (" << node.op() << "): ["
              << str_util::Join(node.input(), ", ") << "]";
    if (node.op() == "Const") {
      Tensor tensor;
      if (node.attr().count("value") &&
          tensor.FromProto(node.attr().at("value").tensor())) {
        std::cout << ", value=" << tensor.DebugString();
      } else {
        LOG(WARNING) << "Decoding Tensor failed for node" << node.name();
      }
    }
    std::cout << std::endl;
  }
  return Status::OK();
}

Status SummarizeGraph(const GraphDef& graph, const string& graph_path,
                      bool print_structure) {
  std::vector<const NodeDef*> placeholders;
  std::vector<const NodeDef*> variables;
  for (const NodeDef& node : graph.node()) {
    if (node.op() == "Placeholder") {
      placeholders.push_back(&node);
    }
    if (node.op() == "Variable" || node.op() == "VariableV2") {
      variables.push_back(&node);
    }
  }

  if (placeholders.empty()) {
    std::cout << "No inputs spotted." << std::endl;
  } else {
    std::cout << "Found " << placeholders.size() << " possible inputs: ";
    for (const NodeDef* node : placeholders) {
      PrintNodeInfo(node);
    }
    std::cout << std::endl;
  }

  if (variables.empty()) {
    std::cout << "No variables spotted." << std::endl;
  } else {
    std::cout << "Found " << variables.size() << " variables: ";
    for (const NodeDef* node : variables) {
      PrintNodeInfo(node);
    }
    std::cout << std::endl;
  }

  std::map<string, std::vector<const NodeDef*>> output_map;
  MapNodesToOutputs(graph, &output_map);
  std::vector<const NodeDef*> outputs;
  std::unordered_set<string> unlikely_output_types = {"Const", "Assign", "NoOp",
                                                      "Placeholder"};
  for (const NodeDef& node : graph.node()) {
    if ((output_map.count(node.name()) == 0) &&
        (unlikely_output_types.count(node.op()) == 0)) {
      outputs.push_back(&node);
    }
  }

  if (outputs.empty()) {
    std::cout << "No outputs spotted." << std::endl;
  } else {
    std::cout << "Found " << outputs.size() << " possible outputs: ";
    for (const NodeDef* node : outputs) {
      std::cout << "(name=" << node->name();
      std::cout << ", op=" << node->op() << ") ";
    }
    std::cout << std::endl;
  }

  int64 const_parameter_count = 0;
  int64 variable_parameter_count = 0;
  int control_edge_count = 0;
  std::map<string, int> device_counts;
  for (const NodeDef& node : graph.node()) {
    for (const string& input : node.input()) {
      if (input.substr(0, 1) == "^") {
        ++control_edge_count;
      }
    }
    if (!node.device().empty()) {
      ++device_counts[node.device()];
    }
    if ((node.op() == "Const") || (node.op() == "Variable") ||
        (node.op() == "VariableV2")) {
      Tensor tensor;
      if (node.attr().count("value") &&
          tensor.FromProto(node.attr().at("value").tensor())) {
        const size_t num_elements = tensor.NumElements();
        if (node.op() == "Const") {
          const_parameter_count += num_elements;
        } else {
          variable_parameter_count += num_elements;
        }
      } else {
        LOG(WARNING) << "Decoding Tensor failed for node" << node.name();
      }
    }
  }

  std::cout << "Found " << const_parameter_count << " ("
            << strings::HumanReadableNum(const_parameter_count)
            << ") const parameters, " << variable_parameter_count << " ("
            << strings::HumanReadableNum(variable_parameter_count)
            << ") variable parameters, and " << control_edge_count
            << " control_edges" << std::endl;
  if (!device_counts.empty()) {
    for (const auto& device_info : device_counts) {
      std::cout << device_info.second << " nodes assigned to device '"
                << device_info.first << "'";
    }
  }

  std::vector<std::pair<string, string>> invalid_inputs;
  FindInvalidInputs(graph, &invalid_inputs);
  if (!invalid_inputs.empty()) {
    for (const std::pair<string, string>& invalid_input : invalid_inputs) {
      std::cout << "Invalid input " << invalid_input.second << " for node "
                << invalid_input.first << std::endl;
    }
    return errors::Internal(
        "Invalid graph with inputs referring to nonexistent nodes");
  }

  std::map<string, int> op_counts;
  for (const NodeDef& node : graph.node()) {
    ++op_counts[node.op()];
  }
  for (const FunctionDef& function : graph.library().function()) {
    for (const NodeDef& node : function.node_def()) {
      ++op_counts[node.op()];
    }
  }
  std::vector<std::pair<string, int>> op_counts_vec(op_counts.begin(),
                                                    op_counts.end());
  std::sort(op_counts_vec.begin(), op_counts_vec.end(),
            [](std::pair<string, int> a, std::pair<string, int> b) {
              return (a.second > b.second);
            });
  std::cout << "Op types used: ";
  bool is_first = true;
  for (const std::pair<string, int>& op_count : op_counts_vec) {
    if (!is_first) {
      std::cout << ", ";
    } else {
      is_first = false;
    }
    std::cout << op_count.second << " " << op_count.first;
  }
  std::cout << std::endl;

  PrintBenchmarkUsage(placeholders, variables, outputs, graph_path);

  if (print_structure) {
    TF_RETURN_IF_ERROR(PrintStructure(graph));
  }

  return Status::OK();
}

int ParseFlagsAndSummarizeGraph(int argc, char* argv[]) {
  string in_graph = "";
  bool print_structure = false;
  std::vector<Flag> flag_list = {
      Flag("in_graph", &in_graph, "input graph file name"),
      Flag("print_structure", &print_structure,
           "whether to print the network connections of the graph"),
  };
  string usage = Flags::Usage(argv[0], flag_list);

  const bool parse_result = Flags::Parse(&argc, argv, flag_list);
  // We need to call this to set up global state for TensorFlow.
  port::InitMain(argv[0], &argc, &argv);

  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << ".\n" << usage;
    return -1;
  }
  if (in_graph.empty()) {
    LOG(ERROR) << "in_graph graph can't be empty.\n" << usage;
    return -1;
  }

  GraphDef graph_def;
  Status load_status = LoadTextOrBinaryGraphFile(in_graph, &graph_def);
  if (!load_status.ok()) {
    LOG(ERROR) << "Loading graph '" << in_graph << "' failed with "
               << load_status.error_message();
    LOG(ERROR) << usage;
    return -1;
  }

  Status summarize_result =
      SummarizeGraph(graph_def, in_graph, print_structure);
  if (!summarize_result.ok()) {
    LOG(ERROR) << summarize_result.error_message() << "\n" << usage;
    return -1;
  }

  return 0;
}

}  // namespace
}  // namespace graph_transforms
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  return tensorflow::graph_transforms::ParseFlagsAndSummarizeGraph(argc, argv);
}
