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

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {
namespace {

Status SummarizeGraph(const GraphDef& graph) {
  std::vector<const NodeDef*> placeholders;
  for (const NodeDef& node : graph.node()) {
    if (node.op() == "Placeholder") {
      placeholders.push_back(&node);
    }
  }

  if (placeholders.empty()) {
    std::cout << "No inputs spotted." << std::endl;
  } else {
    std::cout << "Found " << placeholders.size() << " possible inputs: ";
    for (const NodeDef* node : placeholders) {
      TensorShape shape;
      if (node->attr().count("shape")) {
        TensorShapeProto shape_proto = node->attr().at("shape").shape();
        shape = TensorShape(shape_proto);
      }
      DataType dtype = node->attr().at("dtype").type();
      std::cout << "(name=" << node->name();
      std::cout << ", type=" << DataTypeString(dtype) << "(" << dtype << ")";
      std::cout << ", shape=" << shape.DebugString() << ") ";
    }
    std::cout << std::endl;
  }

  std::map<string, std::vector<const NodeDef*>> output_map;
  MapNodesToOutputs(graph, &output_map);
  std::vector<const NodeDef*> outputs;
  for (const NodeDef& node : graph.node()) {
    if ((output_map.count(node.name()) == 0) && (node.op() != "Const") &&
        (node.op() != "Assign")) {
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
    if (node.device() != "") {
      ++device_counts[node.device()];
    }
    if ((node.op() == "Const") || (node.op() == "Variable")) {
      Tensor tensor;
      if (tensor.FromProto(node.attr().at("value").tensor())) {
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

  return Status::OK();
}

int ParseFlagsAndSummarizeGraph(int argc, char* argv[]) {
  string in_graph = "";
  string out_graph = "";
  string inputs_string = "";
  string outputs_string = "";
  string transforms_string = "";
  std::vector<Flag> flag_list = {
      Flag("in_graph", &in_graph, "input graph file name"),
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
  Status load_status = ReadBinaryProto(Env::Default(), in_graph, &graph_def);
  if (!load_status.ok()) {
    LOG(ERROR) << "Loading graph '" << in_graph << "' failed with "
               << load_status.error_message();
    LOG(ERROR) << usage;
    return -1;
  }

  Status summarize_result = SummarizeGraph(graph_def);
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
