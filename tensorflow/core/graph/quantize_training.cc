/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include <algorithm>
#include <atomic>
#include <set>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/graph/quantize_training.h"

#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {
// Node types to rewrite. Insert quantize_and_dequantize op for their inputs.
const std::unordered_set<string, StringPiece::Hasher> nodes_to_rewrite{
    "MatMul", "Conv2D"};

// Contains necessary parameters to convert an edge.
struct EdgeToConvert {
  // Edge is not owned here.
  const Edge* edge;
  int32 num_bits;
  bool signed_input;
  bool range_given;
  float input_min;
  float input_max;

  EdgeToConvert(const Edge* e, int32 bits, bool sign, bool range, float min,
                float max) {
    edge = e;
    num_bits = bits;
    signed_input = sign;
    range_given = range;
    input_min = min;
    input_max = max;
  }
};

// Decide if a node is in backward pass by checking if its name is led by
// "gradients".
// TODO(jmchen): Make this check more robust as it is not guaranteed that the
// forward node will not be named with a leading "gradients".
inline bool IsGradientNode(const Graph* graph, const Node* node) {
  static const string tag = "gradients";
  return (node->name().compare(0, tag.size(), tag) == 0);
}

// Find the type of the input to set the parameters for the
// quantize_and_dequantize op.
// Returns true if the root tensor op type is known, false otherwise.
bool FindType(const Graph* graph, const Node* node, bool* signed_input,
              bool* range_given, float* input_min, float* input_max) {
  const string src_op = node->type_string();
  if (src_op == "Const" || src_op == "Variable") {
    *signed_input = true;
    *range_given = false;
  } else if (src_op == "Relu") {
    // Range is not given for Relu.
    *signed_input = false;
    *range_given = false;
  } else if (src_op == "Relu6") {
    *signed_input = false;
    *range_given = true;
    *input_min = 0;
    *input_max = 6;
  } else if (src_op == "Sigmoid") {
    *signed_input = false;
    *range_given = true;
    *input_min = 0;
    *input_max = 1;
  } else if (src_op == "Tanh") {
    *signed_input = true;
    *range_given = true;
    *input_min = -1;
    *input_max = 1;
  } else if (src_op == "Reshape") {
    // Reshape has 2 inputs and the first one is the tensor.
    for (const Edge* edge : node->in_edges()) {
      if (edge->src_output() != Graph::kControlSlot && edge->dst_input() == 0) {
        FindType(graph, edge->src(), signed_input, range_given, input_min,
                 input_max);
      }
    }
  } else if (src_op == "Identity" || src_op == "MaxPool" ||
             src_op == "AvgPool" || src_op == "MaxPool3D" ||
             src_op == "AvgPool3D") {
    // All these Ops only have 1 data input.
    for (const Edge* edge : node->in_edges()) {
      if (edge->src_output() != Graph::kControlSlot) {
        FindType(graph, edge->src(), signed_input, range_given, input_min,
                 input_max);
      }
    }
  } else {
    // Unknown type, could be the model input examples.
    // TODO: Set the params for input with user's hint.
    *signed_input = true;
    *range_given = false;
    return false;
  }

  return true;
}

// Insert conversion op, connect it to the graph and remove the old edge.
Status ProcessTargetEdges(Graph* graph,
                          const std::vector<EdgeToConvert>& target_edges) {
  // Remember previous convert ops to avoid duplicated conversion on the same
  // input.
  std::unordered_map<string, Node*, StringPiece::Hasher> name_index;
  for (const EdgeToConvert edge : target_edges) {
    Node* convert_node;
    string name =
        strings::StrCat(edge.edge->src()->name(), "/QuantizeAndDequantize");

    auto iter = name_index.find(name);
    if (iter == name_index.end()) {
      TF_RETURN_IF_ERROR(NodeBuilder(name, "QuantizeAndDequantize")
                             .Input(edge.edge->src())
                             .Attr("signed_input", edge.signed_input)
                             .Attr("num_bits", edge.num_bits)
                             .Attr("range_given", edge.range_given)
                             .Attr("input_min", edge.input_min)
                             .Attr("input_max", edge.input_max)
                             .Finalize(graph, &convert_node));

      name_index[name] = convert_node;
    } else {
      convert_node = iter->second;
    }

    graph->AddEdge(convert_node, 0, edge.edge->dst(), edge.edge->dst_input());
    graph->RemoveEdge(edge.edge);
  }

  return Status::OK();
}

}  // namespace

Status DoQuantizeTraining(int32 num_bits, Graph* graph) {
  if (graph == nullptr) {
    return errors::InvalidArgument("Cannot accept empty graph pointer.");
  }

  if (num_bits < 1 || num_bits > 63) {
    return errors::OutOfRange("num_bits should be in range [1, 63] but is: ",
                              num_bits);
  }
  int potential_input = 0;
  std::vector<EdgeToConvert> target_edges;
  for (Node* node : graph->nodes()) {
    if (nodes_to_rewrite.find(node->type_string()) != nodes_to_rewrite.end() &&
        !IsGradientNode(graph, node)) {
      // Find out which types are the inputs and convert them accordingly.
      // 1. Const/Variable OP: This is quantized as signed tensors with no given
      // range.
      // 2. Activation OP: Set the range accordingly for different types of
      // activations. Currently we handle {Relu, Relu6, Sigmoid, Tanh}
      // 3. Identity OP: The quantization parameters depend on its input.
      // 4. Pooling OPs: various pooling ops. Also depends on its input.
      // 5. Reshape OP: Also depends on the first input to this op.
      // 6. Not-Listed-Above OP: If there is only 1 such op, consider it as the
      // model input. However, if there are >1 unknown ops, then returns an
      // error for now to avoid unexpected bahavior.
      // Note: The list above might not be a complete list. Please let us
      // know if you see the error so we can handle your case.
      for (const Edge* edge : node->in_edges()) {
        if (edge->src_output() == Graph::kControlSlot) {
          // Skip the control dependency input.
          continue;
        } else {
          bool signed_input = false;
          bool range_given = false;
          float input_min = 0;
          float input_max = 0;
          bool known_op = FindType(graph, edge->src(), &signed_input,
                                   &range_given, &input_min, &input_max);
          if (!known_op) {
            // Unknown op is considered as input.
            // Only support one input for now.
            // TODO: Make this configurable if this is the desirable way to find
            // input.
            if (potential_input > 0) {
              return errors::Unimplemented(
                  "Find a second unknown op: ", edge->src()->name(),
                  " with type: ", edge->src()->type_string(),
                  "; Unknown ops are considered as model input for now and "
                  "only 1 input is supported currently.");
            }
            potential_input++;
          }

          target_edges.emplace_back(EdgeToConvert(
              edge, num_bits, signed_input, range_given, input_min, input_max));
        }
      }
    }
  }

  TF_RETURN_IF_ERROR(ProcessTargetEdges(graph, target_edges));

  return Status::OK();
}

Status DoQuantizeTrainingOnSerializedGraphDef(const string& input_graph,
                                              int32 num_bits,
                                              string* result_graph) {
  // First create the graph from the GraphDef.
  Graph graph(OpRegistry::Global());
  GraphConstructorOptions opts;
  GraphDef input_graphdef;
  if (!ParseProtoUnlimited(&input_graphdef, input_graph)) {
    return errors::InvalidArgument("Invalid input graph");
  }
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, input_graphdef, &graph));

  // Call the rewriter on the graph.
  TF_RETURN_IF_ERROR(DoQuantizeTraining(num_bits, &graph));

  // Convert the result graph back to a GraphDef.
  GraphDef output_graphdef;
  graph.ToGraphDef(&output_graphdef);

  if (!output_graphdef.SerializeToString(result_graph)) {
    return errors::InvalidArgument("Invalid output graph");
  }
  return Status::OK();
}

}  // namespace tensorflow
