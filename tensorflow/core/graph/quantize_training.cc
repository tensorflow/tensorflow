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

// TODO(suharshs): If desired, make these values configurable.
const uint32 kAllowedInputs = 2;
const float kEMADecay = 0.999;

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
                float max)
      : edge(e),
        num_bits(bits),
        signed_input(sign),
        range_given(range),
        input_min(min),
        input_max(max) {}
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
  const string& src_op = node->type_string();
  if (src_op == "Const" || src_op == "Variable" || src_op == "VariableV2") {
    *signed_input = true;
    *range_given = false;
  } else if (src_op == "Relu") {
    // Range is not given for Relu.
    *signed_input = false;
    *range_given = false;
  } else if (src_op == "Relu6") {
    // TODO(suharshs): Also the theoretical min and max is 0 and 6, if the
    // actual activations are somewhere in within this range, we can quantize
    // this even further. This is true for other activations like Sigmoid6 too.
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
    // TODO(jmchen): Set the params for input with user's hint.
    *signed_input = true;
    *range_given = false;
    return false;
  }

  return true;
}

// Sets output to the Node that computes reduction axes corresponding to all
// dimensions of input and return.
Status MakeReductionAxes(Graph* graph, string name_prefix, Node* input,
                         Node** output) {
  name_prefix = strings::StrCat(name_prefix, "/ReductionAxes");
  Node* start;
  Tensor zero_tensor(DT_INT32, TensorShape());
  zero_tensor.flat<int32>()(0) = 0;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/RangeStart"), "Const")
          .Attr("dtype", DT_INT32)
          .Attr("value", zero_tensor)
          .Finalize(graph, &start));
  Node* delta;
  Tensor one_tensor(DT_INT32, TensorShape());
  one_tensor.flat<int32>()(0) = 1;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/RangeDelta"), "Const")
          .Attr("dtype", DT_INT32)
          .Attr("value", one_tensor)
          .Finalize(graph, &delta));
  Node* rank;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/InputRank"), "Rank")
          .Input(input)
          .Finalize(graph, &rank));
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/ReductionAxes"), "Range")
          .Input(start)
          .Input(rank)
          .Input(delta)
          .Finalize(graph, output));
  return Status::OK();
}

// Computes the exponential moving average of input, updated in update_variable.
Status MakeExponentialMovingAverage(Graph* graph, string name_prefix,
                                    const NodeBuilder::NodeOut& input,
                                    Node* decay, Node* update_variable,
                                    Node** assign_value) {
  // variable_t+1 = variable_t - [(variable_t - value) * (1 - decay)]
  name_prefix = strings::StrCat(name_prefix, "/EMA");
  Node* one;
  Tensor one_tensor(DT_FLOAT, TensorShape());
  one_tensor.flat<float>()(0) = 1.0;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/OneConst"), "Const")
          .Attr("dtype", DT_FLOAT)
          .Attr("value", one_tensor)
          .Finalize(graph, &one));
  Node* decay_complement;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/DecayComplement"), "Sub")
          .Input(one)
          .Input(decay)
          .Finalize(graph, &decay_complement));

  Node* value_diff;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/ValueDiff"), "Sub")
          .Input(update_variable)
          .Input(input)
          .Finalize(graph, &value_diff));
  Node* update_value;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/UpdateValue"), "Mul")
          .Input(value_diff)
          .Input(decay_complement)
          .Finalize(graph, &update_value));

  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/EMAValue"), "Sub")
          .Input(update_variable)
          .Input(update_value)
          .Finalize(graph, assign_value));
  return Status::OK();
}

// Creates an automatically initialized exponential moving average variable.
// This uses a switch op to assign a value to the variable on the first run,
// and update with the moving average for all other runs:
//                   init_val
//                      |
//      var--is_init--switch
//       |      true /      \ false
//       |          |        |
//       |         EMA    init_val
//       |           \      /
//       +----------- assign
Status MakeInitializedEMAVariable(Graph* graph, const string& name, Node* decay,
                                  Node* init_val, Node** var) {
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name, "/Variable"), "VariableV2")
          .Attr("shape", TensorShape())
          .Attr("dtype", DT_FLOAT)
          .Finalize(graph, var));

  Node* is_initialized;
  TF_RETURN_IF_ERROR(NodeBuilder(strings::StrCat(name, "/IsInitialized"),
                                 "IsVariableInitialized")
                         .Input(*var)
                         .Finalize(graph, &is_initialized));
  Node* switch_node;
  TF_RETURN_IF_ERROR(NodeBuilder(strings::StrCat(name, "/Switch"), "Switch")
                         .Input(init_val)
                         .Input(is_initialized)
                         .Finalize(graph, &switch_node));
  NodeBuilder::NodeOut output_false = NodeBuilder::NodeOut(switch_node, 0);
  NodeBuilder::NodeOut output_true = NodeBuilder::NodeOut(switch_node, 1);

  Node* ema_value;
  TF_RETURN_IF_ERROR(MakeExponentialMovingAverage(graph, name, output_true,
                                                  decay, *var, &ema_value));

  Node* assign_value;
  TF_RETURN_IF_ERROR(NodeBuilder(strings::StrCat(name, "/Merge"), "Merge")
                         .Input({output_false, ema_value})
                         .Finalize(graph, &assign_value));

  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name, "/AssignValue"), "Assign")
          .Input(*var)
          .Input(assign_value)
          .Finalize(graph, var));
  return Status::OK();
}

// Computes the min and max EMA of input and stores them in min_var and max_var.
Status MakeEMAMinMaxVars(Graph* graph, const string& name_prefix, Node* input,
                         Node** min_var, Node** max_var) {
  // TODO(suharshs): The decay will be constant, so we could make only one for
  // all quantize_and_dequantize ops to share, this would have to live outside
  // this function.
  Tensor decay_tensor(DT_FLOAT, TensorShape());
  decay_tensor.flat<float>()(0) = kEMADecay;
  Node* decay;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/Decay"), "Const")
          .Attr("dtype", DT_FLOAT)
          .Attr("value", decay_tensor)
          .Finalize(graph, &decay));

  Node* reduction_axes;
  TF_RETURN_IF_ERROR(
      MakeReductionAxes(graph, name_prefix, input, &reduction_axes));
  Node* min;
  string min_name = strings::StrCat(name_prefix, "/Min");
  TF_RETURN_IF_ERROR(NodeBuilder(min_name, "Min")
                         .Input(input)
                         .Input(reduction_axes)
                         .Finalize(graph, &min));
  Node* max;
  string max_name = strings::StrCat(name_prefix, "/Max");
  TF_RETURN_IF_ERROR(NodeBuilder(max_name, "Max")
                         .Input(input)
                         .Input(reduction_axes)
                         .Finalize(graph, &max));
  TF_RETURN_IF_ERROR(
      MakeInitializedEMAVariable(graph, min_name, decay, min, min_var));
  TF_RETURN_IF_ERROR(
      MakeInitializedEMAVariable(graph, max_name, decay, max, max_var));
  return Status::OK();
}

// Makes an input min and max constant if the range is given. Otherwise, makes
// min and max variables that are updated by an EMA.
Status MakeInputMinMax(Graph* graph, const string& name_prefix,
                       const EdgeToConvert& edge, Node** input_min,
                       Node** input_max) {
  if (edge.range_given) {
    // Make constant nodes for the input_min and input_max if the range is
    // provided.
    Tensor input_min_tensor(DT_FLOAT, TensorShape());
    input_min_tensor.flat<float>()(0) = edge.input_min;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat(name_prefix, "/InputMin"), "Const")
            .Attr("dtype", DT_FLOAT)
            .Attr("value", input_min_tensor)
            .Finalize(graph, input_min));
    Tensor input_max_tensor(DT_FLOAT, TensorShape());
    input_max_tensor.flat<float>()(0) = edge.input_max;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat(name_prefix, "/InputMax"), "Const")
            .Attr("dtype", DT_FLOAT)
            .Attr("value", input_max_tensor)
            .Finalize(graph, input_max));
  } else {
    // If the range is not given, estimate the range with EMA variables.
    TF_RETURN_IF_ERROR(MakeEMAMinMaxVars(graph, name_prefix, edge.edge->src(),
                                         input_min, input_max));
  }

  return Status::OK();
}

// Adds a QuantizeAndDequantizeV2Op (and required input nodes) based on edge.
// The result is stored in convert_node.
Status MakeQuantizeAndDequantizeV2(Graph* graph, const string& name_prefix,
                                   const EdgeToConvert& edge,
                                   Node** convert_node) {
  Node* input_min;
  Node* input_max;
  TF_RETURN_IF_ERROR(
      MakeInputMinMax(graph, name_prefix, edge, &input_min, &input_max));

  string quant_name = strings::StrCat(name_prefix, "/QuantizeAndDequantizeV2");
  TF_RETURN_IF_ERROR(NodeBuilder(quant_name, "QuantizeAndDequantizeV2")
                         .Input(edge.edge->src())
                         .Input(input_min)
                         .Input(input_max)
                         .Attr("signed_input", edge.signed_input)
                         .Attr("num_bits", edge.num_bits)
                         .Attr("range_given", true)
                         .Finalize(graph, convert_node));
  return Status::OK();
}

// Insert conversion op, connect it to the graph and remove the old edge.
Status ProcessTargetEdges(Graph* graph,
                          const std::vector<EdgeToConvert>& target_edges) {
  // Remember previously converted ops to avoid duplicated conversion on the
  // same input.
  std::unordered_map<string, Node*, StringPiece::Hasher> name_index;
  for (const EdgeToConvert edge : target_edges) {
    Node* convert_node;
    string name_prefix = edge.edge->src()->name();

    auto iter = name_index.find(name_prefix);
    if (iter == name_index.end()) {
      TF_RETURN_IF_ERROR(
          MakeQuantizeAndDequantizeV2(graph, name_prefix, edge, &convert_node));
      name_index[name_prefix] = convert_node;
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
            potential_input++;
            if (potential_input > kAllowedInputs) {
              return errors::Unimplemented(
                  "Found an unknown op: ", edge->src()->name(),
                  " with type: ", edge->src()->type_string(),
                  "; Unknown ops are considered as model input for now and "
                  "only ",
                  kAllowedInputs, " inputs are supported currently.");
            }
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
