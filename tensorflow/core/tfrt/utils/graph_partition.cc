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
#include "tensorflow/core/tfrt/utils/graph_partition.h"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/inline_function_utils.h"
#include "tensorflow/core/common_runtime/partitioning_utils.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace tfrt_stub {

namespace {

// An auxiliary struct to record input/output information.
struct NodeInfo {
  Node* node = nullptr;
  DataType data_type;
  int index = -1;
  Node* node_copy = nullptr;
};

// An auxiliary struct for construction a StatefulPartitionedCallOp enclosing an
// IdentityN node.
struct CallNodeInputInfo {
  int index = -1;
  DataType data_type;
  Node* input_node = nullptr;
  int input_node_index = -1;

  Node* arg_node = nullptr;
  Node* ret_node = nullptr;
};

struct OutputNodeInfo {
  absl::flat_hash_map<std::string, NodeInfo> output_nodes;
  std::optional<std::pair<std::string, NodeInfo>> auxiliary_output_node;
};

// Prepares the `subgraph` for the conversion to a function by adding
// _Arg/_Retval nodes for input/output nodes respectively, and records
// input/output info for the following processing.
// TODO(b/217581711): Consider to use another GraphToFunctionDef() helper which
// does not require _Arg and _Retval nodes.
absl::Status PrepareSubgraphForFunctionConversion(
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs, const Device* host_device,
    const std::string& func_name,
    absl::flat_hash_map<std::string, NodeInfo>& input_nodes,
    absl::flat_hash_map<std::string, NodeInfo>& output_nodes,
    std::optional<std::pair<std::string, NodeInfo>>& auxiliary_output_node,
    Graph* subgraph, Graph* graph) {
  std::unordered_map<std::string, Node*> name_to_node_map =
      subgraph->BuildNodeNameIndex();

  int input_index = 0, output_index = 0;

  // For each input node in this subgraph, replace it with an _Arg node.
  for (const auto& input : inputs) {
    int position = -1;
    std::string node_name = grappler::ParseNodeName(input, &position);
    if (position != 0) {
      return errors::Unimplemented(
          "Support for input node with multiple output tensors is not "
          "implemented.");
    }
    if (name_to_node_map.count(node_name) == 0) continue;

    Node* node = name_to_node_map.at(node_name);
    NodeInfo node_info;
    node_info.node = node;
    node_info.data_type = node->output_type(position);
    node_info.index = input_index++;
    // Copy the input node which will be removed from the subgraph below.
    // The copied node will be used in the top-level graph.
    node_info.node_copy = graph->CopyNode(node);

    input_nodes.emplace(node->name(), node_info);

    // Create an _Arg node to replace the input node.
    TF_ASSIGN_OR_RETURN(
        Node * arg_node,
        NodeBuilder(absl::StrCat("arg_", node_info.index, "/", node->name()),
                    "_Arg")
            .Attr("index", node_info.index)
            .Attr("T", node_info.data_type)
            .Finalize(subgraph));

    CHECK_EQ(node->num_inputs(), 0);
    std::vector<const Edge*> out_edges(node->out_edges().begin(),
                                       node->out_edges().end());
    for (const Edge* edge : out_edges) {
      if (edge->IsControlEdge()) {
        subgraph->AddControlEdge(arg_node, edge->dst());
      } else {
        TF_RETURN_IF_ERROR(
            subgraph->UpdateEdge(arg_node, 0, edge->dst(), edge->dst_input()));
      }
    }
    subgraph->RemoveNode(node);
  }

  // For each output node in this subgraph, connect it to a _Retval node.
  for (const auto& output : outputs) {
    int position = -1;
    std::string node_name = grappler::ParseNodeName(output, &position);
    if (position != 0) {
      return errors::Unimplemented(
          "Support for output node with multiple output tensors is not "
          "implemented.");
    }
    if (name_to_node_map.count(node_name) == 0) continue;

    Node* node = name_to_node_map.at(node_name);
    NodeInfo node_info;
    node_info.node = node;
    node_info.data_type = node->output_type(position);
    node_info.index = output_index++;

    output_nodes.emplace(node->name(), node_info);

    // Create a _RetArg node, and append it to the original output node.
    TF_ASSIGN_OR_RETURN(
        Node * ret_node,
        NodeBuilder(absl::StrCat("ret_", node_info.index, "/", node->name()),
                    "_Retval")
            .Attr("index", node_info.index)
            .Attr("T", node_info.data_type)
            .Input(NodeBuilder::NodeOut(node->name(), position,
                                        node_info.data_type))
            .Finalize(subgraph));
    // Rename the output node, as there will be a node in the top level with
    // the same name.
    node->set_name(node->name() + "/partition_renamed");

    subgraph->AddEdge(node, 0, ret_node, 0);
  }

  // If there is no output for this partition, create an auxiliary output, so
  // that we can generate a data dependency from the PartitionedCallOp (the
  // one we are going to create to wrap this partition) to a downstream
  // stateful node. This helps to preserve the stateless PartitionedCallOp in
  // the subsequent MLIR lowering passes; otherwise, it will be pruned if there
  // is only a control dependency between PartitionedCallOp and another op
  // node, because PartitionedCallOp is stateless and the control dependency
  // will get lost during MLIR lowering with current side effect analysis
  // (b/232026253).
  if (output_nodes.empty()) {
    // Create a const node.
    const DataType data_type = DT_INT32;
    TensorShape const_shape;
    Tensor const_tensor(data_type, const_shape);
    const_tensor.flat<int>()(0) = 0;
    TF_ASSIGN_OR_RETURN(
        Node * const_node,
        NodeBuilder(absl::StrCat("const/unused/", func_name), "Const")
            .AssignedDevice(host_device->name())
            .Attr("dtype", data_type)
            .Attr("value", const_tensor)
            .Finalize(subgraph));

    NodeInfo node_info;
    node_info.node = const_node;
    node_info.data_type = data_type;
    node_info.index = output_index++;
    auxiliary_output_node.emplace(const_node->name(), node_info);

    // Create a _RetArg node, and append to the const node created above.
    TF_ASSIGN_OR_RETURN(
        Node * ret_node,
        NodeBuilder(
            absl::StrCat("ret_", node_info.index, "/", const_node->name()),
            "_Retval")
            .Attr("index", node_info.index)
            .Attr("T", data_type)
            .Input(NodeBuilder::NodeOut(const_node->name(), 0, data_type))
            .Finalize(subgraph));

    subgraph->AddEdge(const_node, 0, ret_node, 0);
  }
  return absl::OkStatus();
}

// Converts the subgraph to a function, and builds a PartitionedCallOp
// to invoke the function.
absl::StatusOr<Node*> BuildPartitionedCallOp(
    const std::string& func_name, const Device* host_device,
    const std::string& device,
    const absl::flat_hash_map<std::string, NodeInfo>& input_nodes,
    const absl::flat_hash_map<std::string, NodeInfo>& output_nodes,
    const absl::optional<std::pair<std::string, NodeInfo>>&
        auxiliary_output_node,
    const std::vector<std::string>& control_outputs, Graph* subgraph,
    Graph* graph) {
  // Build the call node.
  std::string call_node_name = absl::StrCat("partitioned_call/", func_name);
  NodeBuilder call_builder(call_node_name, "PartitionedCall");
  call_builder.AssignedDevice(host_device->name());
  call_builder.Attr(tensorflow::kNoInlineAttr, true);

  std::vector<DataType> input_dtypes(input_nodes.size());
  for (const auto& input_node : input_nodes) {
    input_dtypes[input_node.second.index] = input_node.second.data_type;
  }
  call_builder.Attr("Tin", input_dtypes);

  CHECK(auxiliary_output_node ? output_nodes.empty() : !output_nodes.empty());
  std::vector<DataType> output_dtypes(
      auxiliary_output_node ? 1 : output_nodes.size());
  if (auxiliary_output_node) {
    CHECK_EQ(auxiliary_output_node->second.index, 0);
    output_dtypes[auxiliary_output_node->second.index] =
        auxiliary_output_node->second.data_type;
  } else {
    for (const auto& output_node : output_nodes) {
      output_dtypes[output_node.second.index] = output_node.second.data_type;
    }
  }
  call_builder.Attr("Tout", output_dtypes);

  std::vector<NodeBuilder::NodeOut> call_node_inputs(input_nodes.size());
  for (const auto& input_node : input_nodes) {
    call_node_inputs[input_node.second.index] =
        NodeBuilder::NodeOut(input_node.second.node_copy, 0);
  }
  call_builder.Input(call_node_inputs);

  NameAttrList f;
  f.set_name(func_name);
  call_builder.Attr("f", f);
  TF_ASSIGN_OR_RETURN(Node * call_node, call_builder.Finalize(graph));

  // Convert the subgraph to a function.
  absl::flat_hash_set<std::string> control_ret_names(control_outputs.begin(),
                                                     control_outputs.end());
  // After graph partition, there are send ops added as new end nodes.
  // The completion of the graph requires the send ops to be executed.
  for (const Node* node : subgraph->op_nodes()) {
    if (node->IsSend()) {
      control_ret_names.insert(node->name());
    }
  }
  auto control_ret_node_names =
      [&control_ret_names](const Node* node) -> absl::optional<std::string> {
    if (control_ret_names.contains(node->name())) {
      return node->name();
    }
    return std::nullopt;
  };

  FunctionDef new_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*subgraph, func_name,
                                        control_ret_node_names, &new_fdef));
  // Set the `_noinline` attribute for the function to make sure it does not
  // get inlined, and its corresponding function in TF MLIR does not get inlined
  // in lowering passes as well.
  (*new_fdef.mutable_attr())[tensorflow::kNoInlineAttr].set_b(true);
  (*new_fdef.mutable_attr())["device"].set_s(device);
  TF_RETURN_IF_ERROR(graph->mutable_flib_def()->AddFunctionDef(new_fdef));

  return call_node;
}

// Builds a StatefulPartitionedCallOp, and connects all PartitionedCallOps to
// it. This StatefulPartitionedCallOp behaves as a stateful IdentityN.
absl::StatusOr<Node*> BuildStatefulPartitionedCallOp(
    absl::flat_hash_map<std::string, CallNodeInputInfo>& call_node_input_info,
    const absl::flat_hash_map<std::string, Node*>& all_partitioned_call_ops,
    const std::string& stateful_call_func_name, const Device* host_device,
    Graph* graph) {
  std::string call_node_name =
      absl::StrCat("stateful_partitioned_call/", stateful_call_func_name);
  NodeBuilder call_builder(call_node_name, "StatefulPartitionedCall");
  call_builder.Attr(tensorflow::kNoInlineAttr, true);
  call_builder.AssignedDevice(host_device->name());

  int num_output_nodes = call_node_input_info.size();
  std::vector<DataType> input_dtypes(num_output_nodes);
  for (const auto& node_info : call_node_input_info) {
    CHECK(node_info.second.index < num_output_nodes);
    input_dtypes[node_info.second.index] = node_info.second.data_type;
  }
  call_builder.Attr("Tin", input_dtypes);
  // Outputs are the same as inputs.
  call_builder.Attr("Tout", input_dtypes);

  std::vector<NodeBuilder::NodeOut> call_node_inputs(num_output_nodes);
  for (const auto& node_info : call_node_input_info) {
    call_node_inputs[node_info.second.index] = NodeBuilder::NodeOut(
        node_info.second.input_node, node_info.second.input_node_index);
  }
  call_builder.Input(call_node_inputs);

  NameAttrList f;
  f.set_name(stateful_call_func_name);
  call_builder.Attr("f", f);
  TF_ASSIGN_OR_RETURN(Node * stateful_call_node, call_builder.Finalize(graph));

  // Construct a graph that only contains an IdentityN node, and convert the
  // graph to a function.
  auto id_graph = std::make_unique<Graph>(graph->flib_def().default_registry());

  std::vector<NodeBuilder::NodeOut> output_tensors(num_output_nodes);

  // Create an _Arg node for each input.
  for (auto& node_info : call_node_input_info) {
    TF_ASSIGN_OR_RETURN(node_info.second.arg_node,
                        NodeBuilder(absl::StrCat("arg_", node_info.second.index,
                                                 "/", stateful_call_func_name),
                                    "_Arg")
                            .Attr("index", node_info.second.index)
                            .Attr("T", node_info.second.data_type)
                            .Finalize(id_graph.get()));

    output_tensors[node_info.second.index] =
        NodeBuilder::NodeOut(node_info.second.arg_node, 0);
  }

  // Create the Identity Node.
  TF_ASSIGN_OR_RETURN(
      Node * identity_node,
      NodeBuilder(absl::StrCat("identityN", "/", stateful_call_func_name),
                  "IdentityN")
          .AssignedDevice(host_device->name())
          .Input(output_tensors)
          .Finalize(id_graph.get()));

  // Create a _Retval node for each output.
  for (auto& node_info : call_node_input_info) {
    TF_ASSIGN_OR_RETURN(
        node_info.second.ret_node,
        NodeBuilder(absl::StrCat("ret_", node_info.second.index, "/",
                                 stateful_call_func_name),
                    "_Retval")
            .Attr("index", node_info.second.index)
            .Attr("T", node_info.second.data_type)
            .Input(NodeBuilder::NodeOut(identity_node, node_info.second.index))
            .Finalize(id_graph.get()));

    id_graph->AddEdge(identity_node, node_info.second.index,
                      node_info.second.ret_node, 0);
  }

  // Convert the id_graph to a function.
  FunctionDef id_fdef;
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(*id_graph, stateful_call_func_name, &id_fdef));
  (*id_fdef.mutable_attr())[tensorflow::kNoInlineAttr].set_b(true);
  TF_RETURN_IF_ERROR(graph->mutable_flib_def()->AddFunctionDef(id_fdef));

  return stateful_call_node;
}

// Returns true if nodes in the `graph` are assigned to multiple devices.
bool HasMultipleDevices(const Graph* graph) {
  bool has_multiple_devices = false;
  std::optional<std::string> location;
  for (const Node* node : graph->op_nodes()) {
    if (location) {
      if (*location != node->assigned_device_name()) {
        has_multiple_devices = true;
        break;
      }
    } else {
      location = node->assigned_device_name();
    }
  }
  return has_multiple_devices;
}

std::string GetNameFromDevice(const std::string& device) {
  std::string ret = device;
  for (int i = 0; i < ret.size(); ++i) {
    // Replace ':', as it is not allowed in node names.
    if (ret[i] == ':') ret[i] = '_';
  }
  return ret;
}

}  // namespace

// This function performs the following steps:
// 1. Partition the graph and insert send/recv ops on the edges across devices.
// 2. For each partition, convert the subgraph to a function and invoke the
//    function by a PartitionedCallOp, so that these functions can be executed
//    asynchronousely.
// 3. Connect all PartitionedCallOps to a StatefulPartitionedCallOps to make
//    sure PartitionedCallOps are not pruned in the subsequent MLIR lowering
//    passes.
// 4. Create output nodes and control output nodes to match the original graph's
//    nodes.
absl::StatusOr<std::unique_ptr<Graph>> InsertTransferOps(
    const std::string& graph_func_name, const DeviceSet& device_set,
    const Device* host_device, const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& control_outputs,
    std::unique_ptr<Graph> graph) {
  // Skip transfer op insertion if the graph nodes are not assigned to multiple
  // devices.
  if (!HasMultipleDevices(graph.get())) {
    return graph;
  }

  // Step 1: Partition the graph and insert send/recv ops on the edges across
  // devices.
  auto new_graph = std::make_unique<Graph>(graph->flib_def());
  FunctionDefLibrary flib = graph->flib_def().ToProto();

  std::unordered_map<string, std::unique_ptr<Graph>> partitions;
  TF_RETURN_IF_ERROR(
      PartitionFunctionGraph(device_set, std::move(graph), &partitions));

  // Step 2: For each partition, convert the subgraph to a function and invoke
  // the function by PartitionedCallOp from the top-level graph.

  absl::flat_hash_map<std::string, Node*> all_partitioned_call_ops;
  std::map<std::string, OutputNodeInfo> device_to_output_info_map;

  for (auto& partition : partitions) {
    const string& device = partition.first;
    VLOG(1) << "Process the partitioin on device: " << device;

    Graph* subgraph = partition.second.get();
    TF_RETURN_IF_ERROR(subgraph->AddFunctionLibrary(flib));

    FunctionNameGenerator name_generator(
        &new_graph->flib_def(), absl::StrCat(graph_func_name, "-partition-",
                                             GetNameFromDevice(device)));
    std::string func_name = name_generator.GetName();

    absl::flat_hash_map<std::string, NodeInfo> input_nodes;

    OutputNodeInfo& output_node_info = device_to_output_info_map[device];
    absl::flat_hash_map<std::string, NodeInfo>& output_nodes =
        output_node_info.output_nodes;
    std::optional<std::pair<std::string, NodeInfo>>& auxiliary_output_node =
        output_node_info.auxiliary_output_node;

    // Add _Arg and _Retval nodes to the subgraph to prepare for converting it
    // to a function. Meanwhile, record input/output infos for the following
    // processing.
    TF_RETURN_IF_ERROR(PrepareSubgraphForFunctionConversion(
        inputs, outputs, host_device, func_name, input_nodes, output_nodes,
        auxiliary_output_node, subgraph, new_graph.get()));

    // Convert the subgraph to a function, and build a PartitionedCallOp to
    // invoke the function.
    TF_ASSIGN_OR_RETURN(
        Node * call_node,
        BuildPartitionedCallOp(func_name, host_device, device, input_nodes,
                               output_nodes, auxiliary_output_node,
                               control_outputs, subgraph, new_graph.get()));
    all_partitioned_call_ops[device] = call_node;
  }

  // Step 3: Create a StatefulPartitionedCallOp, and connect all
  // PartitionedCallOps to it. The StatefulPartitionedCallOp behaves as a
  // stateful IdentityN. This helps to preserve the PartitionedCallOps
  // (stateless) in the TF MLIR lowering passes; otherwise, without a stateful
  // consumer, PartitionedCallOps will be pruned, as control output info of
  // the graph gets lost during TF MLIR lowering (b/232026253).

  // Collect all outputs from all partitions, and update their indices to be
  // used for constructing StatefulPartitionedCallOp.
  int input_index = 0;
  absl::flat_hash_map<std::string, CallNodeInputInfo> call_node_input_info;
  auto get_call_node_input_info = [&](const std::string& device,
                                      const std::string& node_name,
                                      const NodeInfo& node_info) {
    CHECK(!call_node_input_info.contains(node_name));
    CallNodeInputInfo& info = call_node_input_info[node_name];
    info.index = input_index++;
    info.data_type = node_info.data_type;
    info.input_node = all_partitioned_call_ops.at(device);
    info.input_node_index = node_info.index;
  };
  for (const auto& entry : device_to_output_info_map) {
    const std::string& device = entry.first;
    const OutputNodeInfo& output_info = entry.second;
    for (const auto& node_info : output_info.output_nodes) {
      get_call_node_input_info(device, node_info.first, node_info.second);
    }
    if (output_info.auxiliary_output_node) {
      get_call_node_input_info(device, output_info.auxiliary_output_node->first,
                               output_info.auxiliary_output_node->second);
    }
  }

  FunctionNameGenerator name_generator(
      &new_graph->flib_def(),
      absl::StrCat(graph_func_name, "/output_aggregator"));
  std::string stateful_call_func_name = name_generator.GetName();
  TF_ASSIGN_OR_RETURN(
      Node * stateful_call_node,
      BuildStatefulPartitionedCallOp(
          call_node_input_info, all_partitioned_call_ops,
          stateful_call_func_name, host_device, new_graph.get()));

  // Step 4: Create output nodes and control output nodes corresponding to the
  // original graph's nodes.

  // For each of the original output, construct a corresponding Identity node
  // with the same name.
  for (const auto& node_info : call_node_input_info) {
    TF_RETURN_IF_ERROR(NodeBuilder(node_info.first, "Identity")
                           .Input(NodeBuilder::NodeOut(stateful_call_node,
                                                       node_info.second.index))
                           .Attr("T", node_info.second.data_type)
                           .AssignedDevice(host_device->name())
                           .Finalize(new_graph.get(), nullptr));
  }

  // For each of the original control output, construct a corresponding Identity
  // node with the same name.
  CHECK_GT(stateful_call_node->num_outputs(), 0);
  for (const auto& control_output : control_outputs) {
    TF_RETURN_IF_ERROR(NodeBuilder(control_output, "Identity")
                           .Input(NodeBuilder::NodeOut(stateful_call_node, 0))
                           .Attr("T", stateful_call_node->output_type(0))
                           .AssignedDevice(host_device->name())
                           .Finalize(new_graph.get(), nullptr));
  }

  return new_graph;
}

}  // namespace tfrt_stub
}  // namespace tensorflow
