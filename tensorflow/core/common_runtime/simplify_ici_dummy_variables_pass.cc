/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/simplify_ici_dummy_variables_pass.h"

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/util/device_name_utils.h"
#include "tensorflow/core/common_runtime/function_utils.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/config/flag_defs.h"
#include "tensorflow/core/config/flags.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tsl/platform/errors.h"

namespace tensorflow {
namespace {

constexpr absl::string_view kTpuExecute = "TPUExecute";
constexpr absl::string_view kParallelExecuteIds = "_parallel_execution_ids";
const char kICIWeightDistributionMlirBridgeMarker[] =
    "_ici_weight_distribution_mlir_bridge_marker";

// Get the new op name which is used to replace the old op, the new op name
// contains the index of the input and the task id of the TPUExecute node.
std::string GetNewOpName(std::string op_name, int index, int task_id) {
  return absl::StrCat(op_name, "_ici_specific_index_", std::to_string(index),
                      "_task_id_", std::to_string(task_id));
}

// Find all the TPUExecute nodes that is not on replica 0. In addition, return
// an empty vector if there is a parallel execute id that is not 0, which
// indicates SPMD case. In the meantime, we check if this is a SPMD case.
std::vector<Node*> GetNonMainReplicaIciTPUExecuteNodes(Graph* graph,
                                                       bool& is_spmd) {
  std::vector<Node*> tpu_nodes;
  for (Node* node : graph->nodes()) {
    if (node->type_string() == kTpuExecute &&
        HasNodeAttr(node->def(), kParallelExecuteIds)) {
      auto parallel_exec_ids = node->attrs().Find(kParallelExecuteIds)->s();
      std::vector<std::string> group_vec =
          absl::StrSplit(parallel_exec_ids, ',');
      if (group_vec.empty()) return tpu_nodes;
      std::vector<std::string> replica_vec = absl::StrSplit(group_vec[0], ':');
      int replica_id = std::stoi(replica_vec[1]);
      if (replica_id != 0) tpu_nodes.push_back(node);
      if (group_vec.size() > 1) {
        std::vector<std::string> parallel_vec =
            absl::StrSplit(group_vec[1], ':');
        int parallel_id = std::stoi(parallel_vec[1]);
        if (parallel_id != 0) is_spmd = true;
      }
    }
  }
  return tpu_nodes;
}

// Remove the edge from old_src_node to dst_node, and add the edge from
// new_src_node to dst_node.
void RedirectEdge(Graph* graph, Node* old_src_node, Node* dst_node,
                  Node* new_src_node, int input_index) {
  const Edge* delete_edge;
  for (auto edge : dst_node->in_edges()) {
    if (edge->src() == old_src_node) {
      delete_edge = edge;
      break;
    }
  }
  if (delete_edge == nullptr) return;

  graph->RemoveEdge(delete_edge);
  graph->AddEdge(new_src_node, 0, dst_node, input_index);
}

// Find the corresponding host device name from the TPU device name.
string GetHostDeviceName(Node* tpu_node) {
  auto device_name = tpu_node->requested_device();
  if (device_name.empty()) device_name = tpu_node->assigned_device_name();
  DeviceNameUtils::ParsedName parsed_device_name;
  DeviceNameUtils::ParseFullName(device_name, &parsed_device_name);
  string host_device_name = DeviceNameUtils::FullName(
      parsed_device_name.job, parsed_device_name.replica,
      parsed_device_name.task, /*type=*/"CPU", /*id=*/0);
  return host_device_name;
}

std::optional<std::vector<int>> GetOutputShapeVec(Node* node) {
  auto output_shapes = node->attrs().Find("_output_shapes");
  if (output_shapes == nullptr) return std::nullopt;
  auto output_shape = output_shapes->list().shape()[0];
  std::vector<int> output_shape_vec;
  output_shape_vec.reserve(output_shape.dim_size());
  for (auto i = 0; i < output_shape.dim_size(); i++) {
    output_shape_vec.push_back(output_shape.dim()[i].size());
  }
  return output_shape_vec;
}

int GetTPUTaskId(Node* tpu_node) {
  auto device_name = tpu_node->requested_device();
  if (device_name.empty()) device_name = tpu_node->assigned_device_name();
  DeviceNameUtils::ParsedName parsed_device_name;
  DeviceNameUtils::ParseFullName(device_name, &parsed_device_name);
  return parsed_device_name.task;
}

// Build the fill op. Its value is 0 and the fill op is put on the host device
// with the same task id as the TPUExecute node.
Node* BuildFillOp(GraphDefBuilder::Options& bopts, Node* tpu_node,
                  Node* in_node, int input_index, string host_device_name) {
  // Find the output_shape vector
  auto output_shape_vec = GetOutputShapeVec(in_node);
  if (!output_shape_vec.has_value()) return nullptr;

  // Find the element type
  auto dtype = in_node->attrs().Find("T")->type();

  // Get TPU task id.
  int tpu_task_id = GetTPUTaskId(tpu_node);

  TensorShape tensor_shape;
  tensor_shape.AddDim(output_shape_vec.value().size());
  Tensor const_op_shape_tensor(DT_INT32, tensor_shape);
  for (int i = 0; i < output_shape_vec.value().size(); i++) {
    const_op_shape_tensor.flat<int>()(i) = output_shape_vec.value()[i];
  }

  // Build dim of fill op
  std::string const_1_name = GetNewOpName("const_1", input_index, tpu_task_id);
  Node* fill_dim_input =
      ops::SourceOp("Const", bopts.WithName(const_1_name)
                                 .WithAttr("dtype", DT_INT32)
                                 .WithAttr("value", const_op_shape_tensor));
  TensorShape fill_dim_output_shape;
  fill_dim_output_shape.AddDim(output_shape_vec.value().size());
  fill_dim_input->AddAttr("_output_shapes",
                          std::vector<TensorShape>{fill_dim_output_shape});

  // Build value of fill op
  std::string const_2_name = GetNewOpName("const_2", input_index, tpu_task_id);
  auto scalar_tensor = Tensor(dtype, {});

  if (dtype == DT_FLOAT) {
    scalar_tensor.scalar<float>()() = 0;
  } else if (dtype == DT_BFLOAT16) {
    scalar_tensor.scalar<bfloat16>()() = bfloat16(0);
  } else {
    LOG(ERROR) << "Unsupported data type: ", DataTypeString(dtype);
    return nullptr;
  }
  Node* fill_value_input =
      ops::SourceOp("Const", bopts.WithName(const_2_name)
                                 .WithAttr("dtype", dtype)
                                 .WithAttr("value", scalar_tensor));
  TensorShape fill_value_output_shape;
  fill_value_input->AddAttr("_output_shapes",
                            std::vector<TensorShape>{fill_value_output_shape});

  // Build fill op
  std::string fill_name = GetNewOpName("fill", input_index, tpu_task_id);
  Node* new_fill =
      ops::BinaryOp("Fill", fill_dim_input, fill_value_input,
                    bopts.WithName(fill_name).WithAttr("T", dtype));

  TensorShape new_output_shape;
  for (auto output_shape : output_shape_vec.value()) {
    new_output_shape.AddDim(output_shape);
  }
  new_fill->AddAttr("_output_shapes",
                    std::vector<TensorShape>{new_output_shape});
  new_fill->AddAttr("_xla_inferred_shapes",
                    std::vector<TensorShape>{new_output_shape});

  // Set the device to each node.
  fill_dim_input->set_requested_device(host_device_name);
  fill_value_input->set_requested_device(host_device_name);
  new_fill->set_requested_device(host_device_name);

  return new_fill;
}

// Replace the ici dummy variable with one on the right task id.
absl::Status ReplaceIciDummyVariables(Graph* graph, int input_index,
                                      std::vector<Node*> tpu_nodes,
                                      GraphDefBuilder::Options& bopts) {
  absl::flat_hash_map<std::string, Node*> device_to_node_map;
  for (Node* tpu_node : tpu_nodes) {
    Node* in_node;
    TF_RETURN_IF_ERROR(tpu_node->input_node(input_index, &in_node));

    if (!in_node->attrs().Find(kICIWeightDistributionMlirBridgeMarker)) {
      continue;
    }

    string host_device_name = GetHostDeviceName(tpu_node);

    // If the node corresponding to host_device_name is already in the graph,
    // replace the edge from in_node to tpu_node with the edge from
    // device_to_node_map[host_device_name] to tpu_node.
    if (device_to_node_map.contains(host_device_name)) {
      RedirectEdge(graph, in_node, tpu_node,
                   device_to_node_map[host_device_name], input_index);
      continue;
    }

    Node* new_fill =
        BuildFillOp(bopts, tpu_node, in_node, input_index, host_device_name);
    if (new_fill == nullptr) continue;

    device_to_node_map[host_device_name] = new_fill;
    RedirectEdge(graph, in_node, tpu_node, device_to_node_map[host_device_name],
                 input_index);
  }
  return absl::OkStatus();
}

}  // namespace

bool ShouldRunPass(const GraphOptimizationPassOptions& options) {
  if (!flags::Global().enable_tf2min_ici_weight.value()) {
    VLOG(1) << "SimplifyIciDummyVariablesPass is disabled.";
    return false;
  }
  VLOG(1) << "SimplifyIciDummyVariablesPass is enabled.";

  // find all potential nodes.
  if (options.graph == nullptr) {
    LOG(INFO) << "No graph in simplify_ici_dummy_variables_pass.";
    return false;
  }
  return true;
}

absl::Status SimplifyIciDummyVariablesPass::Run(
    const GraphOptimizationPassOptions& options) {
  if (!ShouldRunPass(options)) {
    return absl::OkStatus();
  }

  Graph* graph = options.graph->get();
  VLOG(1) << DumpGraphToFile("before_simplify_ici_dummy_variables_pass", *graph,
                             options.flib_def);

  absl::Status status;
  GraphDefBuilder::Options bopts(graph, &status);
  if (!status.ok()) {
    LOG(ERROR) << "GraphDefBuilder::Option failed to initialize.";
    return status;
  }

  bool is_spmd = false;

  // Find all the qualified tpu_execute nodes which is not on replica 0.
  std::vector<Node*> tpu_nodes =
      GetNonMainReplicaIciTPUExecuteNodes(graph, is_spmd);

  if (!is_spmd) {
    VLOG(1) << "Not SPMD case, skip SimplifyIciDummyVariablesPass.";
    return absl::OkStatus();
  }

  if (tpu_nodes.empty()) {
    VLOG(1) << "tpu_nodes is empty, skip SimplifyIciDummyVariablesPass.";
    return absl::OkStatus();
  }

  for (int i = 0; i < tpu_nodes[0]->num_inputs(); ++i) {
    auto replace_status = ReplaceIciDummyVariables(graph, i, tpu_nodes, bopts);
    if (!replace_status.ok()) {
      LOG(ERROR) << "Replace ici dummy variables failed.";
      return replace_status;
    }
  }

  // Remove the dead nodes that previously connected to the TPUExecute node.
  RemoveDeadNodes(graph);

  VLOG(1) << DumpGraphToFile("after_simplify_ici_dummy_variables_pass", *graph,
                             options.flib_def);

  return absl::OkStatus();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 49,
                      SimplifyIciDummyVariablesPass);

}  // namespace tensorflow
