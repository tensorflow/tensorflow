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

// Configuration for TPU Embedding.

#include "tensorflow/core/tpu/graph_rewrite/update_tpu_embedding_ops_passes.h"

#include <string>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/tf2xla/side_effect_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace {

constexpr absl::string_view kTPUEmbeddingOps[] = {
    "EnqueueTPUEmbeddingBatch",
    "EnqueueTPUEmbeddingIntegerBatch",
    "EnqueueTPUEmbeddingSparseBatch",
    "EnqueueTPUEmbeddingSparseTensorBatch",
    "EnqueueTPUEmbeddingRaggedTensorBatch",
    "EnqueueTPUEmbeddingArbitraryTensorBatch"};

constexpr absl::string_view kTPURecvOps[] = {"RecvTPUEmbeddingActivations",
                                             "XlaRecvTPUEmbeddingActivations"};

constexpr absl::string_view kTPUGradientSendOps[] = {
    "SendTPUEmbeddingGradients", "XlaSendTPUEmbeddingGradients"};

}  // namespace

Status UpdateTPUEmbeddingEnqueueOrdinalPass::Run(
    const GraphOptimizationPassOptions& options) {
  VLOG(1) << "UpdateTPUEmbeddingEnqueueOrdinalPass::Run";

  // Need the device set to get the number of devices per host.
  TF_RET_CHECK(options.device_set != nullptr);

  std::vector<Device*> tpu_devices;
  DeviceNameUtils::ParsedName tpu_device_spec;
  tpu_device_spec.has_type = true;
  tpu_device_spec.type = "TPU";
  options.device_set->FindMatchingDevices(tpu_device_spec, &tpu_devices);
  if (tpu_devices.empty()) {
    // If there are no TPUs don't run this pass.
    return OkStatus();
  }

  TF_RET_CHECK(options.graph != nullptr);
  Graph* graph = options.graph->get();

  std::vector<Node*> embedding_nodes;
  for (Node* node : graph->op_nodes()) {
    if (absl::c_linear_search(kTPUEmbeddingOps, node->type_string())) {
      embedding_nodes.emplace_back(node);
    }
  }

  // Only run if there are embedding nodes.
  if (embedding_nodes.empty()) {
    return OkStatus();
  }

  DeviceNameUtils::ParsedName single_tpu_device_spec =
      tpu_devices[0]->parsed_name();

  TF_RET_CHECK(single_tpu_device_spec.has_job);

  // Note that TPUEmbedding is only supported on system with a single TPU slice
  // (as determined by the 'job' portion of the device spec). Check for that
  // here just to be sure.
  for (const auto* tpu_device : tpu_devices) {
    TF_RET_CHECK(tpu_device->parsed_name().has_job);
    TF_RET_CHECK(tpu_device->parsed_name().job == single_tpu_device_spec.job)
        << "Multiple TPU jobs detected. This is not supported for now.";
  }

  std::vector<Device*> task_devices;
  single_tpu_device_spec.has_id = false;
  options.device_set->FindMatchingDevices(single_tpu_device_spec,
                                          &task_devices);
  int64 num_tpus_per_task = task_devices.size();

  for (Node* node : embedding_nodes) {
    int64 replica_id;
    if (TryGetNodeAttr(node->attrs(), kXlaReplicaIdAttrName, &replica_id)) {
      node->AddAttr("device_ordinal", replica_id % num_tpus_per_task);
    }
  }

  VLOG(1) << "UpdateTPUEmbeddingEnqueueOrdinalPass::Run() finished";
  return OkStatus();
}

template <typename A, typename N>
Status UpdateMapsForModeOverride(
    const std::string& op, const A& attrs, const N node_identifier,
    std::map<std::string, N>* enqueue_op,
    std::map<std::string, bool>* found_recv_op,
    std::map<std::string, bool>* found_grad_send_op) {
  string layer_call_index;
  if (TryGetNodeAttr(attrs, "_tpu_embedding_layer", &layer_call_index)) {
    if ((op == kTPURecvOps[0]) || (op == kTPURecvOps[1])) {
      // We will prevent users from creating multiple copies of the
      // TPUEmbedding layer so this should never happen.
      TF_RET_CHECK(!(*found_recv_op)[layer_call_index])
          << "Found second receive op for call " << layer_call_index << ". "
          << "This will happen if you create multiple TPUEmbedding layers. "
          << "Please ensure that you have only created one TPUEmbedding "
          << "layer.";
      (*found_recv_op)[layer_call_index] = true;
    } else if ((op == kTPUGradientSendOps[0]) ||
               (op == kTPUGradientSendOps[1])) {
      TF_RET_CHECK(!(*found_grad_send_op)[layer_call_index])
          << "Found second send op for call " << layer_call_index << ". "
          << "This will happen if you create multiple TPUEmbedding layers. "
          << "Please ensure that you have only created one TPUEmbedding "
          << "layer.";
      (*found_grad_send_op)[layer_call_index] = true;
    } else if (absl::c_linear_search(kTPUEmbeddingOps, op)) {
      TF_RET_CHECK(enqueue_op->find(layer_call_index) == enqueue_op->end())
          << "Found second enqueue op for call " << layer_call_index << ". "
          << "This will happen if you create multiple TPUEmbedding layers. "
          << "Please ensure that you have only created one TPUEmbedding "
          << "layer.";
      (*enqueue_op)[layer_call_index] = node_identifier;
    }
  }
  return OkStatus();
}

template <typename M, typename N>
Status ComputeEnqueueTrainingStatus(
    const std::map<std::string, N>& enqueue_op,
    const std::map<std::string, bool>& found_recv_op,
    const std::map<std::string, bool>& found_grad_send_op, M* enqueue) {
  TF_RET_CHECK(enqueue_op.size() == found_recv_op.size())
      << "Enqueue and recv ops should be in a one-to-one corresondence."
      << "Found " << enqueue_op.size() << " enqueue(s) and "
      << found_recv_op.size() << " receive(s).";
  for (const auto& node : enqueue_op) {
    TF_RET_CHECK(found_recv_op.find(node.first) != found_recv_op.end())
        << "No receive for enqueue call " << node.first;
    bool send_exists =
        (found_grad_send_op.find(node.first) != found_grad_send_op.end());
    VLOG(1) << "Found call " << node.first
        << (send_exists ? " with " : " without ") << " send op(s).";
    // If we have found a send gradient op for that is in the same cluster as
    // the enqueue op, then this is a training call so set the output to true
    // for this
    (*enqueue)[node.second] = send_exists;
  }
  return OkStatus();
}

// Get the enqueue ops and their status (training or eval) from a graph.
// enqueue is a map from a Graph Node* for an enqueue op to a bool which is true
// when the enqueue is part of a TPUEmbedding layer call that contains a send
// gradients.
Status UpdateTPUEmbeddingModePass::GetEnqueueOpsFromGraph(
    Graph* graph, absl::flat_hash_map<Node*, bool>* enqueue) {
  // Maps are index by the TPUEmbedding layer's call number.
  std::map<std::string, Node*> enqueue_op;
  std::map<std::string, bool> found_recv_op;
  std::map<std::string, bool> found_grad_send_op;

  for (Node* node : graph->op_nodes()) {
    TF_RETURN_IF_ERROR(UpdateMapsForModeOverride(
        node->type_string(), node->attrs(), node, &enqueue_op, &found_recv_op,
        &found_grad_send_op));
    // Clear attribute so any further executions of this pass don't activate
    // pass.
    node->ClearAttr("_tpu_embedding_layer");
  }

  return ComputeEnqueueTrainingStatus(enqueue_op, found_recv_op,
                                      found_grad_send_op, enqueue);
}

// Update the graph for a specific enqueue op.
Status UpdateTPUEmbeddingModePass::UpdateGraphEnqueueOp(bool training,
                                                     Graph* graph,
                                                     Node* enqueue) {
  // When using the layer, the mode override input is a SelectV2 op (unless this
  // pass has already run), which takes a training and eval op as input. We will
  // simply short circut the SelectV2 and take input from the correct op.
  const Edge* select_edge;
  TF_RETURN_IF_ERROR(
      enqueue->input_edge(enqueue->num_inputs() - 1, &select_edge));
  if (select_edge->src()->type_string() == "SelectV2") {
    TF_RET_CHECK(select_edge->src()->num_inputs() == 3);
    Node* mode;
    TF_RETURN_IF_ERROR(select_edge->src()->input_node(training ? 1 : 2, &mode));
    graph->AddEdge(mode, 0, enqueue, enqueue->num_inputs() - 1);
    graph->RemoveEdge(select_edge);
  }

  return OkStatus();
}

// Get the enqueue ops and their status (training or eval) from a function def.
// The enqueue map is indexed by the position of the enqueue op in the
// function's node_def array.
Status UpdateTPUEmbeddingModePass::GetEnqueueOpsFromFunctionDef(
    FunctionDef* function, std::map<int, bool>* enqueue) {
  std::map<std::string, int> enqueue_op;
  std::map<std::string, bool> found_recv_op;
  std::map<std::string, bool> found_grad_send_op;

  std::string cluster;
  for (int i = 0; i < function->node_def_size(); ++i) {
    const NodeDef& node = function->node_def(i);
    TF_RETURN_IF_ERROR(UpdateMapsForModeOverride(
        node.op(), node, i, &enqueue_op, &found_recv_op, &found_grad_send_op));
    // Clear attribute so any further executions of this pass don't activate
    // pass.
    function->mutable_node_def(i)->mutable_attr()->erase(
        "_tpu_embedding_layer");
  }

  return ComputeEnqueueTrainingStatus(enqueue_op, found_recv_op,
                                      found_grad_send_op, enqueue);
}

// Update the function def for a specific enqueue op.
Status UpdateTPUEmbeddingModePass::UpdateFunctionDefEnqueueOp(
    int enqueue, bool training, FunctionDef* function, bool* updated) {
  // When using the layer, the mode override input is a SelectV2 op,
  // which takes a training and eval op as input. We will simply short circut
  // the SelectV2 and take input from the correct op.
  NodeDef* node = function->mutable_node_def(enqueue);
  int mode_override = node->input_size() - 1;
  while ((mode_override >= 0) && (node->input(mode_override).empty() ||
                                  (node->input(mode_override)[0] == '^'))) {
    mode_override--;
  }
  TF_RET_CHECK(mode_override >= 0) << "Can't find non-control input to "
                                   << "enqueue.";
  TF_RET_CHECK(!node->input(mode_override).empty());

  // Find input node
  string select_name = std::vector<std::string>(
      absl::StrSplit(node->input(mode_override), ':'))[0];
  int select = 0;
  while ((select < function->node_def_size()) &&
         (function->node_def(select).name() != select_name)) {
    select++;
  }
  TF_RET_CHECK(select < function->node_def_size())
      << "Unable to find enqueue input node " << select_name << " in function "
      << function->signature().name();
  if (function->node_def(select).op() == "SelectV2") {
    // Make the mode override input the same as the correct input of the
    // select v2.
    (*node->mutable_input(mode_override)) =
        function->node_def(select).input(training ? 1 : 2);
    *updated = true;
  }

  return OkStatus();
}

Status UpdateTPUEmbeddingModePass::Run(
    const GraphOptimizationPassOptions& options) {
  // Updates the Enqueue ops when using a layer to set the mode override
  // behavior depending on the existence of send gradients ops.
  // Note we only do this when a layer is used (all BC ops with an integer
  // attribute "_tpu_embedding_layer" that is incremented per call, so we can
  // easily associate the various ops).
  //
  // Note that the BC ops can be in the Graph or in the FunctionDef.
  // If they are in the graph at stage 0, this means that there as no control
  // flow containing them (i.e. a host loop). In this case, we group together
  // ops with the same "_tpu_embedding_layer" tag.
  //
  // We also search all FunctionDefs. Note that as the ops are all created in
  // the layer's call, a cluster of TPUEmbedding ops won't be split across
  // different FunctionDefs.

  VLOG(1) << "UpdateTPUEmbeddingModePass::Run";

  TF_RET_CHECK(options.graph != nullptr);

  // First process the graph
  Graph* graph = options.graph->get();
  absl::flat_hash_map<Node*, bool> enqueue_nodes;
  TF_RETURN_IF_ERROR(GetEnqueueOpsFromGraph(graph, &enqueue_nodes));
  for (const auto& enqueue : enqueue_nodes) {
    TF_RETURN_IF_ERROR(
        UpdateGraphEnqueueOp(enqueue.second, graph, enqueue.first));
  }

  for (const auto& fname : options.flib_def->ListFunctionNames()) {
    FunctionDef fdef_copy(*options.flib_def->Find(fname));
    std::map<int, bool> enqueue_nodes;
    TF_RETURN_IF_ERROR(
        GetEnqueueOpsFromFunctionDef(&fdef_copy, &enqueue_nodes));
    bool updated = false;
    for (const auto& enqueue : enqueue_nodes) {
      TF_RETURN_IF_ERROR(UpdateFunctionDefEnqueueOp(
          enqueue.first, enqueue.second, &fdef_copy, &updated));
    }

    if (updated) {
      TF_RETURN_IF_ERROR(options.flib_def->ReplaceFunction(fname, fdef_copy));
    }
  }

  VLOG(1) << "UpdateTPUEmbeddingModePass::Run() finished";
  return OkStatus();
}

}  // namespace tensorflow
