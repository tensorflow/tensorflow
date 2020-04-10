/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_BUILDER_HELPER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_BUILDER_HELPER_H_

#include <set>
#include <string>
#include <unordered_map>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace gpu {
inline absl::Status GetNodeAndRegistration(TfLiteContext* context, int node_id,
                                           TfLiteNode** tflite_node,
                                           TfLiteRegistration** registration) {
  if (context->GetNodeAndRegistration(context, node_id, tflite_node,
                                      registration) != kTfLiteOk) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Couldn't get node and registration info for op: ", node_id));
  }
  return absl::OkStatus();
}

using IsNodeSupportedFn = tflite::delegates::IsNodeSupportedFn;

class GraphWithDequantPartitionHelper
    : public tflite::delegates::GraphPartitionHelper {
 public:
  GraphWithDequantPartitionHelper(TfLiteContext* context,
                                  IsNodeSupportedFn is_node_supported_fn)
      : GraphPartitionHelper(context, std::move(is_node_supported_fn)) {}

  TfLiteStatus Partition(
      std::set<std::string>* unsupported_nodes_info) override {
    const auto status = GraphPartitionHelper::Partition(unsupported_nodes_info);
    // Clean up those partitions that have a single dequant op. NoteThose
    // removed dequant ops have to be reserved in the graph and should not be
    // delegated.
    RemoveSingleDequantNodePartitions();
    return status;
  }

  // Returns a list of node indices of all nodes from the first n largest
  // partitions. If there are fewer paritions than n, all nodes will be
  // returned. The partition is ranked according to the number of nodes.
  std::vector<int> GetNodesOfFirstNLargestPartitions(int n) {
    // We first get partitions to reduce the number of nodes to be checked in
    // deciding which dequant ops could actually be replaced. And then we
    // remap input-tensor to dequant nodes' inputs and remove those
    // to-be-reserved dequant nodes.
    auto first_nps = GetFirstNLargestPartitions(n);
    std::vector<int> ops_to_replace;
    for (const auto p : first_nps) {
      auto nodes = p->nodes_to_replace;
      ops_to_replace.insert(ops_to_replace.end(), nodes->data,
                            nodes->data + nodes->size);
    }
    RemapInputTensors(ops_to_replace);
    RemoveReservedDequantsFromNodes(&ops_to_replace);
    return ops_to_replace;
  }

 protected:
  bool IsNodeSupported(TfLiteContext* context, TfLiteNode* node,
                       TfLiteRegistration* registration, int node_id,
                       std::string* unsupported_details) override {
    // If we need to handle dequant nodes, we have to remap input tensors of
    // this node if some of them come from a dequant node before testing if
    // the node is supported.
    std::vector<int> orig_inputs;
    if (RecordAndRemapInputTensors(registration->builtin_code, node_id, node,
                                   &orig_inputs)) {
      // We have a dequant op here. Note that we retrun an Ok status because a
      // dequant node is first added as supported. Later, this dequant node
      // will be removed if it has to be preserved in the graph which happens
      // when its immediate downstream nodes cannot be supported.
      return true;
    }
    const auto status = GraphPartitionHelper::IsNodeSupported(
        context, node, registration, node_id, unsupported_details);
    RestoreToOrigInputTensors(node, orig_inputs);
    return status;
  }

 private:
  // Record 'node' if it is a dequant op (i.e. a fp16 one here) and return true.
  // When it's not a dequant op, remap its inputs to the inputs of the preceding
  // dequant if there's a one and returns false. 'orig_inputs' records original
  // input tensor ids of this node if any input is remapped.
  bool RecordAndRemapInputTensors(int32_t op_code, int node_id,
                                  TfLiteNode* node,
                                  std::vector<int>* orig_inputs) {
    orig_inputs->clear();
    // Record the dequant node.
    if (op_code == kTfLiteBuiltinDequantize &&
        context_->tensors[node->inputs->data[0]].type ==
            TfLiteType::kTfLiteFloat16) {
      dequant_nodes_[node->outputs->data[0]] = node->inputs->data[0];
      return true;
    }
    // For a dequantize op, there's no need to remap its input tensors.
    if (dequant_nodes_.empty()) return false;
    RemapInputTensors(node, orig_inputs);
    return false;
  }

  // Restore inputs of 'node' to 'orig_inputs' only if two sizes match.
  void RestoreToOrigInputTensors(TfLiteNode* node,
                                 const std::vector<int>& orig_inputs) {
    if (node->inputs->size != orig_inputs.size()) return;
    for (int j = 0; j < node->inputs->size; ++j) {
      node->inputs->data[j] = orig_inputs[j];
    }
  }

  // Remap input tensors of every node in 'nodes' (i.e. node indices) if some of
  // them are from dequant ops.
  void RemapInputTensors(const std::vector<int>& nodes) const {
    for (int node_id : nodes) {
      TfLiteNode* node;
      TfLiteRegistration* registration;
      GetNodeAndRegistration(context_, node_id, &node, &registration)
          .IgnoreError();
      RemapInputTensors(node, nullptr /* orig_inputs*/);
    }
  }

  void RemoveSingleDequantNodePartitions() {
    auto it = partitions_.begin();
    while (it != partitions_.end()) {
      auto p = *it;
      if (p->nodes_to_replace->size != 1) {
        ++it;
        continue;
      }
      int node_id = p->nodes_to_replace->data[0];
      TfLiteNode* node = nullptr;
      TfLiteRegistration* registration = nullptr;
      GetNodeAndRegistration(context_, node_id, &node, &registration)
          .IgnoreError();
      if (registration->builtin_code != kTfLiteBuiltinDequantize ||
          context_->tensors[node->inputs->data[0]].type !=
              TfLiteType::kTfLiteFloat16) {
        ++it;
        continue;
      }
      // Note such dequant nodes have to be preserved in the graph as dequant
      // ops are not actually supported in the GPU delegate.
      dequant_nodes_to_save_.insert(node_id);
      it = partitions_.erase(it);
    }
  }

  void RemoveReservedDequantsFromNodes(std::vector<int>* nodes) {
    if (dequant_nodes_to_save_.empty()) return;
    auto it = nodes->begin();
    while (it != nodes->end()) {
      if (dequant_nodes_to_save_.find(*it) == dequant_nodes_to_save_.end()) {
        ++it;
        continue;
      }
      it = nodes->erase(it);
    }
  }

  // Remap input tensors of a single 'node' if some of come from a dequant op.
  // If 'orig_inputs' isn't nullptr, it records original input tensor ids of
  // this node if any input is remapped.
  void RemapInputTensors(TfLiteNode* node,
                         std::vector<int>* orig_inputs) const {
    TfLiteIntArray* inputs = node->inputs;
    auto inputs_view = TfLiteIntArrayView(inputs);
    // Prepopulate 'orig_inputs' first and clear it if there's no input from a
    // dequant op.
    if (orig_inputs) {
      orig_inputs->clear();
      orig_inputs->reserve(inputs->size);
      for (auto tid : inputs_view) {
        orig_inputs->push_back(tid);
      }
    }
    // Fix this node's inputs (i.e. prune out the preceding dequantize node) in
    // order to test if it is supported.
    bool is_remapped = false;
    for (int j = 0; j < inputs->size; ++j) {
      const int input_tid = inputs->data[j];
      const auto it = dequant_nodes_.find(input_tid);
      if (it != dequant_nodes_.end()) {
        inputs->data[j] = it->second;
        is_remapped = true;
      }
    }
    if (!is_remapped && orig_inputs) orig_inputs->clear();
  }

  // A map recording dequantize nodes's input/output tensors of this selected
  // graph. The key is the output tensor id, and the value is the input tensor
  // id.
  std::unordered_map<int, int> dequant_nodes_;

  // A set of dequant nodes as in node indices that have to be preserved in the
  // graph.
  std::set<int> dequant_nodes_to_save_;
};
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_BUILDER_HELPER_H_
