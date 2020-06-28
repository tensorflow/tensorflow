/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/utils.h"

#include <algorithm>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"

namespace tflite {
namespace delegates {

TfLiteStatus CreateNewTensorWithDifferentType(TfLiteContext* context,
                                              const int original_tensor_index,
                                              TfLiteType new_type,
                                              TfLiteTensor** new_tensor,
                                              int* new_tensor_index) {
  const TfLiteTensor& original_tensor = context->tensors[original_tensor_index];
  TF_LITE_ENSURE_STATUS(context->AddTensors(context, 1, new_tensor_index));
  *new_tensor = &context->tensors[*new_tensor_index];
  (*new_tensor)->type = new_type;
  (*new_tensor)->allocation_type = kTfLiteArenaRw;
  const auto* original_dims = original_tensor.dims;
  TfLiteIntArray* dims = TfLiteIntArrayCreate(original_dims->size);
  for (int i = 0; i < original_dims->size; ++i) {
    dims->data[i] = original_dims->data[i];
  }
  if (context->ResizeTensor(context, *new_tensor, dims) != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context, "Could not resize new delegate tensor");
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus GraphPartitionHelper::Partition(
    std::set<std::string>* unsupported_nodes_info) {
  const auto prepare_status = PrepareSupportedNodes(unsupported_nodes_info);
  if (prepare_status != kTfLiteOk) return prepare_status;

  TfLiteDelegateParams* partition_params_array_ = nullptr;
  int num_partitions_ = 0;
  if (context_->PreviewDelegatePartitioning(context_, supported_nodes_,
                                            &partition_params_array_,
                                            &num_partitions_) != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context_, "Unable to preview delegate partition.\n");
    return kTfLiteError;
  }

  for (int i = 0; i < num_partitions_; ++i) {
    partitions_.push_back(partition_params_array_ + i);
  }

  return kTfLiteOk;
}

std::vector<TfLiteDelegateParams*>
GraphPartitionHelper::GetFirstNLargestPartitions(
    int n, int min_nodes_per_partition) const {
  // In general, the number of partitions in a delegate is never likely to be
  // high enough to cause latency issues. Also considering this is generally a
  // one-time work, we simply unconditionally sort partitions here according to
  // the size.
  std::vector<TfLiteDelegateParams*> sorted_partitions(partitions_);
  std::sort(sorted_partitions.begin(), sorted_partitions.end(),
            [](TfLiteDelegateParams* left, TfLiteDelegateParams* right) {
              // Reverse sort
              return left->nodes_to_replace->size >
                     right->nodes_to_replace->size;
            });

  std::vector<TfLiteDelegateParams*> results;
  auto p_it = sorted_partitions.begin();
  const int total = sorted_partitions.size();
  for (int i = 0; i < std::min(total, n); ++i, ++p_it) {
    auto* p = (*p_it);
    if (p->nodes_to_replace->size < min_nodes_per_partition) {
      break;
    }
    results.push_back(p);
  }
  return results;
}

std::vector<int> GraphPartitionHelper::GetNodesOfFirstNLargestPartitionsImpl(
    int n, int min_nodes_per_partition) {
  auto first_n_partitions =
      GetFirstNLargestPartitions(n, min_nodes_per_partition);
  std::vector<int> ops_to_replace;
  for (const auto p : first_n_partitions) {
    auto nodes = p->nodes_to_replace;
    ops_to_replace.insert(ops_to_replace.end(), nodes->data,
                          nodes->data + nodes->size);
  }
  return ops_to_replace;
}

TfLiteStatus GraphPartitionHelper::PrepareSupportedNodes(
    std::set<std::string>* unsupported_nodes_info) {
  if (!is_node_supported_fn_) return kTfLiteOk;

  TfLiteIntArray* execution_plan = nullptr;
  auto status = context_->GetExecutionPlan(context_, &execution_plan);
  if (status != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context_, "Unable to get graph execution plan.\n");
    return status;
  }

  num_total_nodes_ = execution_plan->size;
  supported_nodes_ = TfLiteIntArrayCreate(num_total_nodes_);
  supported_nodes_->size = 0;
  for (int node_id : TfLiteIntArrayView(execution_plan)) {
    TfLiteNode* node;
    TfLiteRegistration* registration;

    status = context_->GetNodeAndRegistration(context_, node_id, &node,
                                              &registration);
    if (status != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context_,
                         "Couldn't get node and registration info for op: %d\n",
                         node_id);
      supported_nodes_->size = 0;
      return status;
    }

    std::string unsupported_details;
    if (IsNodeSupported(context_, node, registration, node_id,
                        &unsupported_details)) {
      supported_nodes_->data[supported_nodes_->size++] = node_id;
    } else if (unsupported_nodes_info) {
      std::string node_info = GetOpNameByRegistration(*registration);
      node_info.append(": ");
      node_info.append(unsupported_details);
      unsupported_nodes_info->insert(node_info);
    }
  }
  return kTfLiteOk;
}

std::vector<int>
FP16GraphPartitionHelper::GetNodesOfFirstNLargestPartitionsImpl(
    int n, int min_nodes_per_partition) {
  auto first_n_partitions =
      GetFirstNLargestPartitions(n, min_nodes_per_partition);
  std::vector<int> ops_to_replace;
  if (first_n_partitions.empty()) return ops_to_replace;

  // Handle the first delegated partition specially.
  // All fp16 DEQUANTIZE nodes whose consumers exist only in this partition can
  // be added to the ops to delegate. Others have to be preserved in the graph,
  // since the partitioning algorithm will put such nodes greedily in the first
  // partition.
  const auto* first_partition = first_n_partitions[0];
  std::unordered_map<int, int> delegated_dequant_consumers;
  for (int i = 0; i < first_partition->nodes_to_replace->size; ++i) {
    const int node_id = first_partition->nodes_to_replace->data[i];
    ops_to_replace.push_back(node_id);
    TfLiteNode* node;
    TfLiteRegistration* registration;
    const auto status = context_->GetNodeAndRegistration(context_, node_id,
                                                         &node, &registration);
    if (status != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context_,
                         "Couldn't get node and registration info for op: %d\n",
                         node_id);
      ops_to_replace.clear();
      return ops_to_replace;
    }
    // See if any input to the op is a (converted) fp16 value. If yes, increment
    // its value in delegated_dequant_consumers.
    for (int j = 0; j < node->inputs->size; ++j) {
      const int input_tid = node->inputs->data[j];
      if (dequant_consumers_.find(input_tid) != dequant_consumers_.end()) {
        delegated_dequant_consumers[input_tid] += 1;
      }
    }
  }
  // Check all dequant nodes that have some consumers in the first partition.
  // If the number of delegated consumers is same as total number of consumers,
  // add the corresponding DEQUANTIZE op to the delegated nodes.
  for (auto tensor_and_consumers : delegated_dequant_consumers) {
    if (dequant_consumers_[tensor_and_consumers.first] ==
        tensor_and_consumers.second) {
      ops_to_replace.emplace_back(dequant_nodes_[tensor_and_consumers.first]);
    }
  }

  // For all other partitions after the first one, insert all nodes into
  // ops_to_replace.
  for (int i = 1; i < first_n_partitions.size(); ++i) {
    auto nodes = first_n_partitions[i]->nodes_to_replace;
    ops_to_replace.insert(ops_to_replace.end(), nodes->data,
                          nodes->data + nodes->size);
  }

  // Modify the inputs of relevant ops that support fp16 constants.
  // TODO(b/156707497): Ensure that these inputs are remapped during the
  // delegate's 'free', so that CPU fallback works for fp16 models.
  RemapFp16InputTensors(ops_to_replace);
  return ops_to_replace;
}

bool FP16GraphPartitionHelper::IsNodeSupported(
    TfLiteContext* context, TfLiteNode* node, TfLiteRegistration* registration,
    int node_id, std::string* unsupported_details) {
  if (registration->builtin_code == kTfLiteBuiltinDequantize &&
      context_->tensors[node->inputs->data[0]].type ==
          TfLiteType::kTfLiteFloat16) {
    // Update mappings if this node is a fp16 DEQUANTIZE node.
    dequant_map_[node->outputs->data[0]] = node->inputs->data[0];
    dequant_nodes_[node->outputs->data[0]] = node_id;
    // We do not accept these ops right now.
    // This is done to support use-cases where a DEQUANTIZE output might be
    // consumed by a CPU op.
    return false;
  }

  // To check if a (possibly) FP16 node is supported, we temporarily point the
  // node's inputs to the original fp16 tensors. This 'mutated' node is then
  // passed to the base IsNodeSupported function for checking. After the check,
  // we remap the original node inputs, so that the TFLite graph remains the
  // same.
  std::vector<int> orig_inputs;
  if (!dequant_nodes_.empty()) {
    RemapFp16InputTensors(node, &orig_inputs);
  }

  const auto is_supported = GraphPartitionHelper::IsNodeSupported(
      context, node, registration, node_id, unsupported_details);

  if (!orig_inputs.empty() && node->inputs->size == orig_inputs.size()) {
    // Remapping happened. Restore original inputs.
    for (int j = 0; j < node->inputs->size; ++j) {
      node->inputs->data[j] = orig_inputs[j];
      if (dequant_nodes_.find(orig_inputs[j]) != dequant_nodes_.end()) {
        // If its a fp16 tensor, increment number of consumers of the
        // corresponding DEQUANTIZE.
        dequant_consumers_[orig_inputs[j]] += 1;
      }
    }
  }
  return is_supported;
}

void FP16GraphPartitionHelper::RemapFp16InputTensors(
    const std::vector<int>& nodes) const {
  for (int node_id : nodes) {
    TfLiteNode* node;
    TfLiteRegistration* registration;
    TfLiteStatus status = context_->GetNodeAndRegistration(
        context_, node_id, &node, &registration);
    if (status != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context_,
                         "Couldn't get node and registration info for op: %d\n",
                         node_id);
    }
    RemapFp16InputTensors(node, nullptr /* orig_inputs*/);
  }
}

void FP16GraphPartitionHelper::RemapFp16InputTensors(
    TfLiteNode* node, std::vector<int>* orig_inputs) const {
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
    const auto it = dequant_map_.find(input_tid);
    if (it != dequant_map_.end()) {
      inputs->data[j] = it->second;
      is_remapped = true;
    }
  }
  if (!is_remapped && orig_inputs) orig_inputs->clear();
}

}  // namespace delegates
}  // namespace tflite
