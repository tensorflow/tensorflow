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

TfLiteStatus FP16GraphPartitionHelper::Partition(
    std::set<std::string>* unsupported_nodes_info) {
  const auto status = GraphPartitionHelper::Partition(unsupported_nodes_info);
  // Clean up those partitions that have a single dequant op. NoteThose
  // removed dequant ops have to be reserved in the graph and should not be
  // delegated.
  RemoveSingleDequantNodePartitions();
  return status;
}

std::vector<int>
FP16GraphPartitionHelper::GetNodesOfFirstNLargestPartitionsImpl(
    int n, int min_nodes_per_partition) {
  std::vector<int> ops_to_replace =
      GraphPartitionHelper::GetNodesOfFirstNLargestPartitionsImpl(
          n, min_nodes_per_partition);
  RemapInputTensors(ops_to_replace);
  RemoveReservedDequantsFromNodes(&ops_to_replace);
  return ops_to_replace;
}

bool FP16GraphPartitionHelper::IsNodeSupported(
    TfLiteContext* context, TfLiteNode* node, TfLiteRegistration* registration,
    int node_id, std::string* unsupported_details) {
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

bool FP16GraphPartitionHelper::RecordAndRemapInputTensors(
    int32_t op_code, int node_id, TfLiteNode* node,
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

void FP16GraphPartitionHelper::RestoreToOrigInputTensors(
    TfLiteNode* node, const std::vector<int>& orig_inputs) {
  if (node->inputs->size != orig_inputs.size()) return;
  for (int j = 0; j < node->inputs->size; ++j) {
    node->inputs->data[j] = orig_inputs[j];
  }
}

void FP16GraphPartitionHelper::RemapInputTensors(
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
    RemapInputTensors(node, nullptr /* orig_inputs*/);
  }
}

void FP16GraphPartitionHelper::RemoveSingleDequantNodePartitions() {
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

    TfLiteStatus status = context_->GetNodeAndRegistration(
        context_, node_id, &node, &registration);
    if (status != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context_,
                         "Couldn't get node and registration info for op: %d\n",
                         node_id);
    }
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

void FP16GraphPartitionHelper::RemoveReservedDequantsFromNodes(
    std::vector<int>* nodes) {
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

void FP16GraphPartitionHelper::RemapInputTensors(
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
    const auto it = dequant_nodes_.find(input_tid);
    if (it != dequant_nodes_.end()) {
      inputs->data[j] = it->second;
      is_remapped = true;
    }
  }
  if (!is_remapped && orig_inputs) orig_inputs->clear();
}

}  // namespace delegates
}  // namespace tflite
