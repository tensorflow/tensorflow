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
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {

TfLiteStatus CreateNewTensorWithDifferentType(TfLiteContext* context,
                                              const int original_tensor_index,
                                              TfLiteType new_type,
                                              TfLiteTensor** new_tensor,
                                              int* new_tensor_index) {
  TF_LITE_ENSURE_STATUS(context->AddTensors(context, 1, new_tensor_index));
  const TfLiteTensor& original_tensor = context->tensors[original_tensor_index];
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

TfLiteStatus GraphPartitionHelper::PartitionImpl(
    std::set<std::string>* unsupported_nodes_info, int start_node_index,
    int end_node_index) {
  const auto prepare_status = PrepareSupportedNodes(
      unsupported_nodes_info, start_node_index, end_node_index);
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
    std::set<std::string>* unsupported_nodes_info, int start_node_index,
    int end_node_index) {
  if (!is_node_supported_fn_) return kTfLiteOk;

  TfLiteIntArray* execution_plan = nullptr;
  auto status = context_->GetExecutionPlan(context_, &execution_plan);
  if (status != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context_, "Unable to get graph execution plan.\n");
    return status;
  }
  // context->GetExecutionPlan invalidates memory obtained from previous calls,
  // which is dangerous if a delegate's IsNodeSupportedFn uses it anywhere.
  // So we store a copy to ensure validity.
  num_total_nodes_ = execution_plan->size;
  original_execution_plan_ = TfLiteIntArrayCreate(execution_plan->size);
  std::memcpy(original_execution_plan_->data, execution_plan->data,
              num_total_nodes_ * sizeof(int32_t));

  supported_nodes_ = TfLiteIntArrayCreate(num_total_nodes_);
  supported_nodes_->size = 0;
  for (int node_id : TfLiteIntArrayView(original_execution_plan_)) {
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
#ifdef TFLITE_DEBUG_DELEGATE
      if (node_id < start_node_index) {
        continue;
      } else if (node_id > end_node_index) {
        break;
      }
#endif  // TFLITE_DEBUG_DELEGATE
      supported_nodes_->data[supported_nodes_->size++] = node_id;
    } else if (unsupported_nodes_info) {
      std::string node_info = GetOpNameByRegistration(*registration);
      node_info.append(": ");
      node_info.append(unsupported_details);
      unsupported_nodes_info->insert(node_info);
    }
  }

  num_supported_nodes_ = supported_nodes_->size;
  return kTfLiteOk;
}

std::vector<int>
FP16GraphPartitionHelper::GetNodesOfFirstNLargestPartitionsImpl(
    int n, int min_nodes_per_partition) {
  std::vector<int> ops_to_replace;

  if (num_supported_nodes() + constant_dequant_nodes_.size() ==
      num_total_nodes()) {
    // Scenario 1: Full Delegation.
    // We delegate all nodes in this case to avoid unnecessary partitions due to
    // FP16 DEQUANT nodes. This is safe to do since no non-delegated op needs
    // the output of such a DEQUANT.
    for (int node_id : TfLiteIntArrayView(original_execution_plan_)) {
      ops_to_replace.push_back(node_id);
    }
  } else {
    // Scenario 2: Partial Delegation.
    // In this case, we just select the top 'n' applicable node subsets to
    // delegate, devoid of any FP16 DEQUANT ops. Handling the latter is tricky
    // in partial delegation cases & causes edge cases if non-delegated nodes
    // consume their output. So we keep all of them on CPU.
    auto first_n_partitions =
        GetFirstNLargestPartitions(n, min_nodes_per_partition);
    if (first_n_partitions.empty()) return ops_to_replace;
    for (int i = 0; i < first_n_partitions.size(); ++i) {
      auto nodes = first_n_partitions[i]->nodes_to_replace;
      ops_to_replace.insert(ops_to_replace.end(), nodes->data,
                            nodes->data + nodes->size);
    }
  }

  // Modify the inputs of relevant ops that support fp16 constants.
  RemapFp16InputTensors(ops_to_replace);
  return ops_to_replace;
}

bool FP16GraphPartitionHelper::IsNodeSupported(
    TfLiteContext* context, TfLiteNode* node, TfLiteRegistration* registration,
    int node_id, std::string* unsupported_details) {
  if (registration->builtin_code == kTfLiteBuiltinDequantize) {
    auto& dequantize_input = context_->tensors[node->inputs->data[0]];
    if (dequantize_input.type == kTfLiteFloat16 &&
        IsConstantTensor(&dequantize_input)) {
      // Update mappings if this node is a fp16 DEQUANTIZE node that
      // works on a **constant** input tensor.
      // If the input is not a constant, the remapping that we do here will
      // cause bugs due to preceding ops such as DENSIFY.
      constant_dequant_map_[node->outputs->data[0]] = node->inputs->data[0];
      constant_dequant_nodes_[node->outputs->data[0]] = node_id;
      // We do not accept these ops right now.
      // This is done to support use-cases where a DEQUANTIZE output might be
      // consumed by a CPU op.
      return false;
    }
  }

  // To check if a (possibly) FP16 node is supported, we temporarily point the
  // node's inputs to the original fp16 tensors. This 'mutated' node is then
  // passed to the base IsNodeSupported function for checking. After the check,
  // we remap the original node inputs, so that the TFLite graph remains the
  // same.
  std::vector<int> orig_inputs;
  if (!constant_dequant_nodes_.empty()) {
    RemapFp16InputTensors(node, &orig_inputs);
  }

  const auto is_supported = GraphPartitionHelper::IsNodeSupported(
      context, node, registration, node_id, unsupported_details);

  if (!orig_inputs.empty() && node->inputs->size == orig_inputs.size()) {
    // Remapping happened. Restore original inputs.
    for (int j = 0; j < node->inputs->size; ++j) {
      node->inputs->data[j] = orig_inputs[j];
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
    const auto it = constant_dequant_map_.find(input_tid);
    if (it != constant_dequant_map_.end()) {
      inputs->data[j] = it->second;
      is_remapped = true;
    }
  }
  if (!is_remapped && orig_inputs) orig_inputs->clear();
}

}  // namespace delegates
}  // namespace tflite
