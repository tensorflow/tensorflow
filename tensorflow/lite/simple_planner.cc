/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/simple_planner.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/graph_info.h"

namespace tflite {

namespace {

constexpr int32_t kNodeNotAssigned = std::numeric_limits<int32_t>::max();

}  // namespace

SimplePlanner::SimplePlanner(TfLiteContext* context,
                             std::unique_ptr<GraphInfo> graph_info)
    : context_(context), graph_info_(std::move(graph_info)) {}

SimplePlanner::~SimplePlanner() { FreeAllAllocations(); }

void SimplePlanner::FreeAllAllocations() {
  for (int i = 0; i < static_cast<int>(allocs_.size()); ++i) {
    allocs_[i].free();
  }
}

TfLiteStatus SimplePlanner::ResetAllocations() {
  FreeAllAllocations();
  allocs_.clear();
  allocs_.resize(graph_info_->num_tensors());
  return kTfLiteOk;
}

TfLiteStatus SimplePlanner::ResetAllocationsAfter(int node) {
  TfLiteTensor* tensors = graph_info_->tensors();
  for (int i = 0; i < static_cast<int>(allocs_.size()); ++i) {
    if (allocs_[i].node > node && allocs_[i].size > 0) {
      TfLiteTensor& tensor = tensors[i];
      if (tensor.allocation_type == kTfLiteArenaRw) {
        allocs_[i].free();
        tensor.data.raw = nullptr;
      }
    }
  }

  return kTfLiteOk;
}

TfLiteStatus SimplePlanner::PlanAllocations() {
  // Invalidate any existing data.
  TF_LITE_ENSURE_STATUS(ResetAllocations());
  // Maybe other verb instead of 'Assigned'
  alloc_node_.assign(graph_info_->num_tensors(), kNodeNotAssigned);
  dealloc_node_.assign(graph_info_->num_tensors(), kNodeNotAssigned);

  // Keeps track of references to each tensor.
  std::vector<int> refcounts(graph_info_->num_tensors(), 0);

  auto allocate = [this](int node, int tensor) -> TfLiteStatus {
    if (alloc_node_[tensor] != kNodeNotAssigned) {
      // Tensor has already been allocated.
      return kTfLiteOk;
    }
    TF_LITE_ENSURE(context_, dealloc_node_[tensor] == kNodeNotAssigned);
    alloc_node_[tensor] = node;
    return kTfLiteOk;
  };

  auto deallocate = [this](int node, int tensor) -> TfLiteStatus {
    if (alloc_node_[tensor] == kNodeNotAssigned) {
      // We don't need to deallocate the tensor, that is never allocated.
      // This happened with the constant tensors.
      return kTfLiteOk;
    }
    TF_LITE_ENSURE(context_, dealloc_node_[tensor] == kNodeNotAssigned);
    dealloc_node_[tensor] = node;
    return kTfLiteOk;
  };

  // We must make sure the output tensors are never overwritten. We do that by
  // artificially adding one to their ref-counts so they are never selected
  // for deallocation.
  for (int tensor_index : graph_info_->outputs()) {
    if (tensor_index != kTfLiteOptionalTensor) {
      refcounts[tensor_index]++;
    }
  }

  // Variable tensors also should be ensured to be never overwritten and need to
  // be alive all the time.
  for (int tensor_index : graph_info_->variables()) {
    // Increase the reference count for variable tensors by one, so it will
    // never be deallocated.
    refcounts[tensor_index]++;
    // `variables` is a subgraph-level list and it should never be
    // kTfLiteOptionalTensor.
    TF_LITE_ENSURE(context_, tensor_index != kTfLiteOptionalTensor);
    // Variable tensor should be allocated at the very beginning.
    TF_LITE_ENSURE_STATUS(allocate(0, tensor_index));
  }

  // Queue all graph inputs for allocation and make sure they are never
  // overwritten.
  for (int tensor_index : graph_info_->inputs()) {
    if (tensor_index != kTfLiteOptionalTensor) {
      refcounts[tensor_index]++;
      TF_LITE_ENSURE_STATUS(allocate(0, tensor_index));
    }
  }

  // Count references to node input tensors.
  const size_t num_execution_nodes = graph_info_->num_execution_nodes();
  for (size_t i = 0; i < num_execution_nodes; ++i) {
    const TfLiteNode& node = graph_info_->node(i);
    TfLiteIntArray* node_inputs = node.inputs;
    for (int j = 0; j < node_inputs->size; ++j) {
      int tensor_index = node_inputs->data[j];
      if (tensor_index != kTfLiteOptionalTensor) {
        refcounts[tensor_index]++;
      }
    }
  }

  // Go through the graph in execution order.
  for (size_t i = 0; i < num_execution_nodes; ++i) {
    const TfLiteNode& node = graph_info_->node(i);

    // First queue output tensors for allocation.
    TfLiteIntArray* node_outputs = node.outputs;
    for (int j = 0; j < node_outputs->size; ++j) {
      int tensor_index = node_outputs->data[j];
      TF_LITE_ENSURE_STATUS(allocate(i, tensor_index));
    }

    // Then update the ref-counts of the node's inputs, and if necessary queue
    // them for deallocation.
    TfLiteIntArray* node_inputs = node.inputs;
    for (int j = 0; j < node_inputs->size; ++j) {
      int tensor_index = node_inputs->data[j];
      if (tensor_index != kTfLiteOptionalTensor) {
        refcounts[tensor_index]--;
        if (refcounts[tensor_index] == 0) {
          TF_LITE_ENSURE_STATUS(deallocate(i, tensor_index));
        }
      }
    }
  }

  // Note that graph outputs will never be scheduled for deallocation. We
  // could do that here for completeness, but it won't have any effect.
  return kTfLiteOk;
}

TfLiteStatus SimplePlanner::ExecuteAllocations(int first_node, int last_node) {
  alloc_node_.resize(graph_info_->num_tensors(), kNodeNotAssigned);
  dealloc_node_.resize(graph_info_->num_tensors(), kNodeNotAssigned);
  allocs_.resize(graph_info_->num_tensors());
  // Set allocation and deallocation for temporary tensors.
  const size_t num_execution_nodes = graph_info_->num_execution_nodes();
  for (size_t i = first_node;
       i <= static_cast<size_t>(last_node) && i < num_execution_nodes; ++i) {
    const TfLiteNode& node = graph_info_->node(i);
    TfLiteIntArray* node_temporaries = node.temporaries;
    for (int j = 0; j < node_temporaries->size; ++j) {
      int tensor_index = node_temporaries->data[j];
      alloc_node_[tensor_index] = i;
      dealloc_node_[tensor_index] = i;
    }
  }

  // Conduct the planned allocations.
  const int num_tensors = static_cast<int>(graph_info_->num_tensors());
  TfLiteTensor* tensors = graph_info_->tensors();
  for (int i = 0; i < num_tensors; ++i) {
    bool allocated = false;
    if (alloc_node_[i] >= first_node && alloc_node_[i] <= last_node) {
      TfLiteTensor& tensor = tensors[i];
      if (tensor.allocation_type == kTfLiteArenaRw) {
        if (allocs_[i].size != 0) {
          allocs_[i].free();
        }
        allocated = allocs_[i].alloc(tensor.bytes, alloc_node_[i]);
      } else if (tensor.allocation_type == kTfLiteArenaRwPersistent &&
                 allocs_[i].size == 0) {
        allocated = allocs_[i].alloc(tensor.bytes, alloc_node_[i]);
      }
    }
    if (allocated) {
      TF_LITE_ENSURE_STATUS(ResolveTensorAllocation(i));
    }
  }
  // TODO(b/191631156): Dealloc node if it's not needed.

  return kTfLiteOk;
}

TfLiteStatus SimplePlanner::ReleaseNonPersistentMemory() {
  // Set data pointers for all non-persistent tensors to nullptr.
  const int num_tensors = static_cast<int>(graph_info_->num_tensors());
  TfLiteTensor* tensors = graph_info_->tensors();
  for (int i = 0; i < num_tensors; ++i) {
    TfLiteTensor& tensor = tensors[i];
    if (tensor.allocation_type == kTfLiteArenaRw) {
      allocs_[i].free();
      tensor.data.raw = nullptr;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus SimplePlanner::AcquireNonPersistentMemory() {
  // Resolve allocations for all tensors not on the persistent arena.
  const int num_tensors = static_cast<int>(graph_info_->num_tensors());
  TfLiteTensor* tensors = graph_info_->tensors();
  for (int i = 0; i < num_tensors; ++i) {
    TfLiteTensor& tensor = tensors[i];
    if (tensor.allocation_type == kTfLiteArenaRw) {
      TF_LITE_ENSURE_STATUS(ResolveTensorAllocation(i));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus SimplePlanner::ResolveTensorAllocation(int tensor_index) {
  TfLiteTensor& tensor = *graph_info_->tensor(tensor_index);
  if (tensor.allocation_type == kTfLiteArenaRw) {
    // Skip resolution if the size of the tensor is zero, leaving it as a
    // nullptr.
    if (allocs_[tensor_index].size != 0) {
      tensor.data.raw = allocs_[tensor_index].ptr;
    }
  }
  if (tensor.allocation_type == kTfLiteArenaRwPersistent) {
    tensor.data.raw = allocs_[tensor_index].ptr;
  }
  return kTfLiteOk;
}

}  // namespace tflite
