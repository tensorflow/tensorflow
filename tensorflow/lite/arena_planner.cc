/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/arena_planner.h"

#include <stddef.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/simple_memory_arena.h"

namespace tflite {
namespace {

constexpr int32_t kNodeNotAssigned = std::numeric_limits<int32_t>::max();

}  // namespace

ArenaPlanner::ArenaPlanner(TfLiteContext* context,
                           std::unique_ptr<GraphInfo> graph_info,
                           bool preserve_all_tensors, int tensor_alignment,
                           int subgraph_index)
    : context_(context),
      graph_info_(std::move(graph_info)),
      arena_(kDefaultArenaAlignment, subgraph_index),
      persistent_arena_(kDefaultArenaAlignment, subgraph_index),
      preserve_all_tensors_(preserve_all_tensors),
      tensor_alignment_(tensor_alignment) {}

ArenaPlanner::~ArenaPlanner() {}

std::intptr_t ArenaPlanner::BasePointer(TfLiteAllocationType type) {
  if (type == kTfLiteArenaRwPersistent) {
    return persistent_arena_.BasePointer();
  }
  if (type == kTfLiteArenaRw) {
    return arena_.BasePointer();
  }
  return 0;
}

TfLiteStatus ArenaPlanner::ResetAllocations() {
  TF_LITE_ENSURE_STATUS(arena_.ClearPlan());
  TF_LITE_ENSURE_STATUS(persistent_arena_.ClearPlan());
  allocs_.clear();
  allocs_.resize(graph_info_->num_tensors());
  return kTfLiteOk;
}

TfLiteStatus ArenaPlanner::ResetAllocationsAfter(int node) {
  for (int i = 0; i < static_cast<int>(allocs_.size()); ++i) {
    if (allocs_[i].first_node > node && allocs_[i].size > 0) {
      TfLiteTensor& tensor = *graph_info_->tensor(i);
      if (tensor.allocation_type == kTfLiteArenaRw) {
        TF_LITE_ENSURE_STATUS(arena_.Deallocate(context_, allocs_[i]));
        allocs_[i].reset();
        tensor.data.raw = nullptr;
      }
    }
  }

  return kTfLiteOk;
}

TfLiteStatus ArenaPlanner::PlanAllocations() {
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
    refcounts[tensor_index]++;
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
  for (size_t i = 0; i < graph_info_->num_execution_nodes(); ++i) {
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
  for (size_t i = 0; i < graph_info_->num_execution_nodes(); ++i) {
    const TfLiteNode& node = graph_info_->node(i);

    // First queue output tensors for allocation.
    TfLiteIntArray* node_outputs = node.outputs;
    for (int j = 0; j < node_outputs->size; ++j) {
      int tensor_index = node_outputs->data[j];
      TF_LITE_ENSURE_STATUS(allocate(i, tensor_index));
    }

    // Then update the ref-counts of the node's inputs, and if necessary queue
    // them for deallocation.
    if (!preserve_all_tensors_) {
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
  }

  // Note that graph outputs will never be scheduled for deallocation. We
  // could do that here for completeness, but it won't have any effect.
  return kTfLiteOk;
}

TfLiteStatus ArenaPlanner::ExecuteAllocations(int first_node, int last_node) {
  // Grow the size of `allocs_` if necessary. This allows allocating temporary
  // tensors in op's `prepare` function.
  TF_LITE_ENSURE(context_, graph_info_->num_tensors() >= allocs_.size());
  alloc_node_.resize(graph_info_->num_tensors(), kNodeNotAssigned);
  dealloc_node_.resize(graph_info_->num_tensors(), kNodeNotAssigned);
  allocs_.resize(graph_info_->num_tensors());
  // Set allocation and deallocation for temporary tensors.
  for (size_t i = first_node; i <= static_cast<size_t>(last_node) &&
                              i < graph_info_->num_execution_nodes();
       ++i) {
    const TfLiteNode& node = graph_info_->node(i);
    TfLiteIntArray* node_temporaries = node.temporaries;
    for (int j = 0; j < node_temporaries->size; ++j) {
      int tensor_index = node_temporaries->data[j];
      alloc_node_[tensor_index] = i;
      if (!preserve_all_tensors_) {
        dealloc_node_[tensor_index] = i;
      }
    }
  }

  TF_LITE_ENSURE_STATUS(CalculateAllocations(first_node, last_node));
  TF_LITE_ENSURE_STATUS(Commit());

  for (int i = 0; i < static_cast<int>(graph_info_->num_tensors()); ++i) {
    TF_LITE_ENSURE_STATUS(ResolveTensorAllocation(i));
  }

  return kTfLiteOk;
}

TfLiteStatus ArenaPlanner::ReleaseNonPersistentMemory() {
  // Clear non-persistent arena's buffer.
  TF_LITE_ENSURE_STATUS(arena_.ReleaseBuffer());
  // Set data pointers for all non-persistent tensors to nullptr.
  for (int i = 0; i < static_cast<int>(graph_info_->num_tensors()); ++i) {
    TfLiteTensor& tensor = *graph_info_->tensor(i);
    if (tensor.allocation_type == kTfLiteArenaRw) {
      tensor.data.raw = nullptr;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus ArenaPlanner::AcquireNonPersistentMemory() {
  // First commit arena_ to allocate underlying buffer.
  TF_LITE_ENSURE_STATUS(arena_.Commit(context_));
  // Resolve allocations for all tensors not on the persistent arena.
  for (int i = 0; i < static_cast<int>(graph_info_->num_tensors()); ++i) {
    TfLiteTensor& tensor = *graph_info_->tensor(i);
    if (tensor.allocation_type == kTfLiteArenaRw) {
      TF_LITE_ENSURE_STATUS(ResolveTensorAllocation(i));
    }
  }
  return kTfLiteOk;
}

bool ArenaPlanner::HasNonPersistentMemory() {
  return arena_.GetBufferSize() != 0;
}

void ArenaPlanner::DumpDebugInfo(const std::vector<int>& execution_plan) const {
  arena_.DumpDebugInfo("kTfLiteArenaRw Dump:", execution_plan);
  persistent_arena_.DumpDebugInfo("kTfLiteArenaRwPersistent Dump:",
                                  execution_plan);
}

void ArenaPlanner::GetAllocInfo(size_t* arena_size,
                                size_t* arena_persist_size) const {
  *arena_size = arena_.GetBufferSize();
  *arena_persist_size = persistent_arena_.GetBufferSize();
}

TfLiteStatus ArenaPlanner::Commit() {
  TF_LITE_ENSURE_STATUS(arena_.Commit(context_));
  TF_LITE_ENSURE_STATUS(persistent_arena_.Commit(context_));
  return kTfLiteOk;
}

std::vector<int32_t> ArenaPlanner::CreateTensorAllocationVector(int first_node,
                                                                int last_node) {
  auto tensor_compare = [this](int idx1, int idx2) {
    // Tensors that have lifespan through the whole model inference time are
    // allocated at the beginning of memory slice. Their respective order
    // doesn't matter in fact, so here they are sorted by index.
    if (this->alloc_node_[idx1] == 0 &&
        this->dealloc_node_[idx1] == kNodeNotAssigned) {
      if (this->alloc_node_[idx2] == 0 &&
          this->dealloc_node_[idx2] == kNodeNotAssigned) {
        return idx1 < idx2;
      }
      return true;
    }
    if (this->alloc_node_[idx2] == 0 &&
        this->dealloc_node_[idx2] == kNodeNotAssigned) {
      return false;
    }

    // All other tensors are sorted in non-increasing order of their size.
    auto size1 = this->graph_info_->tensor(idx1)->bytes;
    auto size2 = this->graph_info_->tensor(idx2)->bytes;
    if (size1 != size2) {
      return size1 > size2;
    }
    // Tensors with equal size are sorted in order of their allocation time.
    return this->alloc_node_[idx1] < this->alloc_node_[idx2];
  };

  std::vector<int32_t> tensor_order;
  for (int i = 0; i < static_cast<int>(graph_info_->num_tensors()); ++i) {
    if (alloc_node_[i] >= first_node && alloc_node_[i] <= last_node) {
      tensor_order.push_back(i);
    }
  }
  // Indices of tensors in order their allocation offsets will be calculated.
  std::sort(tensor_order.begin(), tensor_order.end(), tensor_compare);

  return tensor_order;
}

TfLiteStatus ArenaPlanner::CalculateAllocations(int first_node, int last_node) {
  // Indices of tensors in order their allocation offsets will be calculated.
  const std::vector<int32_t> tensor_order =
      CreateTensorAllocationVector(first_node, last_node);

  // Deallocate if the tensor was already allocated.
  for (const auto& tensor_index : tensor_order) {
    TfLiteTensor& tensor = *graph_info_->tensor(tensor_index);
    if (tensor.allocation_type == kTfLiteArenaRw &&
        allocs_[tensor_index].size != 0) {
      TF_LITE_ENSURE_STATUS(arena_.Deallocate(context_, allocs_[tensor_index]));
    }
  }

  // Vector of ids of already allocated tensors, ordered by offset.
  for (const auto& tensor_index : tensor_order) {
    TfLiteTensor& tensor = *graph_info_->tensor(tensor_index);
    if (tensor.allocation_type == kTfLiteArenaRw) {
      TF_LITE_ENSURE_STATUS(
          arena_.Allocate(context_, tensor_alignment_, tensor.bytes,
                          tensor_index, alloc_node_[tensor_index],
                          dealloc_node_[tensor_index], &allocs_[tensor_index]));
    }
    // Check allocs_[].size to prevent from reallocation of persistent tensors.
    if (tensor.allocation_type == kTfLiteArenaRwPersistent &&
        allocs_[tensor_index].size == 0) {
      TF_LITE_ENSURE_STATUS(persistent_arena_.Allocate(
          context_, tensor_alignment_, tensor.bytes, tensor_index,
          /*first_node=*/alloc_node_[tensor_index],
          /*last_node=*/std::numeric_limits<int32_t>::max(),
          &allocs_[tensor_index]));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus ArenaPlanner::ResolveTensorAllocation(int tensor_index) {
  TfLiteTensor& tensor = *graph_info_->tensor(tensor_index);
  if (tensor.allocation_type == kTfLiteArenaRw) {
    // Skip resolution if the size of the tensor is zero, leaving it as a
    // nullptr.
    if (allocs_[tensor_index].size != 0) {
      TF_LITE_ENSURE_STATUS(arena_.ResolveAlloc(context_, allocs_[tensor_index],
                                                &tensor.data.raw));
    }
  }
  if (tensor.allocation_type == kTfLiteArenaRwPersistent) {
    TF_LITE_ENSURE_STATUS(persistent_arena_.ResolveAlloc(
        context_, allocs_[tensor_index], &tensor.data.raw));
  }
  return kTfLiteOk;
}

}  // namespace tflite
