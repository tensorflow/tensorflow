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

#include <algorithm>
#include <limits>
#include <type_traits>
#include <utility>

namespace tflite {
namespace {

constexpr size_t kNotAssigned = std::numeric_limits<size_t>::max();

}  // namespace

ArenaPlanner::ArenaPlanner(TfLiteContext* context,
                           std::unique_ptr<GraphInfo> graph_info,
                           bool preserve_inputs, bool preserve_intermediates,
                           int tensor_alignment)
    : context_(context),
      graph_info_(std::move(graph_info)),
      arena_(kDefaultArenaAlignment),
      persistent_arena_(kDefaultArenaAlignment),
      preserve_inputs_(preserve_inputs),
      preserve_intermediates_(preserve_intermediates),
      tensor_alignment_(tensor_alignment) {}

ArenaPlanner::~ArenaPlanner() {}

int64_t ArenaPlanner::BasePointer(TfLiteAllocationType type) {
  if (type == kTfLiteArenaRwPersistent) {
    return persistent_arena_.BasePointer();
  }
  if (type == kTfLiteArenaRw) {
    return arena_.BasePointer();
  }
  return 0;
}

TfLiteStatus ArenaPlanner::ResetAllocations() {
  TF_LITE_ENSURE_STATUS(arena_.Clear());
  TF_LITE_ENSURE_STATUS(persistent_arena_.Clear());
  allocs_.clear();
  allocs_.resize(graph_info_->num_tensors());
  order_.clear();
  was_added_.clear();
  return kTfLiteOk;
}

TfLiteStatus ArenaPlanner::PlanAllocations() {
  // Invalidate any existing data.
  TF_LITE_ENSURE_STATUS(ResetAllocations());
  // Maybe other verb instead of 'Assigned'
  alloc_node_.assign(graph_info_->num_tensors(), kNotAssigned);
  dealloc_node_.assign(graph_info_->num_tensors(), kNotAssigned);

  // Keeps track of references to each tensor.
  std::vector<int> refcounts(graph_info_->num_tensors(), 0);

  auto allocate = [this](int node, int tensor) -> TfLiteStatus {
    if (alloc_node_[tensor] != kNotAssigned) {
      // Tensor has already been allocated.
      return kTfLiteOk;
    }
    TF_LITE_ENSURE(context_, dealloc_node_[tensor] == kNotAssigned);
    alloc_node_[tensor] = node;
    return kTfLiteOk;
  };

  auto deallocate = [this](int node, int tensor) -> TfLiteStatus {
    if (alloc_node_[tensor] == kNotAssigned) {
      // We don't need to deallocate the tensor, that is never allocated.
      // This happened with the constant tensors.
      return kTfLiteOk;
    }
    TF_LITE_ENSURE(context_, dealloc_node_[tensor] == kNotAssigned);
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
    refcounts[tensor_index]++;
  }

  // Queue all graph inputs for allocation. If preserve_inputs_ is true, make
  // sure they never be overwritten.
  for (int tensor_index : graph_info_->inputs()) {
    if (tensor_index != kOptionalTensor) {
      if (preserve_inputs_) {
        refcounts[tensor_index]++;
      }
      TF_LITE_ENSURE_STATUS(allocate(0, tensor_index));
    }
  }

  // Queue all graph variable tensors for allocation.
  for (int tensor_index : graph_info_->variables()) {
    if (tensor_index != kOptionalTensor) {
      // Increase the reference count for input tensors by one, so it will
      // never be deallocated.
      TF_LITE_ENSURE_STATUS(allocate(0, tensor_index));
    }
  }

  // Count references to node input tensors.
  for (size_t i = 0; i < graph_info_->num_nodes(); ++i) {
    const TfLiteNode& node = graph_info_->node(i);
    TfLiteIntArray* node_inputs = node.inputs;
    for (int j = 0; j < node_inputs->size; ++j) {
      int tensor_index = node_inputs->data[j];
      if (tensor_index != kOptionalTensor) {
        refcounts[tensor_index]++;
      }
    }
  }

  // Queue all graph inputs for allocation.
  for (int tensor_index : graph_info_->inputs()) {
    if (tensor_index != kOptionalTensor) {
      TF_LITE_ENSURE_STATUS(allocate(0, tensor_index));
    }
  }
  // Go through the graph in execution order.
  for (size_t i = 0; i < graph_info_->num_nodes(); ++i) {
    const TfLiteNode& node = graph_info_->node(i);

    // First queue output tensors for allocation.
    TfLiteIntArray* node_outputs = node.outputs;
    for (int j = 0; j < node_outputs->size; ++j) {
      int tensor_index = node_outputs->data[j];
      TF_LITE_ENSURE_STATUS(allocate(i, tensor_index));
    }

    // Then update the ref-counts of the node's inputs, and if necessary queue
    // them for deallocation.
    if (!preserve_intermediates_) {
      TfLiteIntArray* node_inputs = node.inputs;
      for (int j = 0; j < node_inputs->size; ++j) {
        int tensor_index = node_inputs->data[j];
        if (tensor_index != kOptionalTensor) {
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
  alloc_node_.resize(graph_info_->num_tensors(), kNotAssigned);
  dealloc_node_.resize(graph_info_->num_tensors(), kNotAssigned);
  allocs_.resize(graph_info_->num_tensors());
  was_added_.assign(graph_info_->num_tensors(), false);
  order_.clear();
  // Set allocation and deallocation for temporary tensors.
  for (size_t i = first_node; i <= last_node && i < graph_info_->num_nodes();
       ++i) {
    const TfLiteNode& node = graph_info_->node(i);
    TfLiteIntArray* node_temporaries = node.temporaries;
    for (int j = 0; j < node_temporaries->size; ++j) {
      int tensor_index = node_temporaries->data[j];
      alloc_node_[tensor_index] = i;
      dealloc_node_[tensor_index] = i;
    }
  }

  TF_LITE_ENSURE_STATUS(CalculateAllocations(first_node, last_node));
  TF_LITE_ENSURE_STATUS(Commit());

  for (size_t i = 0; i < graph_info_->num_tensors(); ++i) {
    // TODO(ahentz): we could do this only for the tensors that were modified
    // in CalculateAllocations(), instead of redoing it for tensors that
    // already had proper pointers. However we must be very careful, because
    // SimpleMemoryArena::Commit() could move the base pointer.
    TF_LITE_ENSURE_STATUS(ResolveTensorAllocation(i));
  }

  return kTfLiteOk;
}

TfLiteStatus ArenaPlanner::CalculateAllocations(int first_node, int last_node) {
  for (size_t i = 0; i < graph_info_->num_tensors(); ++i) {
    if (alloc_node_[i] >= first_node && alloc_node_[i] <= last_node) {
      AddTensorIfNeeded(i);
    }
  }
  std::sort(order_.begin(), order_.end(), CompareBySize(this));

  // Vector of ids of already allocated tensors, ordered by offset.
  for (const auto& tensor_index : order_) {
    TfLiteTensor& tensor = *graph_info_->tensor(tensor_index);
    if (tensor.allocation_type == kTfLiteArenaRw) {
      TF_LITE_ENSURE_STATUS(arena_.Allocate(
          context_, tensor_alignment_, tensor.bytes, alloc_node_[tensor_index],
          dealloc_node_[tensor_index], &allocs_[tensor_index]));
    }
    if (tensor.allocation_type == kTfLiteArenaRwPersistent) {
      TF_LITE_ENSURE_STATUS(persistent_arena_.Allocate(
          context_, tensor_alignment_, tensor.bytes, alloc_node_[tensor_index],
          std::numeric_limits<size_t>::max(), &allocs_[tensor_index]));
    }
  }
  return kTfLiteOk;
}

void ArenaPlanner::AddTensorIfNeeded(int tensor_index) {
  if (!was_added_[tensor_index]) {
    was_added_[tensor_index] = true;
    order_.push_back(tensor_index);
  }
}

bool ArenaPlanner::CompareBySize::operator()(const int idx1,
                                             const int idx2) const {
  // Tensors that have lifespan through the whole model inference time are
  // allocated at the beginning of memory slice. Their respective order doesn't
  // matter in fact, so here they are sorted by index.
  if (planner->alloc_node_[idx1] == 0 &&
      planner->dealloc_node_[idx1] == kNotAssigned) {
    if (planner->alloc_node_[idx2] == 0 &&
        planner->dealloc_node_[idx2] == kNotAssigned) {
      return idx1 < idx2;
    }
    return true;
  }
  if (planner->alloc_node_[idx2] == 0 &&
      planner->dealloc_node_[idx2] == kNotAssigned) {
    return false;
  }

  // All other tensors are sorted in non-increasing order of their size.
  auto size1 = planner->graph_info_->tensor(idx1)->bytes;
  auto size2 = planner->graph_info_->tensor(idx2)->bytes;
  if (size1 != size2) {
    return size1 > size2;
  }

  // Tensors with equal size are sorted in order of their allocation time.
  return planner->alloc_node_[idx1] < planner->alloc_node_[idx2];
}

TfLiteStatus ArenaPlanner::Commit() {
  TF_LITE_ENSURE_STATUS(arena_.Commit(context_));
  TF_LITE_ENSURE_STATUS(persistent_arena_.Commit(context_));
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
