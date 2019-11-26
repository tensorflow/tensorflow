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

#include <cstdint>
#include <utility>

namespace tflite {

struct AllocationInfo {
  // The node index requesting this allocation.
  int node;
  // The tensor index to be allocated or deallocated.
  int tensor;
  // Whether to allocate or deallocate
  enum Type { ALLOC, DEALLOC } type;
};

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
  // Note that we only clear the alloc_queue_ when re-planning allocations, as
  // it should only change when the graph topology itself changes.
  return kTfLiteOk;
}

TfLiteStatus ArenaPlanner::PlanAllocations() {
  // Invalidate any existing data.
  TF_LITE_ENSURE_STATUS(ResetAllocations());
  // The alloc_queue_ is specific to the graph topology, and will be
  // completely reconstructed from graph data here.
  alloc_queue_.clear();

  // Keeps track of references to each tensor.
  std::vector<int> refcounts(graph_info_->num_tensors(), 0);
  // `allocated` and `deallocated` are technically list of boolean values.
  // We're saving the compiled binary size by using `vector<int>`.
  std::vector<int> allocated(graph_info_->num_tensors(), false);
  std::vector<int> deallocated(graph_info_->num_tensors(), false);

  auto allocate = [this, &allocated, &deallocated](int node,
                                                   int tensor) -> TfLiteStatus {
    if (allocated[tensor]) {
      return kTfLiteOk;
    }
    TF_LITE_ENSURE(context_, !deallocated[tensor]);
    alloc_queue_.push_back({node, tensor, AllocationInfo::ALLOC});
    allocated[tensor] = true;
    return kTfLiteOk;
  };

  auto deallocate = [this, &allocated, &deallocated](
                        int node, int tensor) -> TfLiteStatus {
    if (!allocated[tensor]) {
      // Do not enqueue a DEALLOC if the tensor is never allocated.
      // This happened with the constant tensors.
      return kTfLiteOk;
    }
    TF_LITE_ENSURE(context_, !deallocated[tensor]);
    alloc_queue_.push_back({node, tensor, AllocationInfo::DEALLOC});
    return kTfLiteOk;
  };

  // There will be an entry in alloc_queue_ for the allocation of each tensor
  // and another for their deallocation.
  alloc_queue_.reserve(2 * graph_info_->num_tensors());

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
    if (tensor_index != kTfLiteOptionalTensor) {
      if (preserve_inputs_) {
        refcounts[tensor_index]++;
      }
      TF_LITE_ENSURE_STATUS(allocate(0, tensor_index));
    }
  }

  // Queue all graph variable tensors for allocation.
  for (int tensor_index : graph_info_->variables()) {
    if (tensor_index != kTfLiteOptionalTensor) {
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
      if (tensor_index != kTfLiteOptionalTensor) {
        refcounts[tensor_index]++;
      }
    }
  }

  // Queue all graph inputs for allocation.
  for (int tensor_index : graph_info_->inputs()) {
    if (tensor_index != kTfLiteOptionalTensor) {
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
  allocs_.resize(graph_info_->num_tensors());

  TF_LITE_ENSURE_STATUS(CalculateAllocations(first_node, last_node));
  TF_LITE_ENSURE_STATUS(Commit());

  for (int i = 0; i < static_cast<int>(graph_info_->num_tensors()); ++i) {
    // TODO(ahentz): we could do this only for the tensors that were modified
    // in CalculateAllocations(), instead of redoing it for tensors that
    // already had proper pointers. However we must be very careful, because
    // SimpleMemoryArena::Commit() could move the base pointer.
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

TfLiteStatus ArenaPlanner::Commit() {
  TF_LITE_ENSURE_STATUS(arena_.Commit(context_));
  TF_LITE_ENSURE_STATUS(persistent_arena_.Commit(context_));
  return kTfLiteOk;
}

TfLiteStatus ArenaPlanner::CalculateAllocations(int first_node, int last_node) {
  int active_node = first_node;
  // When dynamic tensors are present this method is called multiple times.
  // The items in the alloc_queue_ referring to nodes before first_node were
  // processed previously and should be skipped. Entries after last_node are
  // not yet ready to be handled.
  for (const auto& alloc_info : alloc_queue_) {
    if (alloc_info.node < first_node) continue;
    if (alloc_info.node > last_node) break;
    if (alloc_info.node == active_node) {
      // This is the first allocation/deallocation for a given node.  It is
      // time to deallocate the previous temporaries and allocate new ones.
      if (active_node != first_node) {
        TF_LITE_ENSURE_STATUS(
            CalculateDeallocationOfInternalTensors(active_node - 1));
      }
      TF_LITE_ENSURE_STATUS(CalculateAllocationOfInternalTensors(active_node));
      ++active_node;
    }
    // Handle the current item.
    if (alloc_info.type == AllocationInfo::ALLOC) {
      TF_LITE_ENSURE_STATUS(CalculateTensorAllocation(alloc_info.tensor));
    } else {
      TF_LITE_ENSURE_STATUS(CalculateTensorDeallocation(alloc_info.tensor));
    }
  }

  // For the case if the graph is empty the node index can be negative since we
  // substract from the active node, so the node_index can be zero for those
  // cases
  if (active_node > 0) {
    // Don't forget to deallocate temporaries of last node.
    TF_LITE_ENSURE_STATUS(
        CalculateDeallocationOfInternalTensors(active_node - 1));
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

TfLiteStatus ArenaPlanner::CalculateTensorAllocation(int tensor_index) {
  TfLiteTensor& tensor = *graph_info_->tensor(tensor_index);
  if (tensor.allocation_type == kTfLiteArenaRw) {
    TF_LITE_ENSURE_STATUS(arena_.Allocate(
        context_, tensor_alignment_, tensor.bytes, &allocs_[tensor_index]));
  }
  if (tensor.allocation_type == kTfLiteArenaRwPersistent) {
    TF_LITE_ENSURE_STATUS(persistent_arena_.Allocate(
        context_, tensor_alignment_, tensor.bytes, &allocs_[tensor_index]));
  }
  return kTfLiteOk;
}

TfLiteStatus ArenaPlanner::CalculateTensorDeallocation(int tensor_index) {
  TfLiteTensor& tensor = *graph_info_->tensor(tensor_index);
  if (tensor.allocation_type == kTfLiteArenaRw) {
    TF_LITE_ENSURE_STATUS(arena_.Deallocate(context_, allocs_[tensor_index]));
  }
  return kTfLiteOk;
}

TfLiteStatus ArenaPlanner::CalculateAllocationOfInternalTensors(
    int node_index) {
  if (node_index < static_cast<int>(graph_info_->num_nodes())) {
    const TfLiteNode& node = graph_info_->node(static_cast<size_t>(node_index));
    TfLiteIntArray* node_temporaries = node.temporaries;
    for (int i = 0; i < node_temporaries->size; ++i) {
      int tensor_index = node_temporaries->data[i];
      TF_LITE_ENSURE_STATUS(CalculateTensorAllocation(tensor_index));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus ArenaPlanner::CalculateDeallocationOfInternalTensors(
    int node_index) {
  if (node_index < static_cast<int>(graph_info_->num_nodes())) {
    const TfLiteNode& node = graph_info_->node(static_cast<size_t>(node_index));
    TfLiteIntArray* node_temporaries = node.temporaries;
    for (int i = 0; i < node_temporaries->size; ++i) {
      int tensor_index = node_temporaries->data[i];
      TF_LITE_ENSURE_STATUS(CalculateTensorDeallocation(tensor_index));
    }
  }
  return kTfLiteOk;
}

}  // namespace tflite
