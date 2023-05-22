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

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/simple_memory_arena.h"

namespace tflite {
namespace {

constexpr int32_t kLastActiveNodeUndefined =
    std::numeric_limits<int32_t>::max();
constexpr int32_t kNodeNotAssigned = std::numeric_limits<int32_t>::max();

bool ShareFirstInputWithFirstOutputForNode(const TfLiteRegistration& node_reg) {
  // TODO (b/254230751): add support for more ops which support forwarding.
  switch (node_reg.builtin_code) {
    case kTfLiteBuiltinExpandDims:
    case kTfLiteBuiltinReshape:
    case kTfLiteBuiltinSqueeze:
    case kTfLiteBuiltinBitcast:
      return true;
    default:
      return false;
  }
}
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
      tensor_alignment_(tensor_alignment),
      last_active_node_(kLastActiveNodeUndefined) {}

ArenaPlanner::~ArenaPlanner() {
  arena_.ReleaseBuffer();
  persistent_arena_.ReleaseBuffer();
}

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
  // NOMUTANTS -- Setting last_active_node_ to kLastActiveNodeUndefined causes
  // all allocs to be cleared. if this is not set, the slow path is taken
  // (Purge) which inspects each alloc. Both paths give the exact same result.
  last_active_node_ = kLastActiveNodeUndefined;
  return kTfLiteOk;
}

TfLiteStatus ArenaPlanner::ResetAllocationsAfter(int node) {
  TfLiteTensor* tensors = graph_info_->tensors();
  for (int i = 0; i < static_cast<int>(allocs_.size()); ++i) {
    if (allocs_[i].first_node > node && allocs_[i].size > 0) {
      TfLiteTensor& tensor = tensors[i];
      if (tensor.allocation_type == kTfLiteArenaRw) {
        allocs_[i].reset();
        tensor.data.raw = nullptr;
      }
    }
  }
  if (last_active_node_ > node) {
    arena_.CalculateActiveAllocs(allocs_, node);
  } else {
    arena_.PurgeAfter(node);
  }
  last_active_node_ = node;
  return kTfLiteOk;
}

int ArenaPlanner::FindSharedTensor(int tensor_index) {
  auto actual_tensor_it = actual_tensor_id_.find(tensor_index);
  if (actual_tensor_it != actual_tensor_id_.end()) {
    tensor_index = actual_tensor_it->second;
  }
  return tensor_index;
}

void ArenaPlanner::IdentifySharedTensors() {
  actual_tensor_id_.clear();
  TfLiteTensor* tensors = graph_info_->tensors();
  const int num_execution_nodes = graph_info_->num_execution_nodes();
  for (int i = 0; i < num_execution_nodes; ++i) {
    const auto& reg = graph_info_->registration(i);
    const auto& tflite_node = graph_info_->node(i);
    if (ShareFirstInputWithFirstOutputForNode(reg)) {
      int32_t input_tensor = tflite_node.inputs->data[0];
      int32_t output_tensor = tflite_node.outputs->data[0];
      bool is_input_or_output_tensor = false;
      for (int input : graph_info_->inputs()) {
        if (input == input_tensor) {
          is_input_or_output_tensor = true;
          break;
        }
      }
      for (int output : graph_info_->outputs()) {
        if (output == output_tensor) {
          is_input_or_output_tensor = true;
          break;
        }
      }
      if (is_input_or_output_tensor) continue;
      TfLiteAllocationType input_allocation_type =
          tensors[input_tensor].allocation_type;
      TfLiteAllocationType output_allocation_type =
          tensors[output_tensor].allocation_type;
      // Only tensors allocated in the same arena may be shared.
      if (input_allocation_type != output_allocation_type) {
        continue;
      }
      if (input_allocation_type == kTfLiteArenaRw ||
          input_allocation_type == kTfLiteArenaRwPersistent) {
        // Handle the case where a shared tensor is also shared.
        int32_t actual_output_tensor_id = FindSharedTensor(input_tensor);
        actual_tensor_id_[output_tensor] = actual_output_tensor_id;
      }
    }
  }
}

TfLiteStatus ArenaPlanner::PlanAllocations() {
  // Invalidate any existing data.
  const size_t num_tensors = graph_info_->num_tensors();
  TF_LITE_ENSURE_STATUS(ResetAllocations());
  // Maybe other verb instead of 'Assigned'
  alloc_node_.assign(num_tensors, kNodeNotAssigned);
  dealloc_node_.assign(num_tensors, kNodeNotAssigned);
  nodes_to_tensors_.clear();
  nodes_to_tensors_.resize(
      std::max(graph_info_->num_execution_nodes(), (size_t)1), {});

  // Keeps track of references to each tensor.
  std::vector<int> refcounts(num_tensors, 0);

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

  IdentifySharedTensors();
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
    nodes_to_tensors_[0].insert(tensor_index);
  }

  // Queue all graph inputs for allocation and make sure they are never
  // overwritten.
  for (int tensor_index : graph_info_->inputs()) {
    if (tensor_index != kTfLiteOptionalTensor) {
      refcounts[tensor_index]++;
      TF_LITE_ENSURE_STATUS(allocate(0, tensor_index));
      nodes_to_tensors_[0].insert(tensor_index);
    }
  }

  // Count references to node input tensors.
  const int num_execution_nodes = graph_info_->num_execution_nodes();
  for (size_t i = 0; i < num_execution_nodes; ++i) {
    const TfLiteNode& node = graph_info_->node(i);
    TfLiteIntArray* node_inputs = node.inputs;
    for (int j = 0; j < node_inputs->size; ++j) {
      int tensor_index = node_inputs->data[j];
      if (tensor_index != kTfLiteOptionalTensor) {
        // Correctly count references for shared buffers.
        tensor_index = FindSharedTensor(tensor_index);
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
      //  Don't allocate output tensors here for shared memory parts.
      nodes_to_tensors_[i].insert(tensor_index);
      TF_LITE_ENSURE_STATUS(allocate(i, tensor_index));
    }

    // Then update the ref-counts of the node's inputs, and if necessary queue
    // them for deallocation.
    if (!preserve_all_tensors_) {
      TfLiteIntArray* node_inputs = node.inputs;
      for (int j = 0; j < node_inputs->size; ++j) {
        // If the tensor is a ref we decrement the original tensor.
        int tensor_index = node_inputs->data[j];
        if (tensor_index != kTfLiteOptionalTensor) {
          // Correctly count references for shared buffers.
          tensor_index = FindSharedTensor(tensor_index);
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
  const size_t num_tensors = graph_info_->num_tensors();
  TF_LITE_ENSURE(context_, num_tensors >= allocs_.size());
  alloc_node_.resize(num_tensors, kNodeNotAssigned);
  dealloc_node_.resize(num_tensors, kNodeNotAssigned);
  allocs_.resize(num_tensors);
  // Set allocation and deallocation for temporary tensors.
  const int num_execution_nodes = graph_info_->num_execution_nodes();
  for (size_t i = first_node;
       i <= static_cast<size_t>(last_node) && i < num_execution_nodes; ++i) {
    const TfLiteNode& node = graph_info_->node(i);
    TfLiteIntArray* node_temporaries = node.temporaries;
    for (int j = 0; j < node_temporaries->size; ++j) {
      int tensor_index = node_temporaries->data[j];
      alloc_node_[tensor_index] = i;
      nodes_to_tensors_[i].insert(tensor_index);
      if (!preserve_all_tensors_) {
        dealloc_node_[tensor_index] = i;
      }
    }
  }

  std::vector<int32_t> tensors_allocated;
  TF_LITE_ENSURE_STATUS(
      CalculateAllocations(first_node, last_node, &tensors_allocated));
  bool arena_reallocated = false;
  TF_LITE_ENSURE_STATUS(Commit(&arena_reallocated));

  TfLiteTensor* tensors = graph_info_->tensors();
  if (arena_reallocated) {
    for (int i = 0; i < static_cast<int>(num_tensors); ++i) {
      TF_LITE_ENSURE_STATUS(ResolveTensorAllocation(i, tensors));
    }
  } else {
    for (int i = 0; i < static_cast<int>(tensors_allocated.size()); ++i) {
      TF_LITE_ENSURE_STATUS(
          ResolveTensorAllocation(tensors_allocated[i], tensors));
    }
  }

  return kTfLiteOk;
}

TfLiteStatus ArenaPlanner::ReleaseNonPersistentMemory() {
  // Clear non-persistent arena's buffer.
  TF_LITE_ENSURE_STATUS(arena_.ReleaseBuffer());
  // Set data pointers for all non-persistent tensors to nullptr.
  TfLiteTensor* tensors = graph_info_->tensors();
  for (int i = 0; i < static_cast<int>(graph_info_->num_tensors()); ++i) {
    TfLiteTensor& tensor = tensors[i];
    if (tensor.allocation_type == kTfLiteArenaRw) {
      tensor.data.raw = nullptr;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus ArenaPlanner::AcquireNonPersistentMemory() {
  // First commit arena_ to allocate underlying buffer.
  bool reallocated;
  TF_LITE_ENSURE_STATUS(arena_.Commit(context_, &reallocated));
  // Resolve allocations for all tensors not on the persistent arena.
  TfLiteTensor* tensors = graph_info_->tensors();
  for (int i = 0; i < static_cast<int>(graph_info_->num_tensors()); ++i) {
    TfLiteTensor& tensor = tensors[i];
    if (tensor.allocation_type == kTfLiteArenaRw) {
      TF_LITE_ENSURE_STATUS(ResolveTensorAllocation(i, tensors));
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

TfLiteStatus ArenaPlanner::Commit(bool* reallocated) {
  bool arena_reallocated, persistent_arena_reallocated;
  TF_LITE_ENSURE_STATUS(arena_.Commit(context_, &arena_reallocated));
  TF_LITE_ENSURE_STATUS(
      persistent_arena_.Commit(context_, &persistent_arena_reallocated));
  *reallocated = arena_reallocated;
  *reallocated |= persistent_arena_reallocated;
  return kTfLiteOk;
}

void ArenaPlanner::CreateTensorAllocationVector(
    std::vector<int32_t>* tensors_to_allocate) {
  const TfLiteTensor* tensors = this->graph_info_->tensors();
  auto tensor_compare = [&](int idx1, int idx2) {
    // Tensors that have lifespan through the whole model inference time are
    // allocated at the beginning of memory slice. Their respective order
    // doesn't matter in fact, so here they are sorted by index.
    if (alloc_node_[idx1] == 0 && dealloc_node_[idx1] == kNodeNotAssigned) {
      if (alloc_node_[idx2] == 0 && dealloc_node_[idx2] == kNodeNotAssigned) {
        return idx1 < idx2;
      }
      return true;
    }
    if (alloc_node_[idx2] == 0 && dealloc_node_[idx2] == kNodeNotAssigned) {
      return false;
    }

    // All other tensors are sorted in non-increasing order of their size.
    auto size1 = tensors[idx1].bytes;
    auto size2 = tensors[idx2].bytes;
    if (size1 != size2) {
      return size1 > size2;
    }
    // Tensors with equal size are sorted in order of their allocation time.
    return alloc_node_[idx1] < alloc_node_[idx2];
  };

  // Indices of tensors in order their allocation offsets will be calculated.
  std::sort(tensors_to_allocate->begin(), tensors_to_allocate->end(),
            tensor_compare);
}

std::vector<int32_t> ArenaPlanner::GetTensorsToAllocate(int first_node,
                                                        int last_node) {
  int num_tensors = static_cast<int>(graph_info_->num_tensors());
  std::vector<int32_t> tensors_to_allocate;
  tensors_to_allocate.reserve(num_tensors);
  for (int i = first_node; i <= last_node; ++i) {
    tensors_to_allocate.insert(tensors_to_allocate.end(),
                               nodes_to_tensors_[i].begin(),
                               nodes_to_tensors_[i].end());
  }
  return tensors_to_allocate;
}

TfLiteStatus ArenaPlanner::CalculateAllocations(
    int first_node, int last_node, std::vector<int32_t>* tensors_allocated) {
  // Indices of tensors in order their allocation offsets will be calculated.
  const std::vector<int32_t> tensors_to_allocate =
      GetTensorsToAllocate(first_node, last_node);

  tensors_allocated->reserve(tensors_to_allocate.size());
  // Deallocate if the tensor was already allocated.
  TfLiteTensor* tensors = graph_info_->tensors();
  for (const auto& tensor_index : tensors_to_allocate) {
    TfLiteTensor& tensor = tensors[tensor_index];
    // Only arena allocated tensors are allocated here.
    if (tensor.allocation_type == kTfLiteArenaRw) {
      if (allocs_[tensor_index].size < tensor.bytes) {
        tensors_allocated->push_back(tensor_index);
      }
    } else if (tensor.allocation_type == kTfLiteArenaRwPersistent) {
      tensors_allocated->push_back(tensor_index);
    }
  }

  if (tensors_allocated->empty()) {
    last_active_node_ = last_node;
    return kTfLiteOk;
  }
  if (first_node < last_active_node_) {
    arena_.ResetAllocs();
    last_active_node_ = first_node;
  } else {
    // NOMUTANTS -- This function has no impact on the results, it only makes
    // exection faster.
    arena_.PurgeActiveAllocs(first_node);
  }
  CreateTensorAllocationVector(tensors_allocated);
  // Vector of ids of already allocated tensors, ordered by offset.
  for (const auto& tensor_index : *tensors_allocated) {
    TfLiteTensor& tensor = tensors[tensor_index];
    // Only allocate ArenaRw tensors which own their buffer.
    auto it = actual_tensor_id_.find(tensor_index);
    if (it != actual_tensor_id_.end()) {
      // A tensor whose buffer is shared may have had its allocation type
      // changed to kTfLiteCustom or kTfLiteDynamic after `PlanAllocations` was
      // called. This means that the buffer is no longer shareable so remove its
      // index from `actual_tensor_id_`.
      TfLiteAllocationType allocation_type =
          tensors[it->second].allocation_type;
      if (allocation_type != kTfLiteArenaRwPersistent &&
          allocation_type != kTfLiteArenaRw) {
        actual_tensor_id_.erase(it);
      } else {
        // Don't allocate the tensor, it can safely share the input buffer.
        continue;
      }
    }
    if (tensor.allocation_type == kTfLiteArenaRw) {
      TF_LITE_ENSURE_STATUS(
          arena_.Allocate(context_, tensor_alignment_, tensor.bytes,
                          tensor_index, alloc_node_[tensor_index],
                          dealloc_node_[tensor_index], &allocs_[tensor_index]));
    }
    // Check allocs_[].size to prevent from reallocation of persistent tensors.
    // Only allocate ArenaRwPersistent tensors which own their buffer.
    if (tensor.allocation_type == kTfLiteArenaRwPersistent &&
        allocs_[tensor_index].size == 0) {
      if (allocs_[tensor_index].size < tensor.bytes) {
        TF_LITE_ENSURE_STATUS(persistent_arena_.Allocate(
            context_, tensor_alignment_, tensor.bytes, tensor_index,
            /*first_node=*/alloc_node_[tensor_index],
            /*last_node=*/std::numeric_limits<int32_t>::max(),
            &allocs_[tensor_index]));
      }
    }
  }
  last_active_node_ = last_node;
  return kTfLiteOk;
}

bool AreTensorsAllocatedInSameArena(int32_t root_tensor_index,
                                    int32_t tensor_index,
                                    const TfLiteTensor* tensors) {
  if (tensors[root_tensor_index].allocation_type == kTfLiteArenaRw &&
      tensors[tensor_index].allocation_type == kTfLiteArenaRw) {
    return true;
  }
  if (tensors[root_tensor_index].allocation_type == kTfLiteArenaRwPersistent &&
      tensors[tensor_index].allocation_type == kTfLiteArenaRwPersistent) {
    return true;
  }
  return false;
}

TfLiteStatus ArenaPlanner::ResolveTensorAllocation(int32_t tensor_index,
                                                   TfLiteTensor* tensors) {
  // Resolve allocation for tensors which share buffers.
  auto actual_tensor_it = actual_tensor_id_.find(tensor_index);
  TfLiteTensor& tensor = tensors[tensor_index];
  int32_t root_tensor_index = actual_tensor_it == actual_tensor_id_.end()
                                  ? tensor_index
                                  : actual_tensor_it->second;
  const TfLiteTensor& root_tensor = tensors[root_tensor_index];
  if (root_tensor_index != tensor_index) {
    if (AreTensorsAllocatedInSameArena(root_tensor_index, tensor_index,
                                       tensors)) {
      // Make sure that the input tensor has already been allocated.
      ResolveTensorAllocation(root_tensor_index, tensors);
      tensor.data.data = root_tensor.data.data;
      return kTfLiteOk;
    }
  }

  if (tensor.allocation_type == kTfLiteArenaRw) {
    // Skip resolution if the size of the tensor is zero, leaving it as a
    // nullptr.
    if (allocs_[tensor_index].size != 0) {
      return arena_.ResolveAlloc(context_, allocs_[tensor_index],
                                 &tensor.data.raw);
    }
  }
  if (tensor.allocation_type == kTfLiteArenaRwPersistent) {
    return persistent_arena_.ResolveAlloc(context_, allocs_[tensor_index],
                                          &tensor.data.raw);
  }
  return kTfLiteOk;
}

}  // namespace tflite
