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

#include <algorithm>
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
                             std::unique_ptr<GraphInfo> graph_info,
                             bool preserve_all_tensors, bool enable_reclamation,
                             TfLiteAllocator* allocator)
    : context_(context),
      graph_info_(std::move(graph_info)),
      preserve_all_tensors_(preserve_all_tensors),
      reclamation_mode_(enable_reclamation),
      allocator_(allocator),
      non_persistent_memory_available_(true) {}

SimplePlanner::~SimplePlanner() { FreeAllAllocations(); }

void SimplePlanner::FreeAllAllocations() {
  for (SimpleAlloc& alloc : allocs_) {
    if (alloc.is_allocated()) {
      if (current_outstanding_bytes_ >= alloc.size) {
        current_outstanding_bytes_ -= alloc.size;
      } else {
        current_outstanding_bytes_ = 0;
      }
      ++total_deallocations_;
    }
    alloc.free(allocator_);
  }
}

TfLiteStatus SimplePlanner::ResetAllocations() {
  FreeAllAllocations();
  const size_t num_tensors = graph_info_->num_tensors();
  allocs_.clear();
  allocs_.resize(num_tensors);
  non_persistent_memory_available_ = true;
  return kTfLiteOk;
}

TfLiteStatus SimplePlanner::ResetAllocationsAfter(int node) {
  TfLiteTensor* tensors = graph_info_->tensors();
  for (int i = 0; i < static_cast<int>(allocs_.size()); ++i) {
    if (allocs_[i].node > node && allocs_[i].size > 0) {
      TfLiteTensor& tensor = tensors[i];
      if (tensor.allocation_type == kTfLiteArenaRw) {
        if (allocs_[i].is_allocated()) {
          if (current_outstanding_bytes_ >= allocs_[i].size) {
            current_outstanding_bytes_ -= allocs_[i].size;
          } else {
            current_outstanding_bytes_ = 0;
          }
          ++total_deallocations_;
          allocs_[i].free(allocator_);
        }
        tensor.data.raw = nullptr;
      }
    }
  }
  return kTfLiteOk;
}

void SimplePlanner::UpdatePinnedStatus() {
  const size_t num_tensors = graph_info_->num_tensors();
  is_pinned_.assign(num_tensors, false);

  if (preserve_all_tensors_) {
    is_pinned_.assign(num_tensors, true);
    return;
  }

  // Graph inputs, outputs, and variables are always pinned.
  for (int tensor_index : graph_info_->inputs()) {
    if (tensor_index != kTfLiteOptionalTensor &&
        tensor_index < static_cast<int>(num_tensors)) {
      is_pinned_[tensor_index] = true;
    }
  }
  for (int tensor_index : graph_info_->outputs()) {
    if (tensor_index != kTfLiteOptionalTensor &&
        tensor_index < static_cast<int>(num_tensors)) {
      is_pinned_[tensor_index] = true;
    }
  }
  for (int tensor_index : graph_info_->variables()) {
    if (tensor_index != kTfLiteOptionalTensor &&
        tensor_index < static_cast<int>(num_tensors)) {
      is_pinned_[tensor_index] = true;
    }
  }

  // Persistent arena tensors are pinned.
  TfLiteTensor* tensors = graph_info_->tensors();
  if (tensors) {
    for (size_t i = 0; i < num_tensors; ++i) {
      if (tensors[i].allocation_type == kTfLiteArenaRwPersistent) {
        is_pinned_[i] = true;
      }
    }
  }
}

TfLiteStatus SimplePlanner::PlanAllocations() {
  TF_LITE_ENSURE_STATUS(ResetAllocations());
  const size_t num_tensors = graph_info_->num_tensors();
  alloc_node_.assign(num_tensors, kNodeNotAssigned);
  dealloc_node_.assign(num_tensors, kNodeNotAssigned);

  UpdatePinnedStatus();

  const size_t num_execution_nodes = graph_info_->num_execution_nodes();
  allocate_before_.assign(num_execution_nodes, {});
  release_after_.assign(num_execution_nodes, {});

  auto allocate = [this](int node, int tensor) -> TfLiteStatus {
    if (alloc_node_[tensor] != kNodeNotAssigned) {
      return kTfLiteOk;
    }
    TF_LITE_ENSURE(context_, dealloc_node_[tensor] == kNodeNotAssigned);
    alloc_node_[tensor] = node;
    return kTfLiteOk;
  };

  // Variable tensors must be allocated at the very beginning.
  for (int tensor_index : graph_info_->variables()) {
    TF_LITE_ENSURE(context_, tensor_index != kTfLiteOptionalTensor);
    TF_LITE_ENSURE_STATUS(allocate(0, tensor_index));
  }

  // Graph inputs allocated at node 0.
  for (int tensor_index : graph_info_->inputs()) {
    if (tensor_index != kTfLiteOptionalTensor) {
      TF_LITE_ENSURE_STATUS(allocate(0, tensor_index));
    }
  }

  // Find last consumer node for every tensor.
  std::vector<int> last_consumer_node(num_tensors, -1);
  for (size_t i = 0; i < num_execution_nodes; ++i) {
    const TfLiteNode& node = graph_info_->node(i);
    TfLiteIntArray* node_inputs = node.inputs;
    if (node_inputs) {
      for (int j = 0; j < node_inputs->size; ++j) {
        int tensor_index = node_inputs->data[j];
        if (tensor_index != kTfLiteOptionalTensor &&
            tensor_index < static_cast<int>(num_tensors)) {
          last_consumer_node[tensor_index] = static_cast<int>(i);
        }
      }
    }
  }

  // Walk nodes in execution order to assign output alloc_nodes.
  for (size_t i = 0; i < num_execution_nodes; ++i) {
    const TfLiteNode& node = graph_info_->node(i);
    TfLiteIntArray* node_outputs = node.outputs;
    if (node_outputs) {
      for (int j = 0; j < node_outputs->size; ++j) {
        int tensor_index = node_outputs->data[j];
        if (tensor_index != kTfLiteOptionalTensor &&
            tensor_index < static_cast<int>(num_tensors)) {
          TF_LITE_ENSURE_STATUS(allocate(static_cast<int>(i), tensor_index));
          if (reclamation_mode_ && !is_pinned_[tensor_index]) {
            allocate_before_[i].push_back(tensor_index);
          }
        }
      }
    }
  }

  // Calculate deallocation positions for reclaimable tensors.
  for (size_t t = 0; t < num_tensors; ++t) {
    if (alloc_node_[t] != kNodeNotAssigned && !is_pinned_[t]) {
      int last_use = last_consumer_node[t];
      if (last_use != -1) {
        dealloc_node_[t] = last_use;
        if (reclamation_mode_ &&
            last_use < static_cast<int>(num_execution_nodes)) {
          release_after_[last_use].push_back(static_cast<int>(t));
        }
      } else {
        // Output tensor with no consumers and not a graph output.
        // It can be released immediately after its producer executes.
        dealloc_node_[t] = alloc_node_[t];
        if (reclamation_mode_ &&
            alloc_node_[t] < static_cast<int>(num_execution_nodes)) {
          release_after_[alloc_node_[t]].push_back(static_cast<int>(t));
        }
      }
    }
  }

  // Deduplicate schedules.
  if (reclamation_mode_) {
    for (size_t i = 0; i < num_execution_nodes; ++i) {
      std::sort(allocate_before_[i].begin(), allocate_before_[i].end());
      allocate_before_[i].erase(
          std::unique(allocate_before_[i].begin(), allocate_before_[i].end()),
          allocate_before_[i].end());
      std::sort(release_after_[i].begin(), release_after_[i].end());
      release_after_[i].erase(
          std::unique(release_after_[i].begin(), release_after_[i].end()),
          release_after_[i].end());
    }
  }

  return kTfLiteOk;
}

TfLiteStatus SimplePlanner::ExecuteAllocations(int first_node, int last_node) {
  const size_t num_tensors = graph_info_->num_tensors();
  const size_t num_execution_nodes = graph_info_->num_execution_nodes();
  if (alloc_node_.size() != num_tensors ||
      allocate_before_.size() != num_execution_nodes ||
      release_after_.size() != num_execution_nodes) {
    TF_LITE_ENSURE_STATUS(PlanAllocations());
  }

  UpdatePinnedStatus();
  allocs_.resize(num_tensors);

  // Set allocation and deallocation for temporary tensors.
  TfLiteTensor* tensors = graph_info_->tensors();
  for (size_t i = first_node;
       i <= static_cast<size_t>(last_node) && i < num_execution_nodes; ++i) {
    const TfLiteNode& node = graph_info_->node(i);
    TfLiteIntArray* node_temporaries = node.temporaries;
    if (!node_temporaries) continue;
    for (int j = 0; j < node_temporaries->size; ++j) {
      int tensor_index = node_temporaries->data[j];
      if (tensor_index != kTfLiteOptionalTensor &&
          tensor_index < static_cast<int>(num_tensors)) {
        alloc_node_[tensor_index] = i;
        dealloc_node_[tensor_index] = i;
        if (reclamation_mode_ && !is_pinned_[tensor_index]) {
          allocate_before_[i].push_back(tensor_index);
          if (tensors &&
              tensors[tensor_index].allocation_type !=
                  kTfLiteArenaRwPersistent &&
              !preserve_all_tensors_) {
            release_after_[i].push_back(tensor_index);
          }
        }
      }
    }
    if (reclamation_mode_) {
      std::sort(allocate_before_[i].begin(), allocate_before_[i].end());
      allocate_before_[i].erase(
          std::unique(allocate_before_[i].begin(), allocate_before_[i].end()),
          allocate_before_[i].end());
      std::sort(release_after_[i].begin(), release_after_[i].end());
      release_after_[i].erase(
          std::unique(release_after_[i].begin(), release_after_[i].end()),
          release_after_[i].end());
    }
  }

  // Conduct allocations.
  const int total_tensors = static_cast<int>(num_tensors);
  for (int i = 0; i < total_tensors; ++i) {
    if (alloc_node_[i] >= first_node && alloc_node_[i] <= last_node) {
      TfLiteTensor& tensor = tensors[i];
      if (tensor.allocation_type == kTfLiteArenaRw) {
        if (reclamation_mode_ && !is_pinned_[i]) {
          // Defer allocation for reclaimable intermediate tensors.
          if (allocs_[i].is_allocated()) {
            if (current_outstanding_bytes_ >= allocs_[i].size) {
              current_outstanding_bytes_ -= allocs_[i].size;
            } else {
              current_outstanding_bytes_ = 0;
            }
            ++total_deallocations_;
            allocs_[i].free(allocator_);
          }
          allocs_[i].size = tensor.bytes;
          allocs_[i].node = alloc_node_[i];
          allocs_[i].ptr = nullptr;
          tensor.data.raw = nullptr;
          continue;
        }

        // Pinned or eager mode: allocate now.
        if (allocs_[i].is_allocated()) {
          if (current_outstanding_bytes_ >= allocs_[i].size) {
            current_outstanding_bytes_ -= allocs_[i].size;
          } else {
            current_outstanding_bytes_ = 0;
          }
          ++total_deallocations_;
          allocs_[i].free(allocator_);
          tensor.data.raw = nullptr;
        }
        if (tensor.bytes > 0) {
          if (!allocs_[i].alloc(tensor.bytes, alloc_node_[i], allocator_)) {
            if (context_ && context_->ReportError) {
              context_->ReportError(
                  context_,
                  "SimplePlanner failed to allocate %zu bytes for tensor %d",
                  tensor.bytes, i);
            }
            return kTfLiteError;
          }
          current_outstanding_bytes_ += allocs_[i].size;
          if (current_outstanding_bytes_ > peak_outstanding_bytes_) {
            peak_outstanding_bytes_ = current_outstanding_bytes_;
          }
          ++total_allocations_;
          TF_LITE_ENSURE_STATUS(ResolveTensorAllocation(i));
        } else {
          tensor.data.raw = nullptr;
        }
      } else if (tensor.allocation_type == kTfLiteArenaRwPersistent) {
        if (!allocs_[i].is_allocated() && tensor.bytes > 0) {
          if (!allocs_[i].alloc(tensor.bytes, alloc_node_[i], allocator_)) {
            if (context_ && context_->ReportError) {
              context_->ReportError(
                  context_,
                  "SimplePlanner failed to allocate %zu bytes for persistent "
                  "tensor %d",
                  tensor.bytes, i);
            }
            return kTfLiteError;
          }
          current_outstanding_bytes_ += allocs_[i].size;
          if (current_outstanding_bytes_ > peak_outstanding_bytes_) {
            peak_outstanding_bytes_ = current_outstanding_bytes_;
          }
          ++total_allocations_;
          TF_LITE_ENSURE_STATUS(ResolveTensorAllocation(i));
        } else if (allocs_[i].is_allocated()) {
          TF_LITE_ENSURE_STATUS(ResolveTensorAllocation(i));
        } else {
          tensor.data.raw = nullptr;
        }
      }
    }
  }

  non_persistent_memory_available_ = true;
  return kTfLiteOk;
}

TfLiteStatus SimplePlanner::ReleaseNonPersistentMemory() {
  const int num_tensors = static_cast<int>(graph_info_->num_tensors());
  TfLiteTensor* tensors = graph_info_->tensors();
  for (int i = 0; i < num_tensors; ++i) {
    TfLiteTensor& tensor = tensors[i];
    if (tensor.allocation_type == kTfLiteArenaRw) {
      if (allocs_[i].is_allocated()) {
        if (current_outstanding_bytes_ >= allocs_[i].size) {
          current_outstanding_bytes_ -= allocs_[i].size;
        } else {
          current_outstanding_bytes_ = 0;
        }
        ++total_deallocations_;
        allocs_[i].release(allocator_);
      }
      tensor.data.raw = nullptr;
    }
  }
  non_persistent_memory_available_ = false;
  return kTfLiteOk;
}

TfLiteStatus SimplePlanner::AcquireNonPersistentMemory() {
  const int num_tensors = static_cast<int>(graph_info_->num_tensors());
  TfLiteTensor* tensors = graph_info_->tensors();
  for (int i = 0; i < num_tensors; ++i) {
    TfLiteTensor& tensor = tensors[i];
    if (tensor.allocation_type == kTfLiteArenaRw &&
        alloc_node_[i] != kNodeNotAssigned) {
      // In reclamation mode, only reacquire pinned non-persistent tensors.
      // Reclaimable intermediate tensors remain deferred until BeforeNode().
      if (reclamation_mode_ && !is_pinned_[i]) {
        tensor.data.raw = nullptr;
        continue;
      }
      if (allocs_[i].size != 0 && allocs_[i].ptr == nullptr) {
        if (!allocs_[i].alloc(allocs_[i].size, allocs_[i].node, allocator_)) {
          if (context_ && context_->ReportError) {
            context_->ReportError(
                context_,
                "SimplePlanner failed to re-acquire %zu bytes for tensor %d",
                allocs_[i].size, i);
          }
          return kTfLiteError;
        }
        current_outstanding_bytes_ += allocs_[i].size;
        if (current_outstanding_bytes_ > peak_outstanding_bytes_) {
          peak_outstanding_bytes_ = current_outstanding_bytes_;
        }
        ++total_allocations_;
      }
      TF_LITE_ENSURE_STATUS(ResolveTensorAllocation(i));
    }
  }
  non_persistent_memory_available_ = true;
  return kTfLiteOk;
}

bool SimplePlanner::HasNonPersistentMemory() {
  return non_persistent_memory_available_;
}

void SimplePlanner::GetAllocInfo(size_t* arena_size,
                                 size_t* arena_persist_size) const {
  if (arena_size) {
    *arena_size = current_outstanding_bytes_;
  }
  if (arena_persist_size) {
    size_t persist = 0;
    TfLiteTensor* tensors = graph_info_->tensors();
    for (size_t i = 0; i < allocs_.size(); ++i) {
      if (tensors && tensors[i].allocation_type == kTfLiteArenaRwPersistent &&
          allocs_[i].is_allocated()) {
        persist += allocs_[i].size;
      }
    }
    *arena_persist_size = persist;
  }
}

TfLiteStatus SimplePlanner::BeginInvocation() {
  if (!non_persistent_memory_available_) {
    if (context_ && context_->ReportError) {
      context_->ReportError(context_,
                            "SimplePlanner: non-persistent memory is not "
                            "available.");
    }
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus SimplePlanner::BeforeNode(int execution_plan_index) {
  if (!reclamation_mode_) return kTfLiteOk;
  if (execution_plan_index < 0 ||
      execution_plan_index >= static_cast<int>(allocate_before_.size())) {
    return kTfLiteOk;
  }
  TfLiteTensor* tensors = graph_info_->tensors();
  std::vector<int> newly_allocated;
  for (int tensor_index : allocate_before_[execution_plan_index]) {
    if (tensor_index < 0 || tensor_index >= static_cast<int>(allocs_.size())) {
      continue;
    }
    TfLiteTensor& tensor = tensors[tensor_index];
    if (tensor.allocation_type != kTfLiteArenaRw &&
        tensor.allocation_type != kTfLiteArenaRwPersistent) {
      continue;
    }
    if (allocs_[tensor_index].ptr == nullptr && tensor.bytes > 0) {
      if (!allocs_[tensor_index].alloc(tensor.bytes, execution_plan_index,
                                       allocator_)) {
        if (context_ && context_->ReportError) {
          context_->ReportError(
              context_,
              "SimplePlanner failed to allocate %zu bytes for tensor %d",
              tensor.bytes, tensor_index);
        }
        // Rollback any newly allocated buffers in this step.
        for (int rollback_idx : newly_allocated) {
          if (current_outstanding_bytes_ >= allocs_[rollback_idx].size) {
            current_outstanding_bytes_ -= allocs_[rollback_idx].size;
          } else {
            current_outstanding_bytes_ = 0;
          }
          ++total_deallocations_;
          allocs_[rollback_idx].release(allocator_);
          tensors[rollback_idx].data.raw = nullptr;
        }
        return kTfLiteError;
      }
      current_outstanding_bytes_ += allocs_[tensor_index].size;
      if (current_outstanding_bytes_ > peak_outstanding_bytes_) {
        peak_outstanding_bytes_ = current_outstanding_bytes_;
      }
      ++total_allocations_;
      newly_allocated.push_back(tensor_index);
      TF_LITE_ENSURE_STATUS(ResolveTensorAllocation(tensor_index));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus SimplePlanner::AfterNode(int execution_plan_index) {
  if (!reclamation_mode_) return kTfLiteOk;
  if (execution_plan_index < 0 ||
      execution_plan_index >= static_cast<int>(release_after_.size())) {
    return kTfLiteOk;
  }
  TfLiteTensor* tensors = graph_info_->tensors();
  for (int tensor_index : release_after_[execution_plan_index]) {
    if (tensor_index < 0 || tensor_index >= static_cast<int>(allocs_.size())) {
      continue;
    }
    if (is_pinned_[tensor_index] || preserve_all_tensors_) {
      continue;
    }
    TfLiteTensor& tensor = tensors[tensor_index];
    if (tensor.allocation_type == kTfLiteArenaRw &&
        allocs_[tensor_index].is_allocated()) {
      if (current_outstanding_bytes_ >= allocs_[tensor_index].size) {
        current_outstanding_bytes_ -= allocs_[tensor_index].size;
      } else {
        current_outstanding_bytes_ = 0;
      }
      ++total_deallocations_;
      allocs_[tensor_index].release(allocator_);
      tensor.data.raw = nullptr;
    }
  }
  return kTfLiteOk;
}

void SimplePlanner::EndInvocation(bool completed_successfully) {
  if (!reclamation_mode_) return;
  // If invocation failed or was cancelled, clean up any active reclaimable
  // intermediate buffers. Pinned tensors (inputs, outputs, variables,
  // persistent) are preserved.
  if (!completed_successfully) {
    TfLiteTensor* tensors = graph_info_->tensors();
    for (size_t i = 0; i < allocs_.size(); ++i) {
      if (!is_pinned_[i] && allocs_[i].is_allocated()) {
        if (current_outstanding_bytes_ >= allocs_[i].size) {
          current_outstanding_bytes_ -= allocs_[i].size;
        } else {
          current_outstanding_bytes_ = 0;
        }
        ++total_deallocations_;
        allocs_[i].release(allocator_);
        if (tensors) {
          tensors[i].data.raw = nullptr;
        }
      }
    }
  }
}

TfLiteStatus SimplePlanner::ResolveTensorAllocation(int tensor_index) {
  TfLiteTensor& tensor = *graph_info_->tensor(tensor_index);
  if (tensor.allocation_type == kTfLiteArenaRw ||
      tensor.allocation_type == kTfLiteArenaRwPersistent) {
    if (allocs_[tensor_index].is_allocated()) {
      tensor.data.raw = allocs_[tensor_index].ptr;
    } else {
      tensor.data.raw = nullptr;
    }
  }
  return kTfLiteOk;
}

}  // namespace tflite
