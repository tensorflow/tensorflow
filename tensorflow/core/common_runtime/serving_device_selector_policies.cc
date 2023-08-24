/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/serving_device_selector_policies.h"

#include <atomic>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/serving_device_selector.h"

namespace tensorflow {

int RoundRobinPolicy::SelectDevice(
    absl::string_view program_fingerprint,
    const ServingDeviceSelector::DeviceStates& device_states) {
  const int num_devices = device_states.states.size();
  return ordinal_.fetch_add(1, std::memory_order_relaxed) % num_devices;
}

int MinHeapPolicy::SelectDevice(
    absl::string_view program_fingerprint,
    const ServingDeviceSelector::DeviceStates& device_states) {
  if (total_num_ == 0) {
    total_num_ = device_states.states.size();
    min_heap_.resize(total_num_);
    for (int i = 0; i < total_num_; ++i) {
      min_heap_[i] = absl::make_unique<HeapNode>(i);
      id2heap_map_.insert(std::make_pair(i, i));
    }
  }
  return SelectFromHeap();
}

void MinHeapPolicy::swap(const size_t idx1, const size_t idx2) {
  id2heap_map_[min_heap_[idx1]->id_] = idx2;
  id2heap_map_[min_heap_[idx2]->id_] = idx1;
  std::swap(min_heap_[idx1], min_heap_[idx2]);
}

void MinHeapPolicy::reset_accumulators() {
  for (auto& node : min_heap_) {
    node->accumulator_ = 0;
  }
}

int MinHeapPolicy::SelectFromHeap() {
  int ret(min_heap_[0]->id_);
  ++min_heap_[0]->workload_;
  if (++min_heap_[0]->accumulator_ == 0xFFFFFFFFFFFFFFFFull) {
    reset_accumulators();
  }
  size_t ptr(0);
  while (true) {
    if (2 * ptr + 2 >= total_num_) {
      if (2 * ptr + 2 == total_num_ &&
          min_heap_[ptr]->workload_ > min_heap_[2 * ptr + 1]->workload_) {
        swap(ptr, 2 * ptr + 1);
      }
      break;
    }
    if (min_heap_[2 * ptr + 1]->workload_ < min_heap_[2 * ptr + 2]->workload_) {
      if (min_heap_[ptr]->workload_ > min_heap_[2 * ptr + 1]->workload_) {
        swap(ptr, 2 * ptr + 1);
        ptr = 2 * ptr + 1;
      } else {
        break;
      }
    } else if (min_heap_[2 * ptr + 1]->workload_ >
               min_heap_[2 * ptr + 2]->workload_) {
      if (min_heap_[ptr]->workload_ > min_heap_[2 * ptr + 2]->workload_) {
        swap(ptr, 2 * ptr + 2);
        ptr = 2 * ptr + 2;
      } else {
        break;
      }
    } else {
      if (min_heap_[ptr]->workload_ > min_heap_[2 * ptr + 1]->workload_) {
        if (min_heap_[2 * ptr + 1]->accumulator_ <
            min_heap_[2 * ptr + 2]->accumulator_) {
          swap(ptr, 2 * ptr + 1);
          ptr = 2 * ptr + 1;
        } else {
          swap(ptr, 2 * ptr + 2);
          ptr = 2 * ptr + 2;
        }
      } else {
        break;
      }
    }
  }
  return ret;
}

void MinHeapPolicy::FreeDevice(
    absl::string_view program_fingerprint,
    const ServingDeviceSelector::DeviceStates& device_states,
    int device_index) {
  size_t ptr = id2heap_map_[device_index];
  --min_heap_[ptr]->workload_;
  while (ptr != 0) {
    size_t parent = (ptr + 1) / 2 - 1;
    if (min_heap_[ptr]->workload_ < min_heap_[parent]->workload_) {
      swap(ptr, parent);
      ptr = parent;
    } else {
      break;
    }
  }
}

}  // namespace tensorflow
