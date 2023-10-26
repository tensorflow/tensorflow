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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_SERVING_DEVICE_SELECTOR_POLICIES_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_SERVING_DEVICE_SELECTOR_POLICIES_H_

#include <atomic>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/serving_device_selector.h"

namespace tensorflow {

enum class ServingDeviceSelectorPolicy {
  kRoundRobin,
};

class RoundRobinPolicy : public ServingDeviceSelector::Policy {
 public:
  RoundRobinPolicy() : ordinal_(0) {}

  int SelectDevice(
      absl::string_view program_fingerprint,
      const ServingDeviceSelector::DeviceStates& device_states) override;

  void FreeDevice(absl::string_view program_fingerprint,
                  const ServingDeviceSelector::DeviceStates& device_states,
                  int device_index) override {}

 private:
  std::atomic<uint64_t> ordinal_;
};

class MinHeapPolicy : public ServingDeviceSelector::Policy {
 public:
  MinHeapPolicy() : total_num_(0) {}

  int SelectDevice(
      absl::string_view program_fingerprint,
      const ServingDeviceSelector::DeviceStates& device_states) override;

  void FreeDevice(absl::string_view program_fingerprint,
                  const ServingDeviceSelector::DeviceStates& device_states,
                  int device_index) override;

 private:
  // Node structure of the min-heap. A node represents a device. One node
  // contains a workload counter to record how many reservations are there for
  // the device, and contains an accumulator to record how many times has the
  // device been reserved in history. New reservation should be allocated to the
  // node of the lowest load. If two nodes have the same load, the reservation
  // goes to the one with smaller accumulator count.
  struct HeapNode {
    int id_;
    int workload_;
    uint64_t accumulator_;
    HeapNode(const int id, const int workload = 0,
             const uint64_t accumulator = 0)
        : id_(id), workload_(workload), accumulator_(accumulator) {}
  };

  // Select one node from the heap, and return the node id, i.e., the device
  // index.
  int SelectFromHeap();

  // Swap two nodes of the heap.
  void swap(const size_t, const size_t);

  // Reset the accumulator of all nodes.
  void reset_accumulators();

  size_t total_num_;
  std::vector<std::unique_ptr<HeapNode>> min_heap_;
  std::unordered_map<int, size_t> id2heap_map_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SERVING_DEVICE_SELECTOR_POLICIES_H_
