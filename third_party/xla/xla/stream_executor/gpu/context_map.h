/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_CONTEXT_MAP_H_
#define XLA_STREAM_EXECUTOR_GPU_CONTEXT_MAP_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"

namespace stream_executor::gpu {

// Manages a map of gpu contexts that we've created, mapping
// from the gpu-specific contexts to the ContextType that we pass around
// internally.
template <class GpuContext, class ContextType>
class ContextMap {
 public:
  // Takes as a parameter a function that translates a void pointer into a
  // device ordinal.
  explicit ContextMap(absl::AnyInvocable<int(void* ptr)> find_device_ordinal)
      : find_device_ordinal_(std::move(find_device_ordinal)) {}
  // Returns whether context is a member of the live set.
  bool Has(GpuContext context) {
    absl::ReaderMutexLock lock(&mutex_);
    return gpu_context_to_context_type_map_.find(context) !=
           gpu_context_to_context_type_map_.end();
  }

  // Adds context to the live set, or returns it if it's already present.
  ContextType* Add(GpuContext context, int device_ordinal) {
    CHECK_NE(context, nullptr);
    absl::MutexLock lock(&mutex_);

    auto insert_result = gpu_context_to_context_type_map_.insert(
        std::make_pair(context, nullptr));
    auto it = insert_result.first;
    if (insert_result.second) {
      // context was not present in the map.  Add it.
      it->second = std::make_unique<ContextType>(context, device_ordinal);
      ordinal_to_type_map_[device_ordinal].push_back(context);
    }
    return it->second.get();
  }

  // Removes context from the live set.
  void Remove(GpuContext context) {
    absl::MutexLock lock(&mutex_);
    CHECK_NE(context, nullptr);
    auto it = gpu_context_to_context_type_map_.find(context);
    CHECK(it != gpu_context_to_context_type_map_.end()) << context;
    gpu_context_to_context_type_map_.erase(it);
    for (auto p : ordinal_to_type_map_) {
      auto it2 = std::find(p.second.begin(), p.second.end(), context);
      if (it2 != p.second.end()) {
        p.second.erase(it2);
        if (p.second.empty()) {
          ordinal_to_type_map_.erase(p.first);
        }
        break;
      }
    }
  }

  // Returns the context associated to that ptr.
  GpuContext GetAnyContext(void* ptr) {
    absl::ReaderMutexLock lock(&mutex_);
    int device_ordinal = find_device_ordinal_(ptr);
    CHECK_EQ(ordinal_to_type_map_.count(device_ordinal), 1);
    CHECK(!ordinal_to_type_map_.at(device_ordinal).empty())
        << "Need at least one context.";
    return ordinal_to_type_map_.at(device_ordinal)[0];
  }

 private:
  // Mutex protecting concurrent access to the maps.
  absl::Mutex mutex_;

  // A map of GpuContext to ContextType objects.
  absl::flat_hash_map<GpuContext, std::unique_ptr<ContextType>>
      gpu_context_to_context_type_map_ ABSL_GUARDED_BY(mutex_);

  // A map of device ordinal to GpuContext.
  absl::flat_hash_map<int, std::vector<GpuContext>> ordinal_to_type_map_
      ABSL_GUARDED_BY(mutex_);

  // A function that translates a given memory address into an associated device
  // ordinal.
  absl::AnyInvocable<int(void* ptr)> find_device_ordinal_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_CONTEXT_MAP_H_
