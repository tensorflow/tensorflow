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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_RENDEZVOUS_CACHE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_RENDEZVOUS_CACHE_H_

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/refcount.h"
#include "tensorflow/tsl/platform/thread_annotations.h"

namespace tensorflow {

// The class for caching Rendezvous instances per step_id.
// If the Rendezvous object is destroyed for the step, a new one will be
// created on demand.
template <typename T>
class RendezvousCache {
 public:
  RendezvousCache() = default;
  virtual ~RendezvousCache() {
    for (auto& p : table_) {
      auto rendez = p.second.GetNewRef();
      if (rendez) {
        rendez->StartAbort(tsl::errors::Aborted("Shutdown"));
      }
    }
  }

  // Returns a new Reference.
  template <typename RendezvousCreator>
  tsl::core::RefCountPtr<T> FindOrCreate(int64_t step_id,
                                         RendezvousCreator create_fn) {
    tsl::mutex_lock l(table_lock_);
    tsl::core::RefCountPtr<T> rendz = nullptr;
    auto iter = table_.find(step_id);
    if (iter != table_.end()) {
      rendz = iter->second.GetNewRef();
      VLOG(5) << "step_id:" << step_id << " "
              << "WeakPtr returned:" << rendz.get();
      if (!rendz) {
        table_.erase(iter);
      }
    }
    if (!rendz) {  // Deleted or not found
      rendz = create_fn();
      VLOG(5) << "step_id:" << step_id << " "
              << "Rendezvous not found, inserting a new one." << rendz.get();
      // TODO(b/274793840): The WeakPtr still leaks, albeit slower than
      // leaking entire Rendezvous objects.
      table_.insert({step_id, tsl::core::WeakPtr<T>{rendz.get()}});
    }
    return rendz;
  }

  // Returns a new Reference.
  tsl::core::RefCountPtr<T> Find(int64_t step_id) {
    tsl::mutex_lock l(table_lock_);
    auto iter = table_.find(step_id);
    if (iter == table_.end()) return nullptr;
    return iter->second.GetNewRef();
  }

  // Removes a Rendezvous weak reference from table.
  void Remove(int64_t step_id) {
    tsl::core::RefCountPtr<T> rendez = nullptr;
    {
      tsl::mutex_lock l(table_lock_);
      auto iter = table_.find(step_id);
      if (iter != table_.end()) {
        rendez = iter->second.GetNewRef();
        table_.erase(iter);
      }
    }
    if (rendez) {
      rendez->StartAbort(tsl::errors::Aborted("Cleanup ", step_id));
    }
  }

  void RemoveAll() {
    tsl::mutex_lock l(table_lock_);
    table_.clear();
  }

  // Returns a list of active step ids. This result is only informative
  // at time of the call. The returned vector may contain step ids that have
  // been invalidated after the call.
  std::vector<int64_t> GetActiveStepIds() {
    tsl::mutex_lock l(table_lock_);
    std::vector<int64_t> list;
    list.reserve(table_.size());
    for (const auto& iter : table_) {
      if (iter.second.GetNewRef()) {
        list.push_back(iter.first);
      }
    }
    return list;
  }

 private:
  mutable tsl::mutex table_lock_;
  absl::flat_hash_map<int64_t, tsl::core::WeakPtr<T>> table_
      TF_GUARDED_BY(table_lock_);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_RENDEZVOUS_CACHE_H_
