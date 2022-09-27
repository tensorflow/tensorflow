/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_REFCOUNTING_HASH_MAP_H_
#define TENSORFLOW_COMPILER_XLA_REFCOUNTING_HASH_MAP_H_

#include <functional>
#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// RefcountingHashMap is an "eager, thread-safe cache".
//
// Given a key k you can retrieve a shared_ptr to a value v.  If k is not
// already in the map, we construct a new V; if it is already in the map, we'll
// return the existing v.  Once all shared_ptrs are destroyed, the entry is
// removed from the map.
//
// This class is thread-safe.
//
// Word to the wise: You might want an erase() function here that removes a
// value from the map but leaves existing shared_ptrs intact.  My experience is,
// this is extremely complicated to implement correctly.
template <typename K, typename V>
class RefcountingHashMap {
 public:
  // Default-constructs new values.
  RefcountingHashMap() = default;

  // Not copyable or movable because this contains internal pointers (namely,
  // instances of Deleter contain pointers to `this` and into `map_`).
  RefcountingHashMap(const RefcountingHashMap&) = delete;
  RefcountingHashMap(RefcountingHashMap&&) = delete;
  RefcountingHashMap& operator=(const RefcountingHashMap&) = delete;
  RefcountingHashMap& operator=(RefcountingHashMap&&) = delete;

  // Gets the value for the given key.
  //
  // If the map doesn't contain a live value for the key, constructs one
  // using `value_factory`.
  std::shared_ptr<V> GetOrCreateIfAbsent(
      const K& key,
      const std::function<std::unique_ptr<V>(const K&)>& value_factory) {
    absl::MutexLock lock(&mu_);
    auto it = map_.find(key);
    if (it != map_.end()) {
      // We ensure that the entry has not expired in case deleter was running
      // when we have entered this block.
      if (std::shared_ptr<V> value = it->second.lock()) {
        return value;
      }
    }

    // Create entry in the map and then set its value, so the value can
    // contain a pointer back into the map.
    it = map_.emplace(key, std::weak_ptr<V>()).first;
    std::shared_ptr<V> value(value_factory(key).release(),
                             Deleter{it->first, *this});
    it->second = value;  // Set the weak ptr to the shared ptr.
    return value;
  }

 private:
  struct Deleter {
    const K& key;  // Points into parent->map_.
    RefcountingHashMap& parent;

    void operator()(V* v) {
      delete v;
      absl::MutexLock lock(&parent.mu_);
      // We must check if that the entry is still expired in case the value was
      // replaced while the deleter was running.
      auto it = parent.map_.find(key);
      if (it != parent.map_.end() && it->second.expired()) {
        parent.map_.erase(it);
      }
    }
  };

  absl::Mutex mu_;
  absl::node_hash_map<K, std::weak_ptr<V>> map_ ABSL_GUARDED_BY(mu_);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_REFCOUNTING_HASH_MAP_H_
