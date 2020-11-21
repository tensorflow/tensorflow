/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_GCS_EXPIRING_LRU_CACHE_H_
#define TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_GCS_EXPIRING_LRU_CACHE_H_

#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/c/env.h"
#include "tensorflow/c/tf_status.h"

namespace tf_gcs_filesystem {

/// \brief An LRU cache of string keys and arbitrary values, with configurable
/// max item age (in seconds) and max entries.
///
/// This class is thread safe.
template <typename T>
class ExpiringLRUCache {
 public:
  /// A `max_age` of 0 means that nothing is cached. A `max_entries` of 0 means
  /// that there is no limit on the number of entries in the cache (however, if
  /// `max_age` is also 0, the cache will not be populated).
  ExpiringLRUCache(uint64_t max_age, size_t max_entries,
                   std::function<uint64_t()> timer_seconds = TF_NowSeconds)
      : max_age_(max_age),
        max_entries_(max_entries),
        timer_seconds_(timer_seconds) {}

  /// Insert `value` with key `key`. This will replace any previous entry with
  /// the same key.
  void Insert(const std::string& key, const T& value) {
    if (max_age_ == 0) {
      return;
    }
    absl::MutexLock lock(&mu_);
    InsertLocked(key, value);
  }

  // Delete the entry with key `key`. Return true if the entry was found for
  // `key`, false if the entry was not found. In both cases, there is no entry
  // with key `key` existed after the call.
  bool Delete(const std::string& key) {
    absl::MutexLock lock(&mu_);
    return DeleteLocked(key);
  }

  /// Look up the entry with key `key` and copy it to `value` if found. Returns
  /// true if an entry was found for `key`, and its timestamp is not more than
  /// max_age_ seconds in the past.
  bool Lookup(const std::string& key, T* value) {
    if (max_age_ == 0) {
      return false;
    }
    absl::MutexLock lock(&mu_);
    return LookupLocked(key, value);
  }

  typedef std::function<void(const std::string&, T*, TF_Status*)> ComputeFunc;

  /// Look up the entry with key `key` and copy it to `value` if found. If not
  /// found, call `compute_func`. If `compute_func` set `status` to `TF_OK`,
  /// store a copy of the output parameter in the cache, and another copy in
  /// `value`.
  void LookupOrCompute(const std::string& key, T* value,
                       const ComputeFunc& compute_func, TF_Status* status) {
    if (max_age_ == 0) {
      return compute_func(key, value, status);
    }

    // Note: we hold onto mu_ for the rest of this function. In practice, this
    // is okay, as stat requests are typically fast, and concurrent requests are
    // often for the same file. Future work can split this up into one lock per
    // key if this proves to be a significant performance bottleneck.
    absl::MutexLock lock(&mu_);
    if (LookupLocked(key, value)) {
      return TF_SetStatus(status, TF_OK, "");
    }
    compute_func(key, value, status);
    if (TF_GetCode(status) == TF_OK) {
      InsertLocked(key, *value);
    }
  }

  /// Clear the cache.
  void Clear() {
    absl::MutexLock lock(&mu_);
    cache_.clear();
    lru_list_.clear();
  }

  /// Accessors for cache parameters.
  uint64_t max_age() const { return max_age_; }
  size_t max_entries() const { return max_entries_; }

 private:
  struct Entry {
    /// The timestamp (seconds) at which the entry was added to the cache.
    uint64_t timestamp;

    /// The entry's value.
    T value;

    /// A list iterator pointing to the entry's position in the LRU list.
    std::list<std::string>::iterator lru_iterator;
  };

  bool LookupLocked(const std::string& key, T* value)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return false;
    }
    lru_list_.erase(it->second.lru_iterator);
    if (timer_seconds_() - it->second.timestamp > max_age_) {
      cache_.erase(it);
      return false;
    }
    *value = it->second.value;
    lru_list_.push_front(it->first);
    it->second.lru_iterator = lru_list_.begin();
    return true;
  }

  void InsertLocked(const std::string& key, const T& value)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    lru_list_.push_front(key);
    Entry entry{timer_seconds_(), value, lru_list_.begin()};
    auto insert = cache_.insert(std::make_pair(key, entry));
    if (!insert.second) {
      lru_list_.erase(insert.first->second.lru_iterator);
      insert.first->second = entry;
    } else if (max_entries_ > 0 && cache_.size() > max_entries_) {
      cache_.erase(lru_list_.back());
      lru_list_.pop_back();
    }
  }

  bool DeleteLocked(const std::string& key) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return false;
    }
    lru_list_.erase(it->second.lru_iterator);
    cache_.erase(it);
    return true;
  }

  /// The maximum age of entries in the cache, in seconds. A value of 0 means
  /// that no entry is ever placed in the cache.
  const uint64_t max_age_;

  /// The maximum number of entries in the cache. A value of 0 means there is no
  /// limit on entry count.
  const size_t max_entries_;

  /// The callback to read timestamps.
  std::function<uint64_t()> timer_seconds_;

  /// Guards access to the cache and the LRU list.
  absl::Mutex mu_;

  /// The cache (a map from string key to Entry).
  std::map<std::string, Entry> cache_ ABSL_GUARDED_BY(mu_);

  /// The LRU list of entries. The front of the list identifies the most
  /// recently accessed entry.
  std::list<std::string> lru_list_ ABSL_GUARDED_BY(mu_);
};

}  // namespace tf_gcs_filesystem

#endif  // TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_GCS_EXPIRING_LRU_CACHE_H_
