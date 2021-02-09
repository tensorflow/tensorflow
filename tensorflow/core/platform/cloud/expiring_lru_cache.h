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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_EXPIRING_LRU_CACHE_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_EXPIRING_LRU_CACHE_H_

#include <list>
#include <map>
#include <memory>
#include <string>
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

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
  ExpiringLRUCache(uint64 max_age, size_t max_entries,
                   Env* env = Env::Default())
      : max_age_(max_age), max_entries_(max_entries), env_(env) {}

  /// Insert `value` with key `key`. This will replace any previous entry with
  /// the same key.
  void Insert(const string& key, const T& value) {
    if (max_age_ == 0) {
      return;
    }
    mutex_lock lock(mu_);
    InsertLocked(key, value);
  }

  // Delete the entry with key `key`. Return true if the entry was found for
  // `key`, false if the entry was not found. In both cases, there is no entry
  // with key `key` existed after the call.
  bool Delete(const string& key) {
    mutex_lock lock(mu_);
    return DeleteLocked(key);
  }

  /// Look up the entry with key `key` and copy it to `value` if found. Returns
  /// true if an entry was found for `key`, and its timestamp is not more than
  /// max_age_ seconds in the past.
  bool Lookup(const string& key, T* value) {
    if (max_age_ == 0) {
      return false;
    }
    mutex_lock lock(mu_);
    return LookupLocked(key, value);
  }

  typedef std::function<Status(const string&, T*)> ComputeFunc;

  /// Look up the entry with key `key` and copy it to `value` if found. If not
  /// found, call `compute_func`. If `compute_func` returns successfully, store
  /// a copy of the output parameter in the cache, and another copy in `value`.
  Status LookupOrCompute(const string& key, T* value,
                         const ComputeFunc& compute_func) {
    if (max_age_ == 0) {
      return compute_func(key, value);
    }

    // Note: we hold onto mu_ for the rest of this function. In practice, this
    // is okay, as stat requests are typically fast, and concurrent requests are
    // often for the same file. Future work can split this up into one lock per
    // key if this proves to be a significant performance bottleneck.
    mutex_lock lock(mu_);
    if (LookupLocked(key, value)) {
      return Status::OK();
    }
    Status s = compute_func(key, value);
    if (s.ok()) {
      InsertLocked(key, *value);
    }
    return s;
  }

  /// Clear the cache.
  void Clear() {
    mutex_lock lock(mu_);
    cache_.clear();
    lru_list_.clear();
  }

  /// Accessors for cache parameters.
  uint64 max_age() const { return max_age_; }
  size_t max_entries() const { return max_entries_; }

 private:
  struct Entry {
    /// The timestamp (seconds) at which the entry was added to the cache.
    uint64 timestamp;

    /// The entry's value.
    T value;

    /// A list iterator pointing to the entry's position in the LRU list.
    std::list<string>::iterator lru_iterator;
  };

  bool LookupLocked(const string& key, T* value)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return false;
    }
    lru_list_.erase(it->second.lru_iterator);
    if (env_->NowSeconds() - it->second.timestamp > max_age_) {
      cache_.erase(it);
      return false;
    }
    *value = it->second.value;
    lru_list_.push_front(it->first);
    it->second.lru_iterator = lru_list_.begin();
    return true;
  }

  void InsertLocked(const string& key, const T& value)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    lru_list_.push_front(key);
    Entry entry{env_->NowSeconds(), value, lru_list_.begin()};
    auto insert = cache_.insert(std::make_pair(key, entry));
    if (!insert.second) {
      lru_list_.erase(insert.first->second.lru_iterator);
      insert.first->second = entry;
    } else if (max_entries_ > 0 && cache_.size() > max_entries_) {
      cache_.erase(lru_list_.back());
      lru_list_.pop_back();
    }
  }

  bool DeleteLocked(const string& key) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
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
  const uint64 max_age_;

  /// The maximum number of entries in the cache. A value of 0 means there is no
  /// limit on entry count.
  const size_t max_entries_;

  /// The Env from which we read timestamps.
  Env* const env_;  // not owned

  /// Guards access to the cache and the LRU list.
  mutex mu_;

  /// The cache (a map from string key to Entry).
  std::map<string, Entry> cache_ TF_GUARDED_BY(mu_);

  /// The LRU list of entries. The front of the list identifies the most
  /// recently accessed entry.
  std::list<string> lru_list_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_EXPIRING_LRU_CACHE_H_
