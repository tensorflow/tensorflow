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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_LRU_CACHE_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_LRU_CACHE_H_

#include "absl/container/node_hash_map.h"
#include "absl/types/optional.h"

namespace xla {

// A simple LRU cache. Not thread-safe.
// Value must be copyable and moveable. The intent is that Value is typically
// a smart-pointer type.
template <typename Key, typename Value,
          typename Hash = typename absl::node_hash_map<Key, Value>::hasher,
          typename Eq = typename absl::node_hash_map<Key, Value>::key_equal>
class LRUCache {
 public:
  explicit LRUCache(int capacity);

  LRUCache(const LRUCache&) = delete;
  LRUCache(LRUCache&&) = delete;
  LRUCache& operator=(const LRUCache&) = delete;
  LRUCache& operator=(LRUCache&&) = delete;

  // Returns the `value` associated with `key`. Creates a value with `factory`
  // and inserts it if absent.
  Value GetOrCreateIfAbsent(const Key& key,
                            const std::function<Value(const Key&)>& factory);

  // Removes all entries from the cache.
  void Clear();

  int Size() const { return entries_.size(); }
  int Capacity() const { return capacity_; }

 private:
  const int capacity_;

  struct Entry {
    Entry() = default;

    // Pointer to the key in `entries_`. absl::node_hash_map<> promises
    // pointer stability for keys.
    const Key* key;
    absl::optional<Value> value;
    Entry* next;  // Circular list; never nullptr.
    Entry* prev;  // Circular list; ever nullptr.
  };
  // Root of a circular doubly-linked list of entries, in order from least
  // recently used to most recently used. An "empty" cache always contains
  // this element in the LRU list.
  Entry lru_root_;

  // We use `node_hash_map` because we want to guarantee pointer stability for
  // keys and values.
  absl::node_hash_map<Key, Entry, Hash, Eq> entries_;
};

template <typename Key, typename Value, typename Hash, typename Eq>
LRUCache<Key, Value, Hash, Eq>::LRUCache(int capacity) : capacity_(capacity) {
  lru_root_.next = &lru_root_;
  lru_root_.prev = &lru_root_;
}

template <typename Key, typename Value, typename Hash, typename Eq>
void LRUCache<Key, Value, Hash, Eq>::Clear() {
  entries_.clear();
  lru_root_.next = &lru_root_;
  lru_root_.prev = &lru_root_;
}

template <typename Key, typename Value, typename Hash, typename Eq>
Value LRUCache<Key, Value, Hash, Eq>::GetOrCreateIfAbsent(
    const Key& key, const std::function<Value(const Key&)>& factory) {
  typename absl::node_hash_map<Key, Entry, Hash, Eq>::iterator it;
  bool inserted;
  std::tie(it, inserted) = entries_.try_emplace(key);
  Entry& entry = it->second;
  if (inserted) {
    entry.key = &it->first;
    entry.value = factory(*entry.key);
  } else {
    // Removes the entry from the LRU list, in preparation for adding it
    // to the back of the list.
    entry.prev->next = entry.next;
    entry.next->prev = entry.prev;
  }
  // (Re-)adds entry to the back of the LRU list. Since it is now the
  // most recently used element, it goes at the back.
  entry.prev = lru_root_.prev;
  entry.next = &lru_root_;
  lru_root_.prev->next = &entry;
  lru_root_.prev = &entry;

  Value v = *entry.value;

  // Evicts an entry if we are over capacity.
  if (entries_.size() > capacity_) {
    Entry* to_remove = lru_root_.next;
    to_remove->next->prev = &lru_root_;
    lru_root_.next = to_remove->next;
    entries_.erase(*to_remove->key);
  }
  return v;
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_LRU_CACHE_H_
