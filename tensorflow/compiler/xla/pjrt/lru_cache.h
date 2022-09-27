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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_LRU_CACHE_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_LRU_CACHE_H_

#include <optional>

#include "absl/container/node_hash_map.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {

// A simple LRU cache. Not thread-safe.
// Value must be copyable and moveable. The intent is that Value is typically
// a smart-pointer type.
template <typename Key, typename Value,
          typename Hash = typename absl::node_hash_map<Key, Value>::hasher,
          typename Eq = typename absl::node_hash_map<Key, Value>::key_equal>
class LRUCache {
 private:
  struct LRUListEntry {
    LRUListEntry* next;
    LRUListEntry* prev;
  };

 public:
  // Multiple LRUCaches can share a LRU list, meaning that the capacity and
  // eviction policy is shared. The user provides an LRU list
  // to the cache constructor, and must ensure that it remains alive as long
  // as the cache does.
  class LRUList {
   public:
    explicit LRUList(int capacity) : capacity_(capacity) {
      head_.next = &head_;
      head_.prev = &head_;
    }
    ~LRUList() {
      CHECK(head_.next == &head_);
      CHECK(head_.prev == &head_);
    }

    LRUList(const LRUList&) = delete;
    LRUList(LRUList&&) = delete;
    LRUList& operator=(const LRUList&) = delete;
    LRUList& operator=(LRUList&&) = delete;

    int Capacity() const { return capacity_; }
    int Size() const { return size_; }

    void Clear();

   private:
    friend class LRUCache;
    int capacity_;
    int size_ = 0;

    // Root of a circular doubly-linked list of entries, in order from least
    // recently used to most recently used. An "empty" cache always contains
    // this element in the LRU list.
    LRUListEntry head_;
  };

  explicit LRUCache(LRUList* lru_list) : lru_list_(lru_list) {}
  ~LRUCache();

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
  int Capacity() const { return lru_list_->Capacity(); }

 private:
  LRUList* lru_list_;

  struct Entry : public LRUListEntry {
    Entry() = default;

    // Pointer to the key in `entries_`. absl::node_hash_map<> promises
    // pointer stability for keys.
    const Key* key;
    LRUCache* container;
    std::optional<Value> value;
  };

  // We use `node_hash_map` because we want to guarantee pointer stability for
  // keys and values.
  absl::node_hash_map<Key, Entry, Hash, Eq> entries_;
};

template <typename Key, typename Value, typename Hash, typename Eq>
void LRUCache<Key, Value, Hash, Eq>::LRUList::Clear() {
  while (head_.next != &head_) {
    static_cast<Entry*>(head_.next)->container->Clear();
  }
  size_ = 0;
}

template <typename Key, typename Value, typename Hash, typename Eq>
void LRUCache<Key, Value, Hash, Eq>::Clear() {
  for (auto& e : entries_) {
    LRUListEntry* l = &e.second;
    l->next->prev = l->prev;
    l->prev->next = l->next;
    --lru_list_->size_;
  }
  entries_.clear();
}

template <typename Key, typename Value, typename Hash, typename Eq>
LRUCache<Key, Value, Hash, Eq>::~LRUCache() {
  Clear();
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
    ++lru_list_->size_;
  } else {
    // Removes the entry from the LRU list, in preparation for adding it
    // to the back of the list.
    entry.prev->next = entry.next;
    entry.next->prev = entry.prev;
  }
  // (Re-)adds entry to the back of the LRU list. Since it is now the
  // most recently used element, it goes at the back.
  LRUListEntry& lru_head = lru_list_->head_;
  entry.container = this;
  entry.prev = lru_head.prev;
  entry.next = &lru_head;
  lru_head.prev->next = &entry;
  lru_head.prev = &entry;

  Value v = *entry.value;

  // Evict an LRU entry if we are over capacity.
  if (lru_list_->size_ > lru_list_->capacity_) {
    Entry* to_remove = static_cast<Entry*>(lru_head.next);
    to_remove->next->prev = &lru_head;
    lru_head.next = to_remove->next;
    // Extract instead of erase in case the kv pair contains python objects
    // whose destruction could call back into this code. Extract causes the
    // dtor to be delayed until the kv pair is fully removed from the map.
    to_remove->container->entries_.extract(*to_remove->key);
    --lru_list_->size_;
  }
  return v;
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_LRU_CACHE_H_
