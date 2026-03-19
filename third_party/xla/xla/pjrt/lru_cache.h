/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_PJRT_LRU_CACHE_H_
#define XLA_PJRT_LRU_CACHE_H_

#include <cstddef>
#include <functional>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/container/hash_container_defaults.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"

namespace xla {

// A simple LRU cache. Not thread-safe.
// Value must be copyable and moveable. The intent is that Value is typically
// a smart-pointer type.
template <typename Key, typename Value,
          typename Hash = absl::DefaultHashContainerHash<Key>,
          typename Eq = absl::DefaultHashContainerEq<Key>>
class LRUCache {
  // # Implementation Notes
  //
  // LRUCache is used by JAX. The cache keys are Python objects that may be
  // mutated by the user's program. We have to be careful that these mutations
  // don't cause a crash or lead to undefined behavior.
  //
  // LRUCache (like most LRU caches) is implemented with a hash table and a
  // doubly linked list. To handle key mutations, the hash table is *not* keyed
  // by cache keys. Instead, it is keyed by their hashes. To handle hash
  // collisions, the hash table stores vectors of pointers to the linked list
  // nodes. In addition to keys and values, these linked list nodes also store
  // their hashes so that they can be found in the hash table.
  //
  // # An Example
  //
  // Consider unequal keys x, y, and z with hashes 42, 94, and 94 respectively.
  // Here is what the LRUCache data structure looks like after inserting (x, 1),
  // (y, 2), and (z, 3) into the cache and then accessing keys x and z.
  //
  //
  //    hash      node pointers         0x0020        0x0010        0x0030
  //   +----+------------------+       +---------+   +---------+   +---------+
  //   | 42 |         [0x0010] |       | key=y   |   | key=x   |   | key=z   |
  //   +----+------------------+       | val=2   |-->| key=1   |-->| key=3   |
  //   | 94 | [0x0020, 0x0030] |       | hash=94 |   | hash=42 |   | hash=94 |
  //   +----+------------------+       +---------+   +---------+   +---------+
  //
  // # Behavior When Keys Change
  //
  // For an LRUCache to be "correct", cache.GetOrCreateIfAbsent(x, f) should
  // always return f(x). For an LRUCache to be performant, it should call f as
  // infrequently as possible.
  //
  // An LRUCache behaves correctly only when cache keys are not mutated. If a
  // cache key is mutated after being inserted into the cache, then the cache
  // may exhibit anomalous behavior, but the cache will not crash or have
  // undefined behavior.
  //
  // As an example, imagine that key y in the example above is changed to be
  // equal to key z. Looking up the value of key z in the cache may return the
  // value of key y. In other words, if we let y0 be the original value of y,
  // then cache.GetOrCreateIfAbsent(z, f) returns f(y0) != f(z).

 private:
  struct Node {
    Node* next;
    Node* prev;

    LRUCache* container;
    std::size_t hash;
    Key key;
    Value value;
  };

 public:
  // Multiple LRUCaches can share an LRU list, meaning that the capacity and
  // eviction policy is shared. The user provides an LRU list to the cache
  // constructor, and must ensure that it remains alive as long as the cache
  // does.
  class LRUList {
   public:
    explicit LRUList(int capacity) : capacity_(capacity) {}

    ~LRUList() { CHECK_EQ(head_, nullptr); }

    LRUList(const LRUList&) = delete;
    LRUList(LRUList&&) = delete;
    LRUList& operator=(const LRUList&) = delete;
    LRUList& operator=(LRUList&&) = delete;

    int Capacity() const { return capacity_; }
    int Size() const { return size_; }

    void Clear() {
      while (head_ != nullptr) {
        head_->container->Clear();
      }
    }

   private:
    friend class LRUCache;
    int capacity_;
    int size_ = 0;

    // Pointer to a circular doubly-linked list of entries, in order from least
    // recently used to most recently used.
    Node* head_ = nullptr;
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

  void Remove(const Key& key);

  // Removes all entries from the cache.
  void Clear();

  int Size() const { return size_; }
  int Capacity() const { return lru_list_->Capacity(); }

  // Calls f on every key-value pair in the cache.
  void ForEach(std::function<void(const Key&, const Value&)> f) const {
    for (const auto& [unused, bucket] : buckets_) {
      for (const std::unique_ptr<Node>& node : bucket) {
        f(node->key, node->value);
      }
    }
  }

 private:
  using Bucket = absl::InlinedVector<std::unique_ptr<Node>, 1>;

  // Removes a node from lru_list_.
  //
  // REQUIRES: node is in lru_list_.
  void RemoveNode(Node& node) {
    if (lru_list_->head_ == &node) {
      lru_list_->head_ = node.next;
    }
    node.next->prev = node.prev;
    node.prev->next = node.next;
    size_--;
    lru_list_->size_--;
    if (lru_list_->size_ == 0) {
      lru_list_->head_ = nullptr;
    }
  }

  // Appends a node to the end of lru_list_.
  //
  // REQUIRES: node is not in lru_list_.
  void AppendNode(Node& node) {
    if (lru_list_->head_ == nullptr) {
      node.next = &node;
      node.prev = &node;
      lru_list_->head_ = &node;
    } else {
      Node* head = lru_list_->head_;
      Node* tail = head->prev;
      node.next = head;
      node.prev = tail;
      tail->next = &node;
      head->prev = &node;
    }
    size_++;
    lru_list_->size_++;
  }

  // Evicts a node, removing it from lru_list_ and from node->container.
  //
  // REQUIRES: node is in lru_list_ and in node->container.
  std::unique_ptr<Node> EvictNode(Node& node) {
    // We return the node, rather than just removing it, to extend its lifetime
    // and avoid calling a destructor that calls back into this code.
    RemoveNode(node);
    auto bucket_it = node.container->buckets_.find(node.hash);
    auto& [unused, bucket] = *bucket_it;
    CHECK(bucket_it != node.container->buckets_.end());
    for (auto it = bucket.begin(); it != bucket.end(); ++it) {
      if (it->get() != &node) {
        continue;
      }

      // Swap the node with the last element in the bucket, and then remove it.
      std::swap(*it, bucket.back());
      std::unique_ptr<Node> evicted = std::move(bucket.back());
      bucket.pop_back();
      if (bucket.empty()) {
        evicted->container->buckets_.erase(bucket_it);
      }
      return std::move(evicted);
    }
    LOG(FATAL) << "Evicted node not found";
  }

  Hash hasher_;
  Eq equals_;
  int size_ = 0;
  absl::flat_hash_map<std::size_t, Bucket> buckets_;
  LRUList* lru_list_;
};

template <typename Key, typename Value, typename Hash, typename Eq>
void LRUCache<Key, Value, Hash, Eq>::Clear() {
  for (auto& [unused, bucket] : buckets_) {
    for (std::unique_ptr<Node>& node : bucket) {
      RemoveNode(*node);
    }
  }
  // Deleting a cache entry may reentrantly trigger other calls into, say,
  // Clear().
  absl::flat_hash_map<std::size_t, Bucket> buckets;
  std::swap(buckets, buckets_);
}

template <typename Key, typename Value, typename Hash, typename Eq>
LRUCache<Key, Value, Hash, Eq>::~LRUCache() {
  Clear();
}

template <typename Key, typename Value, typename Hash, typename Eq>
void LRUCache<Key, Value, Hash, Eq>::Remove(const Key& key) {
  std::size_t hash = hasher_(key);
  auto it = buckets_.find(hash);
  if (it == buckets_.end()) {
    return;
  }

  // Extend the lifetime of the removed node to avoid any destructors calling
  // back into this code.
  std::unique_ptr<Node> removed;

  auto& [unused, bucket] = *it;
  for (auto node = bucket.begin(); node != bucket.end(); ++node) {
    if (equals_((*node)->key, key)) {
      RemoveNode(**node);
      // Swap the node with the last element in the bucket, and then remove it.
      std::swap(*node, bucket.back());
      removed = std::move(bucket.back());
      bucket.pop_back();
      break;
    }
  }

  if (bucket.empty()) {
    buckets_.erase(it);
  }
}

template <typename Key, typename Value, typename Hash, typename Eq>
Value LRUCache<Key, Value, Hash, Eq>::GetOrCreateIfAbsent(
    const Key& key, const std::function<Value(const Key&)>& factory) {
  std::size_t hash = hasher_(key);
  Bucket& bucket = buckets_[hash];
  for (std::unique_ptr<Node>& node : bucket) {
    if (equals_(key, node->key)) {
      // Move the node to the end of the LRU list because it is now the most
      // recently used key.
      RemoveNode(*node);
      AppendNode(*node);
      return node->value;
    }
  }

  Value value = factory(key);
  // To be compatible with C++17, we need to pass a Node to make_unique<Node>.
  auto node =
      std::make_unique<Node>(Node{nullptr, nullptr, this, hash, key, value});
  AppendNode(*node);
  bucket.push_back(std::move(node));

  // Evict an LRU entry if we are over capacity. Extend the lifetime of the
  // evicted node to avoid any destructors calling back into this code.
  std::unique_ptr<Node> evicted;
  if (lru_list_->size_ > lru_list_->capacity_) {
    evicted = EvictNode(*lru_list_->head_);
  }
  return value;
}

}  // namespace xla

#endif  // XLA_PJRT_LRU_CACHE_H_
