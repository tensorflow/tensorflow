/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_UNION_FIND_H_
#define XLA_UNION_FIND_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/log/check.h"

namespace xla {

// Union-Find data structure.
// Each cluster has an associated value; when merging clusters we can control
// which value becomes the representative of the merged clusters. Values must be
// copyable.
template <typename T>
class UnionFind {
 public:
  explicit UnionFind(const T& value = T())
      : rank_(0), size_(1), parent_(nullptr), value_(value) {}

  // Returns the number of elements in a cluster.
  int Size() { return FindRoot()->size_; }

  // Merges this cluster with 'other'. This cluster's value becomes
  // the value of the merged cluster; the value of 'other' is ignored.
  void Merge(UnionFind* other);

  // Each cluster has an associated value. Retrieves the value associated
  // with this cluster.
  T& Get() { return FindRoot()->value_; }

 private:
  // Finds the root element of the cluster. Performs path compression.
  UnionFind* FindRoot();

  int rank_;
  int size_;  // Size of the cluster.
  UnionFind* parent_;
  T value_;
};

template <typename T>
void UnionFind<T>::Merge(UnionFind* other) {
  UnionFind<T>* a = FindRoot();
  UnionFind<T>* b = other->FindRoot();
  if (a == b) return;
  if (a->rank_ > b->rank_) {
    b->parent_ = a;
    a->size_ += b->size_;
    return;
  }

  a->parent_ = b;
  if (a->rank_ == b->rank_) {
    b->rank_++;
  }
  b->value_ = a->value_;
  b->size_ += a->size_;
}

template <typename T>
UnionFind<T>* UnionFind<T>::FindRoot() {
  if (!parent_) return this;
  // Path compression: update intermediate nodes to point to the root of the
  // equivalence class.
  parent_ = parent_->FindRoot();
  return parent_;
}

// Union-Find optimized for a dense index range [0, num_elements). Use it
// instead of UnionFind<T> when the elements already have dense ids (for
// example the local ids of a computation's instructions). Avoids per element
// allocations and the need for a hash_map from element to cluster: the whole
// structure is two flat arrays.
class DenseUnionFind {
 public:
  // Every element starts in its own cluster: both arrays zero initialized.
  explicit DenseUnionFind(int num_elements)
      : parent_plus_one_(CheckNonNegative(num_elements), 0),
        rank_(num_elements, 0) {}

  int num_elements() const { return parent_plus_one_.size(); }

  // Returns the representative of the cluster containing 'element'. Compresses
  // the path on the way up by path halving, the one pass variant of path
  // compression: every other node on the path is pointed at its grandparent.
  int Find(int element) {
    DCHECK_GE(element, 0);
    DCHECK_LT(element, num_elements());
    while (true) {
      const int parent = Parent(element);
      if (parent == element) {
        return element;
      }
      const int grandparent = Parent(parent);
      parent_plus_one_[element] = grandparent + 1;
      element = grandparent;
    }
  }

  // Merges the clusters of 'a' and 'b' by rank: the root of higher rank
  // becomes the representative of the merged cluster, and equal ranks grow the
  // surviving root's rank by one.
  void Merge(int a, int b) {
    a = Find(a);
    b = Find(b);
    if (a == b) {
      return;
    }
    if (rank_[a] < rank_[b]) {
      std::swap(a, b);
    }
    if (rank_[a] == rank_[b]) {
      ++rank_[a];
    }
    parent_plus_one_[b] = a + 1;
  }

 private:
  static int CheckNonNegative(int count) {
    DCHECK_GE(count, 0);
    return count;
  }

  // A root is its own parent.
  int Parent(int element) const {
    const int stored = parent_plus_one_[element];
    return stored == 0 ? element : stored - 1;
  }

  // The parent of each element plus one; 0 marks a root, so the zero
  // initialized array holds singletons and needs no iota.
  std::vector<int> parent_plus_one_;
  // The rank of a root bounds the height of its tree and stays below
  // log2(num_elements), so a byte holds it.
  std::vector<uint8_t> rank_;
};

}  // namespace xla

#endif  // XLA_UNION_FIND_H_
