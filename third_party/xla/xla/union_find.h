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

#include <algorithm>
#include <cstdint>
#include <numeric>
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

// Union-Find over the dense index range [0, num_elements), in two flat arrays.
// Use it instead of UnionFind<T> when the elements already have dense ids (for
// example the local ids of a computation's instructions): no per element
// allocation and no hash map from element to cluster.
class DenseUnionFind {
 public:
  explicit DenseUnionFind(int64_t num_elements)
      : parent_(num_elements), size_(num_elements, 1) {
    std::iota(parent_.begin(), parent_.end(), 0);
  }

  int64_t num_elements() const { return parent_.size(); }

  // Returns the representative of the cluster containing 'element'. Compresses
  // the path by halving.
  int64_t Find(int64_t element) {
    DCHECK_GE(element, 0);
    DCHECK_LT(element, num_elements());
    while (parent_[element] != element) {
      parent_[element] = parent_[parent_[element]];
      element = parent_[element];
    }
    return element;
  }

  // Merges the clusters of 'a' and 'b'. The larger cluster's representative
  // becomes the representative of the merged cluster.
  void Merge(int64_t a, int64_t b) {
    a = Find(a);
    b = Find(b);
    if (a == b) return;
    if (size_[a] < size_[b]) std::swap(a, b);
    parent_[b] = a;
    size_[a] += size_[b];
  }

  // Returns the number of elements in the cluster containing 'element'.
  int64_t Size(int64_t element) { return size_[Find(element)]; }

 private:
  std::vector<int64_t> parent_;
  std::vector<int64_t> size_;
};

}  // namespace xla

#endif  // XLA_UNION_FIND_H_
