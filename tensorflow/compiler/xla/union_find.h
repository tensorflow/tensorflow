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

#ifndef TENSORFLOW_COMPILER_JIT_UNION_FIND_H_
#define TENSORFLOW_COMPILER_JIT_UNION_FIND_H_

namespace tensorflow {

// Union-Find data structure.
// Each cluster has an associated value; when merging clusters we can control
// which value becomes the representative of the merged clusters. Values must be
// copyable.
template <typename T>
class UnionFind {
 public:
  UnionFind() : rank_(0), size_(1), parent_(nullptr) {}

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

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_UNION_FIND_H_
