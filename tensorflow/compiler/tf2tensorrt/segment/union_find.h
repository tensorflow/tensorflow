/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_SEGMENT_UNION_FIND_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_SEGMENT_UNION_FIND_H_

#include "absl/strings/str_format.h"
#include "absl/types/optional.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
namespace segment {

// ClusterBatchSize is a data structure to record the batch size we have seen
// for a cluster during segmentation.
//
// When constructing clusters for implicit batch mode, we support the
// with both dynamic batch size and static batch size. We restrict nodes inside
// a cluster to either have dynamic batch size or have the same value for static
// batch size. For this reason, we use a field has_dynamic_batch_value_ to keep
// track of whether the cluster has any node with dynamic batch size. We use
// field static_batch_value_ to keep track of whether the cluster has any node
// with static batch size and what the value of the static batch size, if any.
// Examples:
// cluster:  a = a1[1,3] + a1[1,3]
// ClusterBatchSize: has_dynamic_batch_size_ = false
//                   static_batch_value_ = {has value, 1}
//
// cluster:  b = b1[-1,3] + b2[-1, 3]
// ClusterBatchSize: has_dynamic_batch_size_ = true
//                   static_batch_value_ = {has no value}
//
// cluster:  a = a1[1,3] + a1[1,3]; b = b1[-1,3] + b2[-1, 3]
// ClusterBatchSize: has_dynamic_batch_size_ = true
//                   static_batch_value_ = {has value, 1}
//
// When constructing cluster for explicit batch mode, all ClusterBatchSize is
// irrelevant.
//
//
absl::optional<int> static_batch_value_;
class ClusterBatchSize {
 public:
  ClusterBatchSize()
      : has_dynamic_batch_value_(false), static_batch_value_(absl::nullopt) {}

  bool operator==(const ClusterBatchSize& b) {
    return HasDynamicBatchValue() == b.HasDynamicBatchValue() &&
           static_batch_value_ == b.static_batch_value_;
  }

  bool operator!=(const ClusterBatchSize& b) { return !(*this == b); }

  int GetStaticBatchValue() const {
    DCHECK(HasStaticBatchValue());
    return static_batch_value_.value();
  }

  // Sets the batch size value assuming that the object doesn't have a batch
  // size value yet:
  //   a non-negative input value representing a known batch size.
  //   a negative input value representing a dynamic batch size.
  ClusterBatchSize SetBatchSizeValue(int value) {
    if (value < 0) {
      has_dynamic_batch_value_ = true;
      return *this;
    }
    static_batch_value_ = value;
    return *this;
  }

  bool MergeIfCompatible(const ClusterBatchSize& b) {
    bool is_compatible = MergeIfCompatible(b.static_batch_value_);
    if (!is_compatible) return false;

    if (!HasDynamicBatchValue() && b.HasDynamicBatchValue()) {
      has_dynamic_batch_value_ = true;
    }

    return true;
  }

  // Returns a string for the batch size value. If the object has a static
  // batch size value, return a string for the value. If the object has a
  // dynamic size value, return -1. Otherwise, returns -2 to represent that
  // a batch size hasn't been set yet.
  string ToString() const {
    string s;
    absl::StrAppendFormat(&s, "batch_size=(%d,%d,", HasDynamicBatchValue(),
                          HasStaticBatchValue());
    if (HasStaticBatchValue()) {
      absl::StrAppendFormat(&s, "%d", GetStaticBatchValue());
    }
    absl::StrAppend(&s, ")");
    return s;
  }

 private:
  bool HasStaticBatchValue() const { return static_batch_value_.has_value(); }
  bool HasDynamicBatchValue() const { return has_dynamic_batch_value_; }

 private:
  bool MergeIfCompatible(const absl::optional<int>& b) {
    bool is_compatible = !HasStaticBatchValue() || !b.has_value() ||
                         GetStaticBatchValue() == b.value();
    if (!is_compatible) {
      return false;
    }
    if (!HasStaticBatchValue() && b.has_value()) {
      static_batch_value_ = b;
    }
    return true;
  }

 private:
  // To track whether the cluster has any node with dynamic batch size.
  bool has_dynamic_batch_value_;
  // To track whether the cluster has any node with static batch size, and the
  // unique value for static batch size.
  absl::optional<int> static_batch_value_;
};

inline std::ostream& operator<<(std::ostream& os,
                                const ClusterBatchSize& batch_size) {
  return os << batch_size.ToString();
}

// Represents a disjoint set of copyable values with type T. We use this data
// structure to construct clusters for TRTEngineOp. As such, this data structure
// has a field to record the batch size for the current cluster and merges the
// corresponding batch sizes when merging two clusters. Most of the methods in
// this class are side-effecting as they also compress the path from the object
// to the parent of its containing set.
template <typename T>
class UnionFind {
 public:
  UnionFind() : size_(1), parent_(nullptr) {}
  explicit UnionFind(const T& v, ClusterBatchSize batch_size)
      : size_(1),
        cluster_batch_size_(batch_size),
        parent_(nullptr),
        value_(v) {}

  // Returns the number of elements in the cluster and compresses the path from
  // this object to the root of the cluster.
  int Size() { return FindRoot()->size_; }

  // Returns the batch size of the cluster and compress the path from this
  // object to the root object.
  ClusterBatchSize BatchSize() { return FindRoot()->cluster_batch_size_; }

  // Merges this cluster with 'other'. This cluster's size_ is updated to
  // the size of the merged cluster; the size_ of 'other' becomes inaccessible
  // as only the size_ of the root object is accessible.
  Status Merge(UnionFind* other);

  // Retrieves the value for the root of the cluster.
  T& ParentValue() { return FindRoot()->value_; }

  // Returns the value for the object.
  T& Value() { return value_; }

 private:
  // Returns the root object for the cluster and compresses the path from this
  // object to the root object.
  UnionFind* FindRoot();

  int size_;
  ClusterBatchSize cluster_batch_size_;
  UnionFind* parent_;
  T value_;
};

template <typename T>
Status UnionFind<T>::Merge(UnionFind* other) {
  UnionFind<T>* a = FindRoot();
  UnionFind<T>* b = other->FindRoot();
  if (a == b) return Status::OK();

  ClusterBatchSize batch_size = a->cluster_batch_size_;
  bool merged = batch_size.MergeIfCompatible(other->cluster_batch_size_);
  if (!merged) {
    return errors::Internal("trying to merge incompatible cluster.");
  }

  a->cluster_batch_size_ = batch_size;
  b->parent_ = a;
  a->size_ += b->size_;
  return Status::OK();
}

template <typename T>
UnionFind<T>* UnionFind<T>::FindRoot() {
  if (!parent_) return this;
  // Path compression: update intermediate nodes to point to the root of the
  // equivalence class.
  parent_ = parent_->FindRoot();
  return parent_;
}

}  // namespace segment
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_SEGMENT_UNION_FIND_H_
