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

#include "absl/types/optional.h"
#include "tensorflow/core/util/device_name_utils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
namespace segment {

// ClusterBatchSize is a data structure to record the batch size we have seen
// for a cluster during segmentation.
//
// When constructing clusters for implicit batch mode, we support the
// with both dynamic batch size and static batch size. We restrict nodes inside
// a cluster to either have dynamic batch size or have the same value for static
// batch size. For this reason, we use a field has_dynamic_batch_size_ to keep
// track of whether the cluster has any node with dynamic batch size. We use
// field static_batch_size_ to keep track of whether the cluster has any node
// with static batch size and what the value of the static batch size, if any.
// Examples:
// cluster:  a = a1[1,3] + a1[1,3]
// ClusterBatchSize: has_dynamic_batch_size_ = false
//                   static_batch_size_ = {has value, 1}
//
// cluster:  b = b1[-1,3] + b2[-1, 3]
// ClusterBatchSize: has_dynamic_batch_size_ = true
//                   static_batch_size_ = {has no value}
//
// cluster:  a = a1[1,3] + a1[1,3]; b = b1[-1,3] + b2[-1, 3]
// ClusterBatchSize: has_dynamic_batch_size_ = true
//                   static_batch_size_ = {has value, 1}
//
// When constructing cluster for explicit batch mode, all ClusterBatchSize is
// irrelevant.
//
//

class ClusterBatchSize {
 public:
  ClusterBatchSize();

  bool operator==(const ClusterBatchSize& other);
  bool operator!=(const ClusterBatchSize& other) { return !(*this == other); }

  // Sets the batch size assuming that the object doesn't have a batch size yet:
  //   a non-negative input value representing a static batch size.
  //   a negative input value representing a dynamic batch size.
  ClusterBatchSize& SetBatchSize(int batch_size);
  bool HasStaticBatchSize() const { return static_batch_size_.has_value(); }
  int GetStaticBatchSize() const;

  bool MergeIfCompatible(const ClusterBatchSize& other);

  // Returns a string for the batch size.
  //   If the object has a static batch size, return a string for the value.
  //   If the object has a dynamic size, return -1.
  //   Otherwise, returns -2 to represent that the batch size hasn't been set
  //     yet.
  std::string ToString() const;

 private:
  bool HasDynamicBatchSize() const { return has_dynamic_batch_size_; }

  // To track whether the cluster has any node with dynamic batch size.
  bool has_dynamic_batch_size_;
  // To track whether the cluster has any node with static batch size, and the
  // unique value for static batch size.
  absl::optional<int> static_batch_size_;
};

inline std::ostream& operator<<(std::ostream& os,
                                const ClusterBatchSize& batch_size) {
  return os << batch_size.ToString();
}

// Represents the accumulated properties of a cluster during segmentation,
// including information about batch size and device assignment. Clusters shall
// have compatible properties in order to be merged together.
class ClusterProperty {
 public:
  ClusterProperty() {}
  ClusterProperty(const ClusterBatchSize& batch_size,
                  const DeviceNameUtils::ParsedName& device_name);

  // Returns the batch size of the cluster and compresses the path from this
  // object to the root object.
  const ClusterBatchSize& BatchSize() const { return batch_size_; }

  // Returns the device name of the cluster and compresses the path from this
  // object to the root object.
  const DeviceNameUtils::ParsedName& DeviceName() const { return device_name_; }

  Status Merge(const ClusterProperty& other);

 private:
  ClusterBatchSize batch_size_;
  DeviceNameUtils::ParsedName device_name_;
};

// Represents a disjoint set of copyable value with type T and accumulated
// property of the values with type P. Most of the methods in this class are
// side-effecting as they also compress the path from the object to the parent
// of its containing set.
template <typename T, typename P = ClusterProperty>
class UnionFind {
 public:
  UnionFind() : size_(1), parent_(nullptr) {}
  UnionFind(const T& v, const P& p)
      : size_(1), parent_(nullptr), value_(v), property_(p) {}
  UnionFind(const T& v, P&& p)
      : size_(1), parent_(nullptr), value_(v), property_(p) {}

  // Returns the number of elements in the set and compresses the path from
  // this object to the root of the set.
  int Size() { return FindRoot()->size_; }

  // Returns the accumulated property of all the elements in the set and
  // compresses the path from this object to the root of the set.
  const P& Property() { return FindRoot()->property_; }

  // Merges this set with 'other'. This updates the size_ and property_ of the
  // set. The size_ and property_ of 'other' becomes inaccessible as only the
  // size_ and property_ of the root of the set is accessible.
  Status Merge(UnionFind* other);

  // Retrieves the value for the root of the set.
  const T& ParentValue() { return FindRoot()->value_; }

  // Returns the value for the object.
  const T& Value() const { return value_; }

 private:
  // Returns the root object for the set and compresses the path from this
  // object to the root object.
  UnionFind* FindRoot();

  int size_;
  UnionFind* parent_;
  T value_;
  P property_;
};

template <typename T, typename P>
Status UnionFind<T, P>::Merge(UnionFind* other) {
  UnionFind<T>* a = FindRoot();
  UnionFind<T>* b = other->FindRoot();
  if (a == b) return Status::OK();

  P merged_property(a->property_);
  TF_RETURN_IF_ERROR(merged_property.Merge(b->property_));
  b->parent_ = a;
  a->size_ += b->size_;
  a->property_ = std::move(merged_property);
  return Status::OK();
}

template <typename T, typename P>
UnionFind<T, P>* UnionFind<T, P>::FindRoot() {
  if (!parent_) return this;
  // Path compression: update intermediate nodes to point to the root of the
  // equivalence class.
  parent_ = parent_->FindRoot();
  return parent_;
}

}  // namespace segment
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_SEGMENT_UNION_FIND_H_
