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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/device_name_utils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
namespace segment {

// ClusterBatchSize is a data structure to record the batch size we have seen
// for a cluster during segmentation.
//
// With the help of shape inference, all the dynamic batch sizes are converted
// to a negative integer number.
// If the number is -1, then nothing is known about the dynamic batch size.
// Ideally, we should not put nodes with -1 batch size into the same cluster,
// as they will likely have different batch sizes at runtime. However, we
// currently treat -1 as an equivalent class for simple implementation. We may
// need to revise this if it causes performance issues.
// If the number is strictly less than -1, then it represents a equivalent
// class. It is infered that all the nodes with the same equivalent class
// (strictly less than -1) shall have the same batch size at runtime.
//
// When constructing clusters for implicit batch mode, we support both
// dynamic batch sizes and static batch sizes. As all the nodes inside the same
// cluster shall have the same batch size at runtime, we restrict nodes inside a
// cluster to either have the same dynamic batch size equivalent class or the
// same static batch size value.
//
// Besides, all the nodes with an annotated max batch size inside the same
// cluster shall have the same annotated max batch size. (It is allowed if
// part or all the nodes inside the cluster doesn't have annotated max batch
// size). Static batch sizes are treated as max batch size annotations. The
// converter max batch size is used for an OP with a dynamic batch size and no
// annotated max batch size.
//
// cluster:  a = a1[1,3] + a1[1,3]
// ClusterBatchSize: batch_size_ = 1
//                   max_batch_size_ = 1
//
// cluster:  b = b1[-1,3] + b2[-1, 3]
// ClusterBatchSize: batch_size_ = -1
//                   max_batch_size_ = null
//
// cluster:  c = c1[-2,3] + c2[-2, 3](max_batch_size=100)
// ClusterBatchSize: batch_size_ = -2
//                   max_batch_size_ = 100
//
// When constructing cluster for explicit batch mode, all ClusterBatchSize is
// irrelevant.
//

class ClusterBatchSize {
 public:
  ClusterBatchSize();

  bool operator==(const ClusterBatchSize& other);
  bool operator!=(const ClusterBatchSize& other) { return !(*this == other); }

  // Sets the batch size assuming that the object doesn't have a batch size yet:
  //   A non-negative input representing a static batch size value.
  //   A negative input representing a dynamic batch size equivalent class.
  ClusterBatchSize& SetBatchSize(int batch_size);
  bool HasBatchSize() const;
  int GetBatchSize() const;

  // Sets the max batch size assuming that the object doesn't have a max batch
  // size yet.
  ClusterBatchSize& SetMaxBatchSize(int max_batch_size);
  absl::optional<int> GetOptionalMaxBatchSize() const;

  // Merge `other` into the current ClusterBatchSize if the two are not
  // conflicting. Two ClusterBatchSizes are conflicting iff they both have a
  // value and their values are different.
  bool MergeIfCompatible(const ClusterBatchSize& other);

  // Returns a string for the batch size and the annotated max batch size.
  // For the batch size:
  //   If the object has a static batch size, return a string representing a
  //     non-negative integer.
  //   If the object has a dynamic batch size, return a string representing a
  //     negative integer as an equivalent class.
  //   If the object doesn't have a batch size yet, return "?".
  // For the annotated max batch size:
  //   If the cluster has annotated max batch size in at least one of the nodes,
  //     return a string representing the annotated max batch size. Otherwise,
  //     return "?".
  std::string ToString() const;

 private:
  ClusterBatchSize& SetBatchSize(const absl::optional<int>& batch_size);
  ClusterBatchSize& SetMaxBatchSize(const absl::optional<int>& batch_size);

  absl::optional<int> batch_size_;
  absl::optional<int> max_batch_size_;
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
