/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_TYPED_QUEUE_H_
#define TENSORFLOW_CORE_KERNELS_TYPED_QUEUE_H_

#include <deque>
#include <queue>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/queue_base.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

// TypedQueue builds on QueueBase, with backing class (SubQueue)
// known and stored within.  Shared methods that need to have access
// to the backed data sit in this class.
template <typename SubQueue>
class TypedQueue : public QueueBase {
 public:
  TypedQueue(const int32_t capacity, const DataTypeVector& component_dtypes,
             const std::vector<TensorShape>& component_shapes,
             const string& name);

  virtual absl::Status Initialize();  // Must be called before any other method.

  int64_t MemoryUsed() const override;

 protected:
  std::vector<SubQueue> queues_ TF_GUARDED_BY(mu_);
};  // class TypedQueue

template <typename SubQueue>
TypedQueue<SubQueue>::TypedQueue(
    int32_t capacity, const DataTypeVector& component_dtypes,
    const std::vector<TensorShape>& component_shapes, const string& name)
    : QueueBase(capacity, component_dtypes, component_shapes, name) {}

template <typename SubQueue>
absl::Status TypedQueue<SubQueue>::Initialize() {
  if (component_dtypes_.empty()) {
    return errors::InvalidArgument("Empty component types for queue ", name_);
  }
  if (!component_shapes_.empty() &&
      component_dtypes_.size() != component_shapes_.size()) {
    return errors::InvalidArgument(
        "Different number of component types.  ",
        "Types: ", DataTypeSliceString(component_dtypes_),
        ", Shapes: ", ShapeListString(component_shapes_));
  }

  mutex_lock lock(mu_);
  queues_.reserve(num_components());
  for (int i = 0; i < num_components(); ++i) {
    queues_.push_back(SubQueue());
  }
  return absl::OkStatus();
}

template <typename SubQueue>
inline int64_t SizeOf(const SubQueue& sq) {
  static_assert(sizeof(SubQueue) != sizeof(SubQueue), "SubQueue size unknown.");
  return 0;
}

template <>
inline int64_t SizeOf(const std::deque<Tensor>& sq) {
  if (sq.empty()) {
    return 0;
  }
  return sq.size() * sq.front().AllocatedBytes();
}

template <>
inline int64_t SizeOf(const std::vector<Tensor>& sq) {
  if (sq.empty()) {
    return 0;
  }
  return sq.size() * sq.front().AllocatedBytes();
}

using TensorPair = std::pair<int64_t, Tensor>;

template <typename U, typename V>
int64_t SizeOf(const std::priority_queue<TensorPair, U, V>& sq) {
  if (sq.empty()) {
    return 0;
  }
  return sq.size() * (sizeof(TensorPair) + sq.top().second.AllocatedBytes());
}

template <typename SubQueue>
inline int64_t TypedQueue<SubQueue>::MemoryUsed() const {
  int memory_size = 0;
  mutex_lock l(mu_);
  for (const auto& sq : queues_) {
    memory_size += SizeOf(sq);
  }
  return memory_size;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TYPED_QUEUE_H_
