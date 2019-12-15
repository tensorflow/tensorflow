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

#ifndef TENSORFLOW_CORE_KERNELS_TENSOR_FOREST_RESOURCES_H_
#define TENSORFLOW_CORE_KERNELS_TENSOR_FOREST_RESOURCES_H_

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

// Forward declaration for proto class Tree.
namespace boosted_trees {
class Tree;
}  // namespace boosted_trees

// Keep a tree ensemble in memory for efficient evaluation and mutation.
class TensorForestTreeResource : public ResourceBase {
 public:
  TensorForestTreeResource();

  string DebugString() const override {
    return strings::StrCat("TensorForestTree[size=", get_size(), "]");
  }

  mutex* get_mutex() { return &mu_; }

  bool InitFromSerialized(const string& serialized);

  // Resets the resource and frees the proto.
  // Caller needs to hold the mutex lock while calling this.
  void Reset();

  const int32 get_size() const;

  const boosted_trees::Tree& decision_tree() const;

  const float get_prediction(const int32 id, const int32 dimension_id) const;

  const int32 TraverseTree(const int32 example_id,
                           const TTypes<float>::ConstMatrix* dense_data) const;

 protected:
  mutex mu_;
  protobuf::Arena arena_;
  boosted_trees::Tree* decision_tree_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_KERNELS_TENSOR_FOREST_RESOURCES_H_
