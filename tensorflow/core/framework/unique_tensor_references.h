/* Copyright 2016 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_FRAMEWORK_UNIQUE_TENSOR_REFERENCES_H_
#define TENSORFLOW_FRAMEWORK_UNIQUE_TENSOR_REFERENCES_H_

#include <unordered_set>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// Helper class to maintain a unique set of tensor references. In the
// common case there are not many references, so an inline vector is
// used for <= kInVector unique elements, defaulting to 4 since that
// is the inlined size of TensorReferenceVector. To avoid N^2
// operations when adding N items, any larger number of unique tensor
// references switches to using an unordered set.
class UniqueTensorReferences {
 public:
  UniqueTensorReferences() : frozen_(false), referenced_tensors_set_(nullptr) {}

  ~UniqueTensorReferences();

  // Adds a reference to tensor if its buffer is not already referenced.
  void Add(const Tensor& tensor);

  // No more references may be added after this is called. The unique
  // references are returning in out_vector.
  void FreezeAndReturnReferences(TensorReferenceVector* out_vector);

 private:
  // Up to kInVector elements are stored in reference_tensors_vector_
  // to avoid any allocations or hash computations in the common
  // case. When more unique elements are added they move to
  // referenced_tensors_set_ to avoid an N^2 algorithm on insert.
  static const int kInVector = 4;  // Must be >= 1.

  struct TensorReferenceEqualFn {
    bool operator()(const TensorReference& t1,
                    const TensorReference& t2) const {
      return t1.SharesBufferWith(t2);
    }
  };

  struct TensorReferenceHashFn {
    size_t operator()(const TensorReference& t) const { return t.BufferHash(); }
  };

  bool frozen_;
  TensorReferenceVector referenced_tensors_vector_;

  typedef std::unordered_set<TensorReference, TensorReferenceHashFn,
                             TensorReferenceEqualFn>
      ReferencedTensorsSet;
  // Lazily allocated hash set for when the number of tensors becomes too large.
  // If this is non-NULL, then we use the hash set, otherwise, we use the
  // referenced_tensors_vector_ (and do O(N^2) work per insertion).
  ReferencedTensorsSet* referenced_tensors_set_;

  TF_DISALLOW_COPY_AND_ASSIGN(UniqueTensorReferences);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_UNIQUE_TENSOR_REFERENCES_H_
