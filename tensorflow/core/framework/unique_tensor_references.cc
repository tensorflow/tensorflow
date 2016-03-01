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

#include "tensorflow/core/framework/unique_tensor_references.h"

namespace tensorflow {

UniqueTensorReferences::~UniqueTensorReferences() {
  if (!frozen_) {
    // The references were not retrieved so discard them to avoid
    // leaking memory.
    TensorReferenceVector refs;
    FreezeAndReturnReferences(&refs);
    for (auto& tensor : refs) {
      tensor.Unref();
    }
  }
  delete referenced_tensors_set_;
}

void UniqueTensorReferences::Add(const Tensor& tensor) {
  DCHECK(!frozen_);
  // Do nothing if the tensor has a null buffer.
  if (tensor.IsInitialized()) {
    if (referenced_tensors_set_ != nullptr) {
      // There are enough tensors that we are using a hash set to
      // de-duplicate.
      const TensorReference tensor_ref(tensor);
      if (!referenced_tensors_set_->insert(tensor_ref).second) {
        // The tensor was a duplicate, so discard the reference.
        tensor_ref.Unref();
      }
    } else {
      for (size_t i = 0; i < referenced_tensors_vector_.size(); ++i) {
        if (referenced_tensors_vector_[i].SharesBufferWith(tensor)) {
          // tensor is a duplicate, so nothing to do.
          return;
        }
      }
      referenced_tensors_vector_.push_back(TensorReference(tensor));
      if (kInVector == referenced_tensors_vector_.size()) {
        // There are too many tensors to keep using the N^2 algorithm
        // so start de-duplicating using a set.
        // Transfer the refs from the vector to the set.
        DCHECK(referenced_tensors_set_ == nullptr);
        referenced_tensors_set_ = new ReferencedTensorsSet;
        referenced_tensors_set_->reserve(kInVector);
        referenced_tensors_set_->insert(referenced_tensors_vector_.begin(),
                                        referenced_tensors_vector_.end());
        DCHECK_EQ(kInVector, referenced_tensors_set_->size());
        referenced_tensors_vector_.clear();
      }
    }
  }
}

void UniqueTensorReferences::FreezeAndReturnReferences(
    TensorReferenceVector* out_vector) {
  // Prevent any further additions.
  frozen_ = true;
  if (referenced_tensors_set_ != nullptr) {
    DCHECK(referenced_tensors_vector_.empty());
    out_vector->reserve(referenced_tensors_set_->size());
    for (const auto& ref : *referenced_tensors_set_) {
      out_vector->push_back(ref);
    }
    referenced_tensors_set_->clear();
    delete referenced_tensors_set_;
    referenced_tensors_set_ = nullptr;
  } else {
    out_vector->reserve(referenced_tensors_vector_.size());
    for (const auto& ref : referenced_tensors_vector_) {
      out_vector->push_back(ref);
    }
    referenced_tensors_vector_.clear();
  }
}

}  // namespace tensorflow
