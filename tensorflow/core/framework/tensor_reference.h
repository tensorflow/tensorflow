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

#ifndef TENSORFLOW_FRAMEWORK_TENSOR_REFERENCE_H_
#define TENSORFLOW_FRAMEWORK_TENSOR_REFERENCE_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {

// An opaque class that holds a reference to an underlying TensorBuffer.
// Unlike Tensor, it does not have any shape or type information, so
// it is cheaper to construct/move, but the only thing you can really do
// with it is Unref it, which releases one of the references to the underlying
// TensorBuffer.
// IMPORTANT: If you do not call Unref(), you will likely leak tensor memory.
class TensorReference {
 public:
  // Take the reference of the root buffer so the size will be more accurate
  explicit TensorReference(const Tensor& tensor)
      : buf_(tensor.buf_ ? tensor.buf_->root_buffer() : nullptr) {
    if (buf_) buf_->Ref();
  }

  ~TensorReference() {}

  void Unref() const {
    if (buf_) buf_->Unref();
  }

  void FillDescription(AllocationDescription* description) const {
    if (buf_) buf_->FillAllocationDescription(description);
  }

 private:
  TensorBuffer* buf_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_TENSOR_REFERENCE_H_
