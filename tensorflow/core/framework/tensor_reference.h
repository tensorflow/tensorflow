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

  // Return an estimate of the total bytes being kept alive by this reference.
  size_t TotalBytes() const {
    // We add 128 as a baseline to account for per-Tensor metadata
    return 128 + (buf_ ? buf_->size() : 0);
  }

  void FillDescription(AllocationDescription* description) const {
    if (buf_) buf_->FillAllocationDescription(description);
  }

  // Convenience function for de-duplicating tensor references.
  bool SharesBufferWith(const TensorReference& t) const {
    return buf_ == t.buf_;
  }

  // Convenience function for de-duplicating tensor references.
  bool SharesBufferWith(const Tensor& t) const {
    return buf_ == (t.buf_ ? t.buf_->root_buffer() : nullptr);
  }

  // Convenience function for de-duplicating tensor references.
  size_t BufferHash() const { return std::hash<TensorBuffer*>()(buf_); }

  // A constructor used only for tests
  explicit TensorReference(TensorBuffer* test_buffer) : buf_(test_buffer) {
    if (buf_) buf_->Ref();
  }

 private:
  TensorBuffer* buf_;
};

typedef gtl::InlinedVector<TensorReference, 4> TensorReferenceVector;

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_TENSOR_REFERENCE_H_
