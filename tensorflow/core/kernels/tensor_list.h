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
#ifndef TENSORFLOW_CORE_KERNELS_TENSOR_LIST_H_
#define TENSORFLOW_CORE_KERNELS_TENSOR_LIST_H_

#include <utility>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow {

// Variant compatible type for a list of tensors. This is mutable but instances
// should never be mutated after stored in a variant tensor.
//
// **NOTE**: TensorList stores a refcounted container of tf::Tensor objects,
// which are accessible via TensorList::tensors().  Because it is refcounted,
// straight copies of the form:
//
//    TensorList b = a;
//    b.tensors().push_back(t);  // WARNING: This modifies a.tensors().
//
// Do not create a true copy of the underlying container - but instead increment
// a reference count.  Modifying b.tensors() modifies a.tensors().  In this way,
// TensorList should be considered similar to the tf::Tensor object.
//
// In order to get a copy of the underlying list, use the Copy method:
//
//    TensorList b = a.Copy();
//    b.tensors().push_back(t);  // This does not modify a.tensors().
//
// Note that this is not a deep copy: the memory locations of the underlying
// tensors will still point to the same locations of the corresponding tensors
// in the original.  To truly perform a deep copy, Device and Type-specific
// code needs to be applied to the underlying tensors as usual.
//
// The most important implication of RefCounted TLs is that OpKernels
// wishing to reuse TensorList inputs as outputs via context->forward_input()
// need to perform an additional check on the refcount of the TensorList,
// to ensure aliasing can be performed safely.  For example:
//
//     bool can_alias = false;
//     auto fw = c->forward_input(..., DT_VARIANT, {}, ...);
//     if (fw && fw->dtype() == DT_VARIANT && fw->NumElements() == 1) {
//       auto* tl = fw->scalar<Variant>()().get<TensorList>();
//       if (tl && tl->RefCountIsOne()) {
//         can_alias = true;
//       }
//     }
//
class TensorList {
 public:
  TensorList() : tensors_(new Tensors) {}
  ~TensorList();

  TensorList(const TensorList& other)
      : element_shape(other.element_shape),
        element_dtype(other.element_dtype),
        max_num_elements(other.max_num_elements),
        tensors_(other.tensors_) {
    tensors_->Ref();
  }

  TensorList(TensorList&& rhs)
      : element_shape(std::move(rhs.element_shape)),
        element_dtype(rhs.element_dtype),
        max_num_elements(rhs.max_num_elements),
        tensors_(rhs.tensors_) {
    rhs.tensors_ = nullptr;
  }

  TensorList& operator=(const TensorList& rhs) {
    if (this == &rhs) return *this;
    element_shape = rhs.element_shape;
    element_dtype = rhs.element_dtype;
    max_num_elements = rhs.max_num_elements;
    tensors_->Unref();
    tensors_ = rhs.tensors_;
    tensors_->Ref();
    return *this;
  }

  TensorList& operator=(TensorList&& rhs) {
    if (this == &rhs) return *this;
    element_shape = rhs.element_shape;
    element_dtype = rhs.element_dtype;
    max_num_elements = rhs.max_num_elements;
    std::swap(tensors_, rhs.tensors_);
    return *this;
  }

  static const char kTypeName[];

  string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const;

  bool Decode(const VariantTensorData& data);

  // TODO(apassos) fill this out
  string DebugString() const { return "TensorList"; }

  PartialTensorShape element_shape;

  DataType element_dtype;

  // The maximum allowed size of `tensors`. Defaults to -1 meaning that the size
  // of `tensors` is unbounded.
  int max_num_elements = -1;

  // Access to the underlying tensor container.
  std::vector<Tensor>& tensors() { return tensors_->values_; }
  const std::vector<Tensor>& tensors() const { return tensors_->values_; }

  // Get a new TensorList containing a copy of the underlying tensor container.
  TensorList Copy() const {
    TensorList out;
    out.element_shape = element_shape;
    out.element_dtype = element_dtype;
    out.max_num_elements = max_num_elements;
    // This performs a copy of the std::vector.
    out.tensors_->values_ = tensors_->values_;
    return out;
  }

  // Is this TensorList the only one with a reference to the underlying
  // container?
  bool RefCountIsOne() const { return tensors_->RefCountIsOne(); }

 private:
  class Tensors : public core::RefCounted {
   public:
    std::vector<Tensor> values_;
  };
  Tensors* tensors_;
};

#if defined(PLATFORM_GOOGLE)
// TODO(ebrevdo): Identify why Variant inline size is smaller on mobile devices.
static_assert(Variant::CanInlineType<TensorList>(),
              "Must be able to inline TensorList into a Variant");
#endif
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TENSOR_LIST_H_
