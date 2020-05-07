/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_TENSOR_H_
#define TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_TENSOR_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>

#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_tensor.h"

namespace tensorflow {
namespace cc {

// Tensor represents an n-dimensional array of values.
class Tensor {
 public:
  // TODO(bmzhao): Add a factory function that constructs a Tensor from a char
  // buffer, with an options struct (to specify the buffer's layout, device?,
  // whether to create a TFRT or TF tensor, whether we should take ownership of
  // the memory, etc). This requires extending TF_NewTensor with an options
  // struct:
  // https://github.com/tensorflow/tensorflow/blob/3c520614a3c056d56afdc79b59979b9b0087f8b9/tensorflow/c/tf_tensor.h#L77-L80

  // TODO(bmzhao): In the case we construct a tensor from non-owned memory,
  // we should offer a way to deep copy the tensor into a new tensor, which
  // owns the underlying memory. This could be a .deepcopy()/clone() method.

  // TODO(bmzhao): In the future, we want to relax the non-copyability
  // constraint. To do so, we can add a C API function that acts like CopyFrom:
  // https://github.com/tensorflow/tensorflow/blob/08931c1e3e9eb2e26230502d678408e66730826c/tensorflow/core/framework/tensor.h#L301-L311

  // Tensor is movable, but not copyable
  Tensor(Tensor&&) = default;
  Tensor& operator=(Tensor&&) = default;

  // Returns the number of dimensions in the tensor. Can be -1, which represents
  // unknown rank.
  int dims() const;

  // Returns the number of elements in in demension `d`.
  // REQUIRES: `0 <= d < dims()`
  int64_t dim_size(int d) const;

  // Returns a pointer to the underlying data buffer.
  void* data() const;

  // Returns the data type of the tensor.
  TF_DataType dtype() const;

  // Returns the number of elements in the tensor. For a tensor with a partially
  // defined shape, -1 means not fully defined.
  int64_t num_elements() const;

  // Returns the size of the underlying data in bytes.
  size_t num_bytes() const;

 private:
  friend class TensorHandle;
  friend class Runtime;

  // Wraps a TF_Tensor. Takes ownership of handle.
  explicit Tensor(TF_Tensor* tensor) : tensor_(tensor) {}

  // Tensor is not copyable
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  // Returns the underlying TF_Tensor that this object wraps.
  // This object retains ownership of the pointer.
  TF_Tensor* GetTFTensor() const { return tensor_.get(); }

  struct TFTensorDeleter {
    void operator()(TF_Tensor* p) const { TF_DeleteTensor(p); }
  };
  std::unique_ptr<TF_Tensor, TFTensorDeleter> tensor_;
};

inline void* Tensor::data() const { return TF_TensorData(tensor_.get()); }

inline int Tensor::dims() const { return TF_NumDims(tensor_.get()); }

inline int64_t Tensor::dim_size(int d) const {
  return TF_Dim(tensor_.get(), d);
}

inline TF_DataType Tensor::dtype() const {
  return TF_TensorType(tensor_.get());
}

inline int64_t Tensor::num_elements() const {
  return TF_TensorElementCount(tensor_.get());
}

inline size_t Tensor::num_bytes() const {
  return TF_TensorByteSize(tensor_.get());
}

}  // namespace cc
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_TENSOR_H_
