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

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/cc/experimental/base/public/status.h"

namespace tensorflow {
namespace experimental {
namespace cc {

// Tensor represents an n-dimensional array of values.
class Tensor {
 public:
  using DeleterCallback = std::function<void(void*, size_t)>;

  // Constructs a Tensor from user provided buffer.
  //
  // Params:
  //  dtype - The dtype of the tensor's data.
  //  shape - A shape vector, where each element corresponds to the size of
  //          the tensor's corresponding dimension.
  //  data - Pointer to a buffer of memory to construct a Tensor out of.
  //  len - The length (in bytes) of `data`
  //  deleter - A std::function to be called when the Tensor no longer needs the
  //            memory in `data`. This can be used to free `data`, or
  //            perhaps decrement a refcount associated with `data`, etc.
  //  status - Set to OK on success and an error on failure.
  // Returns:
  // If an error occurred, status->ok() will be false, and the returned
  // Tensor must not be used.
  // TODO(bmzhao): Add Runtime as an argument to this function so we can swap to
  // a TFRT backed tensor.
  // TODO(bmzhao): Add benchmarks on overhead for this function; we can
  // consider using int64_t* + length rather than vector.
  static Tensor FromBuffer(TF_DataType dtype, const std::vector<int64_t>& shape,
                           void* data, size_t len, DeleterCallback deleter,
                           Status* status);

  // TODO(bmzhao): In the case we construct a tensor from non-owned memory,
  // we should offer a way to deep copy the tensor into a new tensor, which
  // owns the underlying memory. This could be a .deepcopy()/clone() method.

  // TODO(bmzhao): In the future, we want to relax the non-copyability
  // constraint. To do so, we can add a C API function that acts like
  // CopyFrom:
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

  struct DeleterStruct {
    std::function<void(void*, size_t)> deleter;
  };

  static void DeleterFunction(void* memory, size_t len, void* deleter_struct) {
    DeleterStruct* deleter = reinterpret_cast<DeleterStruct*>(deleter_struct);
    deleter->deleter(memory, len);
    delete deleter;
  }

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

inline Tensor Tensor::FromBuffer(TF_DataType dtype,
                                 const std::vector<int64_t>& shape, void* data,
                                 size_t len, DeleterCallback deleter,
                                 Status* status) {
  // Credit to apassos@ for this technique:
  // Despite the fact that our API takes a std::function deleter, we are able
  // to maintain ABI stability because:
  // 1. Only a function pointer is sent across the C API (&DeleterFunction)
  // 2. DeleterFunction is defined in the same build artifact that constructed
  //    the std::function (so there isn't confusion about std::function ABI).
  // Note that 2. is satisifed by the fact that this is a header-only API, where
  // the function implementations are inline.

  DeleterStruct* deleter_struct = new DeleterStruct{deleter};
  TF_Tensor* tensor = TF_NewTensor(dtype, shape.data(), shape.size(), data, len,
                                   &DeleterFunction, deleter_struct);
  if (tensor == nullptr) {
    status->SetStatus(TF_INVALID_ARGUMENT,
                      "Failed to create tensor for input buffer");
    return Tensor(nullptr);
  }
  return Tensor(tensor);
}

}  // namespace cc
}  // namespace experimental
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_TENSOR_H_
