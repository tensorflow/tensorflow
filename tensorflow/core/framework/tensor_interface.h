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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_INTERFACE_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_INTERFACE_H_

#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/framework/tensor.h"

// Abstract interface to a Tensor.
//
// This allows us to hide concrete implementations of Tensor from header
// files. The interface lists the common functionality that must be provided by
// any concrete implementation. However, in cases where the true concrete class
// is needed a static_cast can be applied.
class AbstractTensorInterface {
 public:
  virtual ~AbstractTensorInterface() {}

  // Returns tensor dtype.
  virtual TF_DataType Type() const = 0;
  // Returns number of dimensions.
  virtual int NumDims() const = 0;
  // Returns size of specified dimension
  virtual int64_t Dim(int dim_index) const = 0;
  // Returns number of elements across all dimensions.
  virtual int64_t NumElements() const = 0;
  // Return size in bytes of the Tensor
  virtual size_t ByteSize() const = 0;
  // Returns a pointer to tensor data
  virtual void* Data() const = 0;

  // Returns if the tensor is aligned
  virtual bool IsAligned() const = 0;
  // Returns if their is sole ownership of this Tensor and thus it can be moved.
  virtual bool CanMove() const = 0;
};

namespace tensorflow {

class TensorInterface : public AbstractTensorInterface {
 public:
  TensorInterface() {}
  explicit TensorInterface(Tensor t) : tensor_(std::move(t)) {}
  ~TensorInterface() override {}

  TF_DataType Type() const override;
  int NumDims() const override;
  int64_t Dim(int dim_index) const override;
  int64_t NumElements() const override;
  size_t ByteSize() const override;
  void* Data() const override;
  bool IsAligned() const override;
  bool CanMove() const override;

  Status ToTensor(Tensor* dst) const;
  Status BitcastFrom(const TensorInterface& from, TF_DataType type,
                     const int64_t* new_dims, int num_new_dims);

 private:
  Tensor tensor_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_INTERFACE_H_
