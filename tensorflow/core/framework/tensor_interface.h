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

// Internal structures used by the C API. These are likely to change and should
// not be depended on.

namespace tensorflow {

class TensorInterface {
 public:
  TensorInterface() {}
  explicit TensorInterface(Tensor t) : tensor_(std::move(t)) {}

  TF_DataType Type() const;
  int NumDims() const;
  int64_t Dim(int dim_index) const;
  int64_t NumElements() const;
  size_t ByteSize() const;
  void* Data() const;
  bool IsAligned() const;

  Status ToTensor(Tensor* dst) const;
  bool CopyFrom(const Tensor& other, const TensorShape& shape);
  Status BitcastFrom(const TensorInterface& from, TF_DataType type,
                     const int64_t* new_dims, int num_new_dims);

  bool CanMove() const;

 private:
  Tensor tensor_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_INTERFACE_H_
