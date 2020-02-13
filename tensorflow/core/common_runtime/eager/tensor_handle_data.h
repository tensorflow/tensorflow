/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_DATA_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_DATA_H_

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class TensorHandleData {
 public:
  virtual ~TensorHandleData() {}

  // Different tensor handles support a set of these calls. In some cases these
  // are resolved with a Tensor or TensorShape. Typically if the handle is not
  // ready, none of these are supported operations.
  virtual Status Tensor(const tensorflow::Tensor** t) const = 0;
  virtual Status TensorValue(tensorflow::TensorValue* t) = 0;
  virtual Status Shape(TensorShape* shape) const = 0;
  virtual Status NumDims(int* num_dims) const = 0;
  virtual Status Dim(int dim_index, int64* dim) const = 0;
  virtual Status NumElements(int64* num_elements) const = 0;

  virtual string DebugString() const = 0;
};

// Local Tensor Handle: Handle to a Tensor present on the local host.
class LocalTensorHandleData : public TensorHandleData {
 public:
  explicit LocalTensorHandleData(const tensorflow::Tensor& t) : tensor_(t) {}
  ~LocalTensorHandleData() override {}

  // A local tensor handle should be able to satisfy all of these requests.
  Status Tensor(const tensorflow::Tensor** t) const override;
  Status TensorValue(tensorflow::TensorValue* t) override;
  Status Shape(TensorShape* shape) const override;
  Status NumDims(int* num_dims) const override;
  Status Dim(int dim_index, int64* dim) const override;
  Status NumElements(int64* num_elements) const override;

  string DebugString() const override { return tensor_.DebugString(); }

 private:
  tensorflow::Tensor tensor_;
};

// Empty Local Tensor Handle: Once the execution is complete this is replaced by
// a local tensor handle.
class EmptyLocalTensorHandleData : public TensorHandleData {
 public:
  EmptyLocalTensorHandleData() {}
  ~EmptyLocalTensorHandleData() override {}

  // Empty tensor handles are not ready and hence cannot satisfy any of these
  // requests.
  Status Tensor(const tensorflow::Tensor** t) const override;
  Status TensorValue(tensorflow::TensorValue* t) override;
  Status Shape(TensorShape* shape) const override;
  Status NumDims(int* num_dims) const override;
  Status Dim(int dim_index, int64* dim) const override;
  Status NumElements(int64* num_elements) const override;

  string DebugString() const override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_DATA_H_
