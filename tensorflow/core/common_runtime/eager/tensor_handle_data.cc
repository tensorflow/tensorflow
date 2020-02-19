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
#include "tensorflow/core/common_runtime/eager/tensor_handle_data.h"

#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

class Status;

Status LocalTensorHandleData::Tensor(const tensorflow::Tensor** t) const {
  *t = &tensor_;

  return Status::OK();
}

Status LocalTensorHandleData::TensorValue(tensorflow::TensorValue* t) {
  tensorflow::Tensor& tensor = tensor_;
  *t = tensorflow::TensorValue(&tensor);

  return Status::OK();
}

Status LocalTensorHandleData::Shape(TensorShape* shape) const {
  *shape = tensor_.shape();

  return Status::OK();
}

Status LocalTensorHandleData::NumDims(int* num_dims) const {
  *num_dims = tensor_.dims();

  return Status::OK();
}

Status LocalTensorHandleData::Dim(int dim_index, int64* dim) const {
  *dim = tensor_.dim_size(dim_index);

  return Status::OK();
}

Status LocalTensorHandleData::NumElements(int64* num_elements) const {
  *num_elements = tensor_.NumElements();

  return Status::OK();
}

Status EmptyLocalTensorHandleData::Tensor(const tensorflow::Tensor** t) const {
  return errors::Unavailable(
      "Unable to get a tensor for an empty handle. "
      "Please wait until it is ready");
}

Status EmptyLocalTensorHandleData::TensorValue(tensorflow::TensorValue* t) {
  return errors::Unavailable(
      "Unable to get a tensor for an empty handle. "
      "Please wait until it is ready");
}

Status EmptyLocalTensorHandleData::Shape(TensorShape* shape) const {
  return errors::Unavailable(
      "Unable to get shape information for an empty handle. "
      "Please wait until it is ready");
}

Status EmptyLocalTensorHandleData::NumDims(int* num_dims) const {
  return errors::Unavailable(
      "Unable to get shape information for an empty handle. "
      "Please wait until it is ready");
}

Status EmptyLocalTensorHandleData::Dim(int dim_index, int64* dim) const {
  return errors::Unavailable(
      "Unable to get shape information for an empty handle. "
      "Please wait until it is ready");
}

Status EmptyLocalTensorHandleData::NumElements(int64* num_elements) const {
  return errors::Unavailable(
      "Unable to get shape information for an empty handle. "
      "Please wait until it is ready");
}

string EmptyLocalTensorHandleData::DebugString() const {
  return "EmptyLocalTensorHandleData";
}

}  // namespace tensorflow
