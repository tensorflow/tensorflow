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

#include "tensorflow/c/eager/context_interface.h"

#include "tensorflow/c/eager/operation_interface.h"
#include "tensorflow/c/eager/tensor_handle_interface.h"
#include "tensorflow/core/framework/tensor_interface.h"
#include "tensorflow/core/platform/casts.h"

namespace tensorflow {

std::unique_ptr<AbstractTensorInterface> ContextInterface::CreateInt64Scalar(
    int64 value) {
  return std::make_unique<TensorInterface>(Tensor(value));
}

std::unique_ptr<AbstractTensorInterface> ContextInterface::CreateUint64Scalar(
    uint64 value) {
  return std::make_unique<TensorInterface>(Tensor(value));
}

std::unique_ptr<AbstractTensorInterface> ContextInterface::CreateInt32Scalar(
    int32 value) {
  return std::make_unique<TensorInterface>(Tensor(value));
}

std::unique_ptr<AbstractTensorInterface> ContextInterface::CreateFloatScalar(
    float value) {
  return std::make_unique<TensorInterface>(Tensor(value));
}

std::unique_ptr<AbstractTensorInterface> ContextInterface::CreateDoubleScalar(
    double value) {
  return std::make_unique<TensorInterface>(Tensor(value));
}

std::unique_ptr<AbstractTensorInterface> ContextInterface::CreateHalfScalar(
    Eigen::half value) {
  return std::make_unique<TensorInterface>(Tensor(value));
}

std::unique_ptr<AbstractTensorInterface> ContextInterface::CreateStringScalar(
    tstring value) {
  return std::make_unique<TensorInterface>(Tensor(value));
}

std::unique_ptr<AbstractTensorInterface>
ContextInterface::CreateComplex128Scalar(complex128 value) {
  return std::make_unique<TensorInterface>(Tensor(value));
}

std::unique_ptr<AbstractTensorInterface> ContextInterface::CreateBoolScalar(
    bool value) {
  return std::make_unique<TensorInterface>(Tensor(value));
}

std::unique_ptr<AbstractTensorInterface> ContextInterface::CreateInt64Tensor(
    absl::Span<const int64> dim_sizes) {
  return std::make_unique<TensorInterface>(
      Tensor(DT_INT64, TensorShape(dim_sizes)));
}

std::unique_ptr<AbstractTensorInterface> ContextInterface::CreateUint64Tensor(
    absl::Span<const int64> dim_sizes) {
  return std::make_unique<TensorInterface>(
      Tensor(DT_UINT64, TensorShape(dim_sizes)));
}

std::unique_ptr<AbstractTensorInterface> ContextInterface::CreateInt32Tensor(
    absl::Span<const int64> dim_sizes) {
  return std::make_unique<TensorInterface>(
      Tensor(DT_INT32, TensorShape(dim_sizes)));
}

std::unique_ptr<AbstractTensorInterface> ContextInterface::CreateFloatTensor(
    absl::Span<const int64> dim_sizes) {
  return std::make_unique<TensorInterface>(
      Tensor(DT_FLOAT, TensorShape(dim_sizes)));
}

std::unique_ptr<AbstractTensorInterface> ContextInterface::CreateDoubleTensor(
    absl::Span<const int64> dim_sizes) {
  return std::make_unique<TensorInterface>(
      Tensor(DT_DOUBLE, TensorShape(dim_sizes)));
}

std::unique_ptr<AbstractTensorInterface> ContextInterface::CreateHalfTensor(
    absl::Span<const int64> dim_sizes) {
  return std::make_unique<TensorInterface>(
      Tensor(DT_HALF, TensorShape(dim_sizes)));
}

std::unique_ptr<AbstractTensorInterface> ContextInterface::CreateStringTensor(
    absl::Span<const int64> dim_sizes) {
  return std::make_unique<TensorInterface>(
      Tensor(DT_STRING, TensorShape(dim_sizes)));
}

std::unique_ptr<AbstractTensorInterface>
ContextInterface::CreateComplex128Tensor(absl::Span<const int64> dim_sizes) {
  return std::make_unique<TensorInterface>(
      Tensor(DT_COMPLEX128, TensorShape(dim_sizes)));
}

std::unique_ptr<AbstractTensorInterface> ContextInterface::CreateBoolTensor(
    absl::Span<const int64> dim_sizes) {
  return std::make_unique<TensorInterface>(
      Tensor(DT_BOOL, TensorShape(dim_sizes)));
}

std::unique_ptr<AbstractTensorHandleInterface>
ContextInterface::CreateLocalHandle(
    const std::unique_ptr<AbstractTensorInterface> t) {
  Tensor tensor = tensorflow::down_cast<TensorInterface*>(t.get())->Tensor();
  return std::make_unique<TensorHandleInterface>(
      TensorHandle::CreateLocalHandle(std::move(tensor), /*d=*/ctx_->HostCPU(),
                                      /*op_device=*/nullptr, ctx_));
}

std::unique_ptr<AbstractOperationInterface>
ContextInterface::CreateOperation() {
  return std::make_unique<tensorflow::OperationInterface>(ctx_);
}

void ContextInterface::ListDevices(
    std::vector<tensorflow::DeviceAttributes>* devices) {
  ctx_->ListDevices(devices);
}

}  // namespace tensorflow
