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
#ifndef TENSORFLOW_C_EAGER_CONTEXT_INTERFACE_H_
#define TENSORFLOW_C_EAGER_CONTEXT_INTERFACE_H_

#include <memory>

#include "tensorflow/c/eager/operation_interface.h"
#include "tensorflow/c/eager/tensor_handle_interface.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_interface.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/tstring.h"

namespace tensorflow {

// Abstract interface to a context.
//
// A context is responsible for creating key objects such as Tensors,
// TensorHandles & Operations.
class AbstractContextInterface {
 public:
  virtual ~AbstractContextInterface() {}

  // Scalar creation functions
  virtual std::unique_ptr<AbstractTensorInterface> CreateInt64Scalar(
      int64 value) = 0;
  virtual std::unique_ptr<AbstractTensorInterface> CreateUint64Scalar(
      uint64 value) = 0;
  virtual std::unique_ptr<AbstractTensorInterface> CreateInt32Scalar(
      int32 value) = 0;
  virtual std::unique_ptr<AbstractTensorInterface> CreateFloatScalar(
      float value) = 0;
  virtual std::unique_ptr<AbstractTensorInterface> CreateDoubleScalar(
      double value) = 0;
  virtual std::unique_ptr<AbstractTensorInterface> CreateHalfScalar(
      Eigen::half value) = 0;
  virtual std::unique_ptr<AbstractTensorInterface> CreateStringScalar(
      tstring value) = 0;
  virtual std::unique_ptr<AbstractTensorInterface> CreateComplex128Scalar(
      complex128 value) = 0;
  virtual std::unique_ptr<AbstractTensorInterface> CreateBoolScalar(
      bool value) = 0;

  // Tensor creation functions
  virtual std::unique_ptr<AbstractTensorInterface> CreateInt64Tensor(
      absl::Span<const int64> dim_sizes) = 0;
  virtual std::unique_ptr<AbstractTensorInterface> CreateUint64Tensor(
      absl::Span<const int64> dim_sizes) = 0;
  virtual std::unique_ptr<AbstractTensorInterface> CreateInt32Tensor(
      absl::Span<const int64> dim_sizes) = 0;
  virtual std::unique_ptr<AbstractTensorInterface> CreateFloatTensor(
      absl::Span<const int64> dim_sizes) = 0;
  virtual std::unique_ptr<AbstractTensorInterface> CreateDoubleTensor(
      absl::Span<const int64> dim_sizes) = 0;
  virtual std::unique_ptr<AbstractTensorInterface> CreateHalfTensor(
      absl::Span<const int64> dim_sizes) = 0;
  virtual std::unique_ptr<AbstractTensorInterface> CreateStringTensor(
      absl::Span<const int64> dim_sizes) = 0;
  virtual std::unique_ptr<AbstractTensorInterface> CreateComplex128Tensor(
      absl::Span<const int64> dim_sizes) = 0;
  virtual std::unique_ptr<AbstractTensorInterface> CreateBoolTensor(
      absl::Span<const int64> dim_sizes) = 0;

  // Create a handle to wrap and manage a Tensor
  virtual std::unique_ptr<AbstractTensorHandleInterface> CreateLocalHandle(
      const std::unique_ptr<AbstractTensorInterface> t) = 0;

  // Create an operation to perform op execution
  virtual std::unique_ptr<AbstractOperationInterface> CreateOperation() = 0;

  // List attributes of available devices
  virtual void ListDevices(std::vector<DeviceAttributes>* devices) = 0;
};

// TODO(gjn): Try to move these all to EagerContext and make it implement
// AbstractContextInterface. Currently, this is not so straightforward because
// of various BUILD file dependencies.
class ContextInterface : public AbstractContextInterface {
 public:
  explicit ContextInterface(EagerContext* ctx) : ctx_(ctx) {}
  ~ContextInterface() override {}

  std::unique_ptr<AbstractTensorInterface> CreateInt64Scalar(
      int64 value) override;
  std::unique_ptr<AbstractTensorInterface> CreateUint64Scalar(
      uint64 value) override;
  std::unique_ptr<AbstractTensorInterface> CreateInt32Scalar(
      int32 value) override;
  std::unique_ptr<AbstractTensorInterface> CreateFloatScalar(
      float value) override;
  std::unique_ptr<AbstractTensorInterface> CreateDoubleScalar(
      double value) override;
  std::unique_ptr<AbstractTensorInterface> CreateHalfScalar(
      Eigen::half value) override;
  std::unique_ptr<AbstractTensorInterface> CreateStringScalar(
      tensorflow::tstring value) override;
  std::unique_ptr<AbstractTensorInterface> CreateComplex128Scalar(
      tensorflow::complex128 value) override;
  std::unique_ptr<AbstractTensorInterface> CreateBoolScalar(
      bool value) override;

  std::unique_ptr<AbstractTensorInterface> CreateInt64Tensor(
      absl::Span<const int64> dim_sizes) override;
  std::unique_ptr<AbstractTensorInterface> CreateUint64Tensor(
      absl::Span<const int64> dim_sizes) override;
  std::unique_ptr<AbstractTensorInterface> CreateInt32Tensor(
      absl::Span<const int64> dim_sizes) override;
  std::unique_ptr<AbstractTensorInterface> CreateFloatTensor(
      absl::Span<const int64> dim_sizes) override;
  std::unique_ptr<AbstractTensorInterface> CreateDoubleTensor(
      absl::Span<const int64> dim_sizes) override;
  std::unique_ptr<AbstractTensorInterface> CreateHalfTensor(
      absl::Span<const int64> dim_sizes) override;
  std::unique_ptr<AbstractTensorInterface> CreateStringTensor(
      absl::Span<const int64> dim_sizes) override;
  std::unique_ptr<AbstractTensorInterface> CreateComplex128Tensor(
      absl::Span<const int64> dim_sizes) override;
  std::unique_ptr<AbstractTensorInterface> CreateBoolTensor(
      absl::Span<const int64> dim_sizes) override;

  std::unique_ptr<AbstractTensorHandleInterface> CreateLocalHandle(
      const std::unique_ptr<AbstractTensorInterface> t) override;
  std::unique_ptr<AbstractOperationInterface> CreateOperation() override;

  void ListDevices(std::vector<DeviceAttributes>* devices) override;

  // For runtime specific APIs, provide ability to get the underlying context.
  EagerContext* Context() const { return ctx_; }

 private:
  EagerContext* ctx_;
};

inline EagerContext* ContextFromInterface(
    const std::unique_ptr<AbstractContextInterface>& context) {
  return down_cast<ContextInterface*>(context.get())->Context();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_CONTEXT_INTERFACE_H_
