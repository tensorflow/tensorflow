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
#ifndef TENSORFLOW_C_EAGER_TENSOR_HANDLE_INTERFACE_H_
#define TENSORFLOW_C_EAGER_TENSOR_HANDLE_INTERFACE_H_

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"

// Abstract interface to a TensorHandle.
//
// A TensorHandle is management class around a Tensor which may track additional
// metadata and synchronization.
//
// This allows us to hide concrete implementations of TensorHandle from header
// files. The interface lists the common functionality that must be provided by
// any concrete implementation. However, in cases where the true concrete class
// is needed a static_cast can be applied.
class AbstractTensorHandleInterface {
 public:
  virtual ~AbstractTensorHandleInterface() {}

  // Check if the handle is in a valid initialized state.
  virtual bool IsValid(tensorflow::Status* status) const = 0;
  // Returns tensor dtype.
  virtual TF_DataType DataType() const = 0;
  // Returns number of dimensions.
  virtual int NumDims(tensorflow::Status* status) const = 0;
  // Returns number of elements across all dimensions.
  virtual int64_t NumElements(tensorflow::Status* status) const = 0;
  // Returns size of specified dimension
  virtual int64_t Dim(int dim_index, tensorflow::Status* status) const = 0;

  // Returns the device which created the handle.
  virtual const char* DeviceName(tensorflow::Status* status) const = 0;
  // Returns the device where the tensor was placed.
  virtual const char* BackingDeviceName(tensorflow::Status* status) const = 0;
  // Returns a tensor for the handle. If tensor is remote, it will be copied.
  virtual TF_Tensor* Resolve(tensorflow::Status* status) = 0;
  // Returns debug information about the tensor.
  virtual TFE_TensorDebugInfo* TensorDebugInfo(tensorflow::Status* status) = 0;

  // Return a copy of the handle.
  virtual AbstractTensorHandleInterface* Copy() = 0;

  // Maintain mirror tensors for any implicit copies to local devices. This
  // setting is offered on a per tensor handle basis to avoid potential memory
  // over utilization due to holding on to mirrors as well as the original
  // tensor. Note this setting overrides the context mirroring policy whereby if
  // the mirroring policy is MIRRORING_NONE, we will still continue to mirror
  // this tensor.
  virtual void EnableImplicitMirroring() = 0;
};

namespace tensorflow {

class TensorHandleInterface : public AbstractTensorHandleInterface {
 public:
  explicit TensorHandleInterface(TensorHandle* h) : handle_(h) {}
  ~TensorHandleInterface() override;

  bool IsValid(Status* status) const override;
  TF_DataType DataType() const override;
  int NumDims(Status* status) const override;
  int64_t NumElements(Status* status) const override;
  int64_t Dim(int dim_index, Status* status) const override;

  const char* DeviceName(Status* status) const override;
  const char* BackingDeviceName(Status* status) const override;
  TF_Tensor* Resolve(Status* status) override;
  TFE_TensorDebugInfo* TensorDebugInfo(Status* status) override;

  AbstractTensorHandleInterface* Copy() override;

  void EnableImplicitMirroring() override;

  // TODO(gjn): This is not a very generic interface, but is needed for specific
  // use cases.
  TensorHandle* Handle() { return handle_; }

 private:
  TensorHandle* handle_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_TENSOR_HANDLE_INTERFACE_H_
