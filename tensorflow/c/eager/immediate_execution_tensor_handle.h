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
#ifndef TENSORFLOW_C_EAGER_IMMEDIATE_EXECUTION_TENSOR_HANDLE_H_
#define TENSORFLOW_C_EAGER_IMMEDIATE_EXECUTION_TENSOR_HANDLE_H_

#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// Abstract interface to a TensorHandle.
//
// A TensorHandle is management class around a Tensor which may track additional
// metadata and synchronization.
//
// This allows us to hide concrete implementations of TensorHandle from header
// files. The interface lists the common functionality that must be provided by
// any concrete implementation. However, in cases where the true concrete class
// is needed a static_cast can be applied.
class ImmediateExecutionTensorHandle : public AbstractTensorHandle {
 public:
  // Returns tensor dtype.
  virtual tensorflow::DataType DataType() const = 0;
  // Returns number of dimensions.
  virtual Status NumDims(int* num_dims) const = 0;
  // Returns number of elements across all dimensions.
  virtual Status NumElements(int64* num_elements) const = 0;
  // Returns size of specified dimension
  virtual Status Dim(int dim_index, int64* dim) const = 0;

  // Returns the device which created the handle.
  virtual const char* DeviceName(Status* status) const = 0;
  // Returns the device where the tensor was placed.
  virtual const char* BackingDeviceName(Status* status) const = 0;
  // Returns a tensor for the handle. If tensor is remote, it will be copied.
  virtual AbstractTensorInterface* Resolve(Status* status) = 0;

  // Return a copy of the handle.
  virtual ImmediateExecutionTensorHandle* Copy() = 0;

  // For LLVM style RTTI.
  static bool classof(const AbstractTensorHandle* ptr) {
    return ptr->getKind() == kEager || ptr->getKind() == kTfrt;
  }

 protected:
  explicit ImmediateExecutionTensorHandle(AbstractTensorHandleKind kind)
      : AbstractTensorHandle(kind) {}
  ~ImmediateExecutionTensorHandle() override {}
};

namespace internal {
struct ImmediateExecutionTensorHandleDeleter {
  void operator()(ImmediateExecutionTensorHandle* p) const {
    if (p != nullptr) {
      p->Release();
    }
  }
};
}  // namespace internal

using ImmediateTensorHandlePtr =
    std::unique_ptr<ImmediateExecutionTensorHandle,
                    internal::ImmediateExecutionTensorHandleDeleter>;

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_IMMEDIATE_EXECUTION_TENSOR_HANDLE_H_
