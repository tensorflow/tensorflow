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

namespace tensorflow {

class TensorHandleInterface {
 public:
  explicit TensorHandleInterface(TensorHandle* h) : handle_(h) {}
  ~TensorHandleInterface();

  bool IsValid(Status* status) const;
  TF_DataType DataType() const;
  int NumDims(Status* status) const;
  int64_t NumElements(Status* status) const;
  int64_t Dim(int dim_index, Status* status) const;

  const char* DeviceName(Status* status) const;
  const char* BackingDeviceName(Status* status) const;
  TFE_TensorHandle* Copy();
  TF_Tensor* Resolve(Status* status);
  TFE_TensorDebugInfo* TensorDebugInfo(Status* status);

  // TODO(gjn): This is not a very generic interface, but is needed for specific
  // use cases.
  TensorHandle* Handle() { return handle_; }

 private:
  TensorHandle* handle_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_TENSOR_HANDLE_INTERFACE_H_
