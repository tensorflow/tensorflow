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

#ifndef TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_TENSORHANDLE_H_
#define TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_TENSORHANDLE_H_

#include <memory>
#include <vector>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/cc/experimental/base/public/runtime.h"
#include "tensorflow/cc/experimental/base/public/status.h"
#include "tensorflow/cc/experimental/base/public/tensor.h"

namespace tensorflow {
namespace experimental {
namespace cc {

// An opaque representation of a tensor computed/managed by the Tensorflow
// runtime (tensorflow:cc::Runtime). Unlike a tensor, a Tensorhandle may refer
// to tensors placed in memory of different devices or remote address spaces.
// Note that tensorflow::cc::Runtime MUST outlive all TensorHandles created
// from it.
class TensorHandle {
 public:
  // Unwraps a Tensor from the given TensorHandle. If an error occurred,
  // status->ok() will be false, and the returned Tensor must not be used.
  Tensor Resolve(Status* status);

  // Constructs a TensorHandle from a Tensor. If an error occurred,
  // status->ok() will be false, and the returned TensorHandle must not be used.
  static TensorHandle FromTensor(const Tensor& tensor, const Runtime& runtime,
                                 Status* status);

  // TensorHandle is movable, and not copyable
  TensorHandle(TensorHandle&&) = default;
  TensorHandle& operator=(TensorHandle&&) = default;

 private:
  // Wraps a TFE_TensorHandle. Takes ownership of handle.
  explicit TensorHandle(TFE_TensorHandle* handle) : handle_(handle) {}

  // TensorHandle is not copyable
  TensorHandle(const TensorHandle&) = delete;
  TensorHandle& operator=(const TensorHandle&) = delete;

  // Returns the underlying TFE_TensorHandle that this object wraps.
  // This object retains ownership of the pointer.
  TFE_TensorHandle* GetTFETensorHandle() const { return handle_.get(); }

  // Deletes the currently wrapped TFE_TensorHandle, and swaps it with handle,
  // and takes ownership of handle.
  void Reset(TFE_TensorHandle* handle) { handle_.reset(handle); }

  struct TFETensorHandleDeleter {
    void operator()(TFE_TensorHandle* p) const { TFE_DeleteTensorHandle(p); }
  };
  std::unique_ptr<TFE_TensorHandle, TFETensorHandleDeleter> handle_;
};

inline Tensor TensorHandle::Resolve(Status* status) {
  TF_Tensor* tensor =
      TFE_TensorHandleResolve(handle_.get(), status->GetTFStatus());
  if (!status->ok()) {
    return Tensor(nullptr);
  }
  return Tensor(tensor);
}

inline TensorHandle TensorHandle::FromTensor(const Tensor& tensor,
                                             const Runtime& runtime,
                                             Status* status) {
  TFE_TensorHandle* tensor_handle = TFE_NewTensorHandleFromTensor(
      runtime.GetTFEContext(), tensor.GetTFTensor(), status->GetTFStatus());
  if (!status->ok()) {
    return TensorHandle(nullptr);
  }
  return TensorHandle(tensor_handle);
}

}  // namespace cc
}  // namespace experimental
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_TENSORHANDLE_H_
