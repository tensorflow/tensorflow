/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file declares TF runtime fallback tensor.

#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_RUNTIME_FALLBACK_TENSOR_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_RUNTIME_FALLBACK_TENSOR_H_

#include "llvm/ADT/STLExtras.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

class RuntimeFallbackTensor final
    : public tfrt::Tensor,
      public tfrt::TensorTraits<RuntimeFallbackTensor> {
 public:
  explicit RuntimeFallbackTensor(const tfrt::TensorShape& shape,
                                 tfrt::DType dtype, OwnedTensorHandle th);

  void Print(tfrt::raw_ostream& os) const override;

  // Note that this method does not add ref to the return tensor_handle.
  TensorHandle* GetTensorHandle() const { return tensor_handle_.get(); }

  // Tensor type name for RuntimeFallbackTensor.
  static const char* name() { return "RuntimeFallback"; }

 private:
  template <typename T>
  static void PrintTensorValues(void* data, ssize_t size,
                                llvm::raw_ostream& os) {
    llvm::ArrayRef<T> elements =
        llvm::makeArrayRef(static_cast<T*>(data), size);
    llvm::interleaveComma(elements, os);
  }

  OwnedTensorHandle tensor_handle_;
};

llvm::SmallVector<tfrt::Index, 4> GetShape(
    AbstractTensorInterface* tensor_interface);

tfrt::Expected<tfrt::StringHostTensor> CopyTfStringTensorToStringHostTensor(
    AbstractTensorInterface* tensor_interface, tfrt::HostContext* host);

tfrt::Expected<RuntimeFallbackTensor>
CreateRuntimeFallbackTensorFromTfTensorHandle(OwnedTensorHandle owned_th,
                                              tfrt::HostContext* host);

RuntimeFallbackTensor MoveDHTToRuntimeFallbackTensor(
    tfrt::DenseHostTensor&& dht, tfrt::HostContext* host);

RuntimeFallbackTensor CopyRefDHTToRuntimeFallbackTensor(
    const tfrt::DenseHostTensor& dht, tfrt::HostContext* host);

RuntimeFallbackTensor CopySHTToRuntimeFallbackTensor(
    const tfrt::StringHostTensor& sht, tfrt::HostContext* host);

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_RUNTIME_FALLBACK_TENSOR_H_
