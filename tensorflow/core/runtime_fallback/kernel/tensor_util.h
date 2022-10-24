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
#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_TENSOR_UTIL_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_TENSOR_UTIL_H_

#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime

namespace tfrt {
class Device;
class ExecutionContext;
}  // namespace tfrt

namespace tensorflow {
class KernelFallbackTensor;
class Device;
namespace tfd {

tfrt::AsyncValueRef<KernelFallbackTensor> TransferTensorToDevice(
    const tfrt::ExecutionContext& exec_ctx, const KernelFallbackTensor& tensor,
    const tfrt::Device& src_device, const tfrt::Device& dst_device);

llvm::Expected<Device*> GetTfDevice(const tfrt::ExecutionContext& exec_ctx,
                                    const tfrt::Device& device);

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_TENSOR_UTIL_H_
