/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_UTILS_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_UTILS_H_

#include "llvm/ADT/ArrayRef.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/sync_kernel_utils.h"  // from @tf_runtime
#include "tfrt/host_context/value.h"  // from @tf_runtime
#include "tfrt/support/variant.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

using TfInputs =
    tfrt::Variant<tfrt::Monostate, llvm::ArrayRef<tfrt::AsyncValue*>,
                  tfrt::RepeatedSyncArguments<tfrt_stub::FallbackTensor>&>;

// Sets up the OpKernelcontext::Params in `run_state` with the objects and data
// in `runner`, `fallback_request_state` and `device`.
void SetUpParams(const tensorflow::tfrt_stub::OpKernelRunner& runner,
                 const KernelFallbackCompatRequestState& fallback_request_state,
                 tensorflow::Device* device,
                 tensorflow::tfrt_stub::OpKernelRunState& run_state);

// Return the device to be used for the fallback kernel execution. The device is
// guaranteed to be alive during the graph execution.
tensorflow::Device* GetDeviceFromFallbackState(
    const KernelFallbackCompatRequestState& fallback_request_state,
    const tfrt_stub::OpKernelRunner& kernel_runner);

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_UTILS_H_
