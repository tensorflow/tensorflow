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

// This file declares kernels for running TFRT ops/kernels via TF runtime
// fallback.

#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_RUNTIME_FALLBACK_KERNELS_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_RUNTIME_FALLBACK_KERNELS_H_

#include <memory>

#include "llvm/Support/Error.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/shared_context.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

// Injecting TF resources to TFRT GPU resource maps for tfrt::GpuOpHandler to
// use.s
Status InjectTfGpuResources();

// Create an EagerOperation to run the op, taking tensorflow::TensorHandle and
// returning tensorflow::AbstractTensorHandle*.
Status CallEagerExecute(const tfrt::ExecutionContext& exec_ctx,
                        EagerContext* eager_ctx, const char* op_name,
                        const char* device_name,
                        llvm::ArrayRef<TensorHandle*> input_tensor_handles,
                        const tfrt::OpAttrsRef& attrs,
                        llvm::MutableArrayRef<tensorflow::AbstractTensorHandle*>
                            result_tensor_handles);

// Take and return RuntimeFallbackTensors.
tfrt::AsyncValueRef<tfrt::Chain> RuntimeFallbackExecute(
    const tfrt::ExecutionContext& exec_ctx, const char* op_name,
    const char* device_name, tfrt::ArrayRef<tfrt::Tensor*> arguments,
    const tfrt::OpAttrsRef& attrs,
    tfrt::MutableArrayRef<tfrt::RCReference<tfrt::AsyncValue>> results);

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_RUNTIME_FALLBACK_KERNELS_H_
