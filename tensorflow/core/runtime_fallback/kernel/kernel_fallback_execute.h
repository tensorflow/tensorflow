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

// Provides a way to execute a TensorFlow kernel using TFRT kernel fallback.

#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_EXECUTE_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_EXECUTE_H_

#include "llvm/ADT/ArrayRef.h"
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace tfrt {
class AsyncKernelFrame;
}  // namespace tfrt

namespace tensorflow {
namespace tfd {

enum KernelFallbackOutputType {
  TENSOR = 0,                 // Output type is tensorflow::Tensor
  KERNEL_FALLBACK_TENSOR = 1  // Output type is KernelFallbackTensor
};

// Runs kernel asynchronously.
// `frame` must contain tensorflow::Tensor inputs and pre-allocated
// tensorflow::Tensor or tfrt::KernelFallbackTensor outputs.
bool KernelFallbackExecute(
    const tfrt::ExecutionContext& exec_ctx, tfrt::string_view op_name,
    llvm::ArrayRef<tfrt::AsyncValue*> arguments,
    llvm::MutableArrayRef<tfrt::RCReference<tfrt::AsyncValue>> results,
    const tfrt::OpAttrsRef& attrs, KernelFallbackOutputType output_type);
}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_EXECUTE_H_
