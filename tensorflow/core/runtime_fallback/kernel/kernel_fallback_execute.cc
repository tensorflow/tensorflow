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
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute.h"

#include <assert.h>

#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.h"
#include "tfrt/common/compat/eigen/thread_pool_device.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

namespace {
using tensorflow::KernelFallbackTensor;
using tfrt::AsyncValue;
using tfrt::RCReference;
}  // namespace

void SetError(const tfrt::ExecutionContext& exec_ctx,
              llvm::SmallVector<RCReference<AsyncValue>, 4>* results,
              tfrt::string_view message) {
  RCReference<tfrt::ErrorAsyncValue> error = EmitErrorAsync(exec_ctx, message);
  for (auto& result : *results) {
    result->SetError(error->GetError());
  }
}

bool KernelFallbackExecute(
    const tfrt::ExecutionContext& exec_ctx, tfrt::string_view op_name,
    tfrt::ArrayRef<AsyncValue*> arguments,
    tfrt::MutableArrayRef<RCReference<AsyncValue>> results,
    const tfrt::OpAttrsRef& attrs, KernelFallbackOutputType output_type) {
  // Remove tf. prefix.
  op_name.consume_front("tf.");
  std::string op_name_str = op_name.str();

  llvm::SmallVector<RCReference<AsyncValue>, 4> inputs;
  inputs.reserve(arguments.size());
  for (AsyncValue* input : arguments) {
    inputs.push_back(FormRef(input));
  }
  llvm::SmallVector<RCReference<AsyncValue>, 4> outputs(results.begin(),
                                                        results.end());

  // Always run TFRTOpKernel::Compute on the blocking thread pool to
  // avoid deadlock. Many TF kernels block until their intra-op closures
  // complete.
  bool work_enqueued = EnqueueBlockingWork(
      exec_ctx.host(),
      [exec_ctx, inputs = std::move(inputs), outputs = std::move(outputs),
       op_name_str = std::move(op_name_str), attrs = attrs.freeze(),
       output_type = output_type]() mutable {
        TFRTOpKernelConstruction op_kernel_construction(attrs);
        std::unique_ptr<TFRTOpKernel> op =
            tfrt_forwarding_kernel_factories->CreateKernel(
                op_name_str, &op_kernel_construction);

        // Forward kernel construction error.
        if (op_kernel_construction.error().hasValue()) {
          SetError(exec_ctx, &outputs,
                   op_kernel_construction.error().getValue());
          return;
        }

        const TFRTOpMeta* op_meta =
            tfrt_forwarding_op_meta_map->GetOpMeta(op_name_str);
        if (op_meta == nullptr) {
          SetError(exec_ctx, &outputs,
                   tfrt::StrCat("No TFRTOpMeta for op_name ", op_name_str));
          return;
        }

        TFRTOpKernelContext op_kernel_context(inputs, outputs.size(), op_meta,
                                              exec_ctx.host());
        op->Compute(&op_kernel_context);

        // Forward the context's error or outputs to raii_frame.
        if (op_kernel_context.error().hasValue()) {
          SetError(exec_ctx, &outputs, op_kernel_context.error().getValue());
          return;
        } else {
          for (int i = 0, e = outputs.size(); i != e; ++i) {
            // Expected result could be either a tensorflow::Tensor
            // (in case we call kernel directly), or KernelFallbackTensor
            // (if called from OpHandler).
            if (output_type == KernelFallbackOutputType::TENSOR) {
              outputs[i]->emplace<tensorflow::Tensor>(
                  op_kernel_context.output(i));
            } else {
              assert(output_type ==
                     KernelFallbackOutputType::KERNEL_FALLBACK_TENSOR);
              outputs[i]->emplace<KernelFallbackTensor>(
                  KernelFallbackTensor::Create(op_kernel_context.output(i)));
            }
          }
        }
      });

  return work_enqueued;
}
}  // namespace tfd
}  // namespace tensorflow
