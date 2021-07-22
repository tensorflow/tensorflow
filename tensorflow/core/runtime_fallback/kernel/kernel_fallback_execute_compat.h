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
#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_EXECUTE_COMPAT_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_EXECUTE_COMPAT_H_

#include <optional>
#include <string>

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/tfrt/utils/model_metadata.h"
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime

namespace tfrt {
class SyncKernelFrame;
}  // namespace tfrt

namespace tensorflow {
namespace tfd {

// `builder`, `eager_context`, and `pflr` can't be null.
Status SetUpKernelFallbackCompatRequestContext(
    tfrt::RequestContextBuilder* builder,
    const tensorflow::DeviceMgr* device_manager,
    const tensorflow::ProcessFunctionLibraryRuntime* pflr,
    tensorflow::thread::ThreadPoolInterface* user_intra_op_threadpool = nullptr,
    const absl::optional<tfrt::ModelMetadata>& model_metadata = absl::nullopt);

// Runner_table can be nullptr. In that case, kernel_fallback will use
// the default runner_table.
Status SetUpKernelFallbackCompatRequestContext(
    tfrt::RequestContextBuilder* builder, OpKernelRunnerTable* runner_table,
    tensorflow::EagerContext* eager_context,
    tensorflow::thread::ThreadPoolInterface* user_intra_op_threadpool = nullptr,
    const absl::optional<tfrt::ModelMetadata>& model_metadata = absl::nullopt);

// The CoreRuntime dispatch function to run a TF kernel in kernel fallback
// compat mode.
tfrt::AsyncValueRef<tfrt::Chain> KernelFallbackExecuteCompatCoreRuntimeDispatch(
    const tfrt::ExecutionContext& exec_ctx, tfrt::string_view op_name,
    tfrt::string_view device_name, llvm::ArrayRef<tfrt::Tensor*> arguments,
    llvm::MutableArrayRef<tfrt::RCReference<tfrt::AsyncValue>> results,
    const tfrt::OpAttrsRef& attrs);

// `frame` is used to consume the inputs and hold the outputs from kernel
// execution.
//
// TODO(tfrt-devs): switch `attrs` to using tfrt::AggregateAttr after
// cl/343983780.
Status KernelFallbackSyncExecuteCompat(const tfrt::ExecutionContext& exec_ctx,
                                       absl::string_view op_name,
                                       absl::string_view device_name,
                                       tfrt::SyncKernelFrame* frame,
                                       const tfrt::OpAttrsRef& attrs);

// TODO(tfrt-devs): Consider moving following method to a separate file.
llvm::Expected<Device*> GetTfDevice(const tfrt::ExecutionContext& exec_ctx,
                                    const tfrt::Device& device);

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_EXECUTE_COMPAT_H_
