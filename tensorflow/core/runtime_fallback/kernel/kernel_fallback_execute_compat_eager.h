/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_EXECUTE_COMPAT_EAGER_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_EXECUTE_COMPAT_EAGER_H_

#include <optional>

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

// Runner_table can be nullptr. In that case, kernel_fallback will use
// the default runner_table.
Status SetUpKernelFallbackCompatRequestContext(
    tfrt::RequestContextBuilder* builder,
    tfrt_stub::OpKernelRunnerTable* runner_table,
    tensorflow::EagerContext* eager_context,
    tensorflow::thread::ThreadPoolInterface* user_intra_op_threadpool = nullptr,
    const absl::optional<SessionMetadata>& model_metadata = std::nullopt);

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_EXECUTE_COMPAT_EAGER_H_
