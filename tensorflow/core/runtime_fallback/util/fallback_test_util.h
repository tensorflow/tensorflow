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
#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_FALLBACK_TEST_UTIL_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_FALLBACK_TEST_UTIL_H_

#include "tensorflow/core/platform/threadpool_interface.h"
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

tfrt::ExecutionContext CreateFallbackTestExecutionContext(
    tfrt::HostContext* host, tfrt::ResourceContext* resource_context,
    tensorflow::thread::ThreadPoolInterface* user_intra_op_threadpool =
        nullptr);

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_FALLBACK_TEST_UTIL_H_
