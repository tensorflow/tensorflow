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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_UTILS_HOST_CONTEXT_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_UTILS_HOST_CONTEXT_H_

#include <memory>

#include "absl/base/attributes.h"
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace tensorflow {

// The name of the default host device for running fallback kernels.
ABSL_CONST_INIT extern const char* const kDefaultHostDeviceName;

std::unique_ptr<tfrt::HostContext> CreateSingleThreadedHostContext();
std::unique_ptr<tfrt::HostContext> CreateMultiThreadedHostContext(
    int64_t num_threads);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_UTILS_HOST_CONTEXT_H_
