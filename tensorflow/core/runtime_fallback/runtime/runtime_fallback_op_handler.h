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

// This file declares RuntimeFallbackOpHandler, responsible for running TFRT ops
// on Tensorflow.

#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_RUNTIME_FALLBACK_OP_HANDLER_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_RUNTIME_FALLBACK_OP_HANDLER_H_

#include <memory>

#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_handler.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

llvm::Expected<tfrt::OpHandler*> CreateRuntimeFallbackOpHandler(
    tfrt::CoreRuntime* runtime, tfrt::string_view tf_device_name);
}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_RUNTIME_FALLBACK_OP_HANDLER_H_
