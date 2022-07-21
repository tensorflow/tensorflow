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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TF_JITRT_REQUEST_CONTEXT_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TF_JITRT_REQUEST_CONTEXT_H_

#include "tensorflow/core/platform/status.h"
#include "tfrt/jitrt/jitrt.h"  // from @tf_runtime

namespace tensorflow {

struct TfJitRtRequestState {
  explicit TfJitRtRequestState(
      tfrt::jitrt::JitExecutableCache* jit_executable_cache)
      : jit_executable_cache(jit_executable_cache) {}

  // A pointer to the Jit Executable cache owned by the resource context.
  tfrt::jitrt::JitExecutableCache* jit_executable_cache;
};

// Sets up RequestContext with the JitRt state required for running `tf_jitrt`
// kernels (enables fast lookup of required resources in the request context).
Status SetUpTfJitRtRequestContext(tfrt::RequestContextBuilder* builder);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TF_JITRT_REQUEST_CONTEXT_H_
