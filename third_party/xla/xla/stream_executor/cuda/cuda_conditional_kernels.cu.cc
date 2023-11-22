/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "third_party/gpus/cuda/include/cuda.h"

namespace stream_executor {
namespace cuda {
namespace {

#if CUDA_VERSION >= 12030

__global__ void SetCondition(cudaGraphConditionalHandle handle,
                             bool* predicate) {
#if defined(XLA_GPU_USE_CUDA_GRAPH_CONDITIONAL)
  if (*predicate) {
    cudaGraphSetConditional(handle, 1);
  } else {
    cudaGraphSetConditional(handle, 0);
  }
#endif  // defined(XLA_GPU_USE_CUDA_GRAPH_CONDITIONAL)
}

#else
__global__ void SetCondition() {}
#endif  // CUDA_VERSION >= 12030

}  // namespace
}  // namespace cuda

namespace gpu {
void* GetSetConditionKernel() {
  return reinterpret_cast<void*>(&cuda::SetCondition);
}
}  // namespace gpu

}  // namespace stream_executor
