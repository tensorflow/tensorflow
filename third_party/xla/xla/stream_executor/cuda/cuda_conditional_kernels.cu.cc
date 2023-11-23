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

#include "third_party/gpus/cuda/include/cuda.h"

namespace stream_executor {
namespace cuda {
namespace {

#if defined(STREAM_EXECUTOR_CUDA_ENABLE_GRAPH_CONDITIONAL) && \
    CUDA_VERSION >= 12030

__global__ void SetIfCondition(cudaGraphConditionalHandle then_handle,
                               bool* predicate) {
  if (*predicate) {
    cudaGraphSetConditional(then_handle, 1);
  } else {
    cudaGraphSetConditional(then_handle, 0);
  }
}

__global__ void SetIfElseCondition(cudaGraphConditionalHandle then_handle,
                                   cudaGraphConditionalHandle else_handle,
                                   bool* predicate) {
  if (*predicate) {
    cudaGraphSetConditional(then_handle, 1);
    cudaGraphSetConditional(else_handle, 0);
  } else {
    cudaGraphSetConditional(then_handle, 0);
    cudaGraphSetConditional(else_handle, 1);
  }
}

#else  // CUDA graph conditionals are not available

__global__ void SetIfCondition() {}
__global__ void SetIfElseCondition() {}

#endif

}  // namespace
}  // namespace cuda

namespace gpu {

void* GetSetIfConditionKernel() {
  return reinterpret_cast<void*>(&cuda::SetIfCondition);
}

void* GetSetIfElseConditionKernel() {
  return reinterpret_cast<void*>(&cuda::SetIfElseCondition);
}

}  // namespace gpu

}  // namespace stream_executor
