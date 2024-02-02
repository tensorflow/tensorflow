/* Copyright 2023 The OpenXLA Authors.

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

#include <array>
#include <cstdint>

#include "third_party/gpus/cuda/include/cuda.h"

namespace stream_executor {
namespace cuda {
namespace {

// In all kernels defined below we set conditional handle value to `1` when we
// want to execute a CUDA graph tied to it, and to `0` otherwise. For loops, the
// graph will keep being executed until the conditional handle becomes `0`.

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

__global__ void SetCaseCondition(
    cudaGraphConditionalHandle h0, cudaGraphConditionalHandle h1,
    cudaGraphConditionalHandle h2, cudaGraphConditionalHandle h3,
    cudaGraphConditionalHandle h4, cudaGraphConditionalHandle h5,
    cudaGraphConditionalHandle h6, cudaGraphConditionalHandle h7,
    int32_t* index, int32_t num_handles) {
  // Only handles in [0, num_handles) range are valid.
  //
  // We can't define a device function with dynamic number of handle arguments,
  // so we always pass 8 handles, but only some of them are valid. Size 8 picked
  // as a reasonable (but random) upper bound for what we see in XLA uses.
  std::array<cudaGraphConditionalHandle, 8> handles = {h0, h1, h2, h3,
                                                       h4, h5, h6, h7};

  // If branch index is out of range activate the last valid handle.
  int32_t branch_index = *index;
  if (branch_index < 0 || branch_index >= num_handles) {
    branch_index = num_handles - 1;
  }

  for (int32_t i = 0; i < num_handles; ++i) {
    if (branch_index == i) {
      cudaGraphSetConditional(handles[i], 1);
    } else {
      cudaGraphSetConditional(handles[i], 0);
    }
  }
}

__global__ void SetForCondition(cudaGraphConditionalHandle handle,
                                int32_t* loop_index, int32_t num_iterations) {
  if (*loop_index < num_iterations) {
    cudaGraphSetConditional(handle, 1);
  } else {
    cudaGraphSetConditional(handle, 0);
  }
  *loop_index += 1;
}

#else  // CUDA graph conditionals are not available

__global__ void SetIfCondition() {}
__global__ void SetIfElseCondition() {}
__global__ void SetCaseCondition() {}
__global__ void SetForCondition() {}

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

void* GetSetCaseConditionKernel() {
  return reinterpret_cast<void*>(&cuda::SetCaseCondition);
}

void* GetSetForConditionKernel() {
  return reinterpret_cast<void*>(&cuda::SetForCondition);
}

void* GetSetWhileConditionKernel() {
  // While condition kernel is the same as an `If` with a single branch.
  return reinterpret_cast<void*>(&cuda::SetIfCondition);
}

}  // namespace gpu

}  // namespace stream_executor
