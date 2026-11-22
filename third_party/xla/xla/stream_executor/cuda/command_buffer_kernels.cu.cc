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

#include "xla/stream_executor/cuda/command_buffer_kernels.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda_device_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "xla/stream_executor/cuda/cudart_kernel_registry.h"
#include "xla/stream_executor/kernel_spec.h"

namespace stream_executor::cuda {
namespace {

// In all kernels defined below we set conditional handle value to `1` when we
// want to execute a CUDA graph tied to it, and to `0` otherwise. For loops, the
// graph will keep being executed until the conditional handle becomes `0`.

extern "C" __global__ void set_case_condition(
    cudaGraphConditionalHandle h0, cudaGraphConditionalHandle h1,
    cudaGraphConditionalHandle h2, cudaGraphConditionalHandle h3,
    cudaGraphConditionalHandle h4, cudaGraphConditionalHandle h5,
    cudaGraphConditionalHandle h6, cudaGraphConditionalHandle h7,
    const uint8_t* index, bool index_is_bool, int32_t index_offset,
    int32_t num_handles, bool enable_default_condition) {
  // Only handles in [0, num_handles) range are valid.
  //
  // index_offset specifies an offset that will be applied to the index value
  // to determine which conditional handle to set:
  //
  // effective_index = index - index_offset
  // handle[effective_index] = 1
  //
  // When enable_default_condition is true, the handle[num_handles-1] is set
  // for any index < 0 or effective_index >= num_handles.
  //
  // We can't define a device function with dynamic number of handle
  // arguments, so we always pass 8 handles, but only some of them are valid.
  // Size 8 picked as a reasonable (but random) upper bound for what we see
  // in XLA uses.
  cudaGraphConditionalHandle handles[8] = {h0, h1, h2, h3, h4, h5, h6, h7};

  // If branch index is out of range activate the last valid handle.
  const int32_t index_int32 = index_is_bool
                                  ? (*index != 0 ? 1 : 0)
                                  : *reinterpret_cast<const int32_t*>(index);

  int64_t effective_index =
      static_cast<int64_t>(index_int32) - static_cast<int64_t>(index_offset);
  if (enable_default_condition &&
      (index_int32 < 0 || effective_index >= num_handles)) {
    effective_index = num_handles - 1;
  }

  for (int32_t i = 0; i < num_handles; ++i) {
    if (effective_index == i) {
      cudaGraphSetConditional(handles[i], 1);
    } else {
      cudaGraphSetConditional(handles[i], 0);
    }
  }
}

extern "C" __global__ void set_while_condition(
    cudaGraphConditionalHandle handle, const bool* predicate) {
  if (*predicate) {
    cudaGraphSetConditional(handle, 1);
  } else {
    cudaGraphSetConditional(handle, 0);
  }
}

__global__ void NoOp() {}

}  // namespace

absl::StatusOr<KernelLoaderSpec> GetSetCaseConditionKernelLoaderSpec() {
  return FindCudaRuntimeKernel(&set_case_condition);
}

absl::StatusOr<KernelLoaderSpec> GetSetWhileConditionKernelLoaderSpec() {
  return FindCudaRuntimeKernel(&set_while_condition);
}

absl::StatusOr<KernelLoaderSpec> GetNoOpKernelLoaderSpec() {
  return FindCudaRuntimeKernel(&NoOp);
}

}  // namespace stream_executor::cuda
