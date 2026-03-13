/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUB_PREFIX_SUM_KERNEL_CUDA_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUB_PREFIX_SUM_KERNEL_CUDA_H_

#include <cstddef>

#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/driver_types.h"

namespace stream_executor::cuda {

template <typename KeyT>
cudaError_t CubPrefixSum(void* d_temp_storage, size_t& temp_bytes,
                         const void* d_in, void* d_out, size_t num_items,
                         CUstream stream);

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUB_PREFIX_SUM_KERNEL_CUDA_H_
