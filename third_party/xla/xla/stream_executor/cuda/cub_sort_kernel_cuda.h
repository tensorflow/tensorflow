/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUB_SORT_KERNEL_CUDA_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUB_SORT_KERNEL_CUDA_H_

#include <cstddef>

#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

namespace stream_executor::cuda {

// The CUB sort kernel registration is split into 2 compilation units
// (cub_sort_kernel_cuda.cc and cub_sort_kernel_impl.cu.cc) because NVCC (prior
// to CUDA 12.4 Update 1) trips over the XLA FFI headers.
// Doing the FFI handle registration in a compilation unit that is not compiled
// by NVCC fixes the issue.
// The following functions declare the interface between the two compilation
// units.

template <typename KeyT>
cudaError_t CubSortKeys(void* d_temp_storage, size_t& temp_bytes,
                        const void* d_keys_in, void* d_keys_out,
                        size_t num_items, bool descending, size_t batch_size,
                        CUstream stream);

template <typename KeyT, typename ValT>
cudaError_t CubSortPairs(void* d_temp_storage, size_t& temp_bytes,
                         const void* d_keys_in, void* d_keys_out,
                         const void* d_values_in, void* d_values_out,
                         size_t num_items, bool descending, size_t batch_size,
                         CUstream stream);

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUB_SORT_KERNEL_CUDA_H_
