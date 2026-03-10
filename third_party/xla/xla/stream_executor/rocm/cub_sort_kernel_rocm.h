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

#ifndef XLA_STREAM_EXECUTOR_ROCM_CUB_SORT_KERNEL_ROCM_H_
#define XLA_STREAM_EXECUTOR_ROCM_CUB_SORT_KERNEL_ROCM_H_

#include <cstddef>

#include "absl/status/status.h"
#include "rocm/include/hip/hip_runtime.h"

namespace stream_executor {
namespace rocm {

// The CUB sort kernel registration is split into 2 compilation units
// (cub_sort_kernel_rocm.cc and cub_sort_kernel_rocm_impl.cu.cc) to separate
// the heavy hipCUB/rocPRIM template instantiations from the XLA FFI
// registration macros, which significantly speeds up compilation.
// The following functions declare the interface between the two compilation
// units.

template <typename KeyT>
absl::Status CubSortKeys(void* d_temp_storage, size_t& temp_bytes,
                         const void* d_keys_in, void* d_keys_out,
                         size_t num_items, bool descending, size_t batch_size,
                         hipStream_t stream);

template <typename KeyT, typename ValT>
absl::Status CubSortPairs(void* d_temp_storage, size_t& temp_bytes,
                          const void* d_keys_in, void* d_keys_out,
                          const void* d_values_in, void* d_values_out,
                          size_t num_items, bool descending, size_t batch_size,
                          hipStream_t stream);

}  // namespace rocm
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_CUB_SORT_KERNEL_ROCM_H_
