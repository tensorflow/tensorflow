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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_KERNELS_IMPL_H_
#define XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_KERNELS_IMPL_H_

#include <cstddef>

#include "xla/xla_data.pb.h"

namespace xla::gpu::internal {

// Sort ordering for embeddable CUDA CUB radix sort kernels.
enum class SortDirection {
  kAscending,
  kDescending,
};

// Host function pointer and argument arity of an embeddable CUDA kernel.
struct KernelHostRef {
  const void* host_fun = nullptr;
  size_t arity = 0;
};

// All per-type sort kernels defined in a single sharded CUDA compilation unit.
struct CubSortTypeKernelRefs {
  KernelHostRef single_tile;
};

// Per-type kernel accessors explicitly instantiated across sharded CUDA
// compilation units in cub_sort_kernels_impl.cu.cc.
template <PrimitiveType kKeyType>
CubSortTypeKernelRefs GetCubSortKeysRefs(SortDirection direction);

template <PrimitiveType kKeyType, int kValueBitWidth>
CubSortTypeKernelRefs GetCubSortPairsRefs(SortDirection direction);

}  // namespace xla::gpu::internal

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_KERNELS_IMPL_H_
