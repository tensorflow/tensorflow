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

#include "xla/backends/gpu/runtime/topk.h"

#include <sys/types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/topk_kernel.h"
#include "xla/stream_executor/kernel_args_packing_spec.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::kernel::topk {

namespace {

// The optimal number of threads is the smaller value between the number of
// threads available per block and the number of slices of data.
size_t EstimateOptimalNumThreads(size_t n, size_t k, size_t batch_size) {
  // Estimate number of threads per block that can run concurrently given the
  // register footprint (k elements are kept in registers at all times).
  constexpr size_t kEstimatedThreadsPerBlock = 512;
  constexpr size_t kMaxKValue = 16;
  size_t simultaneous_threads_per_block =
      kEstimatedThreadsPerBlock * (kMaxKValue / k);
  size_t threads_per_block =
      std::min(simultaneous_threads_per_block,
               stream_executor::gpu::kTopKMaxThreadsPerBlock);
  // Minimum amount of data that each thread needs to receive for the algorithm.
  size_t min_slice = absl::bit_floor(n / absl::bit_ceil(k));
  return std::min(threads_per_block, min_slice);
}

// Returns a packing spec for invoking the TopK kernel.
se::KernelArgsPackingSpec CreateTopKArgsPacking(size_t num_elements, size_t k) {
  se::KernelArgsPackingSpec spec;
  spec.AddAddressArgument(0);  // data
  spec.AddConstantArgument(num_elements);
  spec.AddAddressArgument(1);  // top_elements
  spec.AddAddressArgument(2);  // top_indices
  spec.AddConstantArgument(k);
  return spec;
}

// Finds the TopK kernel for the given platform registered in the global
// registry.
template <size_t K, typename T, typename VT>
absl::StatusOr<se::KernelLoaderSpec> GetTopKKernelForPlatform(
    se::Platform::Id id) {
  return se::gpu::GpuKernelRegistry::GetGlobalRegistry()
      .FindKernel<se::gpu::TopKKernel<K, T, VT>>(id);
}

// Gets the right version of TopK kernel based on the value of `k`.
template <typename T, typename VT>
absl::StatusOr<se::KernelLoaderSpec> GetTopKKernelForKAndPlatform(
    size_t k, se::Platform::Id id) {
  if (k <= 1) {
    return GetTopKKernelForPlatform<1, T, VT>(id);
  }
  if (k <= 2) {
    return GetTopKKernelForPlatform<2, T, VT>(id);
  }
  if (k <= 4) {
    return GetTopKKernelForPlatform<4, T, VT>(id);
  }
  if (k <= 8) {
    return GetTopKKernelForPlatform<8, T, VT>(id);
  }
  if (k <= 16) {
    return GetTopKKernelForPlatform<16, T, VT>(id);
  }
  return absl::UnimplementedError(absl::StrCat("Unsupported K: ", k));
}

// Gets the right version of TopK kernel based on the value of `n`.
template <typename T>
absl::StatusOr<se::KernelLoaderSpec> GetTopKKernelForKAndPlatformAndN(
    size_t k, se::Platform::Id id, size_t n) {
  // TODO(doak): Switch to uint32_t if we don't have an efficient
  // implementation for uint16_t.
  if (n < std::numeric_limits<uint16_t>::max()) {
    return GetTopKKernelForKAndPlatform<T, uint16_t>(k, id);
  }
  return GetTopKKernelForKAndPlatform<T, uint32_t>(k, id);
}

// GetTopKKernelForKAndPlatformAndN specialization for float type.
template <>
absl::StatusOr<se::KernelLoaderSpec> GetTopKKernelForKAndPlatformAndN<float>(
    size_t k, se::Platform::Id id, size_t n) {
  // For float data on the H100, using uint32_t indices provides better overall
  // performance than uint16_t, even for smaller values of n.
  return GetTopKKernelForKAndPlatform<float, uint32_t>(k, id);
}

// Implementation for creating a CustomKernel for TopK operation with element
// type `T`.
template <typename T>
absl::StatusOr<CustomKernel> GetTypedTopK(std::string name, size_t num_elements,
                                          size_t k, size_t batch_size,
                                          absl::string_view platform_name,
                                          size_t wavefront_size) {
  constexpr size_t kMaxKVSize = sizeof(uint64_t);
  // Allocate shmem assuming we have a full reduction.
  int shmem_size = absl::bit_ceil(k) * kMaxKVSize * wavefront_size;
  int num_threads = EstimateOptimalNumThreads(num_elements, k, batch_size);
  if (num_threads == 0) {
    return absl::FailedPreconditionError(
        "Invalid kernel parameters. This is likely a bug in the "
        "TopkSpecializer.");
  }

  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      se::PlatformManager::PlatformWithName(platform_name));
  TF_ASSIGN_OR_RETURN(
      se::KernelLoaderSpec spec,
      GetTopKKernelForKAndPlatformAndN<T>(k, platform->id(), num_elements));

  spec.set_kernel_args_packing(CreateTopKArgsPacking(num_elements, k));
  return CustomKernel(std::move(name), std::move(spec),
                      se::BlockDim(batch_size, 1, 1),
                      se::ThreadDim(num_threads, 1, 1), shmem_size);
}

}  // namespace

absl::StatusOr<CustomKernel> GetTopKKernel(
    std::string name, PrimitiveType dtype, size_t num_elements, size_t k,
    size_t batch_size, absl::string_view platform_name, size_t wavefront_size) {
  switch (dtype) {
    case PrimitiveType::F32:
      return GetTypedTopK<float>(std::move(name), num_elements, k, batch_size,
                                 platform_name, wavefront_size);
    case PrimitiveType::BF16:
      return GetTypedTopK<bfloat16>(std::move(name), num_elements, k,
                                    batch_size, platform_name, wavefront_size);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported GpuTopK data type: ", dtype));
  }
}

}  // namespace xla::gpu::kernel::topk
