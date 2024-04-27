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

#include "xla/service/gpu/kernels/topk_custom_kernel.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#include "xla/service/gpu/kernels/topk_kernel_common.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace xla::gpu::kernel::topk {

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)

namespace {

using KernelArgsPacking = se::MultiKernelLoaderSpec::KernelArgsPacking;

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
      std::min(simultaneous_threads_per_block, kTopKMaxThreadsPerBlock);
  // Minimum amount of data that each thread needs to receive for the algorithm.
  size_t min_slice = absl::bit_floor(n / absl::bit_ceil(k));
  return std::min(threads_per_block, min_slice);
}

// Gets the right version of TopK kernel based on the value of `k`.
template <typename T>
absl::StatusOr<void*> GetKernel(int n, int k) {
  if (k <= 1) return GetTopKKernelForK<T, 1>(n);
  if (k <= 2) return GetTopKKernelForK<T, 2>(n);
  if (k <= 4) return GetTopKKernelForK<T, 4>(n);
  if (k <= 8) return GetTopKKernelForK<T, 8>(n);
  if (k <= 16) return GetTopKKernelForK<T, 16>(n);
  return absl::UnimplementedError(absl::StrCat("Unsupported K: ", k));
}

// Returns the function creating packed arguments for TopK kernel.
template <typename T>
KernelArgsPacking CreateTopKArgsPacking(size_t num_elements, size_t k) {
  using Packed = absl::StatusOr<std::unique_ptr<se::KernelArgsPackedArrayBase>>;

  return [=](const se::Kernel& kernel, const se::KernelArgs& args) -> Packed {
    auto* mem_args = se::Cast<se::KernelArgsDeviceMemoryArray>(&args);

    se::DeviceMemory<T> data(mem_args->device_memory_args()[0]);
    se::DeviceMemory<T> top_elements(mem_args->device_memory_args()[1]);
    se::DeviceMemory<uint32_t> top_indices(mem_args->device_memory_args()[2]);

    return se::PackKernelArgs(args.number_of_shared_bytes(), data, num_elements,
                              top_elements, top_indices, k);
  };
}

// Implementation for creating a CustomKernel for TopK operation with element
// type `T`.
template <typename T>
absl::StatusOr<CustomKernel> GetTypedTopK(std::string name, size_t num_elements,
                                          size_t k, size_t batch_size) {
  constexpr size_t kMaxKVSize = sizeof(uint64_t);
  // Allocate shmem assuming we have a full reduction.
  int shmem_size = absl::bit_ceil(k) * kMaxKVSize * GetTopKWaveFrontSize<T>();
  int num_threads = EstimateOptimalNumThreads(num_elements, k, batch_size);
  if (num_threads == 0) {
    return absl::FailedPreconditionError(
        "Invalid kernel parameters. This is likely a bug in the "
        "TopkSpecializer.");
  }

  auto packing = CreateTopKArgsPacking<T>(num_elements, k);

  se::MultiKernelLoaderSpec spec(/*arity=*/5, std::move(packing));
  TF_ASSIGN_OR_RETURN(void* kernel_symbol, GetKernel<T>(num_elements, k));
  spec.AddInProcessSymbol(kernel_symbol, name);

  return CustomKernel(std::move(name), std::move(spec),
                      se::BlockDim(batch_size, 1, 1),
                      se::ThreadDim(num_threads, 1, 1), shmem_size);
}

}  // namespace

absl::StatusOr<CustomKernel> GetTopKKernel(std::string name,
                                           PrimitiveType dtype,
                                           size_t num_elements, size_t k,
                                           size_t batch_size) {
  switch (dtype) {
    case PrimitiveType::F32:
      return GetTypedTopK<float>(std::move(name), num_elements, k, batch_size);
    case PrimitiveType::BF16:
      return GetTypedTopK<bfloat16>(std::move(name), num_elements, k,
                                    batch_size);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported GpuTopK data type: ", dtype));
  }
}

#else

// Fallback implementation of creating a CustomKernel for TopK operation.
absl::StatusOr<CustomKernel> GetTopKKernel(std::string name,
                                           PrimitiveType dtype,
                                           size_t num_elements, size_t k,
                                           size_t batch_size) {
  return absl::InternalError("XLA compiled without CUDA support");
}

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace xla::gpu::kernel::topk
