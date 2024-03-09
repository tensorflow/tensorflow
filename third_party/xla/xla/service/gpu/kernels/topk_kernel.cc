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

// This file contains bespoke and optimized implementation for TopK shapes. When
// adding support for new shapes/dtypes, you also need to modify the rewriter
// on topk_specializer.cc for these changes to be picked up.

#include "xla/service/gpu/kernels/topk_kernel.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/kernels/topk_kernel_common.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

size_t NumThreads(size_t n, size_t k, size_t batch_size) {
  // Estimate number of threads per block that can run concurrently given the
  // register footprint.
  size_t simultaneous_threads_per_block = 512 * (16 / k);
  size_t threads_per_block =
      std::min(simultaneous_threads_per_block, kTopKMaxThreadsPerBlock);
  // Minimum amount of data that each thread needs to receive for the algorithm.
  size_t min_slice = absl::bit_floor(n / absl::bit_ceil(k));
  return std::min(threads_per_block, min_slice);
}

template <typename T>
absl::StatusOr<void*> GetKernel(int n, int k) {
  if (k <= 1) return GetTopKKernelForK<T, 1>(n);
  if (k <= 2) return GetTopKKernelForK<T, 2>(n);
  if (k <= 4) return GetTopKKernelForK<T, 4>(n);
  if (k <= 8) return GetTopKKernelForK<T, 8>(n);
  if (k <= 16) return GetTopKKernelForK<T, 16>(n);
  return absl::UnimplementedError(absl::StrCat("Unsupported K: ", k));
}

template <typename T>
absl::Status TypedTopK(se::Stream* stream, se::DeviceMemoryBase data,
                       size_t num_elements, se::DeviceMemoryBase top_elements,
                       se::DeviceMemoryBase top_indices, size_t k,
                       size_t batch_size) {
  constexpr size_t max_kv_size = sizeof(uint64_t);
  // Allocate shmem assuming we have a full reduction.
  int shmem_size = absl::bit_ceil(k) * max_kv_size * GetTopKWaveFrontSize<T>();
  int num_threads = NumThreads(num_elements, k, batch_size);
  if (num_threads == 0) {
    return absl::FailedPreconditionError(
        "Invalid kernel parameters. This is likely a bug in the "
        "TopkSpecializer.");
  }
  se::StreamExecutor* executor = stream->parent();
  se::DeviceMemory<T> data_typed(data);
  se::DeviceMemory<T> top_elements_typed(top_elements);
  se::DeviceMemory<uint32_t> top_indices_typed(top_indices);

  TF_ASSIGN_OR_RETURN(void* kernel_symbol, GetKernel<T>(num_elements, k));
  TF_ASSIGN_OR_RETURN(
      auto kernel,
      (se::TypedKernel<se::DeviceMemory<T>, size_t, se::DeviceMemory<T>,
                       se::DeviceMemory<uint32_t>,
                       size_t>::Create(executor, "topk", kernel_symbol)));

  TF_RETURN_IF_ERROR(stream->ThenLaunch(
      se::ThreadDim(num_threads, 1, 1), se::BlockDim(batch_size, 1, 1),
      shmem_size, kernel, data_typed, num_elements, top_elements_typed,
      top_indices_typed, k));

  return absl::OkStatus();
}

}  // namespace

absl::Status RunTopk(se::Stream* stream, PrimitiveType dtype,
                     se::DeviceMemoryBase data, size_t num_elements,
                     se::DeviceMemoryBase top_elements,
                     se::DeviceMemoryBase top_indices, size_t k,
                     size_t batch_size) {
  VLOG(2) << "TopK: " << primitive_util::LowercasePrimitiveTypeName(dtype)
          << ", n: " << num_elements << ", k: " << k << ", bs: " << batch_size;
  switch (dtype) {
    case PrimitiveType::F32:
      return TypedTopK<float>(stream, data, num_elements, top_elements,
                              top_indices, k, batch_size);
    case PrimitiveType::BF16:
      return TypedTopK<bfloat16>(stream, data, num_elements, top_elements,
                                 top_indices, k, batch_size);
    default:
      return absl::UnimplementedError("GpuTopK not implemented for this dtype");
  }
}

}  // namespace xla::gpu
