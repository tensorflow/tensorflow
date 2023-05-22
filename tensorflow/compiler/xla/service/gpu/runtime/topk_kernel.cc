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

// This file contains bespoke and optimized implementation for TopK shapes. When
// adding support for new shapes/dtypes, you also need to modify the rewritter
// on topk_specializer.cc for these changes to be picked up.

#include "tensorflow/compiler/xla/service/gpu/runtime/topk_kernel.h"

#include <algorithm>

#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/topk_kernel_common.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

using ::stream_executor::gpu::GpuStreamHandle;

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

// Helper type for converting the untyped arguments of RunTopk to TypedTopk
template <typename T>
struct TopkArgs {
  TopkArgs(GpuStreamHandle stream, PrimitiveType dtype, T* data,
           size_t num_elements, T* top_elements, uint32_t* top_indices,
           size_t k, size_t batch_size)
      : stream(stream),
        dtype(dtype),
        data(data),
        num_elements(num_elements),
        top_elements(top_elements),
        top_indices(top_indices),
        k(k),
        batch_size(batch_size) {}

  template <typename T2>
  TopkArgs<T2> Convert() const {
    return TopkArgs<T2>(stream, dtype, static_cast<T2*>(data), num_elements,
                        static_cast<T2*>(top_elements), top_indices, k,
                        batch_size);
  }

  GpuStreamHandle stream;
  PrimitiveType dtype;
  T* data;
  size_t num_elements;
  T* top_elements;
  uint32_t* top_indices;
  size_t k;
  size_t batch_size;
};

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
absl::Status TypedTopK(TopkArgs<T> args) {
  int num_threads = NumThreads(args.num_elements, args.k, args.batch_size);
  if (num_threads == 0) {
    return absl::FailedPreconditionError(
        "Invalid kernel pameters. This is likely a bug in the "
        "TopkSpecializer.");
  }
  absl::StatusOr<void*> kernel = GetKernel<T>(args.num_elements, args.k);
  if (!kernel.ok()) return kernel.status();
  int blocks_per_grid = args.batch_size;
  constexpr size_t max_kv_size = sizeof(uint64_t);
  // Allocate shmem assuming we have a full reduction.
  int shmem_size = absl::bit_ceil(args.k) * max_kv_size * 32;
  void* kernel_args[] = {&args.data, &args.num_elements, &args.top_elements,
                         &args.top_indices, &args.k};
  cudaError_t launch_status =
      cudaLaunchKernel(*kernel, blocks_per_grid, num_threads, kernel_args,
                       shmem_size, args.stream);
  if (launch_status != cudaSuccess) {
    return absl::InternalError(absl::StrCat("Failed to launch kernel: ",
                                            cudaGetErrorString(launch_status)));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status RunTopk(GpuStreamHandle stream, PrimitiveType dtype, void* data,
                     size_t num_elements, void* top_elements,
                     uint32_t* top_indices, size_t k, size_t batch_size) {
  VLOG(2) << "TopK: " << primitive_util::LowercasePrimitiveTypeName(dtype)
          << ", n: " << num_elements << ", k: " << k << ", bs: " << batch_size;
  auto args = TopkArgs<void>(stream, dtype, data, num_elements, top_elements,
                             top_indices, k, batch_size);
  switch (dtype) {
    case PrimitiveType::F32:
      return TypedTopK(args.Convert<float>());
    case PrimitiveType::BF16:
      return TypedTopK(args.Convert<Eigen::bfloat16>());
    default:
      return absl::UnimplementedError("GpuTopK not implemented for this dtype");
  }
}

}  // namespace xla::gpu
