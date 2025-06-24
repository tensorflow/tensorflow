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

#ifndef XLA_STREAM_EXECUTOR_GPU_ALL_REDUCE_KERNEL_H_
#define XLA_STREAM_EXECUTOR_GPU_ALL_REDUCE_KERNEL_H_

#include <array>
#include <cstdint>

#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/kernel.h"

namespace stream_executor::gpu {

enum class AllReduceStrategy : uint32_t {
  kOneShot,
  kTwoShot,
};

template <typename Sink>
void AbslStringify(Sink& sink, AllReduceStrategy strategy) {
  switch (strategy) {
    case AllReduceStrategy::kOneShot:
      sink.Append("kOneShot");
      break;
    case AllReduceStrategy::kTwoShot:
      sink.Append("kTwoShot");
      break;
  }
}

// The maximum number of input pointers that can be passed to the all-reduce
// kernel.
inline constexpr int64_t kMaxNumAllReduceInputPtrs = 8;
inline constexpr int64_t kNumElementsPerThread = 4;

// A pointer to a buffer that does not alias with other buffers.
template <typename U>
using RestrictedPtr = U* __restrict__;

template <typename T>
struct AllReduceKernelParams {
  // Shared buffers of all devices ordered by rank.
  std::array<RestrictedPtr<T>, kMaxNumAllReduceInputPtrs> remote_input_buffers;
  // Local buffer of the device.
  RestrictedPtr<T> input_buffer;
  // Output buffer of the device. Can be the same as the local input buffer in
  // case of aliasing.
  RestrictedPtr<T> output_buffer;
  // Local rank of the device.
  int64_t rank;
  // Number of participating ranks in the all-reduce.
  int64_t num_ranks;
  // Size of tensor on each device.
  int64_t num_elements;
  // Elements to be processed by each rank. This is equal to `num_elements` for
  // one-shot all-reduce and `num_elements / num_ranks` for two-shot
  // all-reduce.
  int64_t num_elements_per_rank;
  // Elements to be processed by each block.
  int64_t num_elements_per_block;
  // Signal flags buffers of all devices ordered by rank.
  std::array<RestrictedPtr<uint32_t>, kMaxNumAllReduceInputPtrs>
      signal_flags_buffers;
  // Value to be written to the signal flags. Should be different for different
  // invocations of the kernel with the same signal buffer.
  uint32_t signal_value;
};

// Defines a trait for the AllReduce kernel that can be used to register
// and look up the kernel in the GPU kernel registry.
template <typename ElementT, xla::ReductionKind ReductionKindT,
          AllReduceStrategy kAllReduceStrategy>
struct AllReduceKernel {
  using KernelType =
      stream_executor::TypedKernel<AllReduceKernelParams<ElementT>>;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_ALL_REDUCE_KERNEL_H_
