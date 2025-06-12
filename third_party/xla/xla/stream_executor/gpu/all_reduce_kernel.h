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
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"

namespace stream_executor::gpu {

// The maximum number of input pointers that can be passed to the all-reduce
// kernel.
inline constexpr int64_t kMaxNumAllReduceInputPtrs = 8;

// Defines a trait for the AllReduce kernel that can be used to register
// and look up the kernel in the GPU kernel registry.
template <typename ElementT, xla::ReductionKind ReductionKindT>
struct AllReduceKernel {
  using KernelType = stream_executor::TypedKernel<
      /*remove_input_ptrs=*/std::array<ElementT*, kMaxNumAllReduceInputPtrs>,
      /*local_input_ptr=*/stream_executor::DeviceMemoryBase,
      /*output_ptr=*/stream_executor::DeviceMemoryBase,
      /*rank=*/int64_t,
      /*num_ranks=*/int64_t,
      /*num_elements=*/int64_t,
      /*signal_flags_ptrs=*/std::array<uint32_t*, kMaxNumAllReduceInputPtrs>,
      /*signal_value=*/uint32_t>;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_ALL_REDUCE_KERNEL_H_
