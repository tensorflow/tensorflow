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

#ifndef XLA_TESTS_COLLECTIVE_OPS_FFI_KERNELS_H_
#define XLA_TESTS_COLLECTIVE_OPS_FFI_KERNELS_H_

#include <cstdint>

#include "xla/stream_executor/kernel.h"
#include "xla/types.h"  // IWYU pragma: keep

namespace xla::gpu {

// Returns true if XLA was compiled with NCCL version that supports collective
// kernels.
bool SupportsCollectiveKernels();

// A type-erased wrappers for passing NCCL handles to CUDA kernels. We
// intentionally erase NCCL types, as we don't want to leak NCCL header.

struct NcclDeviceComm {
  void* device_comm;  // must be `ncclDevComm_t`
};

struct NcclWindow {
  void* window;  // must be `ncclWindow_t`
};

// At run time we rely on the C ABI which guarantees that our custom structs
// will be passed exactly as a single pointer to the CUDA kernel.
static_assert(sizeof(NcclDeviceComm) == sizeof(void*));
static_assert(sizeof(NcclWindow) == sizeof(void*));

// Simple LSA all-reduce kernel from NCCL documentation:
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#simple-lsa-kernel
struct CollectiveInPlaceAllReaduce {
  using KernelType =
      se::TypedKernel<NcclDeviceComm, NcclWindow, uint64_t, uint64_t>;
};

}  // namespace xla::gpu

#endif  // XLA_TESTS_COLLECTIVE_OPS_FFI_KERNELS_H_
