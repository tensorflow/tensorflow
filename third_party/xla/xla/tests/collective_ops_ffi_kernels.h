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

#include <cstddef>

#include "xla/stream_executor/kernel.h"
#include "xla/types.h"  // IWYU pragma: keep

// Forward declare symmetric memory and device communicator as we don't want
// to include these headers into source code compiled by device compiler.
namespace xla {
class SymmetricMemory;
namespace gpu {
class GpuDeviceCommunicator;
}  // namespace gpu
}  // namespace xla

namespace xla::gpu {

// Simple LSA all-reduce kernel from NCCL documentation:
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#simple-lsa-kernel
struct CollectiveInPlaceAllReduce {
  using KernelType =
      se::TypedKernel<GpuDeviceCommunicator*, SymmetricMemory*, size_t, size_t>;
};

}  // namespace xla::gpu

#endif  // XLA_TESTS_COLLECTIVE_OPS_FFI_KERNELS_H_
