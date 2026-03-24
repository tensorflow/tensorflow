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
#include <cstdint>

#include "xla/stream_executor/device_address.h"
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

// Simple LSA all-reduce kernel derived from NCCL documentation:
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#simple-lsa-kernel
struct SymmetricAllReduce {
  using KernelType = se::TypedKernel<GpuDeviceCommunicator*,  // dev_comm
                                     SymmetricMemory*,        // src_win
                                     SymmetricMemory*,        // dst_win
                                     size_t,                  // src_offset
                                     size_t,                  // dst_offset
                                     size_t>;                 // count
};

// Trivial multicast all-reduce for U32 data type without any barriers,
// the kernel assumes that data is ready when it is launched.
struct MultimemAllReduce {
  using KernelType =
      se::TypedKernel<stream_executor::DeviceAddress<uint32_t>,  // src_mmem
                      stream_executor::DeviceAddress<uint32_t>,  // dst_mmem
                      size_t,                                    // src_offset
                      size_t>;                                   // count
};

// Trivial peer all-reduce for U32 data type without any barriers,
// the kernel assumes that data is ready when it is launched.
struct Peer2AllReduce {
  using KernelType =
      se::TypedKernel<stream_executor::DeviceAddress<uint32_t>,  // src0
                      stream_executor::DeviceAddress<uint32_t>,  // src1
                      stream_executor::DeviceAddress<uint32_t>,  // dst
                      size_t>;                                   // count
};

}  // namespace xla::gpu

#endif  // XLA_TESTS_COLLECTIVE_OPS_FFI_KERNELS_H_
