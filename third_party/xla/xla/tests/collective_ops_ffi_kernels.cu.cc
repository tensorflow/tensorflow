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

#include "xla/tests/collective_ops_ffi_kernels.h"

#include <cstddef>
#include <cstdint>

#include "absl/base/casts.h"
#include "third_party/nccl/nccl.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/kernel_spec.h"

#if NCCL_VERSION_CODE >= 22800
// Device initiated collective operations were added in NCCL 2.28.0.
#include "third_party/nccl/nccl_device.h"
#endif  // NCCL_VERSION_CODE >= 22800

namespace xla::gpu {

#if NCCL_VERSION_CODE >= 22800
template <typename T>
static __global__ void NcclDevAllReduce(ncclDevComm dev_comm,
                                        ncclWindow_t src_win,
                                        ncclWindow_t dst_win, size_t src_offset,
                                        size_t dst_offset, size_t count) {
  ncclLsaBarrierSession<ncclCoopCta> bar(ncclCoopCta(), dev_comm,
                                         ncclTeamTagLsa(), blockIdx.x);
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  const int rank = dev_comm.lsaRank, nRanks = dev_comm.lsaSize;
  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;

  for (size_t o = globalTid; o < count; o += globalNthreads) {
    T v = 0;
    for (int peer = 0; peer < nRanks; peer++) {
      T* inputPtr =
          static_cast<T*>(ncclGetLsaPointer(src_win, src_offset, peer));
      v += inputPtr[o];
    }
    for (int peer = 0; peer < nRanks; peer++) {
      T* outputPtr =
          static_cast<T*>(ncclGetLsaPointer(dst_win, dst_offset, peer));
      outputPtr[o] = v;
    }
  }

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}
#else
template <typename T>
static __global__ void NcclDevAllReduce(void* dev_comm, ncclWindow_t src_win,
                                        ncclWindow_t dst_win, size_t src_offset,
                                        size_t dst_offset, size_t count) {
  // If device-initiated collectives are not supported, all reduce becomes a
  // no-op kernel. It's up to the caller to check that GPU communicator supports
  // device-initiated collective operations.
}
#endif

// A trivial all-reduce for S32 data type that uses multimem instructions.
//
// WARNING: This kernel doesn't have any barriers and it is a caller
// responsibility to make sure that data is ready on all ranks.
static __global__ void MulticastAllReduce(uint32_t* src_mmem, uint32_t* dst,
                                          size_t src_offset, size_t count) {
#if __CUDA_ARCH__ >= 900
  int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;

  for (int64_t i = offset; i < count; i += stride) {
    uint32_t data = 0;
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.u32 %0, [%1];"
                 : "=r"(data)
                 : "l"(src_mmem + src_offset + offset)
                 : "memory");
    dst[i] = data;
  }
#endif  // __CUDA_ARCH__ >= 900
}

// A trivial all-reduce for S32 data type that uses peer access.
//
// WARNING: This kernel doesn't have any barriers and it is a caller
// responsibility to make sure that data is ready on all ranks.
static __global__ void PeerAllReduce(uint32_t* src0, uint32_t* src1,
                                     uint32_t* dst, size_t count) {
  int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;

  for (int64_t i = offset; i < count; i += stride) {
    uint32_t data = src0[i] + src1[i];
    dst[i] = data;
  }
}

static se::KernelLoaderSpec SymmetricAllReduceKernelSpec(int32_t arity) {
  return se::KernelLoaderSpec::CreateInProcessSymbolSpec(
      absl::bit_cast<void*>(&NcclDevAllReduce<int32_t>),
      "SymmetricAllReduce_S32", arity);
}

static se::KernelLoaderSpec MulticastAllReduceKernelSpec(int32_t arity) {
  return se::KernelLoaderSpec::CreateInProcessSymbolSpec(
      absl::bit_cast<void*>(&MulticastAllReduce), "MulticastAllReduce_S32",
      arity);
}

static se::KernelLoaderSpec Peer2AllReduceKernelSpec(int32_t arity) {
  return se::KernelLoaderSpec::CreateInProcessSymbolSpec(
      absl::bit_cast<void*>(&PeerAllReduce), "Peer2AllReduce_S32", arity);
}

}  // namespace xla::gpu

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    CollectiveSymmetricAllReduce, xla::gpu::SymmetricAllReduce,
    stream_executor::cuda::kCudaPlatformId,
    xla::gpu::SymmetricAllReduceKernelSpec);

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    CollectiveMulticastAllReduce, xla::gpu::MultimemAllReduce,
    stream_executor::cuda::kCudaPlatformId,
    xla::gpu::MulticastAllReduceKernelSpec);

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    CollectivePeer2AllReduce, xla::gpu::Peer2AllReduce,
    stream_executor::cuda::kCudaPlatformId, xla::gpu::Peer2AllReduceKernelSpec);
