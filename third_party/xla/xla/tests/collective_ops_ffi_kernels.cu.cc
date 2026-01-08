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

#include "absl/base/casts.h"
#include "third_party/nccl/nccl.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/kernel_spec.h"

#if NCCL_VERSION_CODE >= 22800
// Device initiated collective operations were added in NCCL 2.28.0.
#include "third_party/nccl/nccl_device.h"
#endif  // NCCL_VERSION_CODE >= 22800

namespace xla::gpu {

bool SupportsCollectiveKernels() {
#if NCCL_VERSION_CODE >= 22800
  return true;  // NCCL_VERSION_CODE >= 22800
#endif
  return false;
}

#if NCCL_VERSION_CODE >= 22800
template <typename T>
static __global__ void InPlaceAllReduce(ncclDevComm dev_comm, ncclWindow_t win,
                                        size_t offset, size_t count) {
  ncclLsaBarrierSession<ncclCoopCta> bar(ncclCoopCta(), dev_comm,
                                         ncclTeamTagLsa(), blockIdx.x);
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  const int rank = dev_comm.lsaRank, nRanks = dev_comm.lsaSize;
  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;

  for (size_t o = globalTid; o < count; o += globalNthreads) {
    T v = 0;
    for (int peer = 0; peer < nRanks; peer++) {
      T* inputPtr = static_cast<T*>(ncclGetLsaPointer(win, offset, peer));
      v += inputPtr[o];
    }
    for (int peer = 0; peer < nRanks; peer++) {
      T* outputPtr = static_cast<T*>(ncclGetLsaPointer(win, offset, peer));
      outputPtr[o] = v;
    }
  }

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}
#else
template <typename T>
static __global__ void InPlaceAllReduce(void* dev_comm, ncclWindow_t win,
                                        size_t offset, size_t count) {
  // If device-initiated collectives are not supported, in-place all reduce
  // becomes a no-op kernel. It's up to the caller to check that XLA was
  // compiled with correct version of NCCL via `SupportsCollectiveKernels`.
}
#endif

static se::KernelLoaderSpec InPlaceAllReduceKernelSpec(int32_t arity) {
  return se::KernelLoaderSpec::CreateInProcessSymbolSpec(
      absl::bit_cast<void*>(&InPlaceAllReduce<float>), "InPlaceAllReducexf32",
      arity);
}

}  // namespace xla::gpu

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    CollectiveInPlaceAllReaduce, xla::gpu::CollectiveInPlaceAllReaduce,
    stream_executor::cuda::kCudaPlatformId,
    xla::gpu::InPlaceAllReduceKernelSpec);
