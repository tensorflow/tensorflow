/* Copyright 2026 The OpenXLA Authors.

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

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <memory>

#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/gpu/gpu_semaphore.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/rocm/delay_kernel.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"

namespace stream_executor::gpu {
namespace {

// Allocates the semaphore in host memory the device can read and write while a
// kernel is running.
//
// `hipHostMalloc(hipHostMallocPortable)`, which StreamExecutor's generic host
// allocation uses, is coarse-grained, so the device may hold the value in an L2
// that host writes do not invalidate. CDNA3/CDNA4 and RDNA4 lower `volatile` to
// a system-scope access and are unaffected. On CDNA1, CDNA2 and RDNA1-3 it is
// only an L1 bypass and the load is answered from the stale L2 line, and no
// cache bit makes a load skip L2 there. SLC does not, despite the name "System
// Level Coherent". It selects an L2 replacement policy, so a load carrying it
// can still hit a stale line (RDNA2 ISA table 38). Only CDNA2 can repair this
// from the kernel side, by invalidating L2 with BUFFER_INVL2, which CDNA1 and
// RDNA1-3 do not have.
//
// Requesting coherent, i.e. fine-grained, memory sidesteps all of that. It is
// not sufficient by itself, since a load carrying no cache bits at all still
// fails even with it, so the `volatile` below is doing the other half.
absl::StatusOr<GpuSemaphore> CreateCoherentSemaphore() {
  void* ptr = nullptr;
  ABSL_RETURN_IF_ERROR(
      ToStatus(hipHostMalloc(&ptr, sizeof(GpuSemaphoreState),
                             hipHostMallocPortable | hipHostMallocCoherent),
               "Failed to allocate coherent host memory for the delay kernel "
               "semaphore"));
  return GpuSemaphore::Create(std::make_unique<GenericMemoryAllocation>(
      ptr, sizeof(GpuSemaphoreState), [](void* location, uint64_t size) {
        hipError_t result = hipHostFree(location);
        if (result != hipSuccess) {
          LOG(ERROR) << "Failed to free delay kernel semaphore: "
                     << ToString(result);
        }
      }));
}

// Wait for the value pointed to by `semaphore` to have value `target`, timing
// out after approximately `timeout_ticks` wall clock ticks if that value is
// not reached. That happens when something stalls the host between launching
// this kernel and releasing it, e.g. the first launch of a not-yet-loaded
// kernel synchronising while its code object is loaded.
//
// Times with `wall_clock64()` rather than `clock64()`: only the former runs at
// a constant rate, reported by `hipDeviceAttributeWallClockRate`, which is what
// lets the host pass tick counts that mean the same duration on every device.
// `clock64()` reads a counter whose rate differs per part, and on some targets
// does not track the shader clock at all.
__global__ void DelayKernel(volatile GpuSemaphoreState* semaphore,
                            GpuSemaphoreState target, int64_t timeout_ticks,
                            int64_t poll_interval_ticks) {
  const int64_t tstart{wall_clock64()};
  bool target_not_reached{true};
  while ((target_not_reached = (*semaphore != target)) &&
         (wall_clock64() - tstart) < timeout_ticks) {
    int64_t elapsed{};
    const int64_t t0{wall_clock64()};
    do {
      elapsed = wall_clock64() - t0;
    } while (elapsed < poll_interval_ticks);
  }
  if (target_not_reached) {
    // We are exiting due to the timeout. Signal this back to the host so that
    // we can emit a warning, as it probably indicates suboptimal usage.
    *semaphore = GpuSemaphoreState::kTimedOut;
  }
}

// Used only when hipDeviceAttributeWallClockRate cannot be queried.
constexpr int64_t kFallbackWallClockHz = 100'000'000;

// Returns the frequency of the device's constant rate wall clock in Hz.
int64_t WallClockHz(int device_ordinal) {
  int rate_khz = 0;
  hipError_t result = hipDeviceGetAttribute(
      &rate_khz, hipDeviceAttributeWallClockRate, device_ordinal);
  if (result != hipSuccess || rate_khz <= 0) {
    LOG_FIRST_N(WARNING, 1)
        << "Could not query the wall clock rate of device " << device_ordinal
        << "; assuming " << kFallbackWallClockHz / 1'000'000
        << "MHz. The delay kernel timeout may be wrong.";
    return kFallbackWallClockHz;
  }
  return int64_t{rate_khz} * 1000;
}
}  // namespace

absl::StatusOr<GpuSemaphore> LaunchDelayKernel(Stream* stream) {
  StreamExecutor* executor = stream->parent();

  // Allocate a semaphore value that will be used to signal to the delay
  // kernel that it may exit. See CreateCoherentSemaphore for why this does not
  // go through StreamExecutor::HostMemoryAllocate.
  ABSL_ASSIGN_OR_RETURN(auto semaphore, CreateCoherentSemaphore());
  *semaphore = GpuSemaphoreState::kHold;
  ABSL_ASSIGN_OR_RETURN(
      auto kernel,
      (TypedKernelFactory<DeviceAddress<GpuSemaphoreState>, GpuSemaphoreState,
                          int64_t, int64_t>::Create(executor, "DelayKernel",
                                                    reinterpret_cast<void*>(
                                                        DelayKernel))));
  // This runs before the timer's start event is recorded, so the attribute
  // query is off the timed path.
  const int64_t wall_clock_hz = WallClockHz(executor->device_ordinal());
  // Launch a delay kernel into this stream, which will spin until
  // GetElapsedDuration() is called, the timer is destroyed, or the timeout
  // in the kernel is reached.
  ABSL_RETURN_IF_ERROR(
      kernel.Launch(ThreadDim(1, 1, 1), BlockDim(1, 1, 1), stream,
                    semaphore.device(), GpuSemaphoreState::kRelease,
                    /*timeout_ticks=*/wall_clock_hz / 10,  // 100ms
                    /*poll_interval_ticks=*/
                    std::max<int64_t>(wall_clock_hz / 1'000'000, 1)));  // 1us

  return semaphore;
}

}  // namespace stream_executor::gpu
